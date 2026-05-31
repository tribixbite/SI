"""SI command-line interface.

Phase 1 is organized as three subcommands the shell orchestrator pings between:
    si rollout --out <dir> --gen <N> [--adapter <path>]
    si train   --outcomes <file> --adapter-out <path> [--adapter-in <path>]
    si anchor  [--adapter <path>] [--max-problems <N>]

Ping-pong is necessary because vLLM (for fast rollouts) and HF+PEFT+TRL
(for GRPO training) each want ~15 GB VRAM for Gemma 4 E4B — they can't
coexist on one 3090. Each subcommand is a fresh Python process.

Plus, convenience commands:
    si smoke   - scripts/smoke_e2e.py equivalent
    si status  - dump last anchor log
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import typer
from rich import print

app = typer.Typer(add_completion=False, no_args_is_help=True, help="SI — self-improving agent stack.")

DEFAULT_MODEL = os.environ.get("SI_MODEL_PATH", "/home/matilda/git/SI/cache/gemma-4-E4B-hf")
DEFAULT_TRAIN_MODEL = os.environ.get(
    "SI_TRAIN_MODEL_PATH", "/home/matilda/git/SI/cache/gemma-4-E4B-unsloth-4bit"
)


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("SI_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@app.command()
def rollout(
    out_dir: str = typer.Option(..., help="directory to write outcomes.jsonl"),
    gen: int = typer.Option(0, help="generation number (for metadata)"),
    proposals_per_type: int = typer.Option(8, help="deduction+abduction proposals per type"),
    mc_rollouts: int = typer.Option(4, help="MC rollouts per proposal for proposer reward"),
    model: str = typer.Option(DEFAULT_MODEL, help="Gemma 4 HF path"),
    adapter: str | None = typer.Option(None, help="LoRA adapter to load (optional)"),
) -> None:
    """Run one generation of AZR self-play: propose + solve + verify, write outcomes.jsonl."""
    _setup_logging()
    from si.llm import GemmaLLM
    from si.match import MatchRunner
    from si.proposer import AZRProposer
    from si.solver import GemmaSolver
    from si.verifier import SandboxContainer, SandboxVerifier

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[bold cyan]SI rollout[/bold cyan] gen={gen} model={model} adapter={adapter or '(base)'}")
    llm = GemmaLLM(model, cuda_visible_devices="1")
    # TODO(Phase 2): if adapter is not None, apply LoRA via llm.set_adapter(adapter)

    proposer = AZRProposer(llm, branch_id=f"p{gen}", gen=gen, temperature=0.8)
    solver = GemmaSolver(llm, branch_id=f"s{gen}", temperature=0.5)
    container = SandboxContainer()
    verifier = SandboxVerifier(container=container)

    runner = MatchRunner(
        proposer=proposer,
        solver=solver,
        verifier=verifier,
        mc_rollouts=mc_rollouts,
        proposals_per_type=proposals_per_type,
    )
    try:
        results = runner.run_generation()
    finally:
        verifier.close()

    outcomes_file = out_path / f"outcomes_gen{gen:04d}.jsonl"
    with outcomes_file.open("w") as f:
        for o in results.outcomes:
            record = {
                "task": asdict(o.task),
                "rollouts": [
                    {
                        "solution": asdict(r.solution),
                        "result": asdict(r.result),
                    }
                    for r in o.rollouts
                ],
                "pass_rate": o.pass_rate,
                "proposer_reward": o.proposer_reward,
            }
            # TaskType is an Enum; json.dumps via default str coercion
            record["task"]["task_type"] = o.task.task_type.value
            f.write(json.dumps(record) + "\n")

    metrics_file = out_path / f"metrics_gen{gen:04d}.json"
    with metrics_file.open("w") as f:
        json.dump(
            {
                "gen": gen,
                "n_outcomes": results.n_tasks,
                "failed_proposals": results.failed_proposals,
                "aggregate_pass_rate": results.aggregate_pass_rate,
                "aggregate_proposer_reward": results.aggregate_proposer_reward,
                "mc_difficulty_histogram": results.mc_difficulty_histogram(),
            },
            f,
            indent=2,
        )
    print(
        f"[green]wrote[/green] {outcomes_file} "
        f"(n={results.n_tasks}, agg_pass={results.aggregate_pass_rate:.2%}, "
        f"prop_reward={results.aggregate_proposer_reward:.3f}, failed={results.failed_proposals})"
    )


@app.command()
def train(
    outcomes: list[str] = typer.Option(..., help="one or more outcomes*.jsonl files"),
    adapter_out: str = typer.Option(..., help="where to save the LoRA adapter"),
    adapter_in: str | None = typer.Option(None, help="existing adapter to continue from (optional)"),
    model: str = typer.Option(DEFAULT_TRAIN_MODEL, help="Gemma 4 HF path (Unsloth 4-bit by default)"),
    epochs: int = typer.Option(1),
    lr: float = typer.Option(5e-5),
    lora_rank: int = typer.Option(32),
    vanilla: bool = typer.Option(False, help="use the non-Unsloth fallback trainer"),
) -> None:
    """Run one GRPO LoRA step on the solver using the given outcomes.

    Default: Unsloth FastVisionModel + TRL 0.24 GRPO recipe
    (Gemma 4 aware, eliminates the HF SDPA NaN bug, uses QLoRA 4-bit base).
    Pass --vanilla to use the older TRL 1.x path.
    """
    _setup_logging()
    from si.contracts import Task, TaskType
    from si.match import ProposalOutcome
    from si.verifier import SandboxContainer, SandboxVerifier

    print(f"[bold cyan]SI train[/bold cyan] n_files={len(outcomes)} adapter_out={adapter_out} vanilla={vanilla}")

    all_outcomes: list[ProposalOutcome] = []
    for path in outcomes:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                task_dict = rec["task"]
                task = Task(
                    task_type=TaskType(task_dict["task_type"]),
                    program=task_dict["program"],
                    input=task_dict["input"],
                    output=task_dict["output"],
                    proposer_branch_id=task_dict["proposer_branch_id"],
                    gen=task_dict["gen"],
                    task_id=task_dict["task_id"],
                )
                all_outcomes.append(ProposalOutcome(task=task, rollouts=[]))
    print(f"loaded {len(all_outcomes)} tasks")

    container = SandboxContainer()
    verifier = SandboxVerifier(container=container)
    try:
        if vanilla:
            from si.trainer import SITrainer, SITrainerConfig
            cfg = SITrainerConfig(
                model_path=model, output_dir=adapter_out, lora_rank=lora_rank, lr=lr
            )
            trainer = SITrainer(cfg, verifier=verifier)
            saved = trainer.train_on_generation(all_outcomes, epochs=epochs)
        else:
            from si.trainer_unsloth import UnslothSITrainer, UnslothTrainerConfig
            cfg = UnslothTrainerConfig(
                model_path=model,
                output_dir=adapter_out,
                lora_rank=lora_rank,
                lr=lr,
                epochs=epochs,
            )
            trainer = UnslothSITrainer(cfg, verifier=verifier)
            saved = trainer.train_on_generation(all_outcomes)
    finally:
        verifier.close()
    print(f"[green]adapter saved to[/green] {saved}")


@app.command()
def anchor(
    adapter: str | None = typer.Option(None, help="LoRA adapter (optional, base if omitted)"),
    max_problems: int | None = typer.Option(None, help="subset size (None = all)"),
    model: str = typer.Option(DEFAULT_MODEL),
    out: str | None = typer.Option(None, help="optional JSON output path"),
    benchmark: str = typer.Option("humaneval", help="humaneval | lcb | lcb-functional | lcb-stdin"),
    bon: int = typer.Option(1, help="Best-of-N: generate N candidates per problem; pass if any passes (LCB only)"),
    parallel_problems: int = typer.Option(8, help="LCB-only: number of problems verified concurrently (each runs sequentially internally)"),
    max_completion_tokens: int = typer.Option(1024, help="LCB-only: max generation tokens per candidate (raise for chain-of-thought models)"),
    chunk_size: int = typer.Option(0, help="LCB-only: split problems into chunks of this size for vLLM stability (0 = single batch)"),
    problem_offset: int = typer.Option(0, help="LCB-only: skip first N problems (for subprocess-chunked anchor)"),
    problem_limit: int | None = typer.Option(None, help="LCB-only: take only first N problems after offset (for subprocess-chunked anchor)"),
) -> None:
    """Evaluate a branch (adapter or base) on HumanEval+ or LiveCodeBench v6."""
    _setup_logging()
    from si.humaneval import humaneval_plus_pass_at_1
    from si.llm import GemmaLLM
    from si.verifier import SandboxContainer

    print(f"[bold cyan]SI anchor[/bold cyan] adapter={adapter or '(base)'} max_problems={max_problems}")
    # vLLM 0.19.1 doesn't support LoRA on Gemma4ForConditionalGeneration yet,
    # so we PEFT-merge adapter into base on disk first, then point vLLM at the
    # merged dir. Subsequent anchors on the same adapter reuse the merged dir.
    load_path = model
    if adapter:
        # Auto-detect nested PEFT dir (trainer saves at <adapter_out>/adapter/).
        adapter_dir = Path(adapter)
        if not (adapter_dir / "adapter_config.json").exists():
            if (adapter_dir / "adapter" / "adapter_config.json").exists():
                adapter_dir = adapter_dir / "adapter"
                adapter = str(adapter_dir)
                print(f"  resolved nested adapter path → {adapter}")
        from hashlib import sha256
        h = sha256(adapter_dir.resolve().as_posix().encode()).hexdigest()[:16]
        merged_path = f"/home/matilda/git/SI/cache/_merged/{h}"
        if not Path(merged_path, "config.json").exists():
            import subprocess, sys
            print(f"  merging adapter into base → {merged_path}")
            # Invoke via absolute path so the anchor command is not sensitive to cwd.
            merge_script = str(Path(__file__).resolve().parent.parent.parent / "scripts" / "merge_and_anchor.py")
            subprocess.check_call(
                [sys.executable, merge_script,
                 "--adapter", adapter, "--merged-out", merged_path, "--skip-anchor"]
            )
        load_path = merged_path

    container = SandboxContainer()
    container.start()
    try:
        llm = GemmaLLM(load_path, cuda_visible_devices="1")
        if benchmark == "humaneval":
            result = humaneval_plus_pass_at_1(
                llm, max_problems=max_problems, timeout_s=10.0, temperature=0.2
            )
            label = "HumanEval+"
            extra = {}
        elif benchmark in ("lcb", "lcb-functional", "lcb-stdin"):
            from si.livecodebench import lcb_pass_at_1
            tt = None
            if benchmark == "lcb-functional":
                tt = "functional"
            elif benchmark == "lcb-stdin":
                tt = "stdin"
            result = lcb_pass_at_1(
                llm,
                version="release_v6",
                max_problems=max_problems,
                testtype_filter=tt,
                temperature=0.2,
                n_candidates=bon,
                parallel_problems=parallel_problems,
                max_completion_tokens=max_completion_tokens,
                chunk_size=chunk_size,
                problem_offset=problem_offset,
                problem_limit=problem_limit,
            )
            label = f"LCB v6 ({tt or 'all'}, BoN={bon})"
            extra = {"per_difficulty": result.per_difficulty, "bon": bon}
        else:
            raise typer.BadParameter(f"unknown benchmark {benchmark!r}")
        print(
            f"[green]{label} pass@1:[/green] "
            f"{result.passed}/{result.total} = {result.pass_at_1:.2%} "
            f"(wall {result.wall_s:.1f}s)"
        )
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(
                    {
                        "adapter": adapter,
                        "benchmark": benchmark,
                        "passed": result.passed,
                        "total": result.total,
                        "pass_at_1": result.pass_at_1,
                        "wall_s": result.wall_s,
                        "per_problem": result.per_problem,
                        **extra,
                    },
                    f,
                    indent=2,
                )
            print(f"wrote {out}")
    finally:
        container.stop()


@app.command(name="ssd-sample")
def ssd_sample(
    outcomes: list[str] = typer.Option(..., help="one or more outcomes*.jsonl files — pool of tasks"),
    samples_out: str = typer.Option(..., help="output samples.jsonl path (passing candidates only)"),
    n_samples: int = typer.Option(16, help="candidates generated per task"),
    temperature: float = typer.Option(1.0, help="sampling temp (Unsloth Gemma 4 default)"),
    top_p: float = typer.Option(0.95),
    top_k: int = typer.Option(64),
    max_tokens: int = typer.Option(512),
    max_tasks: int | None = typer.Option(None, help="subsample tasks; None = all"),
    model: str = typer.Option(DEFAULT_MODEL, help="Gemma 4 HF path for vLLM sampling"),
    keep_failing: bool = typer.Option(False, help="also save failing samples to <samples_out>.failing.jsonl (Mix 2 DPO)"),
    min_pass_rate: float = typer.Option(0.0, help="drop tasks where pass_rate < this (Mix 1 difficulty stratification)"),
    max_pass_rate: float = typer.Option(1.0, help="drop tasks where pass_rate > this"),
) -> None:
    """Stage 1 of SSD: sample N candidates per task, keep verifier-passing ones."""
    _setup_logging()
    from si.contracts import TaskType
    from si.llm import GemmaLLM, GenParams
    from si.prompts import (
        SOLVER_SYSTEM_TEMPLATES,
        solver_abduction_prompt,
        solver_deduction_prompt,
    )
    from si.ssd import SSDSample, load_task_pool, verify_and_pack, write_samples
    from si.verifier import SandboxContainer, SandboxVerifier

    tasks = load_task_pool(outcomes)
    if max_tasks is not None and len(tasks) > max_tasks:
        tasks = tasks[:max_tasks]
    print(f"[bold cyan]SI ssd-sample[/bold cyan] {len(tasks)} unique tasks × {n_samples} candidates")

    # Load vLLM once; sample everything; close; filter (sandbox is CPU-bound).
    llm = GemmaLLM(model, cuda_visible_devices="1")
    params = GenParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n_samples)
    user_prompts = []
    system_prompts = []
    for t in tasks:
        if t.task_type is TaskType.DEDUCTION:
            user_prompts.append(solver_deduction_prompt(t.program, t.input))
        elif t.task_type is TaskType.ABDUCTION:
            user_prompts.append(solver_abduction_prompt(t.program, t.output))
        else:
            user_prompts.append("")  # induction not supported; will filter
        # Rotate across the 6-prompt pool deterministically so SSD's SFT sees
        # the same prompt-augmentation distribution the solver would at train time.
        system_prompts.append(SOLVER_SYSTEM_TEMPLATES[len(system_prompts) % len(SOLVER_SYSTEM_TEMPLATES)])

    # vLLM doesn't accept per-prompt system messages via chat_batch, so we
    # render each task separately. Still batched by vLLM's internal scheduling.
    print(f"  sampling {len(tasks) * n_samples} completions...")
    all_candidates: list[list[str]] = []
    batch = 64
    for i in range(0, len(tasks), batch):
        chunk_users = user_prompts[i : i + batch]
        chunk_syss = system_prompts[i : i + batch]
        # Group by system prompt so we issue one chat_batch call per distinct system.
        by_sys: dict[str, list[int]] = {}
        for j, s in enumerate(chunk_syss):
            by_sys.setdefault(s, []).append(j)
        chunk_out: list[list[str] | None] = [None] * len(chunk_users)
        for sys_text, idxs in by_sys.items():
            subset = [chunk_users[j] for j in idxs]
            outs = llm.chat_batch(subset, params, system=sys_text)
            for local_j, global_j in enumerate(idxs):
                chunk_out[global_j] = outs[local_j]
        all_candidates.extend(chunk_out)  # type: ignore[arg-type]
        print(f"  ...{min(i + batch, len(tasks))}/{len(tasks)} tasks sampled")

    # Free vLLM before booting sandbox (it doesn't really matter VRAM-wise,
    # but clean lifecycle avoids orphaned EngineCore workers).
    del llm

    # Verify + pack.
    container = SandboxContainer()
    verifier = SandboxVerifier(container=container)
    all_passing: list[SSDSample] = []
    all_failing: list[SSDSample] = []
    try:
        for i, task in enumerate(tasks):
            cands = all_candidates[i] or []
            packed = verify_and_pack(
                task=task,
                candidates=cands,
                verifier=verifier,
                system_prompt=system_prompts[i],
                user_prompt=user_prompts[i],
                keep_failing=keep_failing,
            )
            if keep_failing:
                p, f = packed
                all_passing.extend(p)
                all_failing.extend(f)
            else:
                all_passing.extend(packed)
            if (i + 1) % 32 == 0:
                print(f"  verified {i+1}/{len(tasks)} tasks; kept {len(all_passing)} pass / {len(all_failing)} fail")
    finally:
        verifier.close()

    # Difficulty stratification (Mix 1 E): drop tasks where pass_rate falls outside [min, max].
    if min_pass_rate > 0 or max_pass_rate < 1.0:
        per_task_pass: dict[str, list[SSDSample]] = {}
        per_task_fail: dict[str, list[SSDSample]] = {}
        for s in all_passing:
            per_task_pass.setdefault(s.task_id, []).append(s)
        for s in all_failing:
            per_task_fail.setdefault(s.task_id, []).append(s)
        kept_pass: list[SSDSample] = []
        kept_fail: list[SSDSample] = []
        kept_tasks = 0
        for tid in set(per_task_pass) | set(per_task_fail):
            np = len(per_task_pass.get(tid, []))
            nf = len(per_task_fail.get(tid, []))
            total = np + nf if (np + nf) > 0 else n_samples
            pr = np / max(1, total)
            if min_pass_rate <= pr <= max_pass_rate:
                kept_pass.extend(per_task_pass.get(tid, []))
                kept_fail.extend(per_task_fail.get(tid, []))
                kept_tasks += 1
        print(
            f"  difficulty stratification [{min_pass_rate}, {max_pass_rate}]: "
            f"{kept_tasks} tasks kept, {len(kept_pass)}/{len(all_passing)} passing, "
            f"{len(kept_fail)}/{len(all_failing)} failing"
        )
        all_passing, all_failing = kept_pass, kept_fail

    write_samples(all_passing, samples_out)
    if keep_failing and all_failing:
        write_samples(all_failing, samples_out + ".failing.jsonl")
    pass_rate = len(all_passing) / max(1, len(tasks) * n_samples)
    extra = f" + {len(all_failing)} failing" if keep_failing else ""
    print(
        f"[green]wrote[/green] {samples_out} "
        f"({len(all_passing)} passing{extra} / {len(tasks)*n_samples} generated = {pass_rate:.2%})"
    )


@app.command(name="ssd-train")
def ssd_train(
    samples: str = typer.Option(..., help="samples.jsonl from ssd-sample"),
    adapter_out: str = typer.Option(..., help="output adapter directory"),
    model: str = typer.Option(DEFAULT_TRAIN_MODEL, help="Gemma 4 HF 4-bit path for SFT"),
    epochs: int = typer.Option(2),
    lr: float = typer.Option(2e-5),
    lora_rank: int = typer.Option(32),
    lora_dropout: float = typer.Option(0.05),
    max_steps: int = typer.Option(-1, help="if > 0 overrides epochs"),
    warm_start_adapter: str | None = typer.Option(None, help="optional adapter to warm-start from (e.g. ssd_v7)"),
    use_dora: bool = typer.Option(False, help="DoRA (Weight-Decomposed LoRA, Unsloth fused kernels)"),
) -> None:
    """Stage 2 of SSD: SFT on verifier-passing samples via Unsloth FastModel."""
    _setup_logging()
    from si.ssd import read_samples
    from si.trainer_ssd import SSDTrainer, SSDTrainerConfig

    all_samples = read_samples(samples)
    print(f"[bold cyan]SI ssd-train[/bold cyan] {len(all_samples)} samples dora={use_dora}")
    cfg = SSDTrainerConfig(
        model_path=model,
        output_dir=adapter_out,
        lora_rank=lora_rank,
        lr=lr,
        epochs=epochs,
        lora_dropout=lora_dropout,
        max_steps=max_steps,
        use_dora=use_dora,
    )
    trainer = SSDTrainer(cfg, warm_start_adapter=warm_start_adapter)
    saved = trainer.train_on_samples(all_samples)
    print(f"[green]adapter saved to[/green] {saved}")


@app.command(name="proposer-sample")
def proposer_sample(
    samples_out: str = typer.Option(..., help="output proposer training samples.jsonl"),
    proposals_per_type: int = typer.Option(64, help="tasks generated per type per gen"),
    mc_rollouts: int = typer.Option(4, help="solver MC rollouts per task"),
    n_gens: int = typer.Option(4, help="repeat the propose→score loop this many times"),
    min_pass_rate: float = typer.Option(0.3),
    max_pass_rate: float = typer.Option(0.7),
    model: str = typer.Option(DEFAULT_MODEL, help="Gemma 4 path for vLLM"),
) -> None:
    """Mix 1-A stage 1: collect medium-difficulty proposer tasks via MC scoring."""
    _setup_logging()
    from si.llm import GemmaLLM
    from si.proposer_train import build_match_runner, collect_proposer_training_pairs
    from si.ssd import write_samples
    from si.verifier import SandboxContainer, SandboxVerifier

    print(f"[bold cyan]SI proposer-sample[/bold cyan] {n_gens} gens × {proposals_per_type} tasks/type × {mc_rollouts} MC")
    llm = GemmaLLM(model, cuda_visible_devices="1")
    container = SandboxContainer()
    verifier = SandboxVerifier(container=container)
    all_pairs = []
    try:
        for g in range(n_gens):
            runner = build_match_runner(llm=llm, verifier=verifier, gen=g)
            pairs = collect_proposer_training_pairs(
                runner,
                proposals_per_type=proposals_per_type,
                mc_rollouts=mc_rollouts,
                min_pass_rate=min_pass_rate,
                max_pass_rate=max_pass_rate,
            )
            all_pairs.extend(pairs)
            print(f"  gen {g+1}/{n_gens}: kept {len(pairs)} (cum {len(all_pairs)})")
    finally:
        verifier.close()
    write_samples(all_pairs, samples_out)
    print(f"[green]wrote[/green] {samples_out} ({len(all_pairs)} medium-difficulty proposer pairs)")


@app.command(name="dpo-train")
def dpo_train(
    samples_passing: str = typer.Option(..., help="passing samples.jsonl"),
    samples_failing: str = typer.Option(..., help="failing samples.jsonl"),
    adapter_out: str = typer.Option(..., help="output adapter dir"),
    warm_start_adapter: str | None = typer.Option(None, help="optional adapter to load as starting point (e.g. ssd_v5)"),
    model: str = typer.Option(DEFAULT_TRAIN_MODEL),
    epochs: int = typer.Option(1),
    max_steps: int = typer.Option(-1),
    lr: float = typer.Option(5e-6),
    beta: float = typer.Option(0.1),
    max_pairs_per_task: int = typer.Option(4),
) -> None:
    """Mix 2 stage 2: DPO on (passing, failing) preference pairs."""
    _setup_logging()
    from si.dpo import build_dpo_pairs, write_dpo_jsonl
    from si.ssd import read_samples
    from si.trainer_dpo import DPOSITrainer, DPOTrainerConfig

    passing = read_samples(samples_passing)
    failing = read_samples(samples_failing)
    print(f"[bold cyan]SI dpo-train[/bold cyan] {len(passing)} pass / {len(failing)} fail")
    pairs = build_dpo_pairs(passing, failing, max_pairs_per_task=max_pairs_per_task)
    write_dpo_jsonl(pairs, str(Path(adapter_out).parent / "dpo_pairs.jsonl"))
    print(f"  built {len(pairs)} preference pairs")
    cfg = DPOTrainerConfig(
        model_path=model,
        output_dir=adapter_out,
        epochs=epochs,
        max_steps=max_steps,
        lr=lr,
        beta=beta,
    )
    trainer = DPOSITrainer(cfg, warm_start_adapter=warm_start_adapter)
    saved = trainer.train_on_pairs(pairs)
    print(f"[green]DPO adapter saved to[/green] {saved}")


@app.command()
def smoke() -> None:
    """Phase 0 smoke — load base Gemma 4, generate, verify. Equivalent to scripts/smoke_e2e.py."""
    import runpy
    runpy.run_path("scripts/smoke_e2e.py", run_name="__main__")


@app.command()
def status() -> None:
    """Tail anchor_log.csv (if present) and the most recent run's metrics."""
    p = Path("anchor_log.csv")
    if p.exists():
        print("[bold]anchor_log.csv (last 10):[/bold]")
        lines = p.read_text().strip().splitlines()[-10:]
        for line in lines:
            print("  ", line)
    runs = sorted(Path("runs").glob("*/metrics*.json")) if Path("runs").exists() else []
    if runs:
        latest = runs[-1]
        print(f"[bold]{latest}:[/bold]")
        print(json.loads(latest.read_text()))
    if not p.exists() and not runs:
        print("no anchor_log.csv or runs/*/metrics yet")


# ---- Phase 1 legacy stubs kept for scripts/01_phase1_run.sh compatibility ----


@app.command()
def phase1(config: str = typer.Option("configs/tier1_e4b.yaml")) -> None:
    """Phase 1 orchestrator — currently driven by scripts/phase1_loop.sh, not this command."""
    print("[yellow]Use scripts/phase1_loop.sh for the Phase 1 ping-pong orchestration.[/yellow]")
    raise typer.Exit(code=1)


@app.command()
def phase2(config: str = typer.Option("configs/tier2_26b.yaml")) -> None:
    """Phase 2: multi-branch Elo tournament. Not yet implemented."""
    print("Not yet implemented. See docs/04-implementation.md §2.")
    raise typer.Exit(code=1)


@app.command(name="lcb-merge")
def lcb_merge(
    inputs: list[str] = typer.Option(..., "--input", help="LCB anchor JSON file (repeat)"),
    out: str = typer.Option(..., help="merged output JSON path"),
) -> None:
    """Merge multiple chunked LCB anchor JSONs into a single result."""
    import json
    from pathlib import Path

    merged_per_problem: dict[str, bool] = {}
    per_diff: dict[str, list[int]] = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0], "unknown": [0, 0]}
    total_wall = 0.0
    bon_vals: set[int] = set()
    benchmarks: set[str] = set()
    adapter_seen: str | None = None
    # We need difficulty for each problem id; load it from the LCB dataset.
    from si.livecodebench import load_lcb
    all_probs = load_lcb("release_v6", "/home/matilda/git/SI/cache/livecodebench")
    diff_by_pid = {p.problem_id: p.difficulty for p in all_probs}
    for path in inputs:
        d = json.load(open(path))
        merged_per_problem.update(d.get("per_problem", {}))
        total_wall += d.get("wall_s", 0.0)
        if "bon" in d:
            bon_vals.add(d["bon"])
        benchmarks.add(d.get("benchmark", ""))
        if d.get("adapter"):
            adapter_seen = d["adapter"]
    passed = sum(1 for v in merged_per_problem.values() if v)
    total = len(merged_per_problem)
    for pid, ok in merged_per_problem.items():
        diff = diff_by_pid.get(pid, "unknown")
        if diff not in per_diff:
            diff = "unknown"
        per_diff[diff][1] += 1
        if ok:
            per_diff[diff][0] += 1
    out_d = {
        "adapter": adapter_seen,
        "benchmark": "+".join(sorted(benchmarks)),
        "passed": passed,
        "total": total,
        "pass_at_1": passed / max(1, total),
        "wall_s": total_wall,
        "per_problem": merged_per_problem,
        "per_difficulty": {k: v for k, v in per_diff.items() if v[1] > 0},
        "bon": (next(iter(bon_vals)) if len(bon_vals) == 1 else sorted(bon_vals)),
        "merged_from": inputs,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_d, open(out, "w"), indent=2)
    print(f"[green]merged {len(inputs)} files → {out}[/green]: {passed}/{total} = {passed/max(1,total)*100:.2f}%")


if __name__ == "__main__":
    app()
