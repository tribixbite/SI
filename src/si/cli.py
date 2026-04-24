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
    max_problems: int | None = typer.Option(None, help="subset size (None = all 164)"),
    model: str = typer.Option(DEFAULT_MODEL),
    out: str | None = typer.Option(None, help="optional JSON output path"),
) -> None:
    """Evaluate a branch (adapter or base) on HumanEval+."""
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
        from hashlib import sha256
        h = sha256(Path(adapter).resolve().as_posix().encode()).hexdigest()[:16]
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
        result = humaneval_plus_pass_at_1(
            llm, max_problems=max_problems, timeout_s=10.0, temperature=0.2
        )
        print(
            f"[green]HumanEval+ pass@1:[/green] "
            f"{result.passed}/{result.total} = {result.pass_at_1:.2%} "
            f"(wall {result.wall_s:.1f}s)"
        )
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(
                    {
                        "adapter": adapter,
                        "passed": result.passed,
                        "total": result.total,
                        "pass_at_1": result.pass_at_1,
                        "wall_s": result.wall_s,
                        "per_problem": result.per_problem,
                    },
                    f,
                    indent=2,
                )
            print(f"wrote {out}")
    finally:
        container.stop()


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


if __name__ == "__main__":
    app()
