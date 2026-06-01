"""GPU smoke test for the Phase 2 inference loop.

Validates the novel/risky Phase 2 mechanics end to end on real hardware:
proposer -> resolve -> per-branch vLLM LoRA-swap solving -> verify -> Elo ->
reseed-on-replacement (real adapter perturbation). Training (grpo_update) is
DEFERRED here — on one 24GB GPU vLLM and Unsloth can't co-reside, so per-branch
training is a separate subprocess step (Phase 1's proven pattern). This test
proves the inference half turns over; trainer_fn just logs intent.

Run:  CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/phase2_smoke.py
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from si.config import RunConfig  # noqa: E402
from si.contracts import AnchorResult, Branch, EloState, Experience, Island  # noqa: E402
from si.elo import EloSelector  # noqa: E402
from si.islands import RingMigrator  # noqa: E402
from si.llm import GemmaLLM  # noqa: E402
from si.loop import Loop  # noqa: E402
from si.population import BranchManager  # noqa: E402
from si.phase2_ops import perturb_lora_adapter  # noqa: E402
from si.proposer import AZRProposer  # noqa: E402
from si.solver import GemmaSolver  # noqa: E402
from si.verifier import SandboxContainer, SandboxVerifier  # noqa: E402

log = logging.getLogger("phase2_smoke")

MODEL = str(REPO / "cache/gemma-4-E4B-hf")
SEED_ADAPTER = str(REPO / "runs/phase1_v2_20260423_2250/adapter/adapter")
SMOKE_ROOT = REPO / "runs/_phase2_smoke"


class _ConstAnchor:
    """Inert anchor — anchor_every keeps it from firing in a 1-gen smoke."""

    def evaluate(self, population: list[Branch]) -> AnchorResult:
        return AnchorResult(gen=-1, aggregate=1.0, per_branch={})

    def should_revert(self, prev: AnchorResult | None, curr: AnchorResult) -> bool:
        return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
    lora_root = SMOKE_ROOT / "loras"
    lora_root.mkdir(parents=True)

    # Two branches start from the same proven seed adapter; the loser gets
    # reseeded (perturbed) from the winner during replacement.
    branches = [
        Branch(branch_id=f"b{i}", lora_path=SEED_ADAPTER, elo=1500.0, experience=Experience())
        for i in range(2)
    ]
    island = Island(island_id="i0", branches=branches, elo_state=EloState())

    cfg = RunConfig(run_id="phase2_smoke")
    cfg.max_generations = 1
    cfg.proposer.proposals_per_generation = 4
    cfg.islands.enabled = False  # single island
    cfg.anchor.anchor_every = 10  # won't fire at gen 1

    # vLLM 0.19.1 can't serve LoRA on Gemma4ForConditionalGeneration, so the
    # proven path PEFT-merges the adapter into base weights and loads the merged
    # dir. At gen 0 all branches share the seed adapter, so one merged model is
    # correct here; a multi-gen live run needs per-branch merges in subprocesses
    # (the Phase 1 rollout/train pattern — vLLM + Unsloth can't co-reside on 24GB).
    import subprocess
    from hashlib import sha256

    h = sha256(Path(SEED_ADAPTER).resolve().as_posix().encode()).hexdigest()[:16]
    merged = f"/home/matilda/git/SI/cache/_merged/{h}"
    if not Path(merged, "config.json").exists():
        log.info("merging seed adapter -> %s", merged)
        subprocess.check_call(
            [sys.executable, str(REPO / "scripts/merge_and_anchor.py"),
             "--adapter", SEED_ADAPTER, "--merged-out", merged, "--skip-anchor"]
        )

    llm = GemmaLLM(merged, cuda_visible_devices="1")
    proposer = AZRProposer(llm, branch_id="prop", gen=0, temperature=0.8)
    container = SandboxContainer()
    verifier = SandboxVerifier(container=container)

    def solver_factory(branch: Branch) -> GemmaSolver:
        # gen-0 smoke: all branches share the merged seed model.
        return GemmaSolver(llm, branch_id=branch.branch_id, temperature=0.6)

    def trainer_fn(branch: Branch) -> None:
        log.info("[deferred] grpo_update for %s on %d wins (separate subprocess in a full run)",
                 branch.branch_id, len(branch.experience.recent_wins))

    mgr = BranchManager(
        branches, lora_root=str(lora_root), reseed_fn=perturb_lora_adapter,
        perturb_sigma=cfg.elo.mutation_sigma,
    )
    loop = Loop(
        config=cfg, proposer=proposer, solver=None, verifier=verifier,  # type: ignore[arg-type]
        selector=EloSelector(), migrator=RingMigrator(5, 2.0, lambda a, b: a),
        anchor=_ConstAnchor(), branch_manager=mgr,
        solver_factory=solver_factory, trainer_fn=trainer_fn,
    )

    try:
        out_islands = loop.run([island])
    finally:
        verifier.close()

    final = out_islands[0].branches
    ratings = loop.state.elo.ratings if loop.state.elo else {}
    print("\n===== PHASE 2 SMOKE RESULT =====")
    print(f"generation reached: {loop.state.gen}")
    print(f"Elo ratings: { {k: round(v, 1) for k, v in ratings.items()} }")
    for b in final:
        reseeded = b.lora_path != SEED_ADAPTER
        print(f"  {b.branch_id}: elo={b.elo:.1f} lora={'RESEEDED ' + b.lora_path if reseeded else 'seed'} "
              f"wins={len(b.experience.recent_wins)} losses={len(b.experience.recent_losses)}")
    # The loser should have been reseeded to a fresh perturbed adapter on disk.
    reseeded_paths = [b.lora_path for b in final if b.lora_path != SEED_ADAPTER]
    ok = bool(ratings) and any(Path(p, "adapter_model.safetensors").exists() for p in reseeded_paths)
    print(f"\nLOOP TURNED OVER + RESEED ON DISK: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
