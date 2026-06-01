"""Integration tests for the Phase 2 loop wiring.

Focus: the commit -> regression -> revert cycle that ties the anchor revert
rule to the population manager. This is the anti-collapse path; if the loop
commits a snapshot on a good anchor and restores it on a bad one, the
non-negotiable holds end to end.

Only `maybe_anchor_check` / `_apply_replacement` are exercised, so the other
five contracts are inert stubs.
"""

from si.anchor import HeldOutAnchor
from si.config import RunConfig
from si.contracts import Branch, EloState, Experience, Island, ReplacementPlan
from si.loop import Loop
from si.population import BranchManager


class _Stub:
    """Stands in for the contracts maybe_anchor_check doesn't touch."""


def _branch(bid: str) -> Branch:
    return Branch(branch_id=bid, lora_path=f"/init/{bid}", elo=1500.0, experience=Experience())


def _build(score_holder: dict[str, float]):
    branches = [_branch("a"), _branch("b")]
    mgr = BranchManager(branches, lora_root="/loras", reseed_fn=lambda p, d, s: None)
    anchor = HeldOutAnchor(runner=lambda b: score_holder[b.branch_id], benchmark_name="t")
    loop = Loop(
        config=RunConfig(run_id="test"),
        proposer=_Stub(),
        solver=_Stub(),
        verifier=_Stub(),
        selector=_Stub(),
        migrator=_Stub(),
        anchor=anchor,
        branch_manager=mgr,
    )
    elo = EloState(ratings={"a": 1500.0, "b": 1500.0})
    loop.state.elo = elo
    islands = [Island(island_id="i0", branches=branches, elo_state=elo)]
    return loop, mgr, branches, islands


def test_commit_snapshots_population():
    scores = {"a": 0.30, "b": 0.30}
    loop, mgr, _, islands = _build(scores)
    loop.state.gen = 10  # anchor_every default = 10
    assert loop.maybe_anchor_check(islands) is True
    assert 10 in mgr._snapshots
    assert loop.state.committed_gen == 10


def test_regression_triggers_revert_to_committed_snapshot():
    scores = {"a": 0.30, "b": 0.30}
    loop, _mgr, branches, islands = _build(scores)

    # gen 10: good anchor -> commit snapshot of the initial paths.
    loop.state.gen = 10
    assert loop.maybe_anchor_check(islands) is True
    assert branches[0].lora_path == "/init/a"

    # simulate a generation of drift: branch 'a' moved to a new adapter.
    branches[0].lora_path = "/loras/a/gen15"
    branches[0].experience.proposer_seeds.append(object())  # type: ignore[arg-type]

    # gen 20: anchor collapses -> should_revert fires -> restore gen-10 state.
    scores["a"], scores["b"] = 0.05, 0.05
    loop.state.gen = 20
    assert loop.maybe_anchor_check(islands) is False
    assert branches[0].lora_path == "/init/a"
    assert branches[0].experience.proposer_seeds == []
    # committed_gen stays at the last good generation, not the reverted one.
    assert loop.state.committed_gen == 10


def test_apply_replacement_reseeds_via_manager():
    scores = {"a": 0.30, "b": 0.30}
    loop, _mgr, branches, islands = _build(scores)
    loop.state.gen = 3
    plan = ReplacementPlan(keep=["a"], mutate=[], replace=[("b", "a")])
    out = loop._apply_replacement(islands, plan)
    assert out is islands
    assert branches[1].lora_path == "/loras/b/gen3"  # 'b' reseeded from 'a'
    assert loop.state.elo.ratings["b"] == 1500.0
