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
from si.contracts import (
    Branch,
    EloState,
    Experience,
    Island,
    ReplacementPlan,
    Solution,
    Task,
    TaskType,
    VerifyResult,
)
from si.elo import EloSelector
from si.loop import Loop
from si.population import BranchManager


class _Stub:
    """Stands in for the contracts maybe_anchor_check doesn't touch."""


class _FakeSolver:
    def __init__(self, branch_id: str) -> None:
        self.branch_id = branch_id

    def solve(self, task: Task) -> Solution:
        return self.solve_batch([task])[0]

    def solve_batch(self, tasks: list[Task]) -> list[Solution]:
        return [
            Solution(task_id=t.task_id, solver_branch_id=self.branch_id, body="", trace="",
                     walltime_ms=0)
            for t in tasks
        ]


class _FakeVerifier:
    """Passes iff pass_map[(branch_id, task_id)] is True."""

    def __init__(self, pass_map: dict[tuple[str, str], bool]) -> None:
        self.pass_map = pass_map

    def verify(self, task: Task, sol: Solution) -> VerifyResult:
        p = self.pass_map.get((sol.solver_branch_id, task.task_id), False)
        return VerifyResult(task_id=task.task_id, solution_id=sol.task_id, passed=p,
                            stdout="", stderr="", walltime_ms=0, exit_code=0)


def _task(tid: str) -> Task:
    return Task(task_type=TaskType.DEDUCTION, program="p", input="i", output="o",
                proposer_branch_id="x", gen=0, task_id=tid)


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


# ---- rollout methods (solve_and_verify / run_matches / grpo_update) ----------


def _rollout_loop(branches, pass_map, *, selector=None, trainer_fn=None):
    solver_calls: list[str] = []

    def solver_factory(branch):
        solver_calls.append(branch.branch_id)
        return _FakeSolver(branch.branch_id)

    loop = Loop(
        config=RunConfig(run_id="t"),
        proposer=_Stub(),
        solver=_Stub(),
        verifier=_FakeVerifier(pass_map),
        selector=selector or _Stub(),
        migrator=_Stub(),
        anchor=_Stub(),
        solver_factory=solver_factory,
        trainer_fn=trainer_fn,
    )
    return loop, solver_calls


def test_solve_and_verify_records_experience_and_returns_passed():
    br = Branch(branch_id="a", lora_path="/x", elo=1500.0, experience=Experience())
    tasks = [_task("t1"), _task("t2")]
    pass_map = {("a", "t1"): True, ("a", "t2"): False}
    loop, _ = _rollout_loop([br], pass_map)

    passed = loop.solve_and_verify(tasks, br)
    assert passed == {"t1": True, "t2": False}
    assert len(br.experience.recent_wins) == 1
    assert len(br.experience.recent_losses) == 1


def test_run_matches_scores_pairs_and_solves_each_branch_once():
    branches = [
        Branch(branch_id="a", lora_path="/a", elo=1500.0, experience=Experience()),
        Branch(branch_id="b", lora_path="/b", elo=1500.0, experience=Experience()),
        Branch(branch_id="c", lora_path="/c", elo=1500.0, experience=Experience()),
    ]
    tasks = [_task("t1"), _task("t2")]
    # a always passes, b always fails, c passes only t1.
    pass_map = {
        ("a", "t1"): True, ("a", "t2"): True,
        ("b", "t1"): False, ("b", "t2"): False,
        ("c", "t1"): True, ("c", "t2"): False,
    }
    loop, solver_calls = _rollout_loop(branches, pass_map)
    island = Island(island_id="i0", branches=branches, elo_state=EloState())
    loop.state.gen = 1

    matches = loop.run_matches(tasks, [island])
    # mult defaults to 3.0 → 3*3 = 9 matches.
    assert len(matches) == 9
    # each branch solved the task pool exactly once (one LoRA load per branch).
    assert sorted(solver_calls) == ["a", "b", "c"]
    # winner logic is consistent with pass_map for every emitted match.
    for m in matches:
        pa = pass_map[(m.branch_a, m.task_id)]
        pb = pass_map[(m.branch_b, m.task_id)]
        if pa == pb:
            assert m.winner is None
        else:
            assert m.winner == (m.branch_a if pa else m.branch_b)


def test_run_matches_empty_or_singleton_yields_nothing():
    br = Branch(branch_id="a", lora_path="/a", elo=1500.0, experience=Experience())
    loop, _ = _rollout_loop([br], {})
    assert loop.run_matches([], [Island("i0", [br], EloState())]) == []
    assert loop.run_matches([_task("t1")], [Island("i0", [br], EloState())]) == []


def test_grpo_update_trains_keep_and_mutate_skips_replaced():
    trained: list[str] = []
    # 4 branches with distinct ratings → quartile size 1: keep=[top], replace=[bottom], mutate=[2 mid].
    branches = [
        Branch(branch_id=b, lora_path=f"/{b}", elo=1500.0, experience=Experience())
        for b in ("a", "b", "c", "d")
    ]
    loop, _ = _rollout_loop(branches, {}, selector=EloSelector(),
                            trainer_fn=lambda br: trained.append(br.branch_id))
    loop.state.elo = EloState(ratings={"a": 1700.0, "b": 1550.0, "c": 1450.0, "d": 1300.0})
    island = Island(island_id="i0", branches=branches, elo_state=loop.state.elo)

    loop.grpo_update([island])
    # top 'a' (keep) + middle 'b','c' (mutate) trained; bottom 'd' (replaced) skipped.
    assert sorted(trained) == ["a", "b", "c"]
    assert "d" not in trained


def test_grpo_update_without_trainer_raises():
    br = Branch(branch_id="a", lora_path="/a", elo=1500.0, experience=Experience())
    loop, _ = _rollout_loop([br], {}, selector=EloSelector())
    loop.state.elo = EloState(ratings={"a": 1500.0})
    import pytest

    with pytest.raises(RuntimeError):
        loop.grpo_update([Island("i0", [br], loop.state.elo)])
