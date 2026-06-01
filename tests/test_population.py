"""Tests for the Phase 2 branch population manager.

Pins replacement (reseed dead branches from parents) and the snapshot/revert
pair that the anchor revert rule depends on. The LoRA weight op is a recording
stub, so this runs without a GPU.
"""

import pytest

from si.contracts import Branch, EloState, Experience, Solution, Task, TaskType
from si.population import BranchManager


def _branch(bid: str, elo: float = 1500.0) -> Branch:
    return Branch(branch_id=bid, lora_path=f"/init/{bid}", elo=elo, experience=Experience())


def _recording_reseed():
    calls: list[tuple[str, str, float]] = []

    def reseed(parent_path: str, dest_path: str, sigma: float) -> None:
        calls.append((parent_path, dest_path, sigma))

    return reseed, calls


def _mgr(branches, **kw):
    reseed, calls = _recording_reseed()
    m = BranchManager(branches, lora_root="/loras", reseed_fn=reseed, **kw)
    return m, calls


def _plan(keep=(), mutate=(), replace=()):
    from si.contracts import ReplacementPlan

    return ReplacementPlan(keep=list(keep), mutate=list(mutate), replace=list(replace))


def test_empty_population_rejected():
    reseed, _ = _recording_reseed()
    with pytest.raises(ValueError):
        BranchManager([], lora_root="/loras", reseed_fn=reseed)


def test_lora_path_is_versioned_by_gen():
    m, _ = _mgr([_branch("a")])
    assert m.lora_path("a", 7) == "/loras/a/gen7"
    assert m.lora_path("a", 8) != m.lora_path("a", 7)


def test_apply_replacement_reseeds_dead_from_parent():
    m, calls = _mgr([_branch("top", 1700), _branch("dead", 1300)], perturb_sigma=0.002)
    elo = EloState(ratings={"top": 1700.0, "dead": 1300.0})
    reseeded = m.apply_replacement(_plan(keep=["top"], replace=[("dead", "top")]), gen=3, elo=elo)

    assert reseeded == ["dead"]
    # reseed called with parent's current path → dead's gen-versioned path + sigma.
    assert calls == [("/init/top", "/loras/dead/gen3", 0.002)]
    dead = m.branches["dead"]
    assert dead.lora_path == "/loras/dead/gen3"
    assert dead.experience.recent_wins == []  # fresh individual
    assert dead.elo == 1500.0  # rating reset to default
    assert elo.ratings["dead"] == 1500.0
    # parent untouched
    assert m.branches["top"].lora_path == "/init/top"


def test_unknown_branch_in_plan_is_skipped():
    m, calls = _mgr([_branch("a")])
    reseeded = m.apply_replacement(_plan(replace=[("ghost", "a"), ("a", "ghost")]), gen=1)
    assert reseeded == []
    assert calls == []


def test_commit_then_revert_restores_path_and_experience():
    b = _branch("a")
    t = Task(task_type=TaskType.DEDUCTION, program="p", input="i", output="o",
             proposer_branch_id="x", gen=0, task_id="t1")
    s = Solution(task_id="t1", solver_branch_id="a", body="b", trace="", walltime_ms=1)
    b.experience.append_match(t, s, passed=True)
    m, _ = _mgr([b])

    m.commit(gen=10)
    # mutate after commit: reseed moves the path and wipes experience.
    m.apply_replacement(_plan(replace=[("a", "a")]), gen=11)
    assert m.branches["a"].lora_path == "/loras/a/gen11"
    assert m.branches["a"].experience.recent_wins == []

    m.revert_to(10)
    assert m.branches["a"].lora_path == "/init/a"
    assert len(m.branches["a"].experience.recent_wins) == 1


def test_revert_snapshot_is_deep_copied():
    b = _branch("a")
    m, _ = _mgr([b])
    m.commit(gen=5)
    # appending to the live buffer after commit must not leak into the snapshot.
    t = Task(task_type=TaskType.INDUCTION, program="p", input="i", output="o",
             proposer_branch_id="x", gen=0, task_id="t2")
    s = Solution(task_id="t2", solver_branch_id="a", body="b", trace="", walltime_ms=1)
    b.experience.append_match(t, s, passed=False)
    m.revert_to(5)
    assert m.branches["a"].experience.recent_losses == []


def test_revert_to_unknown_gen_raises():
    m, _ = _mgr([_branch("a")])
    m.commit(gen=2)
    with pytest.raises(KeyError):
        m.revert_to(99)
