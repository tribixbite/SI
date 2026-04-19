"""Unit tests for match loop reward math — no GPU needed."""

from si.contracts import Solution, Task, TaskType, VerifyResult
from si.match import GenerationResults, ProposalOutcome, Rollout


def _mk_task(tid: str = "t") -> Task:
    return Task(
        task_type=TaskType.DEDUCTION,
        program="def f(x): return x",
        input="1",
        output=None,
        proposer_branch_id="p",
        gen=0,
        task_id=tid,
    )


def _mk_rollout(passed: bool) -> Rollout:
    sol = Solution(task_id="t", solver_branch_id="s", body="", trace="", walltime_ms=0)
    res = VerifyResult(
        task_id="t", solution_id="t", passed=passed,
        stdout="", stderr="", walltime_ms=0, exit_code=0,
    )
    return Rollout(solution=sol, result=res)


def test_proposer_reward_peaks_at_half_pass_rate():
    outcome = ProposalOutcome(
        task=_mk_task(),
        rollouts=[_mk_rollout(True), _mk_rollout(True), _mk_rollout(False), _mk_rollout(False)],
    )
    assert outcome.pass_rate == 0.5
    assert outcome.proposer_reward == 0.0


def test_proposer_reward_all_pass_is_penalized():
    outcome = ProposalOutcome(
        task=_mk_task(), rollouts=[_mk_rollout(True)] * 4,
    )
    assert outcome.pass_rate == 1.0
    assert outcome.proposer_reward == -0.5


def test_proposer_reward_all_fail_is_penalized():
    outcome = ProposalOutcome(
        task=_mk_task(), rollouts=[_mk_rollout(False)] * 4,
    )
    assert outcome.pass_rate == 0.0
    assert outcome.proposer_reward == -0.5


def test_solver_rewards_are_binary():
    outcome = ProposalOutcome(
        task=_mk_task(),
        rollouts=[_mk_rollout(True), _mk_rollout(False), _mk_rollout(True)],
    )
    assert outcome.solver_rewards() == [1.0, 0.0, 1.0]


def test_generation_aggregates():
    o1 = ProposalOutcome(task=_mk_task("a"), rollouts=[_mk_rollout(True)] * 2)
    o2 = ProposalOutcome(task=_mk_task("b"), rollouts=[_mk_rollout(False)] * 2)
    g = GenerationResults(outcomes=[o1, o2], failed_proposals=3)
    assert g.n_tasks == 2
    assert g.aggregate_pass_rate == 0.5
    # proposer rewards: o1=-0.5 (all pass), o2=-0.5 (all fail); mean -0.5
    assert g.aggregate_proposer_reward == -0.5


def test_mc_difficulty_histogram():
    outcomes = [
        ProposalOutcome(task=_mk_task(f"t{i}"), rollouts=[_mk_rollout(bool(i % 2))] * 4)
        for i in range(4)
    ]
    # pass_rates: 0, 0, 0, 0 ... wait: i%2 is 0,1,0,1 per rollout × 4 rollouts = all same pattern
    # Actually with same i%2 across rollouts (not per-rollout), pass_rate is 0 or 1 per outcome
    # i=0: rollouts all False → 0, i=1: all True → 1, i=2: all False → 0, i=3: all True → 1
    g = GenerationResults(outcomes=outcomes, failed_proposals=0)
    hist = g.mc_difficulty_histogram(bins=10)
    assert hist[0] == 2  # two outcomes at pass_rate 0.0
    assert hist[9] == 2  # two outcomes at pass_rate 1.0
