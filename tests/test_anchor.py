"""Tests for the anchor commit/revert rule.

This is the single non-negotiable defense against mutual-hallucination
collapse (CLAUDE.md §Non-negotiables, docs/00-overview.md). The decision must
be exact at the tolerance boundary and must tighten after the strict gen — so
test it directly rather than trusting it through the loop.
"""

from si.anchor import HeldOutAnchor
from si.contracts import AnchorResult, Branch, Experience


def _anchor(**kw) -> HeldOutAnchor:
    # runner is irrelevant for should_revert; supply a constant.
    return HeldOutAnchor(runner=lambda _b: 0.0, benchmark_name="test", **kw)


def _res(gen: int, aggregate: float) -> AnchorResult:
    return AnchorResult(gen=gen, aggregate=aggregate)


def test_no_prev_never_reverts():
    a = _anchor()
    assert a.should_revert(None, _res(0, 0.10)) is False


def test_improvement_never_reverts():
    a = _anchor()
    assert a.should_revert(_res(1, 0.20), _res(2, 0.25)) is False


def test_flat_never_reverts():
    a = _anchor()
    assert a.should_revert(_res(1, 0.20), _res(2, 0.20)) is False


def test_drop_within_tolerance_holds():
    # tol=0.02 → allowed drop = 0.20 * 0.02 = 0.004. A 0.003 drop holds.
    a = _anchor(tolerance=0.02)
    assert a.should_revert(_res(1, 0.20), _res(2, 0.197)) is False


def test_drop_beyond_tolerance_reverts():
    # 0.005 drop > 0.004 allowed → revert.
    a = _anchor(tolerance=0.02)
    assert a.should_revert(_res(1, 0.20), _res(2, 0.195)) is True


def test_tolerance_is_strict_inequality_at_boundary():
    # drop == allowed should NOT revert (rule is `drop > prev*tol`). Use
    # binary-exact values so the knife-edge isn't decided by float rounding:
    # allowed = 0.5 * 0.5 = 0.25, drop = 0.5 - 0.25 = 0.25, all exact.
    a = _anchor(tolerance=0.5)
    assert a.should_revert(_res(1, 0.5), _res(2, 0.25)) is False


def test_strict_mode_tightens_after_gen():
    # Same 0.0035 drop: tolerated pre-strict (tol 0.02 → allow 0.004),
    # reverts post-strict (tol 0.01 → allow 0.002).
    a = _anchor(tolerance=0.02, strict_tolerance=0.01, strict_after_gen=50)
    lenient = a.should_revert(_res(49, 0.20), _res(49, 0.20 - 0.0035))
    strict = a.should_revert(_res(49, 0.20), _res(50, 0.20 - 0.0035))
    assert lenient is False
    assert strict is True


def test_strict_boundary_uses_curr_gen():
    # The cutover keys off curr.gen, not prev.gen.
    a = _anchor(tolerance=0.02, strict_tolerance=0.01, strict_after_gen=50)
    # curr.gen == strict_after_gen is already strict (>=).
    assert a.should_revert(_res(40, 0.20), _res(50, 0.197)) is True


def test_evaluate_aggregates_mean_over_population():
    runs = {"a": 0.4, "b": 0.6}
    a = HeldOutAnchor(runner=lambda b: runs[b.branch_id], benchmark_name="t")
    pop = [
        Branch(branch_id="a", lora_path="", elo=1500.0, experience=Experience()),
        Branch(branch_id="b", lora_path="", elo=1500.0, experience=Experience()),
    ]
    res = a.evaluate(pop)
    assert res.aggregate == 0.5
    assert res.per_branch == {"a": 0.4, "b": 0.6}


def test_evaluate_empty_population_does_not_divide_by_zero():
    a = _anchor()
    res = a.evaluate([])
    assert res.aggregate == 0.0
