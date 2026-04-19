"""Tests for the Elo selector. Math has to be exact — test it."""

import math

from si.contracts import EloState, Match, ReplacementPlan
from si.elo import EloSelector, apply_match, expected_score, ranked


def test_expected_score_equal_ratings():
    assert expected_score(1500, 1500) == 0.5


def test_expected_score_400_gap_is_ten_to_one():
    # 400-point gap → higher-rated wins with probability ~0.909
    assert math.isclose(expected_score(1900, 1500), 10.0 / 11.0, rel_tol=1e-6)


def test_apply_match_win_updates_both_sides():
    state = EloState()
    m = Match(task_id="t1", branch_a="A", branch_b="B", winner="A", walltime_ms=10)
    apply_match(state, m)
    assert state.ratings["A"] > state.default_rating
    assert state.ratings["B"] < state.default_rating
    # K=32, equal ratings, A wins: A gets +16, B gets -16.
    assert math.isclose(state.ratings["A"], 1516.0, abs_tol=1e-6)
    assert math.isclose(state.ratings["B"], 1484.0, abs_tol=1e-6)


def test_apply_match_draw_is_neutral_at_equal_ratings():
    state = EloState()
    m = Match(task_id="t1", branch_a="A", branch_b="B", winner=None, walltime_ms=10)
    apply_match(state, m)
    assert math.isclose(state.ratings["A"], 1500.0, abs_tol=1e-6)
    assert math.isclose(state.ratings["B"], 1500.0, abs_tol=1e-6)


def test_upset_moves_more_than_expected_win():
    """Low-rated beating high-rated should move ratings more than the reverse case."""
    state_upset = EloState(ratings={"low": 1400.0, "high": 1700.0})
    state_expected = EloState(ratings={"low": 1400.0, "high": 1700.0})
    apply_match(state_upset, Match("t", "low", "high", "low", 1))
    apply_match(state_expected, Match("t", "low", "high", "high", 1))

    upset_delta = state_upset.ratings["low"] - 1400.0
    expected_delta = 1700.0 - state_expected.ratings["high"]
    assert upset_delta > expected_delta


def test_ranked_is_descending():
    state = EloState(ratings={"A": 1600, "B": 1400, "C": 1500})
    assert [bid for bid, _ in ranked(state)] == ["A", "C", "B"]


def test_replacement_plan_splits_quartiles():
    state = EloState(ratings={f"b{i}": 1500 + i * 10 for i in range(8)})
    plan = EloSelector(replacement_quartile_size=0.25).replacement_plan(state)
    assert len(plan.keep) == 2
    assert len(plan.replace) == 2
    assert len(plan.mutate) == 4
    # Dead branches should be the lowest-rated.
    dead = {d for d, _ in plan.replace}
    assert dead == {"b0", "b1"}
    # Each dead branch is paired with a top-quartile parent.
    parents = {p for _, p in plan.replace}
    assert parents.issubset({"b6", "b7"})


def test_replacement_plan_handles_tiny_population():
    """Edge case: 2 branches with q=0.25 rounds to 1 in each quartile."""
    state = EloState(ratings={"a": 1600, "b": 1400})
    plan = EloSelector(replacement_quartile_size=0.25).replacement_plan(state)
    assert len(plan.keep) == 1
    assert len(plan.replace) == 1
    assert plan.replace[0] == ("b", "a")
