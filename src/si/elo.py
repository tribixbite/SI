"""Elo-based selector (RoboPhD pattern, K=32 default).

This one is actually implemented rather than stubbed — the math is small and
self-contained, and getting it wrong would silently poison every experiment.
Reference: Elo (1978), Arpad Elo's chess-rating paper, and RoboPhD §3.2
(Borthwick & Ash 2026, arXiv:2601.01126).
"""

from __future__ import annotations

import math

from si.contracts import EloState, Match, ReplacementPlan, Selector


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected-score formula."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def apply_match(state: EloState, match: Match) -> None:
    """Mutates state in place. K-factor from state; score 1.0/0.0/0.5 for win/loss/draw."""
    a, b = match.branch_a, match.branch_b
    if a not in state.ratings:
        state.ratings[a] = state.default_rating
    if b not in state.ratings:
        state.ratings[b] = state.default_rating

    ra, rb = state.ratings[a], state.ratings[b]
    ea = expected_score(ra, rb)
    eb = 1.0 - ea

    if match.winner is None:
        sa, sb = 0.5, 0.5
    elif match.winner == a:
        sa, sb = 1.0, 0.0
    elif match.winner == b:
        sa, sb = 0.0, 1.0
    else:
        raise ValueError(f"match.winner {match.winner!r} is not {a!r} or {b!r}")

    state.ratings[a] = ra + state.k * (sa - ea)
    state.ratings[b] = rb + state.k * (sb - eb)


def ranked(state: EloState) -> list[tuple[str, float]]:
    """Return [(branch_id, rating)] sorted descending."""
    return sorted(state.ratings.items(), key=lambda p: p[1], reverse=True)


class EloSelector(Selector):
    """Concrete Selector implementation."""

    def __init__(self, replacement_quartile_size: float = 0.25) -> None:
        self.q = replacement_quartile_size

    def update(self, matches: list[Match], state: EloState) -> EloState:
        for m in matches:
            apply_match(state, m)
        return state

    def replacement_plan(self, state: EloState) -> ReplacementPlan:
        rnk = ranked(state)
        n = len(rnk)
        if n == 0:
            return ReplacementPlan(keep=[], mutate=[], replace=[])

        q = max(1, int(round(self.q * n)))
        top_ids = [bid for bid, _ in rnk[:q]]
        bottom_ids = [bid for bid, _ in rnk[-q:]]
        middle_ids = [bid for bid, _ in rnk[q : n - q]]

        # Each dead branch is reseeded from a top-quartile parent (round-robin).
        replace: list[tuple[str, str]] = []
        for i, dead in enumerate(bottom_ids):
            parent = top_ids[i % len(top_ids)]
            replace.append((dead, parent))

        return ReplacementPlan(keep=top_ids, mutate=middle_ids, replace=replace)
