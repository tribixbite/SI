"""Tests for the ring-topology migrator.

The migration math is exact: donor is the upstream (clockwise) island's
top-Elo branch, receiver is the current island's lowest-Elo branch, and the
LoRA merge is delegated to an injected callback. Pin the topology so a future
refactor can't silently flip the ring direction or the donor/receiver roles.
"""

from si.contracts import Branch, EloState, Experience, Island
from si.islands import RingMigrator


def _branch(bid: str, *, wins=None, seeds=None) -> Branch:
    return Branch(
        branch_id=bid,
        lora_path=f"/loras/{bid}",
        elo=1500.0,
        experience=Experience(
            recent_wins=list(wins or []),
            proposer_seeds=list(seeds or []),
        ),
    )


def _island(iid: str, branches: list[Branch], ratings: dict[str, float]) -> Island:
    return Island(island_id=iid, branches=branches, elo_state=EloState(ratings=ratings))


def _recording_merge():
    calls: list[tuple[str, str]] = []

    def merge(receiver_path: str, donor_path: str) -> str:
        calls.append((receiver_path, donor_path))
        return f"{receiver_path}+merged"

    return merge, calls


def test_gen_zero_is_noop():
    merge, calls = _recording_merge()
    m = RingMigrator(migration_every=5, migrant_experience_priority=2.0, lora_merge_fn=merge)
    islands = [_island("i0", [_branch("a")], {}), _island("i1", [_branch("b")], {})]
    assert m.migrate(islands, gen=0) is islands
    assert calls == []


def test_non_migration_gen_is_noop():
    merge, calls = _recording_merge()
    m = RingMigrator(migration_every=5, migrant_experience_priority=2.0, lora_merge_fn=merge)
    islands = [_island("i0", [_branch("a")], {}), _island("i1", [_branch("b")], {})]
    assert m.migrate(islands, gen=3) is islands
    assert calls == []


def test_single_island_is_noop():
    merge, calls = _recording_merge()
    m = RingMigrator(migration_every=5, migrant_experience_priority=2.0, lora_merge_fn=merge)
    islands = [_island("i0", [_branch("a")], {})]
    assert m.migrate(islands, gen=5) is islands
    assert calls == []


def test_donor_is_upstream_and_receiver_is_lowest_elo():
    merge, calls = _recording_merge()
    m = RingMigrator(migration_every=5, migrant_experience_priority=2.0, lora_merge_fn=merge)
    # i0: a(1600) top, b(1400) low.  i1: c(1700) top, d(1300) low.
    i0 = _island("i0", [_branch("a"), _branch("b")], {"a": 1600.0, "b": 1400.0})
    i1 = _island("i1", [_branch("c"), _branch("d")], {"c": 1700.0, "d": 1300.0})
    out = m.migrate([i0, i1], gen=5)

    # i0's donor is i1 (upstream = (0-1)%2 = 1): top donor c → into i0's low b.
    # i1's donor is i0: top donor a → into i1's low d.
    assert set(calls) == {("/loras/b", "/loras/c"), ("/loras/d", "/loras/a")}

    # The lowest-Elo branch in each island got the merged path; others untouched.
    b_new = next(x for x in out[0].branches if x.branch_id == "b")
    a_kept = next(x for x in out[0].branches if x.branch_id == "a")
    assert b_new.lora_path == "/loras/b+merged"
    assert a_kept.lora_path == "/loras/a"


def test_donor_wins_seed_receiver_experience():
    merge, _ = _recording_merge()
    m = RingMigrator(
        migration_every=1, migrant_experience_priority=2.0, lora_merge_fn=merge, top_k_experiences=2
    )
    # donor (top of i1) carries 3 wins; only the last top_k=2 should migrate.
    donor = _branch("c", wins=["w1", "w2", "w3"], seeds=["s1", "s2", "s3", "s4"])
    receiver = _branch("b", wins=["own_win"])
    i0 = _island("i0", [_branch("a"), receiver], {"a": 1600.0, "b": 1400.0})
    i1 = _island("i1", [donor, _branch("d")], {"c": 1700.0, "d": 1300.0})
    out = m.migrate([i0, i1], gen=1)

    b_new = next(x for x in out[0].branches if x.branch_id == "b")
    # receiver keeps its own win, gains donor's last 2 wins.
    assert b_new.experience.recent_wins == ["own_win", "w2", "w3"]
    # proposer seeds: receiver's (none) + donor's first top_k//2 = 1 seed.
    assert b_new.experience.proposer_seeds == ["s1"]


def test_empty_island_branches_skipped():
    merge, calls = _recording_merge()
    m = RingMigrator(migration_every=1, migrant_experience_priority=2.0, lora_merge_fn=merge)
    i0 = _island("i0", [], {})
    i1 = _island("i1", [_branch("c")], {"c": 1700.0})
    out = m.migrate([i0, i1], gen=1)
    # i0 has no receiver (empty); i1's donor i0 has no top branch. No merges.
    assert calls == []
    assert len(out) == 2
