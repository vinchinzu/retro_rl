"""Unit tests for L3 library bomb budget (no emulator)."""

from __future__ import annotations

from zelda_i.level3.bomb_budget import (
    ISOLATED_POKE16_CLOSES_SPINE,
    ISOLATED_RAFT_BOMBS,
    ISOLATED_TO_BOSS_POKE_BOMBS,
    L3_BOMB_BUDGET,
    L3_BOMB_BUDGET_ASSUMED,
    L3_BOMB_BUDGET_VERIFIED,
    L3_BOMB_FARM_ROOM,
    L3_BOMB_WALL_SPEND,
    L3_BOMBS_IN_AT_DEST_5B,
    L3_BOMBS_IN_AT_RAFT,
    MANHANDLA_BOMB_SPEND_ASSUMED,
    MANHANDLA_BOMB_SPEND_EVIDENCE,
    MANHANDLA_HEADS_EVIDENCE,
    MANHANDLA_HEADS_LIVE,
    MANHANDLA_OBJECT_TYPE,
    bomb_budget,
    isolated_raft_requires_poke16,
    planned_bomb_spend,
    report_used_poke16,
)
from zelda_i.level3.dungeon import ROOM_L3_DARKNUTS


def test_planned_spend_positive_and_labeled() -> None:
    spend = planned_bomb_spend()
    assert L3_BOMB_WALL_SPEND == 2
    assert spend.wall_bombs == 2
    assert spend.verified == L3_BOMB_BUDGET_VERIFIED == 2
    assert all(w.evidence == "verified" for w in spend.walls)
    assert MANHANDLA_HEADS_LIVE == 5
    assert MANHANDLA_HEADS_EVIDENCE == "verified"
    assert MANHANDLA_OBJECT_TYPE == 0x3C
    assert MANHANDLA_BOMB_SPEND_ASSUMED == 5
    assert MANHANDLA_BOMB_SPEND_EVIDENCE == "assumed"
    assert spend.manhandla.evidence == "assumed"
    assert spend.assumed == L3_BOMB_BUDGET_ASSUMED == 5
    assert spend.total == L3_BOMB_BUDGET == 7
    assert bomb_budget() == 7
    assert spend.total > 0
    assert spend.verified > 0
    assert spend.bombs_in_at_dest_5b == L3_BOMBS_IN_AT_DEST_5B == 7
    assert spend.bombs_in_at_raft == L3_BOMBS_IN_AT_RAFT == 7
    assert L3_BOMB_FARM_ROOM == ROOM_L3_DARKNUTS == 0x5B
    assert spend.isolated_poke16_closes_spine is False
    assert ISOLATED_POKE16_CLOSES_SPINE is False
    rep = spend.report()
    assert rep["total"] == 7
    assert rep["isolated_poke16_closes_spine"] is False
    assert rep["manhandla_evidence"] == "assumed"


def test_isolated_raft_zero_implies_poke16() -> None:
    assert ISOLATED_RAFT_BOMBS == 0
    assert isolated_raft_requires_poke16(0) is True
    assert isolated_raft_requires_poke16(ISOLATED_RAFT_BOMBS) is True
    assert isolated_raft_requires_poke16(None) is True
    assert isolated_raft_requires_poke16(2) is False
    assert isolated_raft_requires_poke16(8) is False
    # Empty pin cannot cover verified walls; isolated to-boss poke-16 is recon.
    assert L3_BOMB_WALL_SPEND > ISOLATED_RAFT_BOMBS
    assert ISOLATED_TO_BOSS_POKE_BOMBS == 16
    assert ISOLATED_POKE16_CLOSES_SPINE is False
    assert not report_used_poke16({"runner": "run_level3_to_boss.py --trials 2"})
    assert report_used_poke16(
        {"runner": "run_level3_to_boss.py --infinite-life --poke-bombs 16"}
    )
