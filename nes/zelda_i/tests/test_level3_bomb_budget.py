"""Unit tests for L3 library bomb budget (no emulator)."""

from __future__ import annotations

import pytest

from zelda_i.level3_bomb_budget import (
    ISOLATED_POKE16_CLOSES_SPINE,
    ISOLATED_RAFT_BOMBS,
    ISOLATED_RAFT_RECORDING,
    ISOLATED_TO_BOSS_POKE_BOMBS,
    ISOLATED_TO_BOSS_RECORDING,
    L3_BOMB_BUDGET,
    L3_BOMB_BUDGET_ASSUMED,
    L3_BOMB_BUDGET_VERIFIED,
    L3_BOMB_FARM_ROOM,
    L3_BOMB_R_WALLS,
    L3_BOMB_WALL_SPEND,
    L3_BOMBS_IN_AT_DEST_5B,
    L3_BOMBS_IN_AT_RAFT,
    MANHANDLA_BOMB_SPEND_ASSUMED,
    MANHANDLA_BOMB_SPEND_EVIDENCE,
    MANHANDLA_HEADS_EVIDENCE,
    MANHANDLA_HEADS_LIVE,
    MANHANDLA_OBJECT_TYPE,
    SPINE_0X7C_BOMBS,
    SPINE_0X7C_KEYS,
    SPINE_0X7C_ROOM,
    SPINE_ENTRANCE_RECORDING,
    SPINE_L2_ENTRY_BOMBS,
    bomb_budget,
    bombs_from_snapshot,
    isolated_raft_requires_poke16,
    load_isolated_report,
    planned_bomb_spend,
    raft_stop_bombs,
    report_used_poke16,
)
from zelda_i.level3_dungeon import (
    ROOM_L3_BOMB_SHORTCUT,
    ROOM_L3_BOSS,
    ROOM_L3_COMPASS,
    ROOM_L3_DARKNUTS,
    ROOM_L3_ENTRY,
    ROOM_L3_WEST_DARKNUTS,
)
from zelda_i.level3_geometry import BOMB_STAND_59_RIGHT, BOMB_STAND_5B_RIGHT


def test_bomb_stands_match_geometry() -> None:
    assert BOMB_STAND_59_RIGHT == (192, 141)
    assert BOMB_STAND_5B_RIGHT == (192, 141)
    assert L3_BOMB_R_WALLS == (
        (ROOM_L3_WEST_DARKNUTS, ROOM_L3_COMPASS, BOMB_STAND_59_RIGHT),
        (ROOM_L3_DARKNUTS, ROOM_L3_BOMB_SHORTCUT, BOMB_STAND_5B_RIGHT),
    )
    spend = planned_bomb_spend()
    assert spend.walls[0].stand == BOMB_STAND_59_RIGHT
    assert spend.walls[1].stand == BOMB_STAND_5B_RIGHT
    assert spend.walls[0].room == 0x59
    assert spend.walls[1].room == 0x5B
    assert spend.walls[0].dest == 0x5A
    assert spend.walls[1].dest == 0x5C
    assert spend.manhandla.room == ROOM_L3_BOSS == 0x4D


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


def test_spine_0x7c_named_carry() -> None:
    assert SPINE_0X7C_BOMBS == 8
    assert SPINE_0X7C_KEYS == 4
    assert SPINE_0X7C_ROOM == ROOM_L3_ENTRY == 0x7C
    assert SPINE_L2_ENTRY_BOMBS == 0
    # Carry 8 covers planned 7; farm on 0x5b is the natural alternative.
    assert SPINE_0X7C_BOMBS >= bomb_budget()
    assert bombs_from_snapshot({"keys": 4, "room": 0x7C}) is None
    assert bombs_from_snapshot({"bombs": 8}) == 8
    assert bombs_from_snapshot(None) is None


def test_optional_raft_recording_bombs_zero() -> None:
    data = load_isolated_report(ISOLATED_RAFT_RECORDING)
    if data is None:
        pytest.skip("recordings/ gitignored or missing")
    bombs = raft_stop_bombs(data)
    assert bombs == 0
    assert isolated_raft_requires_poke16(bombs) is True


def test_optional_to_boss_recording_poke16_recon() -> None:
    data = load_isolated_report(ISOLATED_TO_BOSS_RECORDING)
    if data is None:
        pytest.skip("recordings/ gitignored or missing")
    assert report_used_poke16(data) is True
    assert "--poke-bombs 16" in str(data.get("runner") or "")
    heads = data.get("manhandla", {})
    if isinstance(heads, dict) and "heads" in heads:
        assert int(heads["heads"]) == MANHANDLA_HEADS_LIVE


def test_optional_spine_entrance_bombs_eight() -> None:
    data = load_isolated_report(SPINE_ENTRANCE_RECORDING)
    if data is None:
        pytest.skip("recordings/ gitignored or missing")
    final = data.get("final")
    assert isinstance(final, dict)
    assert bombs_from_snapshot(final) == SPINE_0X7C_BOMBS == 8
    assert int(final["keys"]) == SPINE_0X7C_KEYS == 4
    assert int(final["room"]) == SPINE_0X7C_ROOM == 0x7C
    l2 = data.get("l2_entry")
    if isinstance(l2, dict):
        assert bombs_from_snapshot(l2) == SPINE_L2_ENTRY_BOMBS == 0
