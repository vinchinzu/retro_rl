"""Wiki dungeon treasures vs Survival spine collection."""

from __future__ import annotations

import inspect

import numpy as np

from zelda_i.dungeon_treasures import (
    KIND_GATE,
    LIVE_SIDE,
    OW_GATES,
    TREASURES,
    assert_through_wired,
    default_spine_collected,
    required_gate_skips_on_default_spine,
    treasure,
)
from zelda_i.level1_bow import LEVEL1_BOW_ROOM, level1_bow_success
from zelda_i.level6_north2c import level6_north2c_success
from zelda_i.ram import (
    ADDR_ARROWS,
    ADDR_BOW,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_RAFT,
    ADDR_ROD,
    ADDR_SCREEN,
    ADDR_TRIFORCE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.survival_spine import SPINE_THROUGH, run_survival_spine


def _ram(*, level: int, screen: int, **fields: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = fields.get("mode", PLAY_MODE)
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = screen
    ram[ADDR_LINK_X] = fields.get("x", 120)
    ram[ADDR_LINK_Y] = fields.get("y", 141)
    ram[ADDR_KEYS] = fields.get("keys", 1)
    ram[ADDR_BOW] = fields.get("bow", 0)
    ram[ADDR_ARROWS] = fields.get("arrows", 0)
    ram[ADDR_ROD] = fields.get("rod", 0)
    ram[ADDR_TRIFORCE] = fields.get("triforce", 0)
    return ram


def test_wiki_dungeon_treasures_cover_l1_through_l9() -> None:
    by_level = {level: [] for level in range(1, 10)}
    for item in TREASURES:
        by_level[item.dungeon].append(item.name)
    assert by_level[1] == ["bow", "wooden_boomerang"]
    assert "magical_boomerang" in by_level[2]
    assert "raft" in by_level[3]
    assert "stepladder" in by_level[4]
    assert "whistle" in by_level[5]
    assert "magical_rod" in by_level[6]
    assert "red_candle" in by_level[7]
    assert "book_of_magic" in by_level[8] and "magical_key" in by_level[8]
    assert "red_ring" in by_level[9] and "silver_arrows" in by_level[9]


def test_bow_is_l1_not_l6() -> None:
    bow = treasure("bow")
    rod = treasure("magical_rod")
    assert bow.dungeon == 1
    assert bow.addr == ADDR_BOW
    assert rod.dungeon == 6
    assert rod.addr == ADDR_ROD
    assert rod.on_default_spine
    assert not bow.on_default_spine


def test_only_required_gate_skip_before_gohma_is_bow() -> None:
    skips = required_gate_skips_on_default_spine()
    assert [item.name for item in skips] == ["bow"]
    assert skips[0].live == LIVE_SIDE
    assert skips[0].through == "level1-bow-cellar"
    assert "level1-bow-cellar" in SPINE_THROUGH


def test_default_spine_collected_l2_through_l6_items() -> None:
    names = set(default_spine_collected())
    assert names == {
        "magical_boomerang",
        "raft",
        "stepladder",
        "whistle",
        "magical_rod",
    }
    assert treasure("raft").addr == ADDR_RAFT


def test_wired_through_names_exist() -> None:
    for item in TREASURES:
        assert_through_wired(item)


def test_default_spine_never_runs_bow_branch() -> None:
    src = inspect.getsource(run_survival_spine)
    bow = src.index('if through in ("level1-bow", "level1-bow-cellar"):')
    ret = src.index("return run", bow)
    tf = src.index("level1_triforce_stages")
    l2 = src.index("level2_entry_stages")
    assert bow < ret < tf < l2


def test_level1_bow_through_is_enter_stop_not_inventory() -> None:
    dest = read_snapshot(_ram(level=1, screen=LEVEL1_BOW_ROOM, x=16, y=141))
    assert level1_bow_success(dest)
    assert dest.bow == 0
    assert dest.arrows == 0


def test_gohma_enter_does_not_require_bow() -> None:
    snap = read_snapshot(
        _ram(
            level=6,
            screen=0x1C,
            x=120,
            y=205,
            keys=3,
            bow=0,
            arrows=0,
            rod=1,
            triforce=0x1F,
        )
    )
    assert level6_north2c_success(snap)
    assert snap.bow == 0


def test_ow_gates_needed_with_dungeon_items() -> None:
    names = [item.name for item in OW_GATES]
    assert names == ["wooden_arrows", "bait", "blue_candle"]
    arrows = OW_GATES[0]
    assert arrows.cost_rupees == 80
    assert "Gohma" in arrows.gates


def test_l7_plus_required_gates_have_no_spine_through_yet() -> None:
    later = [
        item
        for item in TREASURES
        if item.kind == KIND_GATE and item.dungeon >= 7
    ]
    assert [item.name for item in later] == ["red_candle", "silver_arrows"]
    assert all(item.through is None for item in later)
    assert all(not item.on_default_spine for item in later)
