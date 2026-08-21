"""L3 dest-6b spine wiring and 0x6b occupancy policy (no emulator)."""

from __future__ import annotations

import pytest
import numpy as np

from zelda_i.door_graph import DoorDir, L3_DARKNUTS, L3_ENTRY, L3_NORTH_ZOLS, L3_WEST_KEY
from zelda_i.level3_dungeon import ROOM_L3_NORTH_ZOLS as ROOM_6B
from zelda_i.level3_overworld import (
    OverworldPostL2ToLevel3Controller,
    PostL2TriforceSettleController,
)
from zelda_i.level3_path import (
    Level3NorthChainController,
    Level3NorthExit6bController,
    Level3WestKeyController,
)
from zelda_i.level3_spine import (
    dest_6b_room_plan,
    level3_dest_6b_stages,
    level3_dest_6b_success,
    level3_entry_stages,
    level3_entry_success,
)
from zelda_i.ram import (
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.survival_spine import SPINE_THROUGH


def _ram(*, room: int, x: int, y: int) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 3
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    return ram


def test_through_level3_is_wired() -> None:
    assert SPINE_THROUGH == ("level1", "level2", "level3")


def test_dest_6b_room_plan_is_kill_clear_north() -> None:
    path = dest_6b_room_plan()
    rooms = [L3_ENTRY, *[edge.target_room for edge in path]]
    assert rooms == [L3_ENTRY, L3_WEST_KEY, L3_NORTH_ZOLS, L3_DARKNUTS]
    assert path[0].direction is DoorDir.LEFT
    assert path[1].direction is DoorDir.UP
    assert path[2].direction is DoorDir.UP


def test_level3_stage_names_and_controllers() -> None:
    entry = [(name, type(ctrl)) for name, ctrl, _ in level3_entry_stages()]
    dest = [(name, type(ctrl)) for name, ctrl, _ in level3_dest_6b_stages()]
    assert entry == [
        ("settle_l2_tf", PostL2TriforceSettleController),
        ("enter_level3", OverworldPostL2ToLevel3Controller),
    ]
    assert dest == [
        ("west_key", Level3WestKeyController),
        ("north_chain", Level3NorthChainController),
    ]


def test_dest_stages_fail_closed_without_graph_path(monkeypatch) -> None:
    import zelda_i.level3_spine as spine

    monkeypatch.setattr(spine.LEVEL_3_DOOR_GRAPH, "bfs_path", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="0x7c"):
        level3_dest_6b_stages()


def test_dest_6b_success_is_play_5b() -> None:
    snap = read_snapshot(_ram(room=0x5B, x=120, y=205))
    assert level3_dest_6b_success(snap)


def test_entry_success_is_play_7c() -> None:
    snap = read_snapshot(_ram(room=0x7C, x=120, y=205))
    assert level3_entry_success(snap)
    assert not level3_entry_success(read_snapshot(_ram(room=0x5B, x=120, y=205)))


def test_north_exit_miss_sidesteps_and_still_paths() -> None:
    ctrl = Level3NorthExit6bController()
    start = read_snapshot(_ram(room=ROOM_6B, x=120, y=141))
    first = ctrl.step(start)
    assert first.reason == "north6b_path"
    assert ctrl.walker.last_dir == "UP"
    second = ctrl.step(start)
    assert ctrl.misses == 1
    assert (120, 140) in ctrl.grid.blocked
    assert second.reason == "north6b_path"
    assert ctrl.walker.last_dir in {"LEFT", "RIGHT"}
    path = ctrl.grid.shortest_path((120, 141), (120, 93))
    if path is None:
        path = ctrl.grid.shortest_path((120, 141), (120, 109))
    assert path is not None
    assert (120, 140) not in path


def test_north_band_is_not_occupancy_graded() -> None:
    ctrl = Level3NorthExit6bController()
    snap = read_snapshot(_ram(room=ROOM_6B, x=120, y=109))
    assert ctrl.step(snap).reason == "north6b_push"
    assert ctrl.step(snap).reason == "north6b_push"
    assert ctrl.misses == 0
    assert ctrl.walker.last_dir is None
