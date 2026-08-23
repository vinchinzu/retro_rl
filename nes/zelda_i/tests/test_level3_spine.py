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
    level3_compass_stages,
    level3_compass_success,
    level3_south_darknuts_stages,
    level3_south_darknuts_success,
    level3_raft_stages,
    level3_raft_success,
    level3_west_darknuts_stages,
    level3_west_darknuts_success,
    dest_6b_room_plan,
    level3_dest_6b_stages,
    level3_dest_6b_success,
    level3_entry_stages,
    level3_entry_success,
    level3_west_key_stages,
    level3_west_key_success,
)
from zelda_i.ram import (
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_RAFT,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.survival_spine import SPINE_THROUGH
from zelda_i.chain import PredicateStopController
from zelda_i.walk_physics import WALK_DELTA


def _ram(*, room: int, x: int, y: int, keys: int = 0, raft: int = 0) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 3
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    ram[ADDR_RAFT] = raft
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
    west = [(name, type(ctrl)) for name, ctrl, _ in level3_west_key_stages()]
    dest = [(name, type(ctrl)) for name, ctrl, _ in level3_dest_6b_stages()]
    assert entry == [
        ("settle_l2_tf", PostL2TriforceSettleController),
        ("enter_level3", OverworldPostL2ToLevel3Controller),
    ]
    assert west == [("west_key", Level3WestKeyController)]
    assert dest == [
        ("west_key", Level3WestKeyController),
        ("north_chain", Level3NorthChainController),
    ]
    compass = [(name, type(ctrl)) for name, ctrl, _ in level3_compass_stages()]
    assert compass == [("compass_0x5a", PredicateStopController)]
    west_darknuts = [(name, type(ctrl)) for name, ctrl, _ in level3_west_darknuts_stages()]
    assert west_darknuts == [("west_darknuts_0x59", PredicateStopController)]
    south = [(name, type(ctrl)) for name, ctrl, _ in level3_south_darknuts_stages()]
    assert south == [("south_darknuts_0x69", PredicateStopController)]
    raft = [(name, type(ctrl)) for name, ctrl, _ in level3_raft_stages()]
    assert raft == [("raft_0x0f", PredicateStopController)]


def test_dest_stages_fail_closed_without_graph_path(monkeypatch) -> None:
    import zelda_i.level3_spine as spine

    monkeypatch.setattr(spine.LEVEL_3_DOOR_GRAPH, "bfs_path", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="0x7c"):
        level3_dest_6b_stages()


def test_dest_6b_success_is_play_5b() -> None:
    snap = read_snapshot(_ram(room=0x5B, x=120, y=205))
    assert level3_dest_6b_success(snap)


def test_compass_success_is_play_5a() -> None:
    assert level3_compass_success(read_snapshot(_ram(room=0x5A, x=224, y=141)))
    assert not level3_compass_success(read_snapshot(_ram(room=0x5B, x=120, y=205)))


def test_west_darknuts_success_is_play_59() -> None:
    assert level3_west_darknuts_success(read_snapshot(_ram(room=0x59, x=224, y=141)))
    assert not level3_west_darknuts_success(read_snapshot(_ram(room=0x5A, x=224, y=141)))


def test_south_darknuts_success_is_play_69() -> None:
    assert level3_south_darknuts_success(read_snapshot(_ram(room=0x69, x=120, y=77)))
    assert not level3_south_darknuts_success(read_snapshot(_ram(room=0x59, x=120, y=205)))


def test_raft_success_requires_inventory_bit_in_passage() -> None:
    assert not level3_raft_success(read_snapshot(_ram(room=0x0F, x=136, y=141)))
    ram = _ram(room=0x0F, x=136, y=141, raft=1)
    ram[ADDR_MODE] = 9
    assert level3_raft_success(read_snapshot(ram))
    ram[ADDR_MODE] = PLAY_MODE
    assert not level3_raft_success(read_snapshot(ram))


def test_entry_success_is_play_7c() -> None:
    snap = read_snapshot(_ram(room=0x7C, x=120, y=205))
    assert level3_entry_success(snap)
    assert not level3_entry_success(read_snapshot(_ram(room=0x5B, x=120, y=205)))
    assert not level3_west_key_success(snap)


def test_west_key_success_is_play_7b_with_key() -> None:
    empty = read_snapshot(_ram(room=0x7B, x=120, y=141, keys=0))
    keyed = read_snapshot(_ram(room=0x7B, x=120, y=141, keys=1))
    dest = read_snapshot(_ram(room=0x5B, x=120, y=205, keys=1))
    assert not level3_west_key_success(empty)
    assert level3_west_key_success(keyed)
    assert not level3_west_key_success(dest)
    assert not level3_entry_success(keyed)


def test_north_exit_miss_sidesteps_and_still_paths() -> None:
    ctrl = Level3NorthExit6bController()
    start = read_snapshot(_ram(room=ROOM_6B, x=96, y=141))
    first = ctrl.step(start)
    assert first.reason == "north6b_path"
    assert ctrl.walker.last_dir in WALK_DELTA
    blocked_ahead = {
        "UP": (96, 140),
        "DOWN": (96, 142),
        "LEFT": (95, 141),
        "RIGHT": (97, 141),
    }[ctrl.walker.last_dir]
    second = ctrl.step(start)
    assert ctrl.misses == 1
    assert blocked_ahead in ctrl.grid.blocked
    assert second.reason in {"north6b_path", "north6b_thread", "north6b_thread_up"}
    path = ctrl.grid.shortest_path((96, 141), (120, 109))
    assert path is not None
    assert blocked_ahead not in path


def test_north_band_is_not_occupancy_graded() -> None:
    ctrl = Level3NorthExit6bController()
    snap = read_snapshot(_ram(room=ROOM_6B, x=120, y=109))
    assert ctrl.step(snap).reason == "north6b_push"
    assert ctrl.step(snap).reason == "north6b_push"
    assert ctrl.misses == 0
    assert ctrl.walker.last_dir is None
