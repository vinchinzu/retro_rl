"""L3 dest-6b spine wiring and 0x6b occupancy policy (no emulator)."""

from __future__ import annotations

import numpy as np
import pytest

from zelda_i.door_graph import DoorDir, L3_DARKNUTS, L3_ENTRY, L3_NORTH_ZOLS, L3_WEST_KEY
from zelda_i.level3_dungeon import DARKNUT_OBJECT_TYPE, ROOM_5B_SPEC
from zelda_i.level3_dungeon import ROOM_L3_NORTH_ZOLS as ROOM_6B
from zelda_i.level3_path import (
    Level3NorthChainController,
    Level3NorthExit6bController,
)
from zelda_i.level3_spine import (
    dest_6b_room_plan,
    level3_dest_6b_stages,
    level3_dest_6b_success,
)
from zelda_i.ram import (
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.walk_physics import WALK_DELTA


def _ram(
    *,
    room: int,
    x: int,
    y: int,
    keys: int = 0,
    darknuts: int = 0,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 3
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    slot = 1
    for _ in range(darknuts):
        ram[ADDR_OBJ_TYPE + slot] = DARKNUT_OBJECT_TYPE
        ram[ADDR_OBJ_HP + slot] = 64
        ram[ADDR_LINK_X + slot] = 80 + slot * 16
        ram[ADDR_LINK_Y + slot] = 141
        slot += 1
    return ram


def test_dest_6b_room_plan_is_kill_clear_north() -> None:
    path = dest_6b_room_plan()
    rooms = [L3_ENTRY, *[edge.target_room for edge in path]]
    assert rooms == [L3_ENTRY, L3_WEST_KEY, L3_NORTH_ZOLS, L3_DARKNUTS]
    assert path[0].direction is DoorDir.LEFT
    assert path[1].direction is DoorDir.UP
    assert path[2].direction is DoorDir.UP


def test_dest_stages_fail_closed_without_graph_path(monkeypatch) -> None:
    import zelda_i.level3_spine as spine

    monkeypatch.setattr(spine.LEVEL_3_DOOR_GRAPH, "bfs_path", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="0x7c"):
        level3_dest_6b_stages()


def test_dest_6b_success_requires_cleared_5b() -> None:
    empty = read_snapshot(_ram(room=0x5B, x=120, y=205))
    live = read_snapshot(_ram(room=0x5B, x=120, y=205, darknuts=3))
    assert level3_dest_6b_success(empty)
    assert not level3_dest_6b_success(live)
    assert not ROOM_5B_SPEC.live_enemies(empty)
    assert len(ROOM_5B_SPEC.live_enemies(live)) == 3


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


def test_north_chain_live_5b_does_not_succeed() -> None:
    ctrl = Level3NorthChainController()
    action = ctrl.step(
        read_snapshot(_ram(room=0x5B, x=120, y=205, darknuts=3))
    )
    assert not ctrl.success
    assert ctrl.phase == "clear_5b"
    assert action.reason.startswith("combat")
