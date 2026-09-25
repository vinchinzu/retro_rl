"""Unit tests for Level 3 leftover walks that would burn again."""

from __future__ import annotations

import numpy as np

from retro_harness.nes import nes_action
from zelda_i.dungeon.ids import BLADE_TRAP_OBJECT_TYPE
from zelda_i.level3.dungeon import (
    DARKNUT_OBJECT_TYPE,
    ROOM_L3_COMPASS,
    MANHANDLA_OBJECT_TYPE,
    ROOM_5B_SPEC,
    ROOM_L3_BOSS,
    ROOM_L3_DARKNUTS,
    ROOM_L3_RAFT_PASSAGE,
    ROOM_L3_SOUTH_DARKNUTS,
    ROOM_L3_WEST_DARKNUTS,
    ROOM_L3_WEST_KEY,
    level3_manhandla_live,
)
from zelda_i.level3.path import (
    Level3NorthChainController,
    Level3WestDoorController,
)
from zelda_i.level3.raft_path import Level3RaftPathController, raft_passage_step
from zelda_i.ram import (
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_STATE,
    ADDR_OBJ_TYPE,
    ADDR_RAFT,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _ram(
    *,
    level: int = 3,
    room: int = ROOM_L3_WEST_KEY,
    x: int = 120,
    y: int = 205,
    mode: int = PLAY_MODE,
    keys: int = 0,
    darknuts: int = 0,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = mode
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_KEYS] = keys
    for slot in range(1, darknuts + 1):
        ram[ADDR_OBJ_TYPE + slot] = DARKNUT_OBJECT_TYPE
        ram[ADDR_OBJ_HP + slot] = 64
        ram[ADDR_LINK_X + slot] = 80 + slot * 16
        ram[ADDR_LINK_Y + slot] = 141
    return ram


def test_west_door_controller_arrives() -> None:
    ctrl = Level3WestDoorController()
    action = ctrl.step(
        read_snapshot(_ram(room=ROOM_L3_WEST_KEY, x=200, y=141))
    )
    assert ctrl.success
    assert action.reason == "west_arrived"


def test_north_chain_already_in_5b() -> None:
    ctrl = Level3NorthChainController()
    action = ctrl.step(read_snapshot(_ram(room=ROOM_L3_DARKNUTS, x=120, y=205)))
    assert not ctrl.success
    assert ctrl.phase == "spawn_5b"
    assert action.reason == "spawn_5b_wait"
    empty = read_snapshot(_ram(room=ROOM_L3_DARKNUTS, x=120, y=205))
    for _ in range(99):
        ctrl.step(empty)
        assert not ctrl.success
    ctrl.step(empty)
    assert ctrl.success
    assert ctrl.phase == "done"
    assert "spawn_5b_empty" in ctrl.notes


def test_north_chain_live_5b_enters_clear() -> None:
    ctrl = Level3NorthChainController()
    action = ctrl.step(
        read_snapshot(_ram(room=ROOM_L3_DARKNUTS, x=120, y=205, darknuts=3))
    )
    assert not ctrl.success
    assert ctrl.phase == "clear_5b"
    assert action.reason.startswith("combat")
    assert ctrl.clear_5b.spec is ROOM_5B_SPEC


def test_raft_path_controller_phases_and_raft_success() -> None:
    ctrl = Level3RaftPathController()
    assert ctrl.phase == "settle_5b"
    for _ in range(45):
        ctrl.step(read_snapshot(_ram(room=ROOM_L3_DARKNUTS, x=120, y=205)))
    assert ctrl.phase == "left_to_5a"
    ctrl.phase = "key_to_59"
    ctrl.step(
        read_snapshot(
            _ram(room=ROOM_L3_WEST_DARKNUTS, x=200, y=141, keys=0)
        )
    )
    assert ctrl.phase == "spawn_59"
    ctrl2 = Level3RaftPathController()
    action = ctrl2.step(
        read_snapshot(_ram(room=ROOM_L3_RAFT_PASSAGE, x=136, y=141)),
        has_raft=True,
    )
    assert ctrl2.success
    assert ctrl2.phase == "done"
    assert action.reason == "done"
    assert "raft_acquired" in ctrl2.notes

    ctrl3 = Level3RaftPathController()
    handoff = ctrl3.step(
        read_snapshot(_ram(room=ROOM_L3_SOUTH_DARKNUTS, x=120, y=77))
    )
    assert ctrl3.phase == "spawn_69"
    assert handoff.reason == "phase_handoff"

    raft_ram = _ram(room=ROOM_L3_RAFT_PASSAGE, x=136, y=141)
    raft_ram[ADDR_RAFT] = 1
    ctrl4 = Level3RaftPathController()
    assert ctrl4.step(read_snapshot(raft_ram)).reason == "done"
    assert ctrl4.success


def test_raft_passage_aligns_before_crossing_pickup_lane() -> None:
    # Both poses occurred on the last-heart tape. The old ±8 pickup band
    # sent Link horizontally into a wall from each one.
    assert raft_passage_step(
        read_snapshot(_ram(room=ROOM_L3_RAFT_PASSAGE, x=48, y=133, mode=9))
    ).action == nes_action("DOWN")
    # The west ladder crosses the pickup row; the platform starts at x=112
    # (5800 frames pressing RIGHT into the wall at (48, 141), Clean offset 3).
    assert raft_passage_step(
        read_snapshot(_ram(room=ROOM_L3_RAFT_PASSAGE, x=48, y=141, mode=9))
    ).action == nes_action("DOWN")
    assert raft_passage_step(
        read_snapshot(_ram(room=ROOM_L3_RAFT_PASSAGE, x=176, y=149, mode=9))
    ).action == nes_action("UP")
    assert raft_passage_step(
        read_snapshot(_ram(room=ROOM_L3_RAFT_PASSAGE, x=176, y=141, mode=9))
    ).action == nes_action("LEFT")


def _traps(ram: np.ndarray, state: int, top_y: int = 93, bottom_y: int = 189) -> np.ndarray:
    for slot, y in ((1, top_y), (2, bottom_y)):
        ram[ADDR_OBJ_TYPE + slot] = BLADE_TRAP_OBJECT_TYPE
        ram[ADDR_LINK_X + slot] = 32
        ram[ADDR_LINK_Y + slot] = y
        ram[ADDR_OBJ_STATE + slot] = state
    return ram


def test_key_door_springs_the_west_traps_then_waits_them_out() -> None:
    ctrl = Level3RaftPathController()
    ctrl.phase = "key_to_59"
    step = lambda ram: ctrl.step(read_snapshot(ram))  # noqa: E731
    # Home and idle: walk in far enough to spring them.
    act = step(_traps(_ram(room=ROOM_L3_COMPASS, x=60, y=141, keys=1), 0))
    assert act.action == nes_action("LEFT") and act.reason == "key_59_trap_bait"
    # Charging: back out of their column.
    act = step(_traps(_ram(room=ROOM_L3_COMPASS, x=44, y=141, keys=1), 1))
    assert act.action == nes_action("RIGHT")
    # Returning but still on the door row: wait.
    act = step(_traps(_ram(room=ROOM_L3_COMPASS, x=56, y=141, keys=1), 2, 133, 149))
    assert act.reason == "key_59_trap_wait"
    # Returning and clear of the row: go for the door.
    act = step(_traps(_ram(room=ROOM_L3_COMPASS, x=56, y=141, keys=1), 2, 110, 172))
    assert act.action == nes_action("LEFT") and act.reason == "key_59_approach"


def test_manhandla_live_heads() -> None:
    ram = _ram(room=ROOM_L3_BOSS, x=120, y=141)
    for slot, hp in ((1, 64), (2, 64), (3, 0)):
        ram[ADDR_OBJ_TYPE + slot] = MANHANDLA_OBJECT_TYPE
        ram[ADDR_OBJ_HP + slot] = hp
    heads = level3_manhandla_live(read_snapshot(ram))
    assert len(heads) == 2
    assert all(o.hp > 0 for o in heads)
