from __future__ import annotations

import numpy as np

from zelda_i.level1 import (
    FIRST_KEY_ITEM_ID,
    ROOM_ENTRANCE,
    ROOM_FIRST_KEY,
    ROOM_KEY_STALFOS,
    ROOM_NORTH_STALFOS,
    STALFOS_OBJECT_TYPE,
    Level1Clear63Controller,
    Level1Clear63Phase,
    Level1Clear53Controller,
    Level1Clear53Phase,
    Level1FirstKeyController,
    Level1KeyPhase,
    Level1UnlockNorthController,
    return_west_waypoints,
)
from retro_harness.nes import nes_action
from zelda_i.ram import (
    ADDR_HEALTH,
    ADDR_KEYS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_ROOM_ALL_DEAD,
    ADDR_ROOM_ITEM_ID,
    ADDR_ROOM_OBJ_COUNT,
    ADDR_SCREEN,
    PLAY_MODE,
    read_snapshot,
)


def _ram(
    *,
    room: int = ROOM_ENTRANCE,
    keys: int = 0,
    x: int = 120,
    y: int = 205,
    stalfos: int = 0,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = 1
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_HEALTH] = 0x21
    ram[ADDR_KEYS] = keys
    ram[ADDR_ROOM_ITEM_ID] = FIRST_KEY_ITEM_ID
    ram[ADDR_ROOM_OBJ_COUNT] = stalfos
    for slot in range(1, stalfos + 1):
        ram[ADDR_OBJ_TYPE + slot] = STALFOS_OBJECT_TYPE
        ram[ADDR_OBJ_HP + slot] = 0x20
        ram[ADDR_LINK_X + slot] = 48 + slot * 16
        ram[ADDR_LINK_Y + slot] = 109 + slot * 8
    return ram


def test_controller_detects_key_room_combat() -> None:
    controller = Level1FirstKeyController()
    action = controller.step(
        read_snapshot(_ram(room=ROOM_FIRST_KEY, x=16, y=141, stalfos=5))
    )
    assert controller.phase is Level1KeyPhase.FIGHT_KEY_CARRIER
    assert action.reason.startswith("key_room_patrol")


def test_controller_collects_after_room_clear() -> None:
    controller = Level1FirstKeyController(
        phase=Level1KeyPhase.FIGHT_KEY_CARRIER,
        phase_frames=61,
    )
    snap = read_snapshot(_ram(room=ROOM_FIRST_KEY, x=112, y=168, stalfos=5))
    cleared_ram = _ram(room=ROOM_FIRST_KEY, x=112, y=168, stalfos=5)
    cleared_ram[ADDR_OBJ_TYPE + 1] = 0
    cleared_ram[ADDR_OBJ_HP + 1] = 0
    cleared_ram[ADDR_LINK_X + 1] = 107
    cleared_ram[ADDR_LINK_Y + 1] = 189
    controller.step(snap)
    controller.phase_frames = 61
    action = controller.step(read_snapshot(cleared_ram))
    assert controller.phase is Level1KeyPhase.COLLECT_KEY
    assert action.reason == "collect_key"


def test_return_west_from_diamond_y_goes_up_first() -> None:
    """Live stall: first-key DONE at (184, 109) then DOWN into the east diamond."""
    north = return_west_waypoints(184, 109)
    south = return_west_waypoints(184, 173)
    assert north[0] == (184, 101)
    assert south[0] == (184, 181)
    controller = Level1UnlockNorthController()
    action = controller.step(
        read_snapshot(_ram(room=ROOM_FIRST_KEY, keys=1, x=184, y=109))
    )
    assert action.reason.startswith("return_west")
    assert list(action.action) == list(nes_action("UP"))


def test_clear63_controller_engages_nearby_stalfos() -> None:
    controller = Level1Clear63Controller()
    ram = _ram(room=ROOM_NORTH_STALFOS, x=120, y=165, stalfos=3)
    # Place one Stalfos adjacent so engage beats patrol.
    ram[ADDR_LINK_X + 1] = 128
    ram[ADDR_LINK_Y + 1] = 165
    action = controller.step(read_snapshot(ram))
    assert controller.phase is Level1Clear63Phase.FIGHT
    assert action.reason.startswith("clear_engage")
    assert controller.last_live_stalfos == 3


def test_clear53_controller_routes_around_room63_blocks() -> None:
    controller = Level1Clear53Controller()
    action = controller.step(
        read_snapshot(_ram(room=ROOM_NORTH_STALFOS, x=72, y=125, stalfos=0))
    )
    assert controller.phase is Level1Clear53Phase.ROUTE_NORTH
    assert action.reason == "route_room53"


def test_clear53_controller_fights_then_targets_fixed_key() -> None:
    controller = Level1Clear53Controller(
        phase=Level1Clear53Phase.FIGHT,
        initial_keys=0,
    )
    live_ram = _ram(room=ROOM_KEY_STALFOS, x=120, y=205, stalfos=5)
    action = controller.step(read_snapshot(live_ram))
    assert action.reason.startswith("room53_clear_")
    assert controller.max_live_stalfos == 5

    cleared_ram = _ram(room=ROOM_KEY_STALFOS, x=88, y=141, stalfos=0)
    cleared_ram[ADDR_ROOM_ALL_DEAD] = 24
    action = controller.step(read_snapshot(cleared_ram))
    assert controller.phase is Level1Clear53Phase.COLLECT_KEY
    assert controller.clear_signal_seen is True
    assert action.reason == "collect_room53_key"
