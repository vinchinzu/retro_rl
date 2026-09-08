from __future__ import annotations

import numpy as np

from zelda_i.level1.clear import (
    Level1Clear53Controller,
    Level1Clear53Phase,
    Level1Clear63Controller,
    Level1Clear63Phase,
)
from zelda_i.level1.finish import (
    Level1Room42ExitController,
    Room42ExitPhase,
    ROOM_42_LEFT_DOOR_BIT,
    ROOM_GEL_SWITCH,
)
from zelda_i.level1.path import (
    FIRST_KEY_ITEM_ID,
    ROOM_ENTRANCE,
    ROOM_FIRST_KEY,
    ROOM_KEY_STALFOS,
    ROOM_NORTH_STALFOS,
    ROOM_WEST_KEY,
    STALFOS_OBJECT_TYPE,
    Level1FirstKeyController,
    Level1KeyPhase,
    Level1Resume53Controller,
    Level1ToEntranceController,
    Level1UnlockNorthController,
    Level1WestDoorController,
    Level1WestKeyReturnController,
    east_door_step,
    level1_survival_west_key_stages,
    level1_west_key_stages,
    return_west_waypoints,
    west_door_step,
)
from retro_harness.nes import nes_action
from zelda_i.ram import (
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    ADDR_ROOM_ALL_DEAD,
    ADDR_ROOM_OBJ_COUNT,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {
    "mode": PLAY_MODE,
    "level": 1,
    "screen": ROOM_ENTRANCE,
    "x": 120,
    "y": 205,
    "health": 0x21,
    "keys": 0,
    "item": FIRST_KEY_ITEM_ID,
}


def _ram(
    *,
    room: int = ROOM_ENTRANCE,
    keys: int = 0,
    x: int = 120,
    y: int = 205,
    stalfos: int = 0,
) -> np.ndarray:
    ram = make_ram(_DEFAULTS, screen=room, keys=keys, x=x, y=y)
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


def test_west_door_step_from_entrance_leftover() -> None:
    mouth = west_door_step(
        read_snapshot(_ram(room=ROOM_ENTRANCE, x=120, y=205))
    )
    assert mouth.reason == "west_leave_mouth"
    assert list(mouth.action) == list(nes_action("UP"))
    approach = west_door_step(
        read_snapshot(_ram(room=ROOM_ENTRANCE, x=120, y=149))
    )
    assert approach.reason == "west_approach"
    assert list(approach.action) == list(nes_action("LEFT"))
    arrived = west_door_step(
        read_snapshot(_ram(room=ROOM_WEST_KEY, x=224, y=141))
    )
    assert arrived.reason == "west_arrived"


def test_west_door_controller_arrives_in_0x72() -> None:
    ctrl = Level1WestDoorController()
    action = ctrl.step(
        read_snapshot(_ram(room=ROOM_WEST_KEY, x=224, y=141))
    )
    assert ctrl.success is True
    assert action.reason == "west_arrived"


def test_east_door_step_returns_to_entrance() -> None:
    align = east_door_step(
        read_snapshot(_ram(room=ROOM_WEST_KEY, x=128, y=141))
    )
    assert align.reason == "east_approach"
    assert list(align.action) == list(nes_action("RIGHT"))
    arrived = east_door_step(
        read_snapshot(_ram(room=ROOM_ENTRANCE, x=16, y=141))
    )
    assert arrived.reason == "east_arrived"
    ctrl = Level1WestKeyReturnController()
    action = ctrl.step(
        read_snapshot(_ram(room=ROOM_ENTRANCE, x=16, y=141))
    )
    assert ctrl.success is True
    assert action.reason == "east_arrived"


def test_to_entrance_and_resume53_are_instant_on_dest() -> None:
    to_ent = Level1ToEntranceController()
    action = to_ent.step(
        read_snapshot(_ram(room=ROOM_ENTRANCE, x=120, y=205))
    )
    assert to_ent.success is True
    assert action.reason == "at_entrance"
    resume = Level1Resume53Controller()
    action = resume.step(
        read_snapshot(_ram(room=ROOM_KEY_STALFOS, x=128, y=109))
    )
    assert resume.success is True
    assert action.reason == "resumed_53"


def test_to_entrance_walks_south_from_0x53() -> None:
    ctrl = Level1ToEntranceController()
    action = ctrl.step(
        read_snapshot(_ram(room=ROOM_KEY_STALFOS, x=128, y=109))
    )
    assert ctrl.success is False
    assert action.reason == "south53"
    assert list(action.action) == list(nes_action("LEFT"))


def test_west_key_stages_clear_then_return() -> None:
    names = [name for name, _, _ in level1_west_key_stages()]
    assert names == ["enter72", "clear72_key", "return73"]
    survival = [name for name, _, _ in level1_survival_west_key_stages()]
    assert survival[0] == "to_entrance"
    assert survival[-1] == "resume53"
    assert survival.index("clear72_key") < survival.index("return73")


def test_exit42_skips_old_man_hint() -> None:
    assert not hasattr(Room42ExitPhase, "ENTER_HINT")
    ctl = Level1Room42ExitController()
    ram = make_ram(
        _DEFAULTS,
        screen=ROOM_GEL_SWITCH,
        x=112,
        y=149,
        doors=ROOM_42_LEFT_DOOR_BIT,
    )
    ctl.step(read_snapshot(ram))
    assert ctl.phase is Room42ExitPhase.ROUTE_EAST
    assert "center_block_pushed" in ctl.notes


def test_clean_clear44_uses_west_mouth_controller() -> None:
    from zelda_i.level1.east_dungeon import Room44SurvivalController
    from zelda_i.level1.finish import level1_triforce_stages

    clean = {
        name: ctl for name, ctl, _ in level1_triforce_stages(natural_entry=True)
    }
    assert isinstance(clean["clear44"], Room44SurvivalController)
