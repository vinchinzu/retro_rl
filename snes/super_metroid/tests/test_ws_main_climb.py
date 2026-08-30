"""Contract tests for powered Main Shaft → Attic (rr-kw8t hop 2). No emulator."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from retro_harness.controls import pressed_snes_buttons

from super_metroid.combat.enemies import Enemy, list_enemies
from super_metroid.hop_glance import LeaveMiss
from super_metroid.leave_specs import WS_MAIN_GRATE_SEAT
from super_metroid.ram import FACING_LEFT, FACING_RIGHT, GameplayPhase, parse_state
from super_metroid.routes.kpdr.wrecked_ship.ws_main_actions import (
    attic_door_action,
    climb_action,
    wall_up_shot_action,
)
from super_metroid.routes.kpdr.wrecked_ship.ws_main_climb import (
    play_ws_main_to_attic,
    ws_main_attic_settled,
)
from super_metroid.routes.kpdr.wrecked_ship.ws_main_geometry import (
    FIRST_JUMP_LAND_X,
    FIRST_JUMP_LAND_Y,
    GRATE_LAND_X,
    GRATE_LAND_Y,
    SHAFT_HOPS,
    SLOPE_651_TAKEOFF,
    SLOPE_827_TAKEOFF,
    SLOPE_1019_TAKEOFF,
    SLOPE_1130_TAKEOFF,
    STAIRS_1543_TAKEOFF,
    WS_MAIN_ATTIC_DOOR_X,
    WS_MAIN_PHASES,
    ShaftRegion,
    at_ws_main_grate_seat,
    at_ws_main_usable_grate_seat,
    at_ws_main_left_platform,
    at_ws_main_mid_climb,
    at_ws_main_morph_drop,
    at_ws_main_pit,
    at_ws_main_slope_651,
    at_ws_main_slope_827,
    at_ws_main_slope_1019,
    at_ws_main_slope_1130,
    at_ws_main_stairs_1543,
    at_ws_main_west_super_band,
    classify_region,
    classify_ws_main_phase,
    ws_main_phase_index,
)
from super_metroid.routes.kpdr.wrecked_ship.ws_main_ice import (
    ATOMIC_ID,
    ice_keepaway_action,
)
from super_metroid.routes.kpdr.wrecked_ship.ws_main_shaft import (
    SAVE_COLUMN_WJ,
    at_ws_main_save_alcove,
    at_ws_main_save_column_wj,
    climb_until,
    note_upper_wall,
    save_alcove_jump,
    save_column_walljump,
    upper_wall_open,
)
from super_metroid.routes.kpdr.room_ids import ROOM_WS_ATTIC, ROOM_WS_MAIN
from super_metroid.routes.skills.basic_moves import shoot_up_action
from super_metroid.routes.skills.charge_shot import CHARGE_FULL
from super_metroid.routes.skills.geometry import PhaseStop


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_WS_MAIN,
        "samus_x": 1173,
        "samus_y": 1979,
        "pose": 1,
        "facing": FACING_RIGHT,
        "health": 299,
        "max_health": 299,
        "game_state": 8,
        "door_transition": 0,
        "selected_item": 0,
        "boss_bits": (0, 0, 0, 0x01, 0, 0, 0, 0),
    }
    values.update(overrides)
    return replace(base, **values)


class _Session:
    def __init__(self, state):
        self.state = state
        self.frame = state.frame
        self.actions = []
        self.env = None

    def step(self, action, reason):
        self.actions.append((action, reason))
        self.frame += 1
        self.state = replace(self.state, frame=self.frame)
        return self.state


def test_attic_settled_requires_gs8() -> None:
    trans = _state(room_id=ROOM_WS_ATTIC, game_state=11, door_transition=1)
    assert not ws_main_attic_settled(trans)
    gs11 = _state(room_id=ROOM_WS_ATTIC, game_state=11, door_transition=0)
    assert not ws_main_attic_settled(gs11)
    ordinary = _state(
        room_id=ROOM_WS_ATTIC, samus_x=1135, samus_y=184, pose=1, game_state=8
    )
    assert ws_main_attic_settled(ordinary)
    assert not ws_main_attic_settled(_state())
    session = _Session(ordinary)
    assert play_ws_main_to_attic(session).room_id == ROOM_WS_ATTIC
    assert session.actions == []


def test_classifier_regions_and_phases() -> None:
    assert WS_MAIN_PHASES == (
        "pit_shot",
        "grate_seat",
        "west_super",
        "mid_climb",
        "attic_seat",
        "attic_door",
    )
    assert ws_main_phase_index("grate-seat") == 1

    pin = _state()
    assert classify_region(pin) is ShaftRegion.PIT
    assert classify_ws_main_phase(pin) == "pit_shot"
    assert at_ws_main_pit(pin)
    assert not at_ws_main_grate_seat(pin)

    pocket = _state(samus_x=1177, samus_y=1883, pose=2, velocity_y=0)
    assert not at_ws_main_grate_seat(pocket)
    assert classify_region(pocket) is ShaftRegion.PIT
    assert classify_ws_main_phase(pocket) == "pit_shot"

    fire = _state(samus_x=1223, samus_y=1860, pose=3, velocity_y=0)
    assert at_ws_main_grate_seat(fire)
    assert at_ws_main_usable_grate_seat(fire)
    assert classify_region(fire) is ShaftRegion.GRATE_SEAT
    assert classify_ws_main_phase(fire) == "grate_seat"

    take04 = _state(samus_x=1195, samus_y=1883, pose=3, velocity_y=0)
    assert at_ws_main_grate_seat(take04)
    assert not at_ws_main_usable_grate_seat(take04)
    assert classify_ws_main_phase(take04) == "grate_seat"

    land = _state(samus_x=1189, samus_y=1883, pose=2, velocity_y=0)
    assert at_ws_main_grate_seat(land)
    assert not at_ws_main_usable_grate_seat(land)
    assert classify_ws_main_phase(land) == "grate_seat"
    assert GRATE_LAND_X == (1188, 1232)
    assert GRATE_LAND_Y == (1852, 1888)
    assert FIRST_JUMP_LAND_X == GRATE_LAND_X
    assert FIRST_JUMP_LAND_Y == GRATE_LAND_Y
    assert FIRST_JUMP_LAND_X != WS_MAIN_GRATE_SEAT.x
    assert FIRST_JUMP_LAND_Y != WS_MAIN_GRATE_SEAT.y

    stairs = _state(samus_x=1111, samus_y=1899, pose=157, velocity_y=0)
    assert classify_region(stairs) is ShaftRegion.PIT
    assert classify_region(stairs, lip_hit=True) is ShaftRegion.PIT
    assert classify_ws_main_phase(stairs) == "pit_shot"
    assert not at_ws_main_left_platform(1111, 1899, 157)
    assert at_ws_main_left_platform(1082, 1878, 2)
    shelf = _state(samus_x=1082, samus_y=1878, pose=2, velocity_y=0)
    assert classify_region(shelf) is ShaftRegion.PIT
    assert classify_region(shelf, lip_hit=True) is ShaftRegion.SHELF

    west = _state(samus_x=1152, samus_y=1675, pose=10)
    assert at_ws_main_west_super_band(west)
    assert classify_ws_main_phase(west) == "west_super"
    assert classify_region(west) is ShaftRegion.SHAFT
    assert not at_ws_main_west_super_band(_state(room_id=0xCDA8, samus_y=1675))

    mid = _state(samus_x=1152, samus_y=680, pose=1)
    assert at_ws_main_mid_climb(mid)
    assert classify_ws_main_phase(mid) == "mid_climb"
    planted_640 = _state(samus_x=1131, samus_y=640, pose=2)
    assert at_ws_main_mid_climb(planted_640)
    air_640 = _state(samus_x=1117, samus_y=640, pose=47, velocity_y=0)
    assert not at_ws_main_mid_climb(air_640)

    seat = _state(samus_x=1135, samus_y=80, pose=1)
    assert classify_ws_main_phase(seat) == "attic_seat"
    assert classify_region(seat) is ShaftRegion.ATTIC_SEAT

    attic = _state(room_id=ROOM_WS_ATTIC, samus_x=1135, samus_y=184, pose=1)
    assert classify_ws_main_phase(attic) == "attic_door"
    assert classify_region(attic) is ShaftRegion.ATTIC


def test_two_hop_take02() -> None:
    pin = climb_action(1173, 1979, 1, FACING_LEFT)
    assert pin == ("LEFT",)
    assert "A" not in pin
    short = climb_action(1166, 1979, 2, FACING_LEFT)
    assert short == ("A",)
    assert "RIGHT" not in short
    land = climb_action(1166, 1979, 1, FACING_RIGHT)
    assert land == ("LEFT",)
    assert "A" not in land
    committed = climb_action(1156, 1979, 1, FACING_RIGHT)
    assert committed == ("A",)
    rise = climb_action(1156, 1920, 77, FACING_RIGHT, velocity_y=5)
    assert rise == ("A",)
    over = climb_action(1156, 1880, 81, FACING_RIGHT, velocity_y=3)
    assert over == ("RIGHT", "A")
    pocket = climb_action(1177, 1883, 2, FACING_RIGHT)
    assert pocket == ("A",)
    assert "LEFT" not in pocket
    stairs = climb_action(1111, 1899, 157, FACING_LEFT)
    assert stairs[0] == "RIGHT"
    assert "X" not in stairs
    shelf_recover = climb_action(1082, 1878, 2, FACING_RIGHT)
    assert shelf_recover[0] == "RIGHT"
    assert climb_action(1082, 1878, 2, FACING_RIGHT, lip_hit=True) == ("A",)
    for x, y, pose, facing, vy in (
        (1173, 1979, 1, FACING_LEFT, 0),
        (1166, 1979, 2, FACING_LEFT, 0),
        (1166, 1979, 1, FACING_RIGHT, 0),
        (1156, 1979, 1, FACING_RIGHT, 0),
        (1156, 1920, 77, FACING_RIGHT, 5),
        (1111, 1899, 157, FACING_LEFT, 0),
    ):
        names = climb_action(x, y, pose, facing, velocity_y=vy)
        assert "L" not in names
        assert "DOWN" not in names
        assert "X" not in names


def test_fire_slope_walks_to_take02_seat_before_shoot() -> None:
    land = climb_action(1189, 1883, 2, FACING_LEFT)
    assert land == ("RIGHT",)
    assert "X" not in land
    assert "A" not in land
    take04_low = climb_action(1195, 1883, 3, FACING_RIGHT)
    assert take04_low == ("RIGHT",)
    assert "X" not in take04_low
    mid = climb_action(1210, 1868, 9, FACING_RIGHT)
    assert mid == ("RIGHT",)
    near = climb_action(1227, 1856, 4, FACING_LEFT)
    assert near == ("LEFT",)
    assert "X" not in near
    fire_face = climb_action(1223, 1860, 4, FACING_LEFT)
    assert fire_face == ("RIGHT",)
    assert "X" not in fire_face


def test_fire_slope_shoots_up_until_lip_hit() -> None:
    take02 = climb_action(1223, 1860, 3, FACING_RIGHT)
    assert take02 == shoot_up_action()
    assert "DOWN" not in take02
    jumped = climb_action(1231, 1852, 3, FACING_LEFT, lip_hit=True)
    assert jumped == ("LEFT", "A")
    assert "DOWN" not in jumped
    right_facing_take02 = climb_action(
        1231, 1852, 3, FACING_RIGHT, lip_hit=True
    )
    assert right_facing_take02 == ("LEFT", "A")
    assert "DOWN" not in right_facing_take02
    slope_walk = climb_action(1228, 1856, 9, FACING_RIGHT, lip_hit=True)
    assert slope_walk == ("UP", "RIGHT")
    assert "A" not in slope_walk
    coast = climb_action(
        1230,
        1853,
        15,
        FACING_RIGHT,
        lip_hit=True,
        region=ShaftRegion.GRATE_SEAT,
    )
    assert coast == ("UP",)
    above_window = climb_action(
        1228,
        1800,
        77,
        FACING_LEFT,
        velocity_y=3,
        lip_hit=True,
        region=ShaftRegion.GRATE_SEAT,
    )
    assert above_window == ()
    before = climb_action(1223, 1860, 3, FACING_LEFT, lip_hit=True)
    assert before == ("UP", "RIGHT")
    assert "A" not in before
    assert "DOWN" not in climb_action(1223, 1860, 3, FACING_RIGHT, charge=CHARGE_FULL)


def test_morph_drop_only_after_lip_hit() -> None:
    assert at_ws_main_morph_drop(1189, 1785, 2)
    assert not at_ws_main_morph_drop(1223, 1860, 3)
    assert "DOWN" not in climb_action(1189, 1785, 2, FACING_LEFT)
    assert climb_action(1189, 1785, 2, FACING_LEFT, lip_hit=True) == ("DOWN",)
    assert "DOWN" not in climb_action(1223, 1860, 3, FACING_RIGHT, lip_hit=True)


def test_shaft_hops_are_dpad_sides() -> None:
    assert SHAFT_HOPS
    for hop in SHAFT_HOPS:
        assert hop.side in ("LEFT", "RIGHT")
        assert hop.takeoff.side in ("LEFT", "RIGHT")


def test_west_super_hop_aligns_from_recorded_left_edge() -> None:
    hop = SHAFT_HOPS[0]
    assert hop.y == 1675
    assert hop.takeoff.x_range == (1054, 1074)
    assert climb_action(
        1108, 1675, 2, FACING_RIGHT, region=ShaftRegion.SHAFT
    ) == ("LEFT",)
    assert climb_action(
        1062, 1675, 2, FACING_LEFT, region=ShaftRegion.SHAFT
    ) == ("RIGHT",)
    assert climb_action(
        1062, 1675, 2, FACING_RIGHT, region=ShaftRegion.SHAFT
    ) == ("RIGHT", "A")
    assert climb_action(
        1070, 1675, 2, FACING_LEFT, region=ShaftRegion.SHAFT
    ) == ("RIGHT",)
    assert climb_action(
        1071, 1675, 1, FACING_RIGHT, region=ShaftRegion.SHAFT
    ) == ("RIGHT", "A")
    assert climb_action(
        1067, 1675, 1, FACING_RIGHT, region=ShaftRegion.SHAFT
    ) == ("RIGHT", "A")


def test_west_super_airborne_lands_toward_takeoff() -> None:
    """Natural west_super pin is airborne; mount 1675 then walk to 1062."""
    pin = climb_action(
        1094, 1700, 48, FACING_RIGHT, velocity_y=2, region=ShaftRegion.SHAFT
    )
    assert pin == ("LEFT", "A")
    settled = climb_action(
        1095, 1690, 81, FACING_RIGHT, velocity_y=-2, region=ShaftRegion.SHAFT
    )
    assert settled == ("LEFT", "A")
    above = climb_action(
        1106, 1651, 81, FACING_RIGHT, velocity_y=2, region=ShaftRegion.SHAFT
    )
    assert above == ("RIGHT", "A")
    assert climb_action(
        1065, 1675, 75, FACING_RIGHT, velocity_y=6, region=ShaftRegion.SHAFT
    ) == ("RIGHT", "A")
    midair = climb_action(
        1100, 1600, 81, FACING_RIGHT, velocity_y=3, region=ShaftRegion.SHAFT
    )
    assert midair == ("RIGHT", "A")
    low = climb_action(1099, 1711, 2, FACING_LEFT, region=ShaftRegion.SHAFT)
    assert low == ("RIGHT",)
    assert climb_action(
        1099, 1711, 1, FACING_RIGHT, region=ShaftRegion.SHAFT
    ) == ("RIGHT", "A")


def test_stairs_1543_dashes_far_right_not_atomic_overlap() -> None:
    """Takes 02–05 plant then dash to x~1252–1259; guessed x1150 LEFT is out."""
    hop = next(h for h in SHAFT_HOPS if h.y == 1543)
    assert hop.takeoff.x_range == STAIRS_1543_TAKEOFF.x_range
    assert hop.takeoff.x_range == (1248, 1260)
    assert hop.side == "LEFT"
    assert hop.x_hi >= 1259
    assert at_ws_main_stairs_1543(1129, 1587, 9)
    assert at_ws_main_stairs_1543(1255, 1547, 9)
    assert not at_ws_main_stairs_1543(1154, 1561, 76, velocity_y=2)

    plant = climb_action(1129, 1587, 9, FACING_RIGHT, region=ShaftRegion.SHAFT)
    assert plant == ("RIGHT", "B")
    assert "LEFT" not in plant
    run = climb_action(1150, 1547, 9, FACING_RIGHT, region=ShaftRegion.SHAFT)
    assert run == ("RIGHT", "B")
    leftover = climb_action(
        1154, 1561, 76, FACING_RIGHT, velocity_y=2, region=ShaftRegion.SHAFT
    )
    assert leftover == ("RIGHT",)
    assert "LEFT" not in leftover
    assert climb_action(
        1248, 1549, 42, FACING_LEFT, region=ShaftRegion.SHAFT
    ) == ("LEFT", "A")
    air_takeoff = climb_action(
        1232, 1528, 84, FACING_RIGHT, velocity_y=2, region=ShaftRegion.SHAFT
    )
    assert air_takeoff == ("LEFT",)
    peak = climb_action(
        1171, 1503, 84, FACING_RIGHT, velocity_y=-2, region=ShaftRegion.SHAFT
    )
    assert peak == ("RIGHT", "A")
    turn = climb_action(1255, 1547, 9, FACING_RIGHT, region=ShaftRegion.SHAFT)
    assert turn == ("LEFT",)
    launch = climb_action(1255, 1547, 9, FACING_LEFT, region=ShaftRegion.SHAFT)
    assert launch == ("LEFT", "B", "A")
    save_guard = climb_action(1255, 1547, 9, FACING_RIGHT, region=ShaftRegion.SHAFT)
    assert save_guard != ("LEFT", "B")


def test_slope_1130_dashes_left_not_jump_from_plant() -> None:
    """Takes 02/03 plant (1133, 1130) then B+LEFT to wall (1045, 1083)."""
    hop = next(h for h in SHAFT_HOPS if h.y == 1130)
    assert hop.takeoff.x_range == SLOPE_1130_TAKEOFF.x_range
    assert hop.takeoff.x_range == (1044, 1046)
    assert hop.side == "LEFT"
    assert hop.x_lo <= 1045
    assert at_ws_main_slope_1130(1133, 1130, 10)
    assert at_ws_main_slope_1130(1099, 1095, 38)
    assert at_ws_main_slope_1130(1045, 1083, 10)
    assert not at_ws_main_slope_1130(1098, 1019, 9)
    assert not at_ws_main_slope_1130(1136, 1082, 83, velocity_y=-2)

    plant = climb_action(1133, 1130, 10, FACING_LEFT, region=ShaftRegion.SHAFT)
    assert plant == ("LEFT", "B")
    assert "A" not in plant
    leftover = climb_action(
        1099, 1095, 38, FACING_RIGHT, movement_type=14, region=ShaftRegion.SHAFT
    )
    assert leftover == ("LEFT",)
    assert "RIGHT" not in leftover
    assert "A" not in leftover
    wall = climb_action(1045, 1083, 10, FACING_LEFT, region=ShaftRegion.SHAFT)
    assert wall == ("LEFT", "B")
    assert "A" not in wall
    wall_air = climb_action(
        1045, 1083, 26, FACING_LEFT, velocity_y=7, region=ShaftRegion.SHAFT
    )
    assert wall_air == ("LEFT", "A")
    assert "RIGHT" not in wall_air
    bounce = climb_action(
        1045, 1083, 76, FACING_LEFT, velocity_y=6, region=ShaftRegion.SHAFT
    )
    assert bounce == ("LEFT", "A")
    amid = climb_action(
        1045, 1077, 78, FACING_LEFT, velocity_y=5, region=ShaftRegion.SHAFT
    )
    assert amid == ("A",)
    turn = climb_action(
        1045, 1072, 48, FACING_LEFT, velocity_y=5, region=ShaftRegion.SHAFT
    )
    assert turn == ("RIGHT", "A")
    assert "LEFT" not in turn
    rise = climb_action(
        1045, 1066, 48, FACING_RIGHT, velocity_y=5, region=ShaftRegion.SHAFT
    )
    assert rise == ("RIGHT", "A")
    boost = climb_action(
        1045, 1044, 48, FACING_RIGHT, velocity_y=5, region=ShaftRegion.SHAFT
    )
    assert boost == ("B", "RIGHT", "A")
    incoming = climb_action(
        1194, 1130, 26, FACING_LEFT, velocity_y=5, region=ShaftRegion.SHAFT
    )
    assert incoming == ("LEFT", "B", "A")
    land_air = climb_action(
        1135, 1137, 26, FACING_LEFT, velocity_y=4, region=ShaftRegion.SHAFT
    )
    assert "LEFT" in land_air
    assert "RIGHT" not in land_air
    over_1019 = climb_action(
        1136, 988, 47, FACING_LEFT, velocity_y=0, region=ShaftRegion.SHAFT
    )
    assert over_1019 == ("RIGHT", "A")
    assert "LEFT" not in over_1019
    over_1019_face = climb_action(
        1136, 988, 47, FACING_RIGHT, velocity_y=0, region=ShaftRegion.SHAFT
    )
    assert over_1019_face == ("B", "RIGHT")
    assert "A" not in over_1019_face
    peak = climb_action(
        1062, 980, 81, FACING_RIGHT, velocity_y=3, region=ShaftRegion.SHAFT
    )
    assert peak == ("B", "RIGHT", "A")
    land = climb_action(1098, 1019, 9, FACING_RIGHT, region=ShaftRegion.SHAFT)
    assert land == ("RIGHT", "B")
    assert "A" not in land
    assert climb_action(1152, 1131, 2, FACING_LEFT, region=ShaftRegion.SHAFT) == (
        "LEFT",
        "B",
    )


def test_slope_1019_dashes_right_not_jump_from_plant() -> None:
    """Takes 02–05 plant (1098, 1019) then B+RIGHT to wall (1243, 907)."""
    hop = next(h for h in SHAFT_HOPS if h.y == 1019)
    assert hop.takeoff.x_range == SLOPE_1019_TAKEOFF.x_range
    assert hop.takeoff.x_range == (1240, 1246)
    assert hop.side == "LEFT"
    assert hop.x_hi >= 1243
    assert at_ws_main_slope_1019(1098, 1019, 9)
    assert at_ws_main_slope_1019(1228, 907, 9)
    assert at_ws_main_slope_1019(1243, 907, 9)
    assert not at_ws_main_slope_1019(1205, 827, 10)
    assert not at_ws_main_slope_1019(1187, 817, 48, velocity_y=0)

    plant = climb_action(1098, 1019, 9, FACING_RIGHT, region=ShaftRegion.SHAFT)
    assert plant == ("RIGHT", "B")
    assert "A" not in plant
    mid = climb_action(1180, 936, 9, FACING_RIGHT, region=ShaftRegion.SHAFT)
    assert mid == ("RIGHT", "B")
    assert "A" not in mid
    turn = climb_action(1243, 907, 9, FACING_RIGHT, region=ShaftRegion.SHAFT)
    assert turn == ("LEFT",)
    assert "A" not in turn
    bounce = climb_action(
        1243, 890, 47, FACING_LEFT, velocity_y=5, region=ShaftRegion.SHAFT
    )
    assert bounce == ("A",)
    assert "LEFT" not in bounce
    assert "RIGHT" not in bounce
    drift = climb_action(
        1230, 825, 82, FACING_LEFT, velocity_y=4, region=ShaftRegion.SHAFT
    )
    assert drift == ("B", "LEFT")
    assert "A" not in drift
    wall_rise = climb_action(
        1243, 830, 47, FACING_LEFT, velocity_y=4, region=ShaftRegion.SHAFT
    )
    assert "A" in wall_rise
    peak = climb_action(
        1226, 816, 82, FACING_LEFT, velocity_y=0, region=ShaftRegion.SHAFT
    )
    assert peak == ("B", "LEFT")
    assert "A" not in peak
    leftover = climb_action(
        1187, 817, 48, FACING_RIGHT, velocity_y=0, region=ShaftRegion.SHAFT
    )
    assert leftover == ("LEFT",)
    assert "RIGHT" not in leftover
    land = climb_action(1205, 827, 10, FACING_LEFT, region=ShaftRegion.SHAFT)
    assert land == ("LEFT", "B")
    assert "A" not in land
    land165 = climb_action(1220, 827, 165, FACING_LEFT, region=ShaftRegion.SHAFT)
    assert land165 == ("LEFT", "B")
    assert "A" not in land165


def test_slope_827_dashes_left_not_jump_from_plant() -> None:
    """Tape take02 plants (1205, 827) p10 then B+LEFT to (1061, 763)."""
    hop = next(h for h in SHAFT_HOPS if h.y == 827)
    assert hop.takeoff.x_range == SLOPE_827_TAKEOFF.x_range
    assert hop.side == "RIGHT"
    assert at_ws_main_slope_827(1205, 827, 10)
    assert at_ws_main_slope_827(1220, 827, 165)
    assert at_ws_main_slope_827(1061, 763, 10)
    plant = climb_action(1205, 827, 10, FACING_LEFT, region=ShaftRegion.SHAFT)
    assert plant == ("LEFT", "B")
    assert "A" not in plant
    mid = climb_action(1148, 792, 10, FACING_LEFT, region=ShaftRegion.SHAFT)
    assert mid == ("LEFT", "B")
    assert "A" not in mid


def test_slope_651_dashes_right_not_jump_from_plant() -> None:
    """Tape take02 plants (1098, 651) p9 then B+RIGHT to wall (1243, 587)."""
    hop = next(h for h in SHAFT_HOPS if h.y == 651)
    assert hop.takeoff.x_range == SLOPE_651_TAKEOFF.x_range
    assert hop.takeoff.x_range == (1228, 1234)
    assert hop.side == "LEFT"
    assert hop.x_hi >= 1243
    assert at_ws_main_slope_651(1098, 651, 9)
    assert at_ws_main_slope_651(1101, 651, 9)
    assert at_ws_main_slope_651(1243, 587, 9)
    assert at_ws_main_slope_651(1144, 627, 164)
    assert not at_ws_main_slope_651(1061, 752, 77, velocity_y=0)

    plant = climb_action(1101, 651, 9, FACING_RIGHT, region=ShaftRegion.SHAFT)
    assert plant == ("RIGHT", "B")
    assert "A" not in plant
    mid = climb_action(1180, 610, 9, FACING_RIGHT, region=ShaftRegion.SHAFT)
    assert mid == ("RIGHT", "B")
    assert "A" not in mid
    skip = climb_action(
        1131, 640, 83, FACING_RIGHT, velocity_y=5, region=ShaftRegion.SHAFT
    )
    assert skip == ("RIGHT", "B")
    assert "A" not in skip
    wall = climb_action(1243, 587, 9, FACING_RIGHT, region=ShaftRegion.SHAFT)
    assert wall == shoot_up_action()
    assert wall_up_shot_action(0) == shoot_up_action()
    assert wall_up_shot_action(5) == ("UP",)
    assert "X" not in wall_up_shot_action(5)
    assert wall_up_shot_action(0, charge=CHARGE_FULL) == ("UP",)
    release = climb_action(
        1243,
        587,
        9,
        FACING_RIGHT,
        region=ShaftRegion.SHAFT,
        wall_shot_frame=5,
    )
    assert release == ("UP",)
    assert "X" not in release
    charged = climb_action(
        1243,
        587,
        9,
        FACING_RIGHT,
        region=ShaftRegion.SHAFT,
        charge=CHARGE_FULL,
    )
    assert charged == ("UP",)
    opened = climb_action(
        1243, 587, 9, FACING_RIGHT, region=ShaftRegion.SHAFT, ceiling_open=True
    )
    assert opened == ("LEFT",)
    assert "A" not in opened
    takeoff = climb_action(
        1231, 587, 10, FACING_LEFT, region=ShaftRegion.SHAFT, ceiling_open=True
    )
    assert takeoff == ("LEFT", "A")
    aiming = climb_action(
        1231, 587, 4, FACING_LEFT, region=ShaftRegion.SHAFT, ceiling_open=True
    )
    assert aiming == ("LEFT",)
    assert "A" not in aiming
    closed = climb_action(1231, 587, 10, FACING_LEFT, region=ShaftRegion.SHAFT)
    assert closed == shoot_up_action()
    bounce = climb_action(
        1231, 570, 26, FACING_LEFT, velocity_y=5, region=ShaftRegion.SHAFT
    )
    assert bounce == ("LEFT", "A")
    clear = climb_action(
        1210, 560, 26, FACING_LEFT, velocity_y=5, region=ShaftRegion.SHAFT
    )
    assert "A" in clear
    drift = climb_action(
        1210, 510, 82, FACING_LEFT, velocity_y=4, region=ShaftRegion.SHAFT
    )
    assert drift == ("B", "LEFT")
    assert "A" not in drift
    coast = climb_action(1202, 587, 9, FACING_RIGHT, region=ShaftRegion.SHAFT)
    assert coast == ("RIGHT",)
    assert "B" not in coast
    assert "A" not in coast
    leftover = climb_action(
        1061, 752, 77, FACING_LEFT, velocity_y=0, region=ShaftRegion.SHAFT
    )
    # 827 still owns that miss still. 651 must not spin-jump RIGHT from it.
    assert "RIGHT" not in leftover
    ledge = climb_action(1204, 523, 10, FACING_LEFT, region=ShaftRegion.SHAFT)
    assert ledge == ("LEFT", "X")
    ledge_gap = climb_action(
        1204,
        523,
        10,
        FACING_LEFT,
        region=ShaftRegion.SHAFT,
        wall_shot_frame=5,
    )
    assert ledge_gap == ("LEFT",)
    assert "X" not in ledge_gap
    takeoff_523 = climb_action(
        1077, 523, 10, FACING_LEFT, region=ShaftRegion.SHAFT
    )
    assert takeoff_523 == ("UP",)
    assert "X" not in takeoff_523
    assert "A" not in takeoff_523
    gun_523 = climb_action(1077, 523, 4, FACING_LEFT, region=ShaftRegion.SHAFT)
    assert gun_523 == ("UP", "X")
    gun_gap = climb_action(
        1077,
        523,
        4,
        FACING_LEFT,
        region=ShaftRegion.SHAFT,
        wall_shot_frame=5,
    )
    assert gun_gap == ("UP", "A")
    air_523 = climb_action(
        1077,
        500,
        22,
        FACING_LEFT,
        velocity_y=5,
        region=ShaftRegion.SHAFT,
        wall_shot_frame=5,
    )
    assert air_523 == ("UP", "A")
    jam_523 = climb_action(
        1079, 499, 144, FACING_RIGHT, region=ShaftRegion.SHAFT
    )
    assert jam_523 == ("UP", "A")
    assert "B" not in jam_523


def test_save_alcove_jumps_left() -> None:
    takeoff = _state(samus_x=1231, samus_y=1852, pose=3, facing=FACING_LEFT)
    assert not at_ws_main_save_alcove(takeoff)
    planted = _state(samus_x=1235, samus_y=1851, pose=10, facing=FACING_LEFT)
    assert at_ws_main_save_alcove(planted)
    assert classify_region(planted) is ShaftRegion.SAVE_ALCOVE
    session = _Session(planted)
    save_alcove_jump(session, "test")
    assert session.actions[0][1] == "test_alcove_jump"
    face = _Session(_state(samus_x=1235, samus_y=1851, pose=9, facing=FACING_RIGHT))
    save_alcove_jump(face, "test")
    assert face.actions[0][1] == "test_alcove_face"


def test_save_column_wj_band_excludes_lip_pocket_save_door() -> None:
    leftover = _state(samus_x=1220, samus_y=1843, pose=77, velocity_y=0)
    assert at_ws_main_save_column_wj(leftover)
    human = _state(samus_x=1216, samus_y=1852, pose=19, velocity_y=2)
    assert at_ws_main_save_column_wj(human)
    assert not at_ws_main_save_column_wj(
        _state(samus_x=1220, samus_y=1843, pose=77, velocity_y=2, facing=FACING_LEFT)
    )
    assert not at_ws_main_save_column_wj(
        _state(samus_x=1177, samus_y=1883, pose=2, velocity_y=0)
    )
    assert not at_ws_main_save_column_wj(
        _state(samus_x=1232, samus_y=1843, pose=77, velocity_y=0)
    )
    assert SAVE_COLUMN_WJ.into == "RIGHT"
    assert SAVE_COLUMN_WJ.flip == "LEFT"
    session = _Session(human)
    save_column_walljump(session, "test", lambda st: False)
    reasons = [r for _, r in session.actions]
    assert any(r.endswith("_save_wj_into") for r in reasons)
    assert any(r.endswith("_save_wj_flip") for r in reasons)


def test_climb_until_overlay_save_and_lip() -> None:
    def _stop_after_first(sess: _Session):
        return lambda st: sess.frame > 0

    alcove = _Session(
        _state(samus_x=1235, samus_y=1851, pose=10, facing=FACING_LEFT)
    )
    climb_until(alcove, "test", _stop_after_first(alcove))
    assert any("alcove" in r for _, r in alcove.actions)

    column = _Session(
        _state(samus_x=1216, samus_y=1852, pose=19, velocity_y=2)
    )
    climb_until(column, "test", _stop_after_first(column))
    assert any(r.endswith("_save_wj_into") for _, r in column.actions)

    shelf = _Session(
        _state(samus_x=1082, samus_y=1878, pose=2, facing=FACING_RIGHT)
    )
    climb_until(shelf, "test", _stop_after_first(shelf))
    assert not any("shelf_hole" in r for _, r in shelf.actions)
    assert any("_pit" in r for _, r in shelf.actions)

    stairs = _Session(
        _state(samus_x=1111, samus_y=1899, pose=157, facing=FACING_LEFT)
    )
    climb_until(stairs, "test", _stop_after_first(stairs))
    assert not any("shelf_hole" in r for _, r in stairs.actions)
    assert any("_pit" in r for _, r in stairs.actions)

    lip = _Session(
        _state(samus_x=1223, samus_y=1860, pose=3, facing=FACING_RIGHT)
    )
    climb_until(lip, "test", _stop_after_first(lip))
    assert any(r.endswith("_lip_up") for _, r in lip.actions)
    assert not any("DOWN" in str(a) for a, _ in lip.actions)

    wall_kb = _Session(
        _state(samus_x=1045, samus_y=1083, pose=138, facing=FACING_LEFT)
    )
    climb_until(wall_kb, "test", _stop_after_first(wall_kb))
    assert wall_kb.actions[0][1] == "test_slope_1130_wall"
    wall_btns = set(pressed_snes_buttons(wall_kb.actions[0][0]))
    assert "LEFT" in wall_btns
    assert "A" in wall_btns

    wall_1019 = _Session(
        _state(samus_x=1243, samus_y=907, pose=137, facing=FACING_RIGHT)
    )
    climb_until(wall_1019, "test", _stop_after_first(wall_1019))
    assert wall_1019.actions[0][1] == "test_slope_1019_wall"
    wall_1019_btns = set(pressed_snes_buttons(wall_1019.actions[0][0]))
    assert wall_1019_btns == {"A"}

    wall_523 = _Session(
        _state(samus_x=1077, samus_y=523, pose=138, facing=FACING_LEFT)
    )
    climb_until(wall_523, "test", _stop_after_first(wall_523))
    assert wall_523.actions[0][1] == "test_slope_523_wall"
    assert set(pressed_snes_buttons(wall_523.actions[0][0])) == {"LEFT", "A"}

    ram = np.zeros(0x2000, dtype=np.uint8)
    base = 0x0F78
    ram[base] = ATOMIC_ID & 0xFF
    ram[base + 1] = ATOMIC_ID >> 8
    ram[base + 0x02] = 1084 & 0xFF
    ram[base + 0x03] = 1084 >> 8
    ram[base + 0x06] = 514 & 0xFF
    ram[base + 0x07] = 514 >> 8
    ram[base + 0x14] = 250
    blocked_523 = _Session(
        _state(samus_x=1077, samus_y=523, pose=138, facing=FACING_LEFT)
    )
    blocked_523.env = type("E", (), {"get_ram": lambda self: ram})()
    climb_until(blocked_523, "test", _stop_after_first(blocked_523))
    assert blocked_523.actions[0][1] == "test_slope_523_ice"
    assert "A" not in set(pressed_snes_buttons(blocked_523.actions[0][0]))

    wall_651 = _Session(
        _state(samus_x=1243, samus_y=587, pose=137, facing=FACING_RIGHT)
    )
    climb_until(wall_651, "test", _stop_after_first(wall_651))
    assert wall_651.actions[0][1] == "test_slope_651_shot"
    wall_651_btns = set(pressed_snes_buttons(wall_651.actions[0][0]))
    assert wall_651_btns == {"UP", "X"}

    wall_cycle = _Session(
        _state(samus_x=1243, samus_y=587, pose=137, facing=FACING_RIGHT)
    )
    climb_until(wall_cycle, "test", lambda st: wall_cycle.frame >= 12)
    cycle_btns = [set(pressed_snes_buttons(a)) for a, _ in wall_cycle.actions]
    assert {"UP", "X"} in cycle_btns
    assert {"UP"} in cycle_btns


def test_knockback_latches_upper_wall_after_three_spawns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from super_metroid.routes.kpdr.wrecked_ship import ws_main_shaft as mod

    n = {"i": 0}

    def _observe(sess, prev, hit):
        del sess, prev, hit
        n["i"] += 1
        rows = (
            {"i": 1, "id": 0xD074, "px": 1240, "py": 568},
            {"i": 2, "id": 0xD080, "px": 1240, "py": 552},
            {"i": 3, "id": 0xD080, "px": 1240, "py": 536},
        )
        if n["i"] <= 3:
            return False, (), (rows[n["i"] - 1],)
        return False, (), ()

    monkeypatch.setattr(mod, "_observe_shot_blocks", _observe)
    session = _Session(
        _state(samus_x=1243, samus_y=587, pose=137, facing=FACING_RIGHT)
    )
    climb_until(session, "test", lambda st: session.frame >= 4)
    reasons = [r for _, r in session.actions]
    assert reasons[0] == "test_slope_651_shot"
    assert "test_slope_651_wall" in reasons
    wall = next(
        a for a, r in session.actions if r == "test_slope_651_wall"
    )
    assert set(pressed_snes_buttons(wall)) == {"LEFT", "A"}


def test_upper_wall_needs_three_near_spawns() -> None:
    far = {"i": 1, "id": 0xD080, "px": 904, "py": 1112}
    near_a = {"i": 2, "id": 0xD074, "px": 1240, "py": 568}
    near_b = {"i": 3, "id": 0xD080, "px": 1240, "py": 552}
    near_c = {"i": 4, "id": 0xD080, "px": 1240, "py": 536}
    cleared = note_upper_wall(set(), (far,), 1243, 587)
    assert cleared == set()
    assert not upper_wall_open(cleared)
    cleared = note_upper_wall(cleared, (near_a,), 1243, 587)
    assert not upper_wall_open(cleared)
    cleared = note_upper_wall(cleared, (near_b,), 1243, 587)
    assert not upper_wall_open(cleared)
    cleared = note_upper_wall(cleared, (near_c,), 1243, 587)
    assert upper_wall_open(cleared)
    grate = note_upper_wall(set(), (near_a,), 1223, 1860)
    assert grate == set()


def test_take02_slope_owns_moving_aim_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from super_metroid.routes.kpdr.wrecked_ship import ws_main_shaft as mod

    session = _Session(
        _state(
            samus_x=1224,
            samus_y=1859,
            pose=15,
            facing=FACING_RIGHT,
            velocity_y=0,
        )
    )
    save_column_calls: list[str] = []

    monkeypatch.setattr(
        mod,
        "_observe_shot_blocks",
        lambda sess, prev, hit: (False, prev, ()),
    )
    monkeypatch.setattr(
        mod,
        "save_column_walljump",
        lambda sess, label, done: save_column_calls.append(label),
    )
    mod.climb_until(session, "test", lambda st: session.frame > 0)
    assert save_column_calls == []
    assert session.actions[0][1] == "test_climb"


def test_latched_take02_suppresses_save_column_and_ice_keeps_morph_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from super_metroid.routes.kpdr.wrecked_ship import ws_main_shaft as mod

    save_column_calls: list[str] = []
    ice_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        mod,
        "_observe_shot_blocks",
        lambda sess, prev, hit: (True, prev, ()),
    )
    monkeypatch.setattr(
        mod,
        "save_column_walljump",
        lambda sess, label, done: save_column_calls.append(label),
    )
    monkeypatch.setattr(
        mod,
        "ice_keepaway_action",
        lambda x, y, *args, **kwargs: ice_calls.append((x, y)) or ("X",),
    )

    airborne = _Session(
        _state(
            samus_x=1231,
            samus_y=1852,
            pose=75,
            facing=FACING_RIGHT,
            velocity_y=6,
        )
    )
    mod.climb_until(airborne, "test", lambda st: airborne.frame > 0)
    assert save_column_calls == []
    assert airborne.actions[0][1] == "test_climb"
    assert set(pressed_snes_buttons(airborne.actions[0][0])) == {"LEFT", "A"}

    contact = _Session(
        _state(
            samus_x=1209,
            samus_y=1787,
            pose=2,
            facing=FACING_LEFT,
            velocity_y=0,
        )
    )
    mod.climb_until(contact, "test", lambda st: contact.frame > 0)
    assert ice_calls == []
    assert contact.actions[0][1] == "test_drop_plant"
    contact_buttons = set(pressed_snes_buttons(contact.actions[0][0]))
    assert contact_buttons == {"B", "LEFT"}
    assert "X" not in contact_buttons


def test_take02_drop_handoff_matches_tape_rle() -> None:
    from super_metroid.routes.kpdr.wrecked_ship import ws_main_shaft as mod

    actions = [
        mod._take02_drop_handoff_action(frame)
        for frame in range(sum(count for count, _ in mod._TAKE02_DROP_HANDOFF))
    ]
    assert len(actions) == 94
    assert actions[:5] == [("LEFT",)] * 5
    assert actions[5:14] == [("LEFT", "A")] * 9
    assert actions[23:27] == [("X",)] * 4
    assert actions[27:32] == [()] * 5
    assert actions[-12:-1] == [()] * 11
    assert actions[-1] == ("DOWN",)
    assert mod._take02_drop_handoff_action(len(actions)) is None

    tunnel_actions = [
        mod._take02_tunnel_handoff_action(frame)
        for frame in range(sum(count for count, _ in mod._TAKE02_TUNNEL_HANDOFF))
    ]
    assert len(tunnel_actions) == 112
    assert tunnel_actions[:11] == [("UP",)] * 11
    assert tunnel_actions[11:17] == [("UP", "X")] * 6
    assert tunnel_actions[-24:-20] == [("A",)] * 4
    assert tunnel_actions[-20:] == [("RIGHT", "A")] * 20
    assert mod._take02_tunnel_handoff_action(len(tunnel_actions)) is None


def test_play_shots_climbs_jumps(monkeypatch: pytest.MonkeyPatch) -> None:
    from super_metroid.routes.kpdr.wrecked_ship import ws_main_climb as mod

    session = _Session(_state())
    seen: dict[str, object] = {}

    def _require(sess, room, label):
        seen["require"] = (room, label)

    def _shot(sess, label):
        seen["shot"] = label

    def _climb(sess, label, done):
        del done
        seen.setdefault("climbs", []).append(label)
        sess.state = replace(sess.state, samus_x=1135, samus_y=80, pose=1)

    def _jump(sess, label):
        seen["jump"] = label
        sess.state = replace(sess.state, room_id=ROOM_WS_ATTIC, game_state=11)

    def _settle(sess, dest_room, *, label, settle_frames=200, land_frames=90):
        del settle_frames, land_frames
        seen["settle"] = (dest_room, label)
        sess.state = replace(sess.state, room_id=dest_room, game_state=8, door_transition=0)
        return sess.state

    monkeypatch.setattr(mod, "require_room", _require)
    monkeypatch.setattr(mod, "three_shot_tunnel", _shot)
    monkeypatch.setattr(mod, "climb_until", _climb)
    monkeypatch.setattr(mod, "_jump_up_attic", _jump)
    monkeypatch.setattr(mod, "settle_ceiling_dest", _settle)
    out = mod.play_ws_main_to_attic(session)
    assert seen["require"][0] == ROOM_WS_MAIN
    assert seen["shot"] == "ws_main_to_attic_pit_shot"
    assert seen["climbs"] == [
        "ws_main_to_attic_grate_seat",
        "ws_main_to_attic_west_super",
        "ws_main_to_attic_mid_climb",
        "ws_main_to_attic_attic_seat",
    ]
    assert seen["jump"] == "ws_main_to_attic"
    assert seen["settle"] == (ROOM_WS_ATTIC, "ws_main_to_attic")
    assert ws_main_attic_settled(out)


def test_grate_seat_phase_waits_for_usable_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from super_metroid.routes.kpdr.wrecked_ship import ws_main_climb as mod

    session = _Session(_state())
    done_fns: list[object] = []

    def _require(sess, room, label):
        del sess, room, label

    def _shot(sess, label):
        del sess, label

    def _climb(sess, label, done):
        del sess
        if label.endswith("grate_seat"):
            done_fns.append(done)

    monkeypatch.setattr(mod, "require_room", _require)
    monkeypatch.setattr(mod, "three_shot_tunnel", _shot)
    monkeypatch.setattr(mod, "climb_until", _climb)
    with pytest.raises(PhaseStop) as caught:
        mod.play_ws_main_to_attic(session, start="pit_shot", stop="grate_seat")
    assert caught.value.phase == "grate_seat"
    assert done_fns == [at_ws_main_usable_grate_seat]


def test_natural_grate_seat_settles_momentum_before_west_super(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from super_metroid.routes.kpdr.wrecked_ship import ws_main_climb as mod

    session = _Session(_state())
    climbs: list[tuple[str, int]] = []

    def _climb(sess, label, done):
        del done
        climbs.append((label, len(sess.actions)))
        if label.endswith("grate_seat"):
            sess.state = replace(
                sess.state,
                samus_x=1217,
                samus_y=1867,
                pose=9,
                momentum_x=2,
                momentum_x_sub=49152,
            )
        else:
            sess.state = replace(sess.state, samus_x=1094, samus_y=1700, pose=48)

    monkeypatch.setattr(mod, "require_room", lambda *args: None)
    monkeypatch.setattr(mod, "three_shot_tunnel", lambda *args: None)
    monkeypatch.setattr(mod, "climb_until", _climb)

    with pytest.raises(PhaseStop) as caught:
        mod.play_ws_main_to_attic(session, stop="west_super")

    assert caught.value.phase == "west_super"
    assert climbs == [
        ("ws_main_to_attic_grate_seat", 0),
        ("ws_main_to_attic_west_super", 5),
    ]
    assert [reason for _, reason in session.actions] == [
        "ws_main_to_attic_grate_seat_settle"
    ] * 5
    assert all(not pressed_snes_buttons(action) for action, _ in session.actions)


def test_phased_play_stop_at_pit_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    from super_metroid.routes.kpdr.wrecked_ship import ws_main_climb as mod

    session = _Session(_state())
    seen: list[str] = []

    def _require(sess, room, label):
        del sess, room, label

    def _shot(sess, label):
        seen.append(label)
        sess.state = replace(sess.state, samus_x=1104, samus_y=1981, pose=48)

    monkeypatch.setattr(mod, "require_room", _require)
    monkeypatch.setattr(mod, "three_shot_tunnel", _shot)
    monkeypatch.setattr(mod, "climb_until", lambda *a, **k: None)
    with pytest.raises(PhaseStop) as caught:
        mod.play_ws_main_to_attic(session, start="pit_shot", stop="pit_shot")
    assert caught.value.phase == "pit_shot"
    assert seen == ["ws_main_to_attic_pit_shot"]


def test_wrong_room_and_missing_boss_bit() -> None:
    session = _Session(_state(room_id=0xCC6F, samus_x=657, samus_y=91, pose=2))
    with pytest.raises(LeaveMiss, match="ws_main_to_attic") as caught:
        play_ws_main_to_attic(session)
    err = caught.value
    assert err.leftover["xy"] == [657, 91]
    assert "expected Attic 0xCA52, got 0xCC6F" in str(err)
    boss = _Session(_state(boss_bits=(0, 0, 0, 0, 0, 0, 0, 0)))
    with pytest.raises(LeaveMiss, match="not defeated") as caught_boss:
        play_ws_main_to_attic(boss)
    assert caught_boss.value.leftover["xy"] == [1173, 1979]


def test_never_uses_l_as_hop_side() -> None:
    cases = (
        (1173, 1979, 1, FACING_LEFT, 0, False),
        (1166, 1979, 2, FACING_LEFT, 0, False),
        (1156, 1979, 1, FACING_RIGHT, 0, False),
        (1156, 1920, 77, FACING_RIGHT, 5, False),
        (1177, 1883, 2, FACING_RIGHT, 0, False),
        (1111, 1899, 157, FACING_LEFT, 0, False),
        (1223, 1860, 3, FACING_RIGHT, 0, False),
        (1223, 1860, 3, FACING_LEFT, 0, True),
        (1082, 1878, 2, FACING_RIGHT, 0, True),
        (1152, 1675, 2, FACING_LEFT, 0, False),
        (1152, 680, 1, FACING_RIGHT, 0, False),
        (1135, 80, 1, FACING_LEFT, 0, False),
    )
    for x, y, pose, facing, vy, lip in cases:
        names = climb_action(x, y, pose, facing, velocity_y=vy, lip_hit=lip)
        assert "L" not in names
        assert "SUPER" not in names
    for hop in SHAFT_HOPS:
        assert hop.side != "L"
        assert hop.takeoff.side != "L"
    for frame in range(80):
        names = attic_door_action(WS_MAIN_ATTIC_DOOR_X, 80, 1, frame)
        assert "L" not in names
        assert "SUPER" not in names


def test_ice_overlay() -> None:
    from super_metroid.routes.kpdr.wrecked_ship.ws_main_ice import (
        COVERN_ID,
        shelf_covern_ice_action,
    )

    blob = Enemy(0, ATOMIC_ID, 1150, 1160, 250, 0)
    assert ice_keepaway_action(1173, 1979, FACING_LEFT, (blob,)) is None
    overlap = Enemy(0, ATOMIC_ID, 1155, 1561, 250, 0)
    assert ice_keepaway_action(
        1154, 1561, FACING_RIGHT, (overlap,), velocity_y=2
    ) is None
    planted_ice = ice_keepaway_action(1255, 1547, FACING_LEFT, (overlap,))
    assert planted_ice is not None
    slope_atomic = Enemy(0, ATOMIC_ID, 1116, 1112, 250, 0)
    assert ice_keepaway_action(1099, 1095, FACING_RIGHT, (slope_atomic,)) is None
    wall_atomic = Enemy(0, ATOMIC_ID, 1046, 1081, 250, 0)
    wall_ice = ice_keepaway_action(1050, 1083, FACING_LEFT, (wall_atomic,))
    assert wall_ice is not None and ("X" in wall_ice or "A" in wall_ice)
    assert ice_keepaway_action(1050, 1083, FACING_RIGHT, (wall_atomic,)) is None
    assert ice_keepaway_action(
        1050, 1083, FACING_LEFT, (wall_atomic,), movement_type=14
    ) is None
    assert ice_keepaway_action(
        1045, 1066, FACING_RIGHT, (wall_atomic,), velocity_y=5
    ) is None
    atomic_827 = Enemy(0, ATOMIC_ID, 1188, 822, 250, 0)
    assert ice_keepaway_action(1098, 1019, FACING_RIGHT, (atomic_827,)) is None
    assert ice_keepaway_action(1243, 907, FACING_RIGHT, (atomic_827,)) is None
    assert ice_keepaway_action(
        1187, 817, FACING_RIGHT, (atomic_827,), velocity_y=0
    ) is None
    atomic_651 = Enemy(0, ATOMIC_ID, 1176, 680, 80, 0)
    assert ice_keepaway_action(1101, 651, FACING_RIGHT, (atomic_651,)) is None
    assert ice_keepaway_action(1243, 587, FACING_RIGHT, (atomic_651,)) is None
    overlap_523 = Enemy(0, ATOMIC_ID, 1084, 514, 250, 0)
    bounce_ice = ice_keepaway_action(1077, 523, FACING_LEFT, (overlap_523,))
    assert bounce_ice is not None
    assert "A" not in bounce_ice
    take02_far = Enemy(0, ATOMIC_ID, 1164, 525, 150, 0)
    assert ice_keepaway_action(1077, 523, FACING_LEFT, (take02_far,)) is None
    assert ice_keepaway_action(1204, 523, FACING_LEFT, (overlap_523,)) is None
    frozen_523 = Enemy(0, ATOMIC_ID, 1084, 514, 250, 80)
    frozen_ice = ice_keepaway_action(1077, 523, FACING_LEFT, (frozen_523,))
    assert frozen_ice is not None
    frozen_wall = Enemy(0, ATOMIC_ID, 1046, 1081, 250, 80)
    assert ice_keepaway_action(1050, 1083, FACING_LEFT, (frozen_wall,)) is None
    shot = ice_keepaway_action(1152, 1163, FACING_LEFT, (blob,))
    assert shot is not None and ("X" in shot or "A" in shot)
    assert ice_keepaway_action(1152, 1163, FACING_LEFT, ()) is None
    covern = Enemy(0, COVERN_ID, 1129, 1818, 80, 0)
    assert ice_keepaway_action(1075, 1845, FACING_LEFT, (covern,)) is not None
    frozen = Enemy(0, ATOMIC_ID, 1152, 1163, 250, 80)
    assert ice_keepaway_action(1152, 1163, FACING_LEFT, (frozen,)) == ()
    shelf = Enemy(0, COVERN_ID, 1129, 1818, 80, 0)
    stairs = Enemy(1, COVERN_ID, 1048, 1928, 80, 0)
    live = shelf_covern_ice_action(1082, 1878, FACING_RIGHT, (shelf, stairs))
    assert live is not None and "LEFT" not in live
    charged = shelf_covern_ice_action(
        1082, 1878, FACING_RIGHT, (shelf, stairs), charge=CHARGE_FULL
    )
    assert charged and "A" in charged
    assert shelf_covern_ice_action(1082, 1878, FACING_RIGHT, (stairs,)) is None
    frozen_c = Enemy(0, COVERN_ID, 1129, 1818, 80, 80)
    assert shelf_covern_ice_action(1082, 1878, FACING_RIGHT, (frozen_c, stairs)) is None

    ram = np.zeros(0x2000, dtype=np.uint8)
    base = 0x0F78
    ram[base] = ATOMIC_ID & 0xFF
    ram[base + 1] = ATOMIC_ID >> 8
    ram[base + 0x02] = 1150 & 0xFF
    ram[base + 0x03] = 1150 >> 8
    ram[base + 0x06] = 1163 & 0xFF
    ram[base + 0x07] = 1163 >> 8
    ram[base + 0x14] = 250
    ram[base + 0x26] = 40
    session = _Session(_state())
    session.env = type("E", (), {"get_ram": lambda self: ram})()
    found = list_enemies(session)
    assert len(found) == 1
    assert found[0].enemy_id == ATOMIC_ID
    assert found[0].freeze_timer == 40
