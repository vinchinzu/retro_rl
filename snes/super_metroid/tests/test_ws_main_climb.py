"""Unit tests for powered Main Shaft → Attic (rr-kw8t hop 2). No emulator."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from super_metroid.combat.enemies import Enemy, list_enemies
from super_metroid.hop_glance import LeaveMiss
from super_metroid.ram import FACING_LEFT, FACING_RIGHT, GameplayPhase, parse_state
from super_metroid.routes.kpdr.k6.ws_main_actions import (
    SHAFT_HOPS,
    THREE_SHOT_X_MIN,
    WS_MAIN_ATTIC_DOOR_X,
    at_ws_main_attic_door_seat,
    at_ws_main_lip_shot_seat,
    at_ws_main_morph_drop,
    at_ws_main_pit,
    attic_door_action,
    climb_action,
    grate_clear_action,
    grate_lip_action,
    grate_morph_action,
    pit_exit_action,
    three_shot_action,
)
from super_metroid.routes.skills.basic_moves import shoot_up_action
from super_metroid.routes.kpdr.k6.ws_main_climb import (
    play_ws_main_to_attic,
    ws_main_attic_settled,
)
from super_metroid.routes.kpdr.k6.ws_main_ice import (
    ATOMIC_ID,
    ice_keepaway_action,
)
from super_metroid.routes.kpdr.k6.ws_main_grate import POCKET_RELEASE_CHARGE
from super_metroid.routes.skills.charge_shot import CHARGE_FULL
from super_metroid.routes.kpdr.room_ids import ROOM_WS_ATTIC, ROOM_WS_MAIN
from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS


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
    still_main = _state()
    assert not ws_main_attic_settled(still_main)


def test_pit_and_attic_door_seats() -> None:
    pin = _state()
    assert at_ws_main_pit(pin)
    assert not at_ws_main_attic_door_seat(pin)
    west = _state(samus_x=1044, samus_y=1675, pose=10)
    assert not at_ws_main_pit(west)
    assert not at_ws_main_attic_door_seat(west)
    seat = _state(samus_x=1135, samus_y=80, pose=1)
    assert at_ws_main_attic_door_seat(seat)
    peak = _state(samus_x=1135, samus_y=80, pose=21, velocity_y=2)
    assert not at_ws_main_attic_door_seat(peak)
    wrong = _state(room_id=ROOM_WS_ATTIC, samus_x=1135, samus_y=80, pose=1)
    assert not at_ws_main_attic_door_seat(wrong)
    assert not at_ws_main_pit(wrong)


def test_first_jump_hatch_takeoff_lands_right_lip() -> None:
    pin = pit_exit_action(1173, 1979, 1, FACING_LEFT)
    assert pin == ("LEFT",)
    assert "A" not in pin
    assert "X" not in pin
    turn = pit_exit_action(1150, 1979, 1, FACING_LEFT)
    assert turn == ("RIGHT",)
    takeoff = pit_exit_action(1150, 1979, 1, FACING_RIGHT)
    assert takeoff == ("A",)
    assert "X" not in takeoff
    assert "B" not in takeoff
    assert "DOWN" not in takeoff
    rise = pit_exit_action(1150, 1920, 77, FACING_RIGHT, velocity_y=5)
    assert rise == ("A",)
    assert "DOWN" not in rise
    peak = pit_exit_action(1150, 1880, 81, FACING_RIGHT, velocity_y=3)
    assert peak == ("RIGHT", "A")
    land = pit_exit_action(1184, 1883, 9, FACING_RIGHT)
    assert land == ()
    cubby = pit_exit_action(1070, 1962, 81, FACING_LEFT, velocity_y=4)
    assert cubby[0] == "RIGHT"
    assert "A" not in cubby
    bonk = pit_exit_action(1173, 1940, 20, FACING_LEFT, velocity_y=2)
    assert "A" not in bonk
    assert bonk == ("LEFT",)


def test_three_shot_pit_is_first_jump_not_charge_bonk() -> None:
    face = three_shot_action(1173, 1979, 1, FACING_RIGHT, 0)
    assert face == ("LEFT",)
    assert "L" not in face
    floor = three_shot_action(1173, 1979, 1, FACING_LEFT, 0, charge=0)
    assert floor == ("LEFT",)
    assert "A" not in floor
    assert "DOWN" not in floor
    hatch = three_shot_action(1150, 1979, 1, FACING_RIGHT, 0)
    assert hatch == ("A",)
    hole_air = three_shot_action(1126, 1956, 81, FACING_RIGHT, 240)
    assert hole_air[0] == "RIGHT"
    assert "A" in hole_air
    assert "DOWN" not in hole_air
    cubby = three_shot_action(1045, 1940, 82, FACING_LEFT, 240)
    assert cubby[0] == "RIGHT"
    assert "LEFT" not in cubby
    assert "A" not in cubby
    seated = three_shot_action(1173, 1800, 1, FACING_LEFT, 0, charge=0)
    assert "X" in seated
    assert "A" in seated
    seated_rel = three_shot_action(1173, 1800, 1, FACING_LEFT, 64, charge=CHARGE_FULL)
    assert "X" not in seated_rel
    jump_shot = three_shot_action(1173, 1800, 1, FACING_LEFT, 80, charge=0)
    assert "X" in jump_shot
    assert "A" in jump_shot
    morph_floor = three_shot_action(1173, 1979, 31, FACING_LEFT, 0)
    assert morph_floor == ("UP",)
    morph_stair = three_shot_action(1100, 1800, 31, FACING_LEFT, 0)
    assert morph_stair == ("LEFT",)
    east = three_shot_action(1220, 1800, 1, FACING_LEFT, 0)
    assert east == ("LEFT", "B")
    for frame in range(280):
        names = three_shot_action(1173, 1979, 1, FACING_LEFT, frame, charge=0)
        assert "L" not in names
        assert "SUPER" not in names
        assert "DOWN" not in names
        assert "X" not in names


def test_climb_stays_in_shaft_never_down_or_l() -> None:
    over_hatch = climb_action(1150, 1979, 1, FACING_RIGHT)
    assert over_hatch == ("A",)
    hatch_jump = climb_action(1150, 1979, 1, FACING_LEFT)
    assert hatch_jump == ("RIGHT",)
    assert "DOWN" not in hatch_jump
    stair_base = climb_action(1089, 1979, 1, FACING_LEFT)
    assert stair_base == ("RIGHT",)
    cubby = climb_action(1045, 1940, 82, FACING_LEFT, velocity_y=0)
    assert cubby[0] == "RIGHT"
    assert "LEFT" not in cubby
    assert "A" not in cubby
    cubby_face = climb_action(1045, 1940, 1, FACING_LEFT)
    assert cubby_face == ("RIGHT",)
    grate = climb_action(1082, 1878, 2, FACING_LEFT)
    assert grate == ("RIGHT",)
    grate_jump = climb_action(1082, 1878, 2, FACING_RIGHT)
    assert grate_jump == ("A",)
    assert "X" not in grate_jump
    assert "B" not in grate_jump
    assert "LEFT" not in grate_jump
    hop = climb_action(1082, 1878, 2, FACING_RIGHT, frame=56)
    assert hop == ("A",)
    assert "B" not in hop
    turning = climb_action(1082, 1878, 38, FACING_RIGHT, movement_type=14)
    assert turning == ()
    lip = climb_action(1177, 1883, 2, FACING_LEFT)
    assert lip == ("X",)
    assert "LEFT" not in lip
    assert "UP" not in lip
    assert "R" not in lip
    lip_face = climb_action(1177, 1883, 2, FACING_RIGHT)
    assert lip_face == ("LEFT",)
    assert "A" not in lip
    assert "DOWN" not in lip
    assert "DOWN" not in lip_face
    leftover = climb_action(1181, 1883, 1, FACING_RIGHT)
    assert leftover == ("LEFT",)
    walked = climb_action(1169, 1883, 38, FACING_RIGHT, movement_type=14)
    assert walked == ("LEFT",)
    assert "A" not in walked
    through = climb_action(
        1202, 1854, 77, FACING_RIGHT, velocity_y=1, lip_hit=True
    )
    assert through == ("LEFT",)
    assert "A" not in through
    through_left = climb_action(
        1202, 1854, 77, FACING_LEFT, velocity_y=1, lip_hit=True
    )
    assert through_left == ("LEFT", "A")
    assert climb_action(1177, 1883, 2, FACING_RIGHT, lip_hit=True) == ("LEFT",)
    assert climb_action(1177, 1883, 2, FACING_LEFT, lip_hit=True) == ("LEFT", "A")
    assert "DOWN" not in climb_action(1177, 1883, 2, FACING_RIGHT, lip_hit=True)
    ledge_jump = climb_action(1219, 1864, 9, FACING_RIGHT, lip_hit=True)
    assert ledge_jump == ("LEFT",)
    assert "DOWN" not in ledge_jump
    assert "A" not in ledge_jump
    lip_crouch = climb_action(1177, 1883, 39, FACING_LEFT)
    assert lip_crouch == ("UP",)
    save = climb_action(1240, 1675, 2, FACING_RIGHT)
    assert save == ("LEFT", "B")
    # Take02 seat ~(1223,1860) p3: shoot until PLM, never spin/morph here.
    take02 = climb_action(1223, 1860, 3, FACING_RIGHT)
    assert take02 == shoot_up_action()
    assert "DOWN" not in take02
    ledge = climb_action(1219, 1864, 9, FACING_RIGHT)
    assert ledge == shoot_up_action()
    assert "DOWN" not in ledge
    wj = climb_action(1216, 1852, 19, FACING_RIGHT, velocity_y=2)
    assert wj == ("A",)
    jam = climb_action(1220, 1843, 77, FACING_RIGHT, velocity_y=0)
    assert jam == ("A",)
    assert "B" not in jam
    assert "DOWN" not in jam
    jam_face = climb_action(1220, 1843, 77, FACING_LEFT, velocity_y=0)
    assert jam_face == ("LEFT", "A")
    assert "B" not in jam_face
    peak = climb_action(1221, 1827, 77, FACING_RIGHT, velocity_y=2)
    assert peak == ("A",)
    assert "B" not in peak
    door = climb_action(1243, 1851, 9, FACING_RIGHT)
    assert door == ("LEFT", "B")
    west = climb_action(1000, 1675, 2, FACING_LEFT)
    assert west == ("RIGHT", "B")
    air = climb_action(1180, 1400, 25, FACING_LEFT, velocity_y=-4)
    assert "A" in air
    assert "DOWN" not in air
    mid = climb_action(1152, 1163, 1, FACING_RIGHT)
    assert "L" not in mid
    assert "DOWN" not in mid
    for x, y, pose, facing in (
        (1173, 1979, 1, FACING_RIGHT),
        (1150, 1979, 1, FACING_LEFT),
        (1152, 1675, 2, FACING_LEFT),
        (1152, 1163, 1, FACING_RIGHT),
        (1135, 80, 1, FACING_LEFT),
        (1240, 900, 2, FACING_RIGHT),
    ):
        names = climb_action(x, y, pose, facing)
        assert "L" not in names
        assert "DOWN" not in names
        assert "SUPER" not in names


def test_shaft_hops_are_dpad_sides() -> None:
    assert SHAFT_HOPS
    for hop in SHAFT_HOPS:
        assert hop.side in ("LEFT", "RIGHT")
        assert hop.takeoff.side in ("LEFT", "RIGHT")


def test_grate_clear_lip_shoots_up_until_plm_hit() -> None:
    assert grate_clear_action(1075, 1700, 1, FACING_LEFT, 0) is None
    assert grate_clear_action(1075, 1979, 1, FACING_LEFT, 0) is None
    face = grate_clear_action(1082, 1878, 8, FACING_LEFT, 0, charge=0)
    assert face == ("RIGHT",)
    assert "UP" not in face
    assert "B" not in face
    jump = grate_clear_action(1082, 1878, 1, FACING_RIGHT, 0, charge=0)
    assert jump == ("A",)
    assert "X" not in jump
    assert "UP" not in jump
    assert "B" not in jump
    hop = grate_clear_action(1082, 1878, 1, FACING_RIGHT, 56, charge=0)
    assert hop == ("A",)
    assert "B" not in hop
    gap = grate_clear_action(1085, 1843, 78, FACING_LEFT, 0, velocity_y=0)
    assert gap == ("LEFT",)
    assert "A" not in gap
    ice_jump = grate_clear_action(1085, 1843, 78, FACING_RIGHT, 0, velocity_y=3)
    assert ice_jump == ("RIGHT", "A")
    mid = grate_clear_action(1152, 1845, 1, FACING_RIGHT, 0)
    assert mid == ("RIGHT", "A")
    assert "UP" not in mid
    lip = grate_clear_action(1177, 1883, 2, FACING_LEFT, 0)
    assert lip == ("X",)
    assert "LEFT" not in lip
    assert "UP" not in lip
    assert "R" not in lip
    lip_face = grate_clear_action(1177, 1883, 2, FACING_RIGHT, 0)
    assert lip_face == ("LEFT",)
    assert grate_clear_action(
        1177, 1883, 2, FACING_RIGHT, 0, lip_hit=True
    ) == ("LEFT",)
    assert grate_clear_action(
        1177, 1883, 2, FACING_LEFT, 0, lip_hit=True
    ) == ("LEFT", "A")
    crouch = grate_clear_action(1177, 1883, 39, FACING_LEFT, 0)
    assert crouch == ("UP",)
    air = grate_clear_action(1177, 1800, 20, FACING_LEFT, 0)
    assert air == ("LEFT", "A")
    assert "UP" not in air
    landing = grate_clear_action(1177, 1880, 19, FACING_RIGHT, 0)
    assert landing is None
    assert at_ws_main_lip_shot_seat(1223, 1860, 3)
    assert not at_ws_main_lip_shot_seat(1224, 1860, 3)
    assert grate_clear_action(1223, 1860, 3, FACING_RIGHT, 0) == shoot_up_action()
    assert grate_clear_action(
        1223, 1860, 3, FACING_RIGHT, 0, charge=CHARGE_FULL
    ) == ("UP",)
    assert grate_lip_action(2, False) == shoot_up_action()
    assert grate_lip_action(2, False, samus_x=1177) == ("X",)
    assert "LEFT" not in grate_lip_action(2, False, samus_x=1177)
    assert "UP" not in grate_lip_action(2, False, samus_x=1177)
    assert "R" not in grate_lip_action(2, False, samus_x=1177)
    assert grate_lip_action(2, False, FACING_RIGHT, 1177) == ("LEFT",)
    assert grate_lip_action(2, False, FACING_LEFT, 1177, POCKET_RELEASE_CHARGE) == ()
    assert "X" not in grate_lip_action(2, False, FACING_LEFT, 1177, POCKET_RELEASE_CHARGE)
    assert "LEFT" not in grate_lip_action(2, False, FACING_LEFT, 1177, POCKET_RELEASE_CHARGE)
    assert grate_lip_action(2, False, FACING_LEFT, 1177, CHARGE_FULL) == ()
    assert at_ws_main_lip_shot_seat(1169, 1883, 38)
    assert grate_clear_action(1169, 1883, 38, FACING_RIGHT, 0) == ("LEFT",)
    assert at_ws_main_lip_shot_seat(1177, 1883, 6)
    assert grate_clear_action(1177, 1883, 6, FACING_LEFT, 0) == ("X",)
    assert "RIGHT" not in grate_clear_action(1177, 1883, 6, FACING_LEFT, 0)
    assert "R" not in grate_clear_action(1177, 1883, 6, FACING_LEFT, 0)
    assert grate_lip_action(2, False, charge=CHARGE_FULL) == ("UP",)
    assert "X" not in grate_lip_action(2, False, charge=CHARGE_FULL)
    assert grate_lip_action(2, True) == ("LEFT", "A")
    assert "DOWN" not in grate_lip_action(2, True)
    assert grate_lip_action(31, True) == ("LEFT",)
    take02_jump = grate_lip_action(2, True, FACING_LEFT, 1223)
    assert take02_jump == ("LEFT", "A")
    assert "DOWN" not in take02_jump
    assert grate_lip_action(2, True, FACING_RIGHT, 1223) == ("LEFT",)
    assert at_ws_main_morph_drop(1189, 1785, 2)
    assert at_ws_main_morph_drop(1214, 1801, 56)
    assert not at_ws_main_morph_drop(1223, 1860, 3)
    assert not at_ws_main_morph_drop(1235, 1851, 10)
    assert grate_morph_action(2, False) is None
    assert grate_morph_action(2, True) == ("DOWN",)
    assert grate_morph_action(56, True) == ("DOWN",)
    assert grate_morph_action(31, True) == ("X",)
    assert grate_clear_action(1189, 1785, 2, FACING_LEFT, 0) == ("RIGHT",)
    assert grate_clear_action(
        1189, 1785, 2, FACING_LEFT, 0, lip_hit=True
    ) == ("DOWN",)
    assert "DOWN" not in grate_clear_action(
        1223, 1860, 3, FACING_RIGHT, 0, lip_hit=True
    )
    assert climb_action(1189, 1785, 2, FACING_LEFT, lip_hit=True) == ("DOWN",)
    assert "DOWN" not in climb_action(1189, 1785, 2, FACING_LEFT)
    air_hit = grate_clear_action(
        1202, 1854, 77, FACING_RIGHT, 0, velocity_y=1, lip_hit=True
    )
    assert air_hit == ("LEFT",)
    assert grate_clear_action(
        1202, 1854, 77, FACING_LEFT, 0, velocity_y=1, lip_hit=True
    ) == ("LEFT", "A")


def test_attic_door_is_up_a_not_super_or_l() -> None:
    assert attic_door_action(1135, 80, 1, 0) == ("UP", "X")
    assert attic_door_action(1135, 80, 1, 60) == ("UP",)
    assert "A" not in attic_door_action(1135, 80, 1, 64)
    assert attic_door_action(1135, 80, 1, 308) == ("UP", "A")
    in_shaft = attic_door_action(1140, 36, 21, 0)
    assert in_shaft == ("LEFT", "A")
    assert "UP" not in in_shaft
    center = attic_door_action(1135, 36, 21, 0)
    assert center == ("A",)
    assert attic_door_action(1135, 80, 138, 0) == ()
    assert "RIGHT" in attic_door_action(1110, 80, 1, 0)
    for frame in range(80):
        names = attic_door_action(WS_MAIN_ATTIC_DOOR_X, 80, 1, frame)
        assert "L" not in names
        assert "SUPER" not in names


def test_ice_keepaway_skips_pit_and_taps_atomic() -> None:
    blob = Enemy(0, ATOMIC_ID, 1150, 1160, 250, 0)
    pit = ice_keepaway_action(1173, 1979, FACING_LEFT, (blob,))
    assert pit is None
    shot = ice_keepaway_action(1152, 1163, FACING_LEFT, (blob,))
    assert shot is not None
    assert "X" in shot or "A" in shot
    none = ice_keepaway_action(1152, 1163, FACING_LEFT, ())
    assert none is None
    from super_metroid.routes.kpdr.k6.ws_main_ice import COVERN_ID

    covern = Enemy(0, COVERN_ID, 1129, 1818, 80, 0)
    grate = ice_keepaway_action(1075, 1845, FACING_LEFT, (covern,))
    assert grate is not None
    frozen = Enemy(0, ATOMIC_ID, 1152, 1163, 250, 80)
    wait = ice_keepaway_action(1152, 1163, FACING_LEFT, (frozen,))
    assert wait == ()


def test_shelf_covern_ice_skips_stairs_and_jumps_when_frozen() -> None:
    from super_metroid.routes.kpdr.k6.ws_main_ice import (
        COVERN_ID,
        shelf_covern_ice_action,
    )

    shelf = Enemy(0, COVERN_ID, 1129, 1818, 80, 0)
    stairs = Enemy(1, COVERN_ID, 1048, 1928, 80, 0)
    live = shelf_covern_ice_action(1082, 1878, FACING_RIGHT, (shelf, stairs))
    assert live is not None
    assert "LEFT" not in live
    assert "X" in live or "A" in live
    charged = shelf_covern_ice_action(
        1082, 1878, FACING_RIGHT, (shelf, stairs), charge=CHARGE_FULL
    )
    assert charged
    assert "A" in charged
    stairs_only = shelf_covern_ice_action(1082, 1878, FACING_RIGHT, (stairs,))
    assert stairs_only is None
    frozen = Enemy(0, COVERN_ID, 1129, 1818, 80, 80)
    assert (
        shelf_covern_ice_action(1082, 1878, FACING_RIGHT, (frozen, stairs)) is None
    )
    assert shelf_covern_ice_action(1082, 1878, FACING_RIGHT, ()) is None


def test_list_enemies_reads_freeze_timer() -> None:
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

    class _Env:
        def get_ram(self):
            return ram

    session = _Session(_state())
    session.env = _Env()
    found = list_enemies(session)
    assert len(found) == 1
    assert found[0].enemy_id == ATOMIC_ID
    assert found[0].x == 1150
    assert found[0].freeze_timer == 40


def test_already_in_attic_is_noop() -> None:
    session = _Session(
        _state(room_id=ROOM_WS_ATTIC, samus_x=1135, samus_y=184, pose=1)
    )
    out = play_ws_main_to_attic(session)
    assert out.room_id == ROOM_WS_ATTIC
    assert session.actions == []


def test_wrong_room() -> None:
    session = _Session(_state(room_id=0xCC6F, samus_x=657, samus_y=91, pose=2))
    with pytest.raises(LeaveMiss, match="ws_main_to_attic") as caught:
        play_ws_main_to_attic(session)
    err = caught.value
    assert err.leftover["xy"] == [657, 91]
    assert err.leftover["pose"] == 2
    assert err.leftover["gs"] == 8
    assert "expected Attic 0xCA52, got 0xCC6F" in str(err)


def test_requires_boss_bit() -> None:
    session = _Session(_state(boss_bits=(0, 0, 0, 0, 0, 0, 0, 0)))
    with pytest.raises(LeaveMiss, match="not defeated") as caught:
        play_ws_main_to_attic(session)
    assert caught.value.leftover["xy"] == [1173, 1979]


def test_first_jump_loop_does_not_morph_bomb() -> None:
    from super_metroid.routes.kpdr.k6 import ws_main_shaft as shaft

    src = inspect.getsource(shaft.three_shot_tunnel)
    assert "at_ws_main_grate_seat" in src
    assert 'reason=f"{label}_bomb"' not in src
    assert 'hold(session, 3, "A"' not in src
    climb = inspect.getsource(shaft.climb_until)
    assert "lip_hit" in climb
    assert "_update_lip_hit" in climb
    assert "drop_bomb" in climb
    assert "at_ws_main_morph_drop" in climb
    dispatch = inspect.getsource(shaft._dispatch_west_super_band)
    assert "shoot_up_action" in dispatch
    assert 'f"{label}_lip_morph"' not in dispatch


def test_save_column_wj_band_and_pulse() -> None:
    from super_metroid.routes.kpdr.k6.ws_main_shaft import (
        SAVE_COLUMN_WJ,
        at_ws_main_save_column_wj,
        save_column_walljump,
    )

    leftover = _state(samus_x=1220, samus_y=1843, pose=77, velocity_y=0)
    assert at_ws_main_save_column_wj(leftover)
    human = _state(samus_x=1216, samus_y=1852, pose=19, velocity_y=2)
    assert at_ws_main_save_column_wj(human)
    from_alcove = _state(
        samus_x=1220, samus_y=1843, pose=77, velocity_y=2, facing=FACING_LEFT
    )
    assert not at_ws_main_save_column_wj(from_alcove)
    peak = _state(samus_x=1221, samus_y=1827, pose=77, velocity_y=2)
    assert at_ws_main_save_column_wj(peak)
    lip = _state(samus_x=1177, samus_y=1883, pose=2, velocity_y=0)
    assert not at_ws_main_save_column_wj(lip)
    ledge = _state(samus_x=1219, samus_y=1864, pose=9, velocity_y=0)
    assert not at_ws_main_save_column_wj(ledge)
    save = _state(samus_x=1232, samus_y=1843, pose=77, velocity_y=0)
    assert not at_ws_main_save_column_wj(save)
    west = _state(samus_x=1152, samus_y=1675, pose=10)
    assert not at_ws_main_save_column_wj(west)
    assert SAVE_COLUMN_WJ.into == "RIGHT"
    assert SAVE_COLUMN_WJ.flip == "LEFT"
    assert SAVE_COLUMN_WJ.delay_into_frames == 0
    assert SAVE_COLUMN_WJ.into_frames <= 4
    assert "B" not in (SAVE_COLUMN_WJ.into, SAVE_COLUMN_WJ.flip)

    session = _Session(human)
    save_column_walljump(session, "test", lambda st: False)
    reasons = [r for _, r in session.actions]
    assert not any(r.endswith("_save_wj_delay") for r in reasons)
    assert not any("_wj_seek" in r for r in reasons)
    assert any(r.endswith("_save_wj_into") for r in reasons)
    assert any(r.endswith("_save_wj_flip") for r in reasons)
    assert session.frame == (
        SAVE_COLUMN_WJ.into_frames
        + SAVE_COLUMN_WJ.amid_frames
        + SAVE_COLUMN_WJ.flip_frames
    )
    approach = _Session(
        _state(samus_x=1212, samus_y=1843, pose=77, velocity_y=2)
    )
    save_column_walljump(approach, "test", lambda st: False)
    assert any("_wj_seek" in r for _, r in approach.actions)


def test_save_alcove_jumps_left_into_shaft() -> None:
    from super_metroid.routes.kpdr.k6.ws_main_shaft import (
        at_ws_main_save_alcove,
        save_alcove_jump,
    )

    planted = _state(samus_x=1235, samus_y=1851, pose=10, facing=FACING_LEFT)
    assert at_ws_main_save_alcove(planted)
    turning = _state(
        samus_x=1232, samus_y=1851, pose=38, facing=FACING_RIGHT
    )
    turning = replace(turning, movement_type=14)
    assert at_ws_main_save_alcove(turning)
    assert not at_ws_main_save_alcove(_state())
    assert not at_ws_main_save_alcove(
        _state(samus_x=1219, samus_y=1864, pose=9, facing=FACING_RIGHT)
    )
    session = _Session(planted)
    save_alcove_jump(session, "test")
    assert session.actions[0][1] == "test_alcove_jump"
    face = _Session(_state(samus_x=1235, samus_y=1851, pose=9, facing=FACING_RIGHT))
    save_alcove_jump(face, "test")
    assert face.actions[0][1] == "test_alcove_face"


def test_climb_until_overlay_save_cubby_and_shelf() -> None:
    from super_metroid.routes.kpdr.k6.ws_main_shaft import climb_until

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
    reasons = [r for _, r in column.actions]
    assert any(r.endswith("_save_wj_into") for r in reasons)

    shelf = _Session(
        _state(samus_x=1082, samus_y=1878, pose=2, facing=FACING_RIGHT)
    )
    climb_until(shelf, "test", _stop_after_first(shelf))
    assert any("shelf_hole" in r for _, r in shelf.actions)

    lip = _Session(
        _state(samus_x=1223, samus_y=1860, pose=3, facing=FACING_RIGHT)
    )
    climb_until(lip, "test", _stop_after_first(lip))
    assert any(r.endswith("_lip_up") for _, r in lip.actions)
    assert not any("DOWN" in str(a) for a, _ in lip.actions)


def test_probe_uses_repo_headed() -> None:
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "scripts" / "probe" / "ws_main_climb.py"
    text = src.read_text(encoding="utf-8")
    assert "from retro_harness.headed import" in text
    assert "add_headed_flag" in text
    assert "attach_headed" in text
    assert "def _attach_headed" not in text
    assert '"leftover"' in text
    assert "glance_misses" in text
    assert "final_from_state" in text
    assert "--stop-at" in text
    assert "play_ws_main_to_attic_phased" not in text
    assert "play_ws_main_to_attic" in text


def test_never_uses_l_as_left() -> None:
    from super_metroid.routes.kpdr.k6 import ws_main_actions as actions
    from super_metroid.routes.kpdr.k6 import ws_main_climb as mod
    from super_metroid.routes.kpdr.k6 import ws_main_shaft as shaft

    for src in (
        inspect.getsource(mod),
        inspect.getsource(actions),
        inspect.getsource(shaft),
    ):
        assert 'hold(session, 1, "L"' not in src
        assert '", "L"' not in src


def test_registered() -> None:
    from super_metroid.routes.kpdr.k6 import play_ws_main_to_attic as play

    assert KPDR_SEGMENTS["ws_main_to_attic"] is play
    assert callable(play)


def test_play_shots_climbs_jumps(monkeypatch: pytest.MonkeyPatch) -> None:
    from super_metroid.routes.kpdr.k6 import ws_main_climb as mod

    session = _Session(_state())
    seen: dict[str, object] = {}

    def _require(sess, room, label):
        seen["require"] = (room, label)

    def _shot(sess, label):
        seen["shot"] = label
        sess.state = replace(sess.state, samus_x=1044, samus_y=1675, pose=10)

    def _climb(sess, label, done):
        seen.setdefault("climbs", []).append(label)
        del done
        sess.state = replace(sess.state, samus_x=1135, samus_y=80, pose=1)

    def _jump(sess, label):
        seen["jump"] = label
        sess.state = replace(
            sess.state,
            room_id=ROOM_WS_ATTIC,
            game_state=11,
            door_transition=1,
            samus_x=1135,
            samus_y=184,
        )

    def _settle(sess, dest_room, *, label, settle_frames=200, land_frames=90):
        del settle_frames, land_frames
        seen["settle"] = (dest_room, label)
        sess.state = replace(
            sess.state, room_id=dest_room, game_state=8, door_transition=0
        )
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


def test_three_shot_x_band_covers_pin() -> None:
    assert THREE_SHOT_X_MIN <= 1173


def test_main_shaft_has_six_named_seams() -> None:
    from super_metroid.routes.kpdr.k6.ws_main_phases import (
        WS_MAIN_PHASES,
        at_ws_main_grate_seat,
        at_ws_main_mid_climb,
        at_ws_main_west_super_band,
        classify_ws_main_phase,
        ws_main_phase_index,
    )

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
    assert classify_ws_main_phase(pin) == "pit_shot"
    assert not at_ws_main_grate_seat(pin)
    leftover = _state(samus_x=1070, samus_y=1962, pose=81)
    assert classify_ws_main_phase(leftover) == "pit_shot"
    left_guess = _state(samus_x=1075, samus_y=1845, pose=2, velocity_y=0)
    assert not at_ws_main_grate_seat(left_guess)
    grate = _state(samus_x=1184, samus_y=1883, pose=9, velocity_y=0)
    assert at_ws_main_grate_seat(grate)
    assert classify_ws_main_phase(grate) == "grate_seat"
    aim_up = _state(samus_x=1180, samus_y=1883, pose=3, velocity_y=0)
    assert at_ws_main_grate_seat(aim_up)
    west = _state(samus_x=1152, samus_y=1675, pose=10)
    assert at_ws_main_west_super_band(west)
    assert classify_ws_main_phase(west) == "west_super"
    assert not at_ws_main_west_super_band(_state(room_id=0xCDA8, samus_y=1675))
    mid = _state(samus_x=1152, samus_y=680, pose=1)
    assert at_ws_main_mid_climb(mid)
    assert classify_ws_main_phase(mid) == "mid_climb"
    seat = _state(samus_x=1135, samus_y=80, pose=1)
    assert classify_ws_main_phase(seat) == "attic_seat"
    attic = _state(room_id=ROOM_WS_ATTIC, samus_x=1135, samus_y=184, pose=1)
    assert classify_ws_main_phase(attic) == "attic_door"


def test_phased_play_stop_at_pit_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    from super_metroid.routes.kpdr.k6 import ws_main_climb as mod
    from super_metroid.routes.skills.geometry import PhaseStop

    session = _Session(_state())
    seen: list[str] = []

    def _require(sess, room, label):
        del sess, room, label

    def _shot(sess, label):
        seen.append(label)
        sess.state = replace(sess.state, samus_x=1104, samus_y=1981, pose=48)

    def _climb(sess, label, done):
        seen.append(label)
        del done

    monkeypatch.setattr(mod, "require_room", _require)
    monkeypatch.setattr(mod, "three_shot_tunnel", _shot)
    monkeypatch.setattr(mod, "climb_until", _climb)
    with pytest.raises(PhaseStop) as caught:
        mod.play_ws_main_to_attic(session, start="pit_shot", stop="pit_shot")
    assert caught.value.phase == "pit_shot"
    assert seen == ["ws_main_to_attic_pit_shot"]
