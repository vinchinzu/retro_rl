"""Unit tests for powered Main Shaft → Attic (rr-kw8t hop 2). No emulator."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from super_metroid.hop_glance import LeaveMiss
from super_metroid.ram import FACING_LEFT, FACING_RIGHT, GameplayPhase, parse_state
from super_metroid.routes.kpdr.k6.ws_main_climb import (
    ATOMIC_ID,
    SHAFT_HOPS,
    THREE_SHOT_X_MIN,
    WS_MAIN_ATTIC_DOOR_X,
    ShaftEnemy,
    at_ws_main_attic_door_seat,
    at_ws_main_pit,
    attic_door_action,
    climb_action,
    grate_clear_action,
    ice_keepaway_action,
    list_shaft_enemies,
    pit_exit_action,
    play_ws_main_to_attic,
    three_shot_action,
    ws_main_attic_settled,
)
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
    grate = climb_action(1075, 1845, 2, FACING_LEFT)
    assert grate == ("RIGHT",)
    grate_jump = climb_action(1075, 1845, 2, FACING_RIGHT)
    assert grate_jump == ("RIGHT", "B", "A")
    assert "LEFT" not in grate_jump
    lip = climb_action(1177, 1883, 2, FACING_LEFT)
    assert lip == ("DOWN",)
    assert "A" not in lip
    lip_crouch = climb_action(1177, 1883, 39, FACING_LEFT)
    assert lip_crouch == ("DOWN",)
    save = climb_action(1240, 1675, 2, FACING_RIGHT)
    assert save == ("LEFT", "B")
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


def test_grate_clear_jumps_right_without_wave_up() -> None:
    assert grate_clear_action(1075, 1700, 1, FACING_LEFT, 0) is None
    assert grate_clear_action(1075, 1979, 1, FACING_LEFT, 0) is None
    face = grate_clear_action(1082, 1878, 8, FACING_LEFT, 0, charge=0)
    assert face == ("RIGHT",)
    assert "UP" not in face
    jump = grate_clear_action(1075, 1845, 1, FACING_RIGHT, 0, charge=0)
    assert jump == ("RIGHT", "B", "A")
    assert "UP" not in jump
    mid = grate_clear_action(1152, 1845, 1, FACING_RIGHT, 0)
    assert mid == ("RIGHT", "B", "A")
    assert "UP" not in mid
    lip = grate_clear_action(1177, 1883, 2, FACING_LEFT, 0)
    assert lip == ("DOWN",)
    assert "A" not in lip
    assert "UP" not in lip
    crouch = grate_clear_action(1177, 1883, 39, FACING_LEFT, 0)
    assert crouch == ("DOWN",)
    air = grate_clear_action(1177, 1800, 20, FACING_LEFT, 0)
    assert air == ("LEFT", "A")
    assert "UP" not in air
    landing = grate_clear_action(1177, 1880, 19, FACING_RIGHT, 0)
    assert landing is None


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
    blob = ShaftEnemy(0, ATOMIC_ID, 1150, 1160, 250, 0)
    pit = ice_keepaway_action(1173, 1979, FACING_LEFT, (blob,))
    assert pit is None
    shot = ice_keepaway_action(1152, 1163, FACING_LEFT, (blob,))
    assert shot is not None
    assert "X" in shot or "A" in shot
    none = ice_keepaway_action(1152, 1163, FACING_LEFT, ())
    assert none is None
    from super_metroid.routes.kpdr.k6.ws_main_ice import COVERN_ID

    covern = ShaftEnemy(0, COVERN_ID, 1129, 1818, 80, 0)
    grate = ice_keepaway_action(1075, 1845, FACING_LEFT, (covern,))
    assert grate is not None
    frozen = ShaftEnemy(0, ATOMIC_ID, 1152, 1163, 250, 80)
    wait = ice_keepaway_action(1152, 1163, FACING_LEFT, (frozen,))
    assert wait == ()


def test_list_shaft_enemies_reads_freeze_timer() -> None:
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
    found = list_shaft_enemies(session)
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
    from super_metroid.routes.kpdr.k6 import ws_main_climb as mod

    src = inspect.getsource(mod._three_shot_tunnel)
    assert "at_ws_main_grate_seat" in src
    assert 'reason=f"{label}_bomb"' not in src
    assert 'hold(session, 3, "A"' not in src


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
    assert "play_ws_main_to_attic_phased" in text


def test_never_uses_l_as_left() -> None:
    from super_metroid.routes.kpdr.k6 import ws_main_actions as actions
    from super_metroid.routes.kpdr.k6 import ws_main_climb as mod

    for src in (inspect.getsource(mod), inspect.getsource(actions)):
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

    def _climb(sess, label):
        seen["climb"] = label
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

    def _settle(sess, room, **kwargs):
        seen["settle"] = (room, kwargs.get("label"))
        sess.state = replace(
            sess.state, room_id=room, game_state=8, door_transition=0
        )
        return sess.state

    monkeypatch.setattr(mod, "require_room", _require)
    monkeypatch.setattr(mod, "_three_shot_tunnel", _shot)
    monkeypatch.setattr(mod, "_climb_to_attic_door", _climb)
    monkeypatch.setattr(mod, "_jump_up_attic", _jump)
    monkeypatch.setattr(mod, "wait_ordinary_room", _settle)

    out = mod.play_ws_main_to_attic(session)
    assert seen["require"][0] == ROOM_WS_MAIN
    assert seen["shot"] == "ws_main_to_attic"
    assert seen["climb"] == "ws_main_to_attic"
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
    monkeypatch.setattr(mod, "_three_shot_tunnel", _shot)
    monkeypatch.setattr(mod, "_climb_until", _climb)
    with pytest.raises(PhaseStop) as caught:
        mod.play_ws_main_to_attic_phased(session, start="pit_shot", stop="pit_shot")
    assert caught.value.phase == "pit_shot"
    assert seen == ["ws_main_to_attic_pit_shot"]
