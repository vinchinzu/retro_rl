"""Unit tests for powered Basement → Main Shaft (rr-kw8t hop 1). No emulator."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from super_metroid.combat.enemies import Enemy, list_enemies
from super_metroid.combat.enemies.workrobot import stall_reason
from super_metroid.hop_glance import LeaveMiss
from super_metroid.ram import FACING_LEFT, FACING_RIGHT, GameplayPhase, parse_state
from super_metroid.routes.kpdr.k6.ws_basement_ice import (
    ice_keepaway_action,
    workrobot_avoid_action,
)
from super_metroid.routes.kpdr.k6.ws_basement_return import (
    ATOMIC_ID,
    WORKROBOT_ID,
    WS_BASEMENT_HATCH_X_MAX,
    WS_BASEMENT_HATCH_X_MIN,
    WS_BASEMENT_PLATFORM_X,
    WS_BASEMENT_TAKEOFF_X_MIN,
    at_ws_basement_hatch_seat,
    hatch_jump_action,
    hatch_mount_action,
    play_ws_basement_to_main,
    ws_basement_main_settled,
)
from super_metroid.routes.skills.charge_shot import CHARGE_FULL
from super_metroid.routes.kpdr.room_ids import ROOM_WS_BASEMENT, ROOM_WS_MAIN
from super_metroid.routes.kpdr.registry import KPDR_SEGMENTS


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_WS_BASEMENT,
        "samus_x": 1240,
        "samus_y": 139,
        "pose": 10,
        "facing": 8,
        "health": 299,
        "max_health": 299,
        "game_state": 8,
        "door_transition": 0,
        "selected_item": 2,
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


def test_main_settled_requires_gs8() -> None:
    trans = _state(room_id=ROOM_WS_MAIN, game_state=11, door_transition=1)
    assert not ws_basement_main_settled(trans)
    gs11 = _state(room_id=ROOM_WS_MAIN, game_state=11, door_transition=0)
    assert not ws_basement_main_settled(gs11)
    ordinary = _state(
        room_id=ROOM_WS_MAIN, samus_x=1144, samus_y=1900, pose=1, game_state=8
    )
    assert ws_basement_main_settled(ordinary)
    still_basement = _state()
    assert not ws_basement_main_settled(still_basement)


def test_hatch_seat_is_ceiling_band() -> None:
    pin = _state()
    assert not at_ws_basement_hatch_seat(pin)
    seat = _state(samus_x=657, samus_y=91, pose=2)
    assert at_ws_basement_hatch_seat(seat)
    on_plat = _state(samus_x=657, samus_y=163, pose=2)
    assert at_ws_basement_hatch_seat(on_plat)
    floor = _state(samus_x=657, samus_y=187, pose=2)
    assert not at_ws_basement_hatch_seat(floor)
    peak = _state(samus_x=673, samus_y=160, pose=48, velocity_y=2)
    assert not at_ws_basement_hatch_seat(peak)
    lip = _state(samus_x=717, samus_y=163, pose=48)
    assert not at_ws_basement_hatch_seat(lip)
    wrong_room = _state(room_id=ROOM_WS_MAIN, samus_x=657, samus_y=91)
    assert not at_ws_basement_hatch_seat(wrong_room)


def test_hatch_mount_from_floor_under_hatch() -> None:
    """Under-hatch floor still flees east. On-platform lip walks to x=657."""
    under = hatch_mount_action(WS_BASEMENT_PLATFORM_X, 185, 2, 0)
    assert under == ("RIGHT", "B")
    assert "A" not in under
    in_band = hatch_mount_action(630, 185, 2, 0)
    assert in_band == ("RIGHT", "B")
    robot_block = hatch_mount_action(641, 187, 9, 0)
    assert robot_block == ("RIGHT", "B")
    still_in_band = hatch_mount_action(680, 185, 10, 0)
    assert still_in_band == ("RIGHT", "B")
    # Lip seat ~750: facing LEFT jumps; facing RIGHT turns first.
    takeoff = hatch_mount_action(WS_BASEMENT_TAKEOFF_X_MIN, 185, 2, 0)
    assert takeoff == ("LEFT", "B", "A")
    face_first = hatch_mount_action(
        WS_BASEMENT_TAKEOFF_X_MIN, 185, 2, 0, FACING_RIGHT, 0
    )
    assert face_first == ("LEFT",)
    finish_turn = hatch_mount_action(
        WS_BASEMENT_TAKEOFF_X_MIN, 185, 37, 0, FACING_LEFT, 14
    )
    assert finish_turn == ("LEFT",)
    too_close = hatch_mount_action(720, 185, 2, 0)
    assert too_close == ("RIGHT", "B")
    approach = hatch_mount_action(900, 185, 2, 0)
    assert approach == ("LEFT", "B")
    assert "A" not in approach
    on_seat = hatch_mount_action(657, 91, 2, 0)
    assert "A" not in on_seat
    assert hatch_mount_action(717, 163, 48, 2) == ("LEFT", "B", "A")
    lip_air_done = hatch_mount_action(717, 163, 48, 0)
    assert "LEFT" in lip_air_done
    lip = hatch_mount_action(717, 163, 2, 0)
    assert "LEFT" in lip
    assert "RIGHT" not in lip
    for x, y, pose, vy in (
        (630, 185, 81, 2),
        (630, 185, 2, 0),
        (641, 187, 9, 0),
        (680, 185, 10, 0),
        (657, 185, 2, 0),
        (719, 185, 2, 0),
        (740, 185, 2, 0),
        (900, 185, 2, 0),
        (657, 91, 2, 0),
    ):
        names = hatch_mount_action(x, y, pose, vy)
        assert "L" not in names
        assert "SUPER" not in names


def test_hatch_mount_latches_left_turn_across_x_drift() -> None:
    """Leftover just west of 750 p38 mov=14 facing RIGHT must finish LEFT.

    Drift below the 750 band used to resume RIGHT and oscillate. West
    approach that has not started turning still walks RIGHT into the band.
    """
    leftover = hatch_mount_action(748, 187, 38, 0, FACING_RIGHT, 14)
    assert leftover == ("LEFT",)
    assert "RIGHT" not in leftover
    after_turn = hatch_mount_action(748, 187, 2, 0, FACING_LEFT, 0)
    assert after_turn == ("LEFT", "B", "A")
    west_approach = hatch_mount_action(748, 187, 2, 0, FACING_RIGHT, 0)
    assert west_approach == ("RIGHT", "B")
    too_far = hatch_mount_action(787, 187, 9, 0, FACING_RIGHT, 1)
    assert too_far == ("LEFT", "B")
    air = hatch_mount_action(780, 180, 25, -4, FACING_LEFT, 3)
    assert air == ("LEFT", "B", "A")
    wall_spin = hatch_mount_action(728, 175, 26, 3)
    assert wall_spin == ("LEFT", "B", "A")
    leftover_spin = hatch_mount_action(819, 181, 82, 0, FACING_LEFT, 2)
    assert leftover_spin == ("LEFT", "B", "A")
    assert "RIGHT" not in leftover_spin
    lip_bounce = hatch_mount_action(737, 170, 82, 5, FACING_LEFT, 2)
    assert lip_bounce == ("LEFT", "B", "A")


def _atomic(*, x: int, y: int, freeze: int = 0, hp: int = 250) -> Enemy:
    return Enemy(0, ATOMIC_ID, x, y, hp, freeze)


def _robot(*, x: int, y: int = 176, hp: int = 800) -> Enemy:
    return Enemy(1, WORKROBOT_ID, x, y, hp, 0)


def test_ice_keepaway_taps_x_until_atomic_is_dead() -> None:
    shot = ice_keepaway_action(670, 185, FACING_LEFT, (_atomic(x=638, y=168),))
    assert "X" in shot
    assert "LEFT" not in shot
    face = ice_keepaway_action(670, 185, FACING_RIGHT, (_atomic(x=638, y=168),))
    assert face == ("LEFT",)
    frozen_alive = ice_keepaway_action(
        670, 185, FACING_LEFT, (_atomic(x=638, y=168, freeze=180),)
    )
    assert "X" in frozen_alive
    # x=879 cannot shoot through the hatch pillar — walk into a seat.
    blocked = ice_keepaway_action(879, 187, FACING_LEFT, (_atomic(x=638, y=168),))
    assert blocked[0] == "LEFT"
    assert "B" in blocked
    under = ice_keepaway_action(900, 185, FACING_LEFT, (_atomic(x=852, y=73),))
    assert "X" in under
    assert "R" in under
    above = ice_keepaway_action(852, 185, FACING_LEFT, (_atomic(x=852, y=73),))
    assert above == ("UP", "X")
    dead = ice_keepaway_action(
        670, 185, FACING_LEFT, (_atomic(x=638, y=168, hp=0),)
    )
    assert dead is None
    map_side = ice_keepaway_action(900, 185, FACING_LEFT, (_atomic(x=152, y=77),))
    assert map_side is None
    overlap = ice_keepaway_action(638, 168, FACING_LEFT, (_atomic(x=638, y=168),))
    assert overlap == ("X",)


def test_ice_keepaway_charge_release_and_robot_clamp() -> None:
    blob = _atomic(x=638, y=168)
    robot = _robot(x=624)
    release = ice_keepaway_action(
        672, 187, FACING_LEFT, (blob, robot), charge=CHARGE_FULL, velocity_y=2
    )
    assert release is not None
    assert "X" not in release
    assert "A" in release
    turning = ice_keepaway_action(
        672, 187, FACING_LEFT, (blob,), movement_type=14
    )
    assert turning == ("LEFT",)
    assert "X" not in turning
    assert stall_reason(672, 187, 14, 37, (blob, robot)) == "turning"
    assert stall_reason(640, 187, 0, 2, (robot,)) == "workrobot"
    frozen = _atomic(x=670, y=185, freeze=180)
    assert stall_reason(670, 185, 0, 2, (frozen,)) == "frozen_atomic"
    assert stall_reason(879, 187, 0, 2, (blob,)) is None


def test_workrobot_avoid_does_not_walk_into_robot() -> None:
    wait = workrobot_avoid_action(900, 185, (_robot(x=880),))
    assert wait == ()
    flee = workrobot_avoid_action(657, 187, (_robot(x=657),))
    assert flee == ("RIGHT", "B")
    clear = workrobot_avoid_action(740, 185, (_robot(x=657),))
    assert clear is None


def test_list_enemies_reads_freeze_timer() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = 0x0F78
    ram[base] = ATOMIC_ID & 0xFF
    ram[base + 1] = ATOMIC_ID >> 8
    ram[base + 0x02] = 638 & 0xFF
    ram[base + 0x03] = 638 >> 8
    ram[base + 0x06] = 168
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
    assert found[0].x == 638
    assert found[0].freeze_timer == 40


def test_run_to_hatch_does_not_spin_left_in_band() -> None:
    from super_metroid.routes.kpdr.k6 import ws_basement_return as mod

    src = inspect.getsource(mod._run_to_hatch)
    assert "hatch_mount_action" in src
    assert src.count("list_enemies(session)") == 1
    assert src.count("choice = choose(") == 1
    assert "ice_keepaway_action(" not in src
    assert "workrobot_avoid_action(" not in src
    assert "session_beam_charge" in src
    assert "_ICE_TAP_FRAMES" not in src
    assert "_MOVEMENT_STUN" not in src
    assert "pose) in (37, 38)" not in src
    assert f"{'{'}label{'}'}_hop" not in src
    assert "LEFT\", \"B\", \"A\"" not in src

    # Hop-local takeoff is x=750. At x=740 the generic x=720 takeoff idled.
    robot = _robot(x=720)
    choice = mod.choose(
        740,
        185,
        FACING_LEFT,
        (robot,),
        mod._HATCH_OVERLAY,
        takeoff_x_min=mod.WS_BASEMENT_TAKEOFF_X_MIN,
    )
    assert choice.stance.name == "AVOID"
    assert choice.buttons == ("RIGHT", "B")


def test_hatch_jump_is_up_a_not_super_or_l() -> None:
    assert hatch_jump_action(657, 163, 2, 0) == ("UP", "X")
    assert hatch_jump_action(657, 163, 2, 60) == ("UP",)
    assert "A" not in hatch_jump_action(657, 163, 2, 64)
    assert hatch_jump_action(657, 163, 2, 308) == ("UP", "A")
    assert hatch_jump_action(662, 91, 4, 0) == ("UP", "X")
    assert hatch_jump_action(662, 91, 4, 60) == ("UP",)
    assert hatch_jump_action(662, 91, 4, 68) == ("UP", "A")
    open_hatch = hatch_jump_action(672, 68, 22, 0)
    assert open_hatch == ("LEFT", "A")
    assert "UP" not in open_hatch
    in_shaft = hatch_jump_action(663, 36, 22, 0)
    assert in_shaft == ("LEFT", "A")
    assert "UP" not in in_shaft
    shaft_center = hatch_jump_action(657, 36, 22, 0)
    assert shaft_center == ("A",)
    assert "A" not in hatch_jump_action(662, 91, 4, 0)
    assert "LEFT" in hatch_jump_action(690, 176, 22, 0)
    assert hatch_jump_action(640, 91, 4, 0) == ("RIGHT", "UP", "A")
    assert hatch_jump_action(657, 91, 138, 0) == ()
    x_frames = sum(
        1 for frame in range(80) if "X" in hatch_jump_action(657, 163, 2, frame)
    )
    assert 50 <= x_frames <= 62
    for frame in range(80):
        names = hatch_jump_action(657, 163, 2, frame)
        assert "L" not in names
        assert "SUPER" not in names


def test_already_in_main_is_noop() -> None:
    session = _Session(
        _state(room_id=ROOM_WS_MAIN, samus_x=1144, samus_y=1900, pose=1)
    )
    out = play_ws_basement_to_main(session)
    assert out.room_id == ROOM_WS_MAIN
    assert session.actions == []


def test_wrong_room() -> None:
    session = _Session(_state(room_id=0xCD13, samus_x=39, samus_y=128, pose=1))
    with pytest.raises(LeaveMiss, match="ws_basement_to_main") as caught:
        play_ws_basement_to_main(session)
    err = caught.value
    assert err.leftover["xy"] == [39, 128]
    assert err.leftover["pose"] == 1
    assert err.leftover["gs"] == 8
    assert "expected Wrecked Ship Main Shaft 0xCAF6, got 0xCD13" in str(err)


def test_requires_boss_bit() -> None:
    session = _Session(_state(boss_bits=(0, 0, 0, 0, 0, 0, 0, 0)))
    with pytest.raises(RuntimeError, match="not defeated"):
        play_ws_basement_to_main(session)


def test_morph_bombs_are_x_not_a() -> None:
    from super_metroid.routes.kpdr.k6 import ws_basement_return as mod

    bomb_src = inspect.getsource(mod._bomb_tunnel_left)
    assert 'hold(session, 3, "X"' in bomb_src
    assert 'reason=f"{label}_bomb"' in bomb_src
    assert 'hold(session, 3, "A"' not in bomb_src


def test_probe_uses_repo_headed() -> None:
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "scripts" / "probe" / "ws_basement_return.py"
    text = src.read_text(encoding="utf-8")
    assert "from retro_harness.headed import" in text
    assert "add_headed_flag" in text
    assert "attach_headed" in text
    assert "def _attach_headed" not in text
    assert '"leftover"' in text
    assert "glance_misses" in text
    assert "final_from_state" in text


def test_never_uses_l_as_left() -> None:
    from super_metroid.routes.kpdr.k6 import ws_basement_return as mod

    src = inspect.getsource(mod)
    assert 'hold(session, 1, "L"' not in src
    assert '", "L"' not in src


def test_registered() -> None:
    from super_metroid.routes.kpdr.k6 import play_ws_basement_to_main as play

    assert KPDR_SEGMENTS["ws_basement_to_main"] is play
    assert callable(play)


def test_play_drops_bombs_runs_jumps(monkeypatch: pytest.MonkeyPatch) -> None:
    from super_metroid.routes.kpdr.k6 import ws_basement_return as mod

    session = _Session(_state())
    seen: dict[str, object] = {}

    def _require(sess, room, label):
        seen["require"] = (room, label)

    def _drop(sess, label):
        seen["drop"] = label

    def _bomb(sess, label):
        seen["bomb"] = label
        sess.state = replace(sess.state, samus_x=880, samus_y=187, pose=31)

    def _run(sess, label):
        seen["run"] = label
        sess.state = replace(sess.state, samus_x=657, samus_y=91, pose=2)

    def _jump(sess, label):
        seen["jump"] = label
        sess.state = replace(
            sess.state,
            room_id=ROOM_WS_MAIN,
            game_state=11,
            door_transition=1,
            samus_x=1144,
            samus_y=1900,
        )

    def _settle(sess, room, **kwargs):
        seen["settle"] = (room, kwargs.get("label"))
        sess.state = replace(
            sess.state, room_id=room, game_state=8, door_transition=0
        )
        return sess.state

    monkeypatch.setattr(mod, "require_room", _require)
    monkeypatch.setattr(mod, "_drop_to_tunnel_floor", _drop)
    monkeypatch.setattr(mod, "_bomb_tunnel_left", _bomb)
    monkeypatch.setattr(mod, "_run_to_hatch", _run)
    monkeypatch.setattr(mod, "_jump_up_hatch", _jump)
    monkeypatch.setattr(mod, "settle_ceiling_dest", _settle)

    out = mod.play_ws_basement_to_main(session)
    assert seen["require"][0] == ROOM_WS_BASEMENT
    assert seen["drop"] == "ws_basement_to_main"
    assert seen["bomb"] == "ws_basement_to_main"
    assert seen["run"] == "ws_basement_to_main"
    assert seen["jump"] == "ws_basement_to_main"
    assert seen["settle"] == (ROOM_WS_MAIN, "ws_basement_to_main")
    assert ws_basement_main_settled(out)
