"""Unit tests for powered Basement → Main Shaft (rr-kw8t hop 1). No emulator."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from super_metroid.ram import FACING_LEFT, FACING_RIGHT, GameplayPhase, parse_state
from super_metroid.routes.kpdr.k6.ws_basement_return import (
    ATOMIC_ID,
    WORKROBOT_ID,
    WS_BASEMENT_HATCH_X_MAX,
    WS_BASEMENT_HATCH_X_MIN,
    WS_BASEMENT_PLATFORM_X,
    WS_BASEMENT_TAKEOFF_X_MIN,
    BasementEnemy,
    at_ws_basement_hatch_seat,
    hatch_jump_action,
    hatch_mount_action,
    ice_keepaway_action,
    list_basement_enemies,
    play_ws_basement_to_main,
    workrobot_avoid_action,
    ws_basement_main_settled,
)
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
    floor = _state(samus_x=657, samus_y=187, pose=2)
    assert not at_ws_basement_hatch_seat(floor)
    wrong_room = _state(room_id=ROOM_WS_MAIN, samus_x=657, samus_y=91)
    assert not at_ws_basement_hatch_seat(wrong_room)


def test_hatch_mount_from_floor_under_hatch() -> None:
    """Under-hatch is occupied. Takeoff is x≳720, spin-LEFT onto y=91."""
    under = hatch_mount_action(WS_BASEMENT_PLATFORM_X, 185, 2, 0)
    assert under == ("RIGHT", "B")
    assert "A" not in under
    in_band = hatch_mount_action(630, 185, 2, 0)
    assert in_band == ("RIGHT", "B")
    robot_block = hatch_mount_action(641, 187, 9, 0)
    assert robot_block == ("RIGHT", "B")
    still_in_band = hatch_mount_action(680, 185, 10, 0)
    assert still_in_band == ("RIGHT", "B")
    takeoff = hatch_mount_action(WS_BASEMENT_TAKEOFF_X_MIN, 185, 2, 0)
    assert takeoff == ("LEFT", "B", "A")
    approach = hatch_mount_action(880, 185, 2, 0)
    assert approach == ("LEFT", "B")
    assert "A" not in approach
    on_seat = hatch_mount_action(657, 91, 2, 0)
    assert "A" not in on_seat
    for x, y, pose, vy in (
        (630, 185, 81, 2),
        (630, 185, 2, 0),
        (641, 187, 9, 0),
        (680, 185, 10, 0),
        (657, 185, 2, 0),
        (740, 185, 2, 0),
        (880, 185, 2, 0),
        (657, 91, 2, 0),
    ):
        names = hatch_mount_action(x, y, pose, vy)
        assert "L" not in names
        assert "SUPER" not in names


def _atomic(*, x: int, y: int, freeze: int = 0, hp: int = 250) -> BasementEnemy:
    return BasementEnemy(0, ATOMIC_ID, x, y, hp, freeze)


def _robot(*, x: int, y: int = 176, hp: int = 800) -> BasementEnemy:
    return BasementEnemy(1, WORKROBOT_ID, x, y, hp, 0)


def test_ice_keepaway_taps_x_until_atomic_is_dead() -> None:
    shot = ice_keepaway_action(670, 185, FACING_LEFT, (_atomic(x=638, y=168),))
    assert shot == ("X",)
    face = ice_keepaway_action(670, 185, FACING_RIGHT, (_atomic(x=638, y=168),))
    assert face == ("LEFT",)
    frozen_alive = ice_keepaway_action(
        670, 185, FACING_LEFT, (_atomic(x=638, y=168, freeze=180),)
    )
    assert frozen_alive == ("X",)
    under = ice_keepaway_action(900, 185, FACING_LEFT, (_atomic(x=852, y=73),))
    assert under == ("LEFT",)
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


def test_workrobot_avoid_does_not_walk_into_robot() -> None:
    wait = workrobot_avoid_action(900, 185, (_robot(x=880),))
    assert wait == ()
    flee = workrobot_avoid_action(657, 187, (_robot(x=657),))
    assert flee == ("RIGHT", "B")
    clear = workrobot_avoid_action(740, 185, (_robot(x=657),))
    assert clear is None


def test_list_basement_enemies_reads_freeze_timer() -> None:
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
    found = list_basement_enemies(session)
    assert len(found) == 1
    assert found[0].enemy_id == ATOMIC_ID
    assert found[0].x == 638
    assert found[0].freeze_timer == 40


def test_run_to_hatch_does_not_spin_left_in_band() -> None:
    from super_metroid.routes.kpdr.k6 import ws_basement_return as mod

    src = inspect.getsource(mod._run_to_hatch)
    assert "hatch_mount_action" in src
    assert "ice_keepaway_action" in src
    assert "workrobot_avoid_action" in src
    assert "_ICE_TAP_FRAMES" in src
    assert "_ICE_RELEASE_FRAMES" in src
    assert f"{'{'}label{'}'}_hop" not in src
    assert "LEFT\", \"B\", \"A\"" not in src


def test_hatch_jump_is_up_a_not_super_or_l() -> None:
    assert hatch_jump_action(657, 91, 2, 0) == ("UP", "X")
    assert hatch_jump_action(657, 91, 2, 2) == ("UP",)
    assert hatch_jump_action(657, 91, 2, 10) == ("UP", "A")
    assert "LEFT" in hatch_jump_action(WS_BASEMENT_HATCH_X_MAX + 10, 91, 2, 0)
    assert "RIGHT" in hatch_jump_action(WS_BASEMENT_HATCH_X_MIN - 10, 91, 2, 0)
    assert hatch_jump_action(657, 91, 138, 0) == ()
    x_frames = sum(
        1 for frame in range(40) if "X" in hatch_jump_action(657, 91, 2, frame)
    )
    assert 1 <= x_frames <= 4
    for frame in range(40):
        names = hatch_jump_action(657, 91, 2, frame)
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
    session = _Session(_state(room_id=0xCD13))
    with pytest.raises(RuntimeError, match="ws_basement_to_main"):
        play_ws_basement_to_main(session)


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
    monkeypatch.setattr(mod, "wait_ordinary_room", _settle)

    out = mod.play_ws_basement_to_main(session)
    assert seen["require"][0] == ROOM_WS_BASEMENT
    assert seen["drop"] == "ws_basement_to_main"
    assert seen["bomb"] == "ws_basement_to_main"
    assert seen["run"] == "ws_basement_to_main"
    assert seen["jump"] == "ws_basement_to_main"
    assert seen["settle"] == (ROOM_WS_MAIN, "ws_basement_to_main")
    assert ws_basement_main_settled(out)
