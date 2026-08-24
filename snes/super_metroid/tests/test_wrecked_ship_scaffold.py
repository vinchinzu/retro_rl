"""Unit locks for Wrecked Ship: Entrance→Main, Main→basement, Basement→Phantoon room."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from super_metroid.ram import parse_state
from super_metroid.routes.kpdr import wrecked_ship
from super_metroid.routes.kpdr.guides import ROUTE_PRESETS
from super_metroid.source_states import get_source


def test_wrecked_ship_rooms_and_controllers() -> None:
    assert wrecked_ship.ROOM_KIHUNTER == 0x948C
    assert wrecked_ship.ROOM_MOAT == 0x95FF
    assert wrecked_ship.ROOM_WEST_OCEAN == 0x93FE
    assert wrecked_ship.ROOM_WS_ENTRANCE == 0xCA08
    assert wrecked_ship.ROOM_WS_MAIN == 0xCAF6
    assert wrecked_ship.ROOM_WS_BASEMENT == 0xCC6F
    assert wrecked_ship.ROOM_PHANTOON == 0xCD13
    for name in (
        "play_moat_to_west_ocean",
        "play_moat_to_ws",
        "play_west_ocean_to_ws",
        "play_ws_entrance_to_main",
        "play_ws_main_to_basement",
        "play_ws_basement_to_phantoon",
    ):
        assert callable(getattr(wrecked_ship, name))


def test_ws_entrance_to_main_is_not_scaffold() -> None:
    src = inspect.getsource(wrecked_ship.play_ws_entrance_to_main)
    assert "_scaffold_exit" not in src
    assert "super_door=False" in src
    basement = inspect.getsource(wrecked_ship.play_ws_main_to_basement)
    phant = inspect.getsource(wrecked_ship.play_ws_basement_to_phantoon)
    assert "_scaffold_exit" not in basement
    assert "play_script" in basement
    assert sum(n for n, _b in wrecked_ship._WS_MAIN_RLE) == 1091
    assert "_scaffold_exit" not in phant
    assert "wait_ordinary_room" in phant
    assert "Do not fight" in phant or "do not fight" in phant.lower()


def test_moat_to_ws_compose_and_phantoon_recording_wire() -> None:
    """Compose surface + WS pin + recording routes for Phantoon ship tape."""
    assert "west-ocean-to-ws" in ROUTE_PRESETS
    assert "ws-entrance" in ROUTE_PRESETS

    pre = get_source("post_kihunter_pre_moat_spark")
    assert pre.room_id == 0x948C
    assert pre.relative_path.endswith("post_kihunter_pre_moat_spark.state")

    moat_end = get_source("alpha_pb_to_moat_human_end")
    assert moat_end.room_id == 0x95FF

    wo = get_source("post_moat_west_ocean_spark")
    assert wo.room_id == 0x93FE

    ws_pin = get_source("post_west_ocean_ws_spark")
    assert ws_pin.room_id == 0xCA08
    assert "Phantoon" in ws_pin.use_for or "ship free-record" in ws_pin.use_for

    phant_entry = get_source("ws_ship_human_end")
    assert phant_entry.room_id == 0xCD13

    main_leave = get_source("post_ws_entrance_to_main")
    assert main_leave.room_id == 0xCAF6
    assert main_leave.relative_path.endswith("post_ws_entrance_to_main.state")

    basement_leave = get_source("post_ws_main_to_basement")
    assert basement_leave.room_id == 0xCC6F
    assert basement_leave.relative_path.endswith("post_ws_main_to_basement.state")

    phant_leave = get_source("post_ws_basement_to_phantoon")
    assert phant_leave.room_id == 0xCD13
    assert phant_leave.relative_path.endswith("post_ws_basement_to_phantoon.state")


class _FakeSession:
    def __init__(self, state: Any, transitions: list[Any] | None = None) -> None:
        self.state = state
        self.frame = int(getattr(state, "frame", 0))
        self._transitions = list(transitions or [])
        self._step_i = 0

    def step(self, action: np.ndarray, reason: str = "") -> Any:
        del action, reason
        self.frame += 1
        if self._step_i < len(self._transitions):
            self.state = self._transitions[self._step_i]
            self._step_i += 1
        else:
            self.state = replace(self.state, frame=self.frame)
        return self.state


def _state(**kwargs: Any) -> Any:
    base = parse_state(np.zeros(0x10000, dtype=np.uint8))
    return replace(base, **kwargs)


def test_ws_entrance_door_seat() -> None:
    pin = _state(room_id=0xCA08, samus_x=57, samus_y=139, pose=1, game_state=8)
    assert not wrecked_ship.at_ws_entrance_door_seat(pin)
    seat = _state(room_id=0xCA08, samus_x=920, samus_y=139, pose=9, game_state=8)
    assert wrecked_ship.at_ws_entrance_door_seat(seat)
    assert wrecked_ship.WS_ENTRANCE_DOOR_X_MIN >= 900
    wrong_room = _state(room_id=0xCAF6, samus_x=920)
    assert not wrecked_ship.at_ws_entrance_door_seat(wrong_room)


def test_ws_entrance_action_never_supers_blue_door() -> None:
    supers = _state(
        room_id=0xCA08, samus_x=57, selected_item=2, game_state=8
    )
    assert wrecked_ship.ws_entrance_to_main_action(supers) == ("SELECT",)
    assert "X" not in wrecked_ship.ws_entrance_to_main_action(supers)

    dash = _state(
        room_id=0xCA08, samus_x=200, selected_item=wrecked_ship.WEAPON_BEAM
    )
    assert wrecked_ship.ws_entrance_to_main_action(dash) == ("RIGHT", "B")
    assert "X" not in wrecked_ship.ws_entrance_to_main_action(dash)

    door = _state(
        room_id=0xCA08,
        samus_x=wrecked_ship.WS_ENTRANCE_DOOR_X_MIN,
        selected_item=wrecked_ship.WEAPON_BEAM,
    )
    shot = wrecked_ship.ws_entrance_to_main_action(door)
    assert shot == ("RIGHT", "B", "X")
    assert "SELECT" not in shot

    still_supers_at_door = _state(
        room_id=0xCA08, samus_x=980, selected_item=2
    )
    assert wrecked_ship.ws_entrance_to_main_action(still_supers_at_door) == (
        "SELECT",
    )


def test_ws_entrance_main_settled_requires_gs8() -> None:
    trans = _state(
        room_id=0xCAF6, game_state=11, door_transition=1, samus_x=40, samus_y=139
    )
    assert not wrecked_ship.ws_entrance_main_settled(trans)
    gs11 = replace(trans, game_state=11, door_transition=0)
    assert not wrecked_ship.ws_entrance_main_settled(gs11)
    ordinary = replace(trans, game_state=8, door_transition=0)
    assert wrecked_ship.ws_entrance_main_settled(ordinary)
    still_entrance = _state(room_id=0xCA08, game_state=8, door_transition=0)
    assert not wrecked_ship.ws_entrance_main_settled(still_entrance)


def test_ws_main_green_floor_seat() -> None:
    pin = _state(room_id=0xCAF6, samus_x=1063, samus_y=907, pose=9, game_state=8)
    assert not wrecked_ship.at_ws_main_green_floor(pin)
    hatch = _state(
        room_id=0xCAF6,
        samus_x=1143,
        samus_y=wrecked_ship.WS_MAIN_BOTTOM_Y,
        pose=1,
        game_state=8,
    )
    assert wrecked_ship.at_ws_main_green_floor(hatch)
    wrong_room = _state(room_id=0xCC6F, samus_x=1143, samus_y=1700)
    assert not wrecked_ship.at_ws_main_green_floor(wrong_room)


def test_ws_main_action_never_goes_up() -> None:
    pin = _state(room_id=0xCAF6, samus_x=1077, samus_y=907, pose=9, game_state=8)
    act = wrecked_ship.ws_main_to_basement_action(pin)
    assert "UP" not in act
    attic_y = _state(room_id=0xCAF6, samus_x=1150, samus_y=800, pose=1)
    assert wrecked_ship.ws_main_to_basement_action(attic_y) == ("DOWN",)
    assert "UP" not in wrecked_ship.ws_main_to_basement_action(attic_y)


def test_ws_main_action_supers_green_floor_door() -> None:
    beam = _state(
        room_id=0xCAF6,
        samus_x=1143,
        samus_y=wrecked_ship.WS_MAIN_BOTTOM_Y + 20,
        selected_item=0,
        game_state=8,
    )
    assert wrecked_ship.ws_main_to_basement_action(beam) == ("SELECT",)
    assert "X" not in wrecked_ship.ws_main_to_basement_action(beam)

    supers = _state(
        room_id=0xCAF6,
        samus_x=1143,
        samus_y=wrecked_ship.WS_MAIN_BOTTOM_Y + 20,
        selected_item=wrecked_ship.WEAPON_SUPER,
        game_state=8,
    )
    shot = wrecked_ship.ws_main_to_basement_action(supers)
    assert "X" in shot
    assert "L" in shot
    assert "DOWN" not in shot  # DOWN+X morphs; L is shoulder aim-down
    assert "SELECT" not in shot
    # Floor door — not a horizontal Super.
    assert shot != ("RIGHT", "X")
    assert shot != ("RIGHT", "B", "X")


def test_ws_main_basement_settled_requires_gs8() -> None:
    trans = _state(
        room_id=0xCC6F, game_state=11, door_transition=1, samus_x=40, samus_y=139
    )
    assert not wrecked_ship.ws_main_basement_settled(trans)
    gs11 = replace(trans, game_state=11, door_transition=0)
    assert not wrecked_ship.ws_main_basement_settled(gs11)
    ordinary = replace(trans, game_state=8, door_transition=0)
    assert wrecked_ship.ws_main_basement_settled(ordinary)
    still_main = _state(room_id=0xCAF6, game_state=8, door_transition=0)
    assert not wrecked_ship.ws_main_basement_settled(still_main)


def test_play_ws_main_to_basement_plays_human_rle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(
        _state(
            room_id=0xCAF6,
            samus_x=1077,
            samus_y=907,
            pose=9,
            game_state=8,
            door_transition=0,
            selected_item=0,
        )
    )
    seen: dict[str, Any] = {}

    def _require(sess: Any, room: int, label: str) -> None:
        seen["require"] = (room, label)
        assert room == 0xCAF6

    def _script(sess: Any, runs: Any, **kwargs: Any) -> Any:
        seen["rle"] = kwargs.get("reason")
        seen["rle_n"] = sum(n for n, _btns in runs)
        sess.state = replace(
            sess.state,
            room_id=0xCC6F,
            game_state=11,
            door_transition=1,
            samus_x=657,
            samus_y=92,
        )
        return sess.state

    def _settle(sess: Any, room: int, **kwargs: Any) -> Any:
        seen["settle"] = (room, kwargs.get("label"))
        sess.state = replace(
            sess.state, room_id=room, game_state=8, door_transition=0
        )
        return sess.state

    monkeypatch.setattr(wrecked_ship, "require_room", _require)
    monkeypatch.setattr(wrecked_ship, "play_script", _script)
    monkeypatch.setattr(wrecked_ship, "wait_ordinary_room", _settle)

    out = wrecked_ship.play_ws_main_to_basement(session)
    assert seen["require"][0] == 0xCAF6
    assert seen["rle"] == "ws_main_to_basement_body"
    assert seen["rle_n"] == 1091
    assert seen["settle"] == (0xCC6F, "ws_main_to_basement")
    assert wrecked_ship.ws_main_basement_settled(out)


def test_play_ws_entrance_to_main_selects_beam_then_blue_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession(
        _state(
            room_id=0xCA08,
            samus_x=57,
            samus_y=139,
            pose=1,
            game_state=8,
            door_transition=0,
            selected_item=2,
        )
    )
    seen: dict[str, Any] = {}

    def _select(sess: Any, target: int, **_kw: Any) -> None:
        seen["weapon"] = target
        sess.state = replace(sess.state, selected_item=target)

    def _hold_until(
        sess: Any,
        pred: Any,
        *buttons: str,
        timeout: int = 120,
        reason: str = "",
    ) -> Any:
        del timeout
        seen["dash"] = buttons
        seen["dash_reason"] = reason
        sess.state = replace(sess.state, samus_x=920, pose=9, selected_item=0)
        assert pred(sess.state)
        return sess.state

    def _exit(sess: Any, **kwargs: Any) -> Any:
        seen["exit"] = kwargs
        sess.state = replace(
            sess.state,
            room_id=0xCAF6,
            game_state=8,
            door_transition=0,
            samus_x=48,
            samus_y=139,
        )
        return sess.state

    monkeypatch.setattr(wrecked_ship, "select_weapon", _select)
    monkeypatch.setattr(wrecked_ship, "hold_until", _hold_until)
    monkeypatch.setattr(wrecked_ship, "play_run_shoot_exit", _exit)

    out = wrecked_ship.play_ws_entrance_to_main(session)
    assert seen["weapon"] == wrecked_ship.WEAPON_BEAM
    assert seen["weapon"] != 2
    assert seen["dash"] == ("RIGHT", "B")
    assert seen["exit"]["super_door"] is False
    assert seen["exit"]["from_room"] == 0xCA08
    assert seen["exit"]["to_room"] == 0xCAF6
    assert seen["exit"]["direction"] == "RIGHT"
    assert wrecked_ship.ws_entrance_main_settled(out)


def test_ws_basement_phantoon_settled_requires_gs8() -> None:
    trans = _state(
        room_id=0xCD13, game_state=11, door_transition=1, samus_x=40, samus_y=139
    )
    assert not wrecked_ship.ws_basement_phantoon_settled(trans)
    gs11 = replace(trans, game_state=11, door_transition=0)
    assert not wrecked_ship.ws_basement_phantoon_settled(gs11)
    ordinary = replace(trans, game_state=8, door_transition=0)
    assert wrecked_ship.ws_basement_phantoon_settled(ordinary)
    still_basement = _state(room_id=0xCC6F, game_state=8, door_transition=0)
    assert not wrecked_ship.ws_basement_phantoon_settled(still_basement)


def test_ws_basement_bomb_and_eye_seats() -> None:
    pin = _state(room_id=0xCC6F, samus_x=657, samus_y=92, pose=24, game_state=8)
    assert not wrecked_ship.at_ws_basement_bomb_blocks(pin)
    assert not wrecked_ship.at_ws_basement_eye_seat(pin)
    bomb = _state(
        room_id=0xCC6F,
        samus_x=1051,
        samus_y=201,
        pose=30,
        game_state=8,
    )
    assert wrecked_ship.at_ws_basement_bomb_blocks(bomb)
    eye = _state(
        room_id=0xCC6F,
        samus_x=1180,
        samus_y=201,
        pose=30,
        game_state=8,
    )
    assert wrecked_ship.at_ws_basement_eye_seat(eye)
    wrong = _state(room_id=0xCD13, samus_x=1180, samus_y=201)
    assert not wrecked_ship.at_ws_basement_eye_seat(wrong)


def test_ws_basement_product_morph_bombs_are_x_not_a() -> None:
    from super_metroid.routes.kpdr.k6 import ws_basement as k6ws

    assert "ensure_morph" in inspect.getsource(k6ws)
    bomb_src = inspect.getsource(k6ws._bomb_tunnel)
    assert "hold(session, 3, \"X\"" in bomb_src
    assert 'reason=f"{label}_bomb"' in bomb_src
    # Morph bombs are X. A is jump — must not be the bomb button.
    assert "hold(session, 3, \"A\"" not in bomb_src


def test_ws_basement_product_never_uses_l_as_left() -> None:
    from super_metroid.routes.kpdr.k6 import ws_basement as k6ws

    src = inspect.getsource(k6ws)
    assert 'hold(session, 1, "L"' not in src
    assert 'hold(session, 12, "L"' not in src
    assert '"LEFT"' not in inspect.getsource(k6ws._run_to_morph_seat)
    assert "Map station LEFT is dead" in k6ws.play_ws_basement_to_phantoon.__doc__


def test_play_ws_basement_to_phantoon_already_settled() -> None:
    session = _FakeSession(
        _state(
            room_id=0xCD13,
            samus_x=39,
            samus_y=139,
            pose=1,
            game_state=8,
            door_transition=0,
        )
    )
    out = wrecked_ship.play_ws_basement_to_phantoon(session)
    assert wrecked_ship.ws_basement_phantoon_settled(out)
    assert session.frame == 0


def test_play_ws_basement_to_phantoon_lands_morphs_supers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from super_metroid.routes.kpdr.k6 import ws_basement as k6ws

    session = _FakeSession(
        _state(
            room_id=0xCC6F,
            samus_x=657,
            samus_y=92,
            pose=24,
            game_state=8,
            door_transition=0,
            selected_item=2,
            velocity_y=0,
        )
    )
    seen: dict[str, Any] = {}

    def _require(sess: Any, room: int, label: str) -> None:
        seen["require"] = (room, label)
        assert room == 0xCC6F

    def _land(sess: Any, label: str) -> None:
        seen["land"] = label
        sess.state = replace(sess.state, pose=2, samus_y=91, velocity_y=0)

    def _run(sess: Any, label: str) -> None:
        seen["run"] = label
        sess.state = replace(sess.state, samus_x=938, samus_y=187, pose=9)

    def _bomb(sess: Any, label: str) -> None:
        seen["bomb"] = label
        sess.state = replace(sess.state, samus_x=1160, samus_y=201, pose=30)

    def _eye(sess: Any, label: str) -> None:
        seen["eye"] = label
        sess.state = replace(
            sess.state,
            room_id=0xCD13,
            game_state=11,
            door_transition=1,
            samus_x=40,
            samus_y=118,
        )

    def _settle(sess: Any, room: int, **kwargs: Any) -> Any:
        seen["settle"] = (room, kwargs.get("label"))
        sess.state = replace(
            sess.state, room_id=room, game_state=8, door_transition=0
        )
        return sess.state

    monkeypatch.setattr(k6ws, "require_room", _require)
    monkeypatch.setattr(k6ws, "_land", _land)
    monkeypatch.setattr(k6ws, "_run_to_morph_seat", _run)
    monkeypatch.setattr(k6ws, "_bomb_tunnel", _bomb)
    monkeypatch.setattr(k6ws, "_super_eye_door", _eye)
    monkeypatch.setattr(k6ws, "wait_ordinary_room", _settle)

    out = k6ws.play_ws_basement_to_phantoon(session)
    assert seen["require"][0] == 0xCC6F
    assert seen["land"] == "ws_basement_to_phantoon"
    assert seen["run"] == "ws_basement_to_phantoon"
    assert seen["bomb"] == "ws_basement_to_phantoon"
    assert seen["eye"] == "ws_basement_to_phantoon"
    assert seen["settle"] == (0xCD13, "ws_basement_to_phantoon")
    assert wrecked_ship.ws_basement_phantoon_settled(out)


def test_ws_basement_does_not_fight_phantoon() -> None:
    from super_metroid.routes.kpdr.k6 import ws_basement as k6ws

    src = inspect.getsource(k6ws)
    assert "phantoon_hp" not in src.lower()
    assert "play_phantoon" not in src
    assert "farm" not in src.lower()
    assert "Do not fight" in k6ws.play_ws_basement_to_phantoon.__doc__
