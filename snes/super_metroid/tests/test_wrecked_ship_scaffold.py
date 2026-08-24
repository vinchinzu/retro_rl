"""Unit locks for Wrecked Ship: Entrance→Main product + remaining scaffolds."""

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
    assert "_scaffold_exit" in basement
    assert "_scaffold_exit" in phant


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
