"""Wrecked Ship action traps (no emulator): blue door, morph bombs, hop ≠ fight."""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any

import numpy as np

from super_metroid.ram import parse_state
from super_metroid.routes.kpdr.k6 import ws_basement as k6ws
from super_metroid.routes.kpdr.k6.ws_basement import (
    at_ws_basement_bomb_blocks,
    at_ws_basement_eye_seat,
)
from super_metroid.routes.kpdr.k6.ws_entrance import (
    WEAPON_BEAM,
    WS_ENTRANCE_DOOR_X_MIN,
    at_ws_entrance_door_seat,
    ws_entrance_main_settled,
    ws_entrance_to_main_action,
)
from super_metroid.routes.kpdr.k6.ws_main import (
    WEAPON_SUPER,
    WS_MAIN_BOTTOM_Y,
    at_ws_main_green_floor,
    ws_main_to_basement_action,
)


def _state(**kwargs: Any) -> Any:
    base = parse_state(np.zeros(0x10000, dtype=np.uint8))
    return replace(base, **kwargs)


def test_ws_entrance_door_seat() -> None:
    pin = _state(room_id=0xCA08, samus_x=57, samus_y=139, pose=1, game_state=8)
    assert not at_ws_entrance_door_seat(pin)
    seat = _state(room_id=0xCA08, samus_x=920, samus_y=139, pose=9, game_state=8)
    assert at_ws_entrance_door_seat(seat)
    wrong_room = _state(room_id=0xCAF6, samus_x=920)
    assert not at_ws_entrance_door_seat(wrong_room)


def test_ws_entrance_action_never_supers_blue_door() -> None:
    supers = _state(room_id=0xCA08, samus_x=57, selected_item=2, game_state=8)
    assert ws_entrance_to_main_action(supers) == ("SELECT",)
    assert "X" not in ws_entrance_to_main_action(supers)

    dash = _state(room_id=0xCA08, samus_x=200, selected_item=WEAPON_BEAM)
    assert "X" not in ws_entrance_to_main_action(dash)

    door = _state(
        room_id=0xCA08,
        samus_x=WS_ENTRANCE_DOOR_X_MIN,
        selected_item=WEAPON_BEAM,
    )
    shot = ws_entrance_to_main_action(door)
    assert "X" in shot
    assert "SELECT" not in shot


def test_ws_entrance_main_settled_requires_gs8() -> None:
    trans = _state(
        room_id=0xCAF6, game_state=11, door_transition=1, samus_x=40, samus_y=139
    )
    assert not ws_entrance_main_settled(trans)
    gs11 = replace(trans, game_state=11, door_transition=0)
    assert not ws_entrance_main_settled(gs11)
    ordinary = replace(trans, game_state=8, door_transition=0)
    assert ws_entrance_main_settled(ordinary)


def test_ws_main_green_floor_seat() -> None:
    pin = _state(room_id=0xCAF6, samus_x=1063, samus_y=907, pose=9, game_state=8)
    assert not at_ws_main_green_floor(pin)
    hatch = _state(
        room_id=0xCAF6,
        samus_x=1143,
        samus_y=WS_MAIN_BOTTOM_Y,
        pose=1,
        game_state=8,
    )
    assert at_ws_main_green_floor(hatch)


def test_ws_main_action_never_goes_up() -> None:
    pin = _state(room_id=0xCAF6, samus_x=1077, samus_y=907, pose=9, game_state=8)
    assert "UP" not in ws_main_to_basement_action(pin)
    attic_y = _state(room_id=0xCAF6, samus_x=1150, samus_y=800, pose=1)
    assert "UP" not in ws_main_to_basement_action(attic_y)


def test_ws_main_action_supers_green_floor_door() -> None:
    beam = _state(
        room_id=0xCAF6,
        samus_x=1143,
        samus_y=WS_MAIN_BOTTOM_Y + 20,
        selected_item=0,
        game_state=8,
    )
    assert "X" not in ws_main_to_basement_action(beam)

    supers = _state(
        room_id=0xCAF6,
        samus_x=1143,
        samus_y=WS_MAIN_BOTTOM_Y + 20,
        selected_item=WEAPON_SUPER,
        game_state=8,
    )
    shot = ws_main_to_basement_action(supers)
    assert "X" in shot
    assert "L" in shot
    assert "DOWN" not in shot  # DOWN+X morphs; L is shoulder aim-down


def test_ws_basement_bomb_and_eye_seats() -> None:
    pin = _state(room_id=0xCC6F, samus_x=657, samus_y=92, pose=24, game_state=8)
    assert not at_ws_basement_bomb_blocks(pin)
    bomb = _state(room_id=0xCC6F, samus_x=1051, samus_y=201, pose=30, game_state=8)
    assert at_ws_basement_bomb_blocks(bomb)
    eye = _state(room_id=0xCC6F, samus_x=1180, samus_y=201, pose=30, game_state=8)
    assert at_ws_basement_eye_seat(eye)


def test_ws_basement_product_morph_bombs_are_x_not_a() -> None:
    bomb_src = inspect.getsource(k6ws._bomb_tunnel)
    assert 'hold(session, 3, "X"' in bomb_src
    assert 'hold(session, 3, "A"' not in bomb_src


def test_ws_basement_does_not_fight_phantoon() -> None:
    src = inspect.getsource(k6ws)
    assert "play_phantoon" not in src
    assert "Do not fight" in k6ws.play_ws_basement_to_phantoon.__doc__
