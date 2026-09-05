"""L7 bomb walls pause-select bombs from the predecessor B-slot leftover."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
from zelda_i.dungeon.bomb_wall import BombWallController, BombWallPhase
from zelda_i.dungeon.pause_select import B_SLOT_BAIT, B_SLOT_BOMBS, B_SLOT_RECORDER
from zelda_i.level7.hops import (
    make_room0c_east_bomb_controller,
    make_room18_north_bomb_controller,
    make_room69_west_bomb_controller,
)
from zelda_i.level2.puzzles import BOMB_WALL_6F_NORTH
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_LEVEL,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_MODE,
    ADDR_SCREEN,
    ADDR_SELECTED_ITEM,
    PLAY_MODE,
    read_snapshot,
)


def _ram(
    *,
    room: int,
    x: int,
    y: int,
    selected: int,
    bombs: int = 8,
    level: int = 7,
) -> np.ndarray:
    ram = np.zeros(0x800, dtype=np.uint8)
    ram[ADDR_MODE] = PLAY_MODE
    ram[ADDR_LEVEL] = level
    ram[ADDR_SCREEN] = room
    ram[ADDR_LINK_X] = x
    ram[ADDR_LINK_Y] = y
    ram[ADDR_BOMBS] = bombs
    ram[ADDR_SELECTED_ITEM] = selected
    return ram


def _buttons(action) -> list[str]:
    return sorted(
        name
        for name, idx in NES_BUTTON_NAME_TO_INDEX.items()
        if idx is not None and int(action.action[idx])
    )


def _drive_until_place_or_start(ctl: BombWallController, ram: np.ndarray) -> list[str]:
    ctl.bind_env(SimpleNamespace(get_ram=lambda: ram))
    reasons: list[str] = []
    for _ in range(80):
        before = ram.copy()
        act = ctl.step(read_snapshot(ram))
        assert np.array_equal(ram, before), "bomb wall must not write RAM"
        reasons.append(act.reason)
        if act.reason in {"place_bomb", "pause_open"}:
            break
    return reasons


def test_pond_leftover_slot_5_selects_bombs_before_place() -> None:
    ctl = make_room69_west_bomb_controller()
    assert ctl.select_item == B_SLOT_BOMBS
    sx, sy = ctl.stand
    ram = _ram(room=ctl.from_room, x=sx, y=sy, selected=B_SLOT_RECORDER)
    reasons = _drive_until_place_or_start(ctl, ram)
    assert "pause_open" in reasons
    assert "place_bomb" not in reasons
    assert _buttons(ctl.step(read_snapshot(ram))) != ["B"]


def test_hungry_leftover_slot_6_selects_bombs_before_place() -> None:
    ctl = make_room18_north_bomb_controller()
    sx, sy = ctl.stand
    ram = _ram(room=ctl.from_room, x=sx, y=sy, selected=B_SLOT_BAIT)
    reasons = _drive_until_place_or_start(ctl, ram)
    assert "pause_open" in reasons
    assert "place_bomb" not in reasons


def test_digdogger_leftover_slot_5_selects_bombs_on_0x0c() -> None:
    ctl = make_room0c_east_bomb_controller()
    sx, sy = ctl.stand
    ram = _ram(room=ctl.from_room, x=sx, y=sy, selected=B_SLOT_RECORDER)
    reasons = _drive_until_place_or_start(ctl, ram)
    assert "pause_open" in reasons
    assert "place_bomb" not in reasons


def test_l2_omitted_select_item_still_places_without_menu() -> None:
    ctl = BombWallController(wall=BOMB_WALL_6F_NORTH, level=2, face_frames=1)
    assert ctl.select_item is None
    ctl.phase = BombWallPhase.FACE
    sx, sy = ctl.stand
    ram = _ram(
        room=ctl.from_room,
        x=sx,
        y=sy,
        selected=B_SLOT_RECORDER,
        bombs=4,
        level=2,
    )
    reasons = []
    for _ in range(4):
        act = ctl.step(read_snapshot(ram))
        reasons.append(act.reason)
        if act.reason == "place_bomb":
            break
    assert "place_bomb" in reasons
    assert "pause_open" not in reasons
