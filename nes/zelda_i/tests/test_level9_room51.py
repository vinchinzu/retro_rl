"""Unit tests for Level 9 room 0x51 north walk through statue diamond into 0x41."""

from __future__ import annotations

import json
from types import SimpleNamespace

from retro_harness.controls import pressed_nes_buttons
from zelda_i.level9.dungeon import LEVEL9
from zelda_i.level9.room51 import (
    ROOM51,
    in_room_51,
    room51_is_rom_predecessor_of_41,
    room51_loader_avoids_41,
    room51_rom_north_is_open,
    room51_to_41_step,
)
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot


def _mock_snap(
    *,
    level: int = LEVEL9,
    mode: int = PLAY_MODE,
    screen: int = ROOM51,
    link_x: int = 120,
    link_y: int = 205,
    transitioning: bool = False,
) -> ZeldaSnapshot:
    return SimpleNamespace(
        level=level,
        mode=mode,
        screen=screen,
        link_x=link_x,
        link_y=link_y,
        transitioning=transitioning,
    )  # type: ignore[return-value]


def test_room51_rom_properties() -> None:
    assert room51_rom_north_is_open() is True
    assert room51_is_rom_predecessor_of_41() is True
    assert room51_loader_avoids_41() is True


def test_in_room_51() -> None:
    assert in_room_51(_mock_snap(screen=ROOM51)) is True
    assert in_room_51(_mock_snap(screen=0x41)) is False
    assert in_room_51(_mock_snap(mode=4)) is False
    assert in_room_51(_mock_snap(level=1)) is False


def test_room51_to_41_step_waypoints() -> None:
    """South mouth -> west aisle -> centre -> east aisle -> north door."""

    def heading(**kw) -> list[str]:
        act = room51_to_41_step(_mock_snap(**kw))
        return [b for b in pressed_nes_buttons(list(act.action)) if b != "A"]

    assert heading(link_x=120, link_y=205) == ["UP"]  # approach the south aisle
    assert heading(link_x=120, link_y=185) == ["LEFT"]  # into the west aisle
    assert heading(link_x=96, link_y=160) == ["UP"]  # climb the west aisle
    assert heading(link_x=96, link_y=141) == ["RIGHT"]  # cross the centre aisle
    assert heading(link_x=128, link_y=120) == ["UP"]  # climb the east aisle
    assert heading(link_x=128, link_y=93) == ["LEFT"]  # align on the north door
    assert heading(link_x=120, link_y=93) == ["UP"]  # push through
    assert heading(transitioning=True) == ["UP"]  # hold through the scroll
    assert heading(screen=0x41) == []  # arrived


def test_room51_dump_evidence() -> None:
    dump_path = RECORDINGS_DIR / "l9_room51_dump.json"
    assert dump_path.exists(), f"Missing {dump_path}"
    with open(dump_path) as f:
        data = json.load(f)
    assert data.get("bead") == "rr-yxy6"
    assert data.get("ok") is True
    assert data.get("lands_0x41") is True
    assert data.get("dest_screen") == 0x41
    assert data.get("dest_uncleared_0x41") is True
