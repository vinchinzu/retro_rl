"""Unit tests for Level 9 room 0x51 north walk through statue diamond into 0x41."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
    # 1. South mouth approach
    step = room51_to_41_step(_mock_snap(link_x=120, link_y=205))
    assert step.reason == "room51_approach_south_aisle"

    # 2. West aisle navigation
    step = room51_to_41_step(_mock_snap(link_x=120, link_y=185))
    assert step.reason == "room51_nav_west_aisle"

    # 3. Climb west aisle
    step = room51_to_41_step(_mock_snap(link_x=96, link_y=160))
    assert step.reason == "room51_climb_west_aisle"

    # 4. Cross center aisle
    step = room51_to_41_step(_mock_snap(link_x=96, link_y=141))
    assert step.reason == "room51_cross_center_aisle"

    # 5. Climb east aisle
    step = room51_to_41_step(_mock_snap(link_x=128, link_y=120))
    assert step.reason == "room51_climb_east_aisle"

    # 6. Align north door
    step = room51_to_41_step(_mock_snap(link_x=128, link_y=93))
    assert step.reason == "room51_align_north_door"

    # 7. Push north through door
    step = room51_to_41_step(_mock_snap(link_x=120, link_y=93))
    assert step.reason == "room51_push_north"

    # 8. Scroll transition
    step = room51_to_41_step(_mock_snap(transitioning=True))
    assert step.reason == "room41_scroll"

    # 9. Arrived in 0x41
    step = room51_to_41_step(_mock_snap(screen=0x41))
    assert step.reason == "room41_arrived"


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
