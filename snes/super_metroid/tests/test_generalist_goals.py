"""Goal parse + Join (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from super_metroid.generalist.goals import (
    JOIN_XY_BAND,
    Goal,
    goal_from_session,
    is_join,
    parse_goal,
)
from super_metroid.practice_repertoire.catalog import PRODUCT_CATEGORY


def test_parse_any_and_session() -> None:
    any_goal = parse_goal("any")
    assert any_goal.any_door is True
    assert any_goal.resolved is False
    parsed = parse_goal("session:kpdr25/crateria/morph")
    assert parsed.session_id == "kpdr25/crateria/construction_zone"
    assert parsed.room_id == 0x9E9F
    assert parsed.start_room_id == 0x9E9F


def test_parse_bare_session_and_bad() -> None:
    goal = parse_goal("kpdr25/crateria/ship")
    assert goal.session_id.startswith(PRODUCT_CATEGORY)
    with pytest.raises(ValueError, match="unrecognized"):
        parse_goal("nope")
    with pytest.raises(ValueError, match="door"):
        parse_goal("door:left")


def test_join_is_glance_not_room_id() -> None:
    goal = Goal("next", 0x92FD, 437, 923, pose=2)
    miss = SimpleNamespace(
        room_id=0x92FD, samus_x=437, samus_y=923, pose=0x1D, game_state=8,
        door_transition=0, health=99,
    )
    # morph-in-door is not Join when the pin is a stand pose.
    assert is_join(miss, goal) is False
    hit = SimpleNamespace(
        room_id=0x92FD, samus_x=437, samus_y=923, pose=2, game_state=8,
        door_transition=0, health=99,
    )
    assert is_join(hit, goal) is True
    far = SimpleNamespace(
        room_id=0x92FD,
        samus_x=437 + JOIN_XY_BAND + 1,
        samus_y=923,
        pose=2,
        game_state=8,
        door_transition=0,
        health=99,
    )
    assert is_join(far, goal) is False


def test_goal_from_ship_is_parlor_on_landing() -> None:
    goal = goal_from_session("kpdr25/crateria/ship")
    assert goal.session_id == "kpdr25/crateria/parlor"
    assert goal.room_id == 0x91F8
    assert goal.x == 121
