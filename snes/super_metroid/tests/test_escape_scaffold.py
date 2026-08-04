"""Unit tests for the development-only Tourian escape scaffold."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from super_metroid.combat.escape import (
    DEFAULT_ESCAPE_TIMEOUTS,
    ESCAPE_CHAIN_ROOM_IDS,
    ROOM_ESCAPE_1,
    ROOM_ESCAPE_2,
    ROOM_ESCAPE_3,
    ROOM_ESCAPE_4,
    ROOM_LANDING_SITE,
    play_escape_chain_scaffold,
    play_escape_room_1,
)


def test_escape_constants_match_development_room_chain() -> None:
    assert ESCAPE_CHAIN_ROOM_IDS == (
        ROOM_ESCAPE_1,
        ROOM_ESCAPE_2,
        ROOM_ESCAPE_3,
        ROOM_ESCAPE_4,
        ROOM_LANDING_SITE,
    )
    assert all(timeout > 0 for timeout in DEFAULT_ESCAPE_TIMEOUTS.values())


def test_chain_scaffold_reports_each_room_and_timeout() -> None:
    session = SimpleNamespace(state=SimpleNamespace(room_id=ROOM_ESCAPE_1))
    evidence = play_escape_chain_scaffold(session, room_timeouts={ROOM_ESCAPE_1: 99})

    assert evidence["outcome"] == "scaffold_only"
    assert evidence["success"] is False
    rooms = evidence["rooms"]
    assert isinstance(rooms, list)
    assert [room["roomId"] for room in rooms] == list(ESCAPE_CHAIN_ROOM_IDS)
    assert rooms[0]["timeoutFrames"] == 99
    assert all(room["implemented"] is False for room in rooms)


def test_room_stub_is_callable_and_bounded() -> None:
    evidence = play_escape_room_1(SimpleNamespace())

    assert evidence["roomId"] == ROOM_ESCAPE_1
    assert evidence["timeoutFrames"] == DEFAULT_ESCAPE_TIMEOUTS[ROOM_ESCAPE_1]
    assert evidence["status"] == "stub"


def test_scaffold_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="positive"):
        play_escape_chain_scaffold(SimpleNamespace(), room_timeouts={ROOM_ESCAPE_1: 0})
