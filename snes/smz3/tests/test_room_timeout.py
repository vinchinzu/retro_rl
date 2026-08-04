"""Room timeout watchdog: 3× standard room time → game over."""

from __future__ import annotations

from smz3.room_timeout import (
    DEFAULT_BASELINE_FRAMES,
    RoomTimeoutWatchdog,
    TimeoutReason,
)


def test_no_timeout_within_budget() -> None:
    wd = RoomTimeoutWatchdog.from_mapping({"landing": 100}, multiplier=3.0)
    for frame in range(0, 300):  # dwell 0..299, limit=300
        event = wd.observe(frame=frame, room_key="landing")
        assert event is None
    assert not wd.is_game_over


def test_timeout_at_three_x() -> None:
    wd = RoomTimeoutWatchdog.from_mapping({"morph": 10}, multiplier=3.0)
    # entry at frame 0; dwell exceeds 30 at frame 31
    assert wd.observe(frame=0, room_key="morph") is None
    for frame in range(1, 31):
        assert wd.observe(frame=frame, room_key="morph") is None
    event = wd.observe(frame=31, room_key="morph")
    assert event is not None
    assert event.reason is TimeoutReason.ROOM_DWELL
    assert event.dwell_frames == 31
    assert event.limit_frames == 30
    assert event.standard_frames == 10
    assert wd.is_game_over
    # sticky: further observes do not re-fire
    assert wd.observe(frame=100, room_key="morph") is None
    assert len(wd.events) == 1


def test_room_change_resets_dwell() -> None:
    wd = RoomTimeoutWatchdog.from_mapping({"a": 5, "b": 5}, multiplier=3.0)
    for frame in range(0, 14):
        assert wd.observe(frame=frame, room_key="a") is None
    # switch rooms before timeout
    assert wd.observe(frame=14, room_key="b") is None
    for frame in range(15, 29):
        assert wd.observe(frame=frame, room_key="b") is None
    assert not wd.is_game_over
    event = wd.observe(frame=30, room_key="b")  # dwell from 14 → 16 > 15
    assert event is not None


def test_unsettled_does_not_accumulate() -> None:
    wd = RoomTimeoutWatchdog.from_mapping({"r": 1}, multiplier=3.0)
    assert wd.observe(frame=0, room_key="r") is None
    for frame in range(1, 100):
        assert wd.observe(frame=frame, room_key="r", settled=False) is None
    assert not wd.is_game_over


def test_default_baseline_fallback() -> None:
    wd = RoomTimeoutWatchdog(multiplier=3.0)
    bl = wd.baseline_for("unknown_room")
    assert bl.standard_frames == DEFAULT_BASELINE_FRAMES
    assert bl.source == "default_fallback"
    report = wd.report()
    assert report["multiplier"] == 3.0
    assert report["game_over"] is None
