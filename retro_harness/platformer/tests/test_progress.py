"""Tests for progress trackers (no ROM needed)."""

import pytest
from retro_harness.platformer.progress import (
    MonotonicAxisTracker,
    CompositeAxisTracker,
    HighWaterWithBacktrack,
    WaypointTracker,
    make_progress_tracker,
)
from retro_harness.platformer.level_config import LevelConfig, PlatformerRAM


def test_monotonic_axis_basic():
    t = MonotonicAxisTracker(axis="camera_x", direction=1)
    t.reset()
    # First frame sets initial
    p = t.update({"camera_x": 100})
    assert p == 0.0
    # Move forward
    p = t.update({"camera_x": 200})
    assert p == 100.0
    # Move backward - progress stays at high-water
    p = t.update({"camera_x": 150})
    assert p == 100.0
    # Move further
    p = t.update({"camera_x": 300})
    assert p == 200.0


def test_monotonic_axis_reverse():
    t = MonotonicAxisTracker(axis="camera_y", direction=-1)
    t.reset()
    t.update({"camera_y": 500})
    p = t.update({"camera_y": 400})
    assert p == 100.0  # (500-400) * -(-1) = 100


def test_composite_tracker():
    t = CompositeAxisTracker(x_weight=1.0, y_weight=0.5, x_direction=1, y_direction=-1)
    t.reset()
    t.update({"camera_x": 0, "camera_y": 100})
    p = t.update({"camera_x": 100, "camera_y": 50})
    # dx = 100*1*1 = 100, dy = (50-100)*(-1)*0.5 = 25
    assert p == 125.0


def test_high_water_with_backtrack():
    t = HighWaterWithBacktrack(axis="camera_x", direction=1, backtrack_tolerance=50.0)
    t.reset()
    t.update({"camera_x": 0})
    t.update({"camera_x": 200})
    assert t.max_progress == 200.0

    # Backtrack within tolerance
    t.update({"camera_x": 170})
    assert t.max_progress == 200.0
    assert not t.is_stalled  # 30 < 50 tolerance

    # Backtrack beyond tolerance
    t.update({"camera_x": 100})
    assert t.max_progress == 200.0
    assert t.is_stalled  # 100 > 50 tolerance


def test_missing_axis_frame_is_ignored():
    t = HighWaterWithBacktrack(axis="camera_x", direction=1, backtrack_tolerance=50.0)
    t.reset()
    t.update({"camera_x": 1000})
    t.update({"camera_x": 1200})
    # A frame without the tracked axis must not read as regression from 0
    t.update({"camera_y": 5})
    assert t.max_progress == 200.0
    assert not t.is_stalled


def test_missing_axis_first_frame_does_not_anchor():
    t = MonotonicAxisTracker(axis="camera_x", direction=1)
    t.reset()
    # First frame lacks the axis: baseline must anchor on the first real value
    t.update({"camera_y": 42})
    p = t.update({"camera_x": 100})
    assert p == 0.0
    assert t.update({"camera_x": 200}) == 100.0


def test_waypoint_tracker():
    waypoints = [(0, 0), (100, 0), (100, 100), (200, 100)]
    t = WaypointTracker(waypoints=waypoints, capture_radius=20.0)
    t.reset()

    # Start near first waypoint
    p = t.update({"player_x": 5, "player_y": 5})
    assert p >= 0.0

    # Reach second waypoint
    p = t.update({"player_x": 100, "player_y": 5})
    assert t._furthest_wp >= 1
    assert p >= 1.0

    # Reach third
    p = t.update({"player_x": 100, "player_y": 100})
    assert t._furthest_wp >= 2
    assert p >= 2.0


def test_waypoint_needs_at_least_two():
    with pytest.raises(ValueError):
        WaypointTracker(waypoints=[(0, 0)])


def test_factory_monotonic():
    config = LevelConfig(
        level_id="test", display_name="Test", game_name="Test-Snes",
        game_dir_name="test", start_state="test",
        ram=PlatformerRAM(), target_level_id=1,
        progress_axis="camera_x", progress_direction=1,
    )
    tracker = make_progress_tracker(config)
    assert isinstance(tracker, MonotonicAxisTracker)


def test_factory_backtrack():
    config = LevelConfig(
        level_id="test", display_name="Test", game_name="Test-Snes",
        game_dir_name="test", start_state="test",
        ram=PlatformerRAM(), target_level_id=1,
        progress_axis="camera_x", backtrack_tolerance=100.0,
    )
    tracker = make_progress_tracker(config)
    assert isinstance(tracker, HighWaterWithBacktrack)


def test_factory_waypoints():
    config = LevelConfig(
        level_id="test", display_name="Test", game_name="Test-Snes",
        game_dir_name="test", start_state="test",
        ram=PlatformerRAM(), target_level_id=1,
        progress_axis="waypoints",
        waypoints=[(0, 0), (100, 0)],
    )
    tracker = make_progress_tracker(config)
    assert isinstance(tracker, WaypointTracker)


def test_factory_player_x():
    config = LevelConfig(
        level_id="test", display_name="Test", game_name="Test-Snes",
        game_dir_name="test", start_state="test",
        ram=PlatformerRAM(), target_level_id=1,
        progress_axis="player_x", progress_direction=-1,
    )
    tracker = make_progress_tracker(config)
    assert isinstance(tracker, MonotonicAxisTracker)
    tracker.reset()
    tracker.update({"player_x": 500})
    p = tracker.update({"player_x": 400})
    assert p == 100.0  # (500-400) * 1 (direction applied as -1 -> positive progress)


def test_factory_player_y():
    config = LevelConfig(
        level_id="test", display_name="Test", game_name="Test-Snes",
        game_dir_name="test", start_state="test",
        ram=PlatformerRAM(), target_level_id=1,
        progress_axis="player_y", progress_direction=-1,
    )
    tracker = make_progress_tracker(config)
    assert isinstance(tracker, MonotonicAxisTracker)
    tracker.reset()
    tracker.update({"player_y": 1000})
    p = tracker.update({"player_y": 800})
    assert p == 200.0


def test_factory_player_x_backtrack():
    config = LevelConfig(
        level_id="test", display_name="Test", game_name="Test-Snes",
        game_dir_name="test", start_state="test",
        ram=PlatformerRAM(), target_level_id=1,
        progress_axis="player_x", backtrack_tolerance=50.0,
    )
    tracker = make_progress_tracker(config)
    assert isinstance(tracker, HighWaterWithBacktrack)
