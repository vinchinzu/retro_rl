"""Tests for path guide overlay projection helpers."""

from __future__ import annotations

from retro_harness.path_overlay import (
    GuidePoint,
    OverlayTransform,
    nearest_waypoint_index,
    project_points,
    transform_from_session_ctx,
)


def test_world_to_screen_with_camera() -> None:
    t = OverlayTransform(camera_x=100, camera_y=50, scale=2.0, x_off=10, y_off=20)
    sx, sy = t.world_to_screen(100, 50)
    assert sx == 10.0
    assert sy == 20.0
    sx2, sy2 = t.world_to_screen(132, 82)
    assert sx2 == 10.0 + 32 * 2
    assert sy2 == 20.0 + 32 * 2


def test_project_and_nearest() -> None:
    pts = (GuidePoint(0, 0, "a"), GuidePoint(100, 0, "b"), GuidePoint(100, 100, "c"))
    t = OverlayTransform(camera_x=0, camera_y=0, scale=1.0, x_off=0, y_off=0)
    screen = project_points(pts, t)
    assert screen[1] == (100.0, 0.0)
    assert nearest_waypoint_index(pts, 90, 5) == 1
    assert nearest_waypoint_index(pts, 100, 90) == 2


def test_transform_from_session_ctx() -> None:
    ctx = {"scale": 3.0, "x_off": 5, "y_off": 7, "game_w": 256, "game_h": 224}
    t = transform_from_session_ctx(ctx, camera_x=40, camera_y=10)
    assert t.scale == 3.0
    assert t.camera_x == 40
    assert t.game_w == 256


def test_projection_fractional_scale_and_negative_camera() -> None:
    t = OverlayTransform(camera_x=-20, camera_y=300, scale=0.5, x_off=-4, y_off=8)
    sx, sy = t.world_to_screen(10, 268)
    assert sx == (-4.0 + 30 * 0.5)
    assert sy == (8.0 + (268 - 300) * 0.5)


def test_project_points_empty_and_mixed() -> None:
    t = OverlayTransform(camera_x=0, camera_y=0, scale=2.0, x_off=0, y_off=0)
    assert project_points([], t) == []
    mixed: list[GuidePoint | tuple[int, int]] = [GuidePoint(3, 4, "a"), (5, 6), [7, 8]]
    screen = project_points(mixed, t)
    assert screen == [(6.0, 8.0), (10.0, 12.0), (14.0, 16.0)]


def test_nearest_empty_and_mixed_and_tie() -> None:
    assert nearest_waypoint_index([], 0, 0) is None
    mixed: list[GuidePoint | tuple[int, int]] = [(0, 0), GuidePoint(50, 0, "mid"), [100, 100]]
    assert nearest_waypoint_index(mixed, 49, 3) == 1
    assert nearest_waypoint_index([(0, 0), (0, 0), (100, 0)], 1, 1) == 0


def test_on_screen_bounds_with_margin() -> None:
    t = OverlayTransform(camera_x=0, camera_y=0, scale=2.0, x_off=10, y_off=20, game_w=256, game_h=224)
    assert t.on_screen(10.0, 20.0)
    assert t.on_screen(10.0 + 256 * 2.0, 20.0 + 224 * 2.0)
    assert not t.on_screen(1.0, 20.0)
    assert not t.on_screen(10.0, 20.0 + 224 * 2.0 + 20.0)


def test_malformed_waypoint_raises() -> None:
    t = OverlayTransform(camera_x=0, camera_y=0, scale=1.0, x_off=0, y_off=0)
    try:
        project_points([(1,)], t)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for malformed waypoint")
