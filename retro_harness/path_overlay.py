"""World-space guide path overlay for human recording / live play.

Games supply room-local pixel waypoints; this module projects them through a
camera scroll + letterboxed scale and draws a polyline + dots on a pygame
surface. Used so a human recorder can see where the intended route goes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class GuidePoint:
    """One world-pixel waypoint (game room coordinates)."""

    x: int
    y: int
    label: str = ""


@dataclass(frozen=True)
class RoomGuide:
    """Guide polyline for a single room id."""

    room_id: int
    points: tuple[GuidePoint, ...]
    color: tuple[int, int, int] = (80, 255, 120)
    name: str = ""


@dataclass(frozen=True)
class OverlayTransform:
    """Map world pixels → window pixels (camera scroll + scaled letterbox)."""

    camera_x: int
    camera_y: int
    scale: float
    x_off: int
    y_off: int
    game_w: int = 256
    game_h: int = 224

    def world_to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        sx = (wx - self.camera_x) * self.scale + self.x_off
        sy = (wy - self.camera_y) * self.scale + self.y_off
        return sx, sy

    def on_screen(self, sx: float, sy: float, *, margin: float = 8.0) -> bool:
        return (
            self.x_off - margin <= sx <= self.x_off + self.game_w * self.scale + margin
            and self.y_off - margin <= sy <= self.y_off + self.game_h * self.scale + margin
        )


def transform_from_session_ctx(
    ctx: dict[str, Any],
    *,
    camera_x: int = 0,
    camera_y: int = 0,
) -> OverlayTransform:
    """Build a transform from a PlaySession ``on_overlay`` context dict."""
    return OverlayTransform(
        camera_x=int(camera_x),
        camera_y=int(camera_y),
        scale=float(ctx.get("scale", 1.0)),
        x_off=int(ctx.get("x_off", 0)),
        y_off=int(ctx.get("y_off", 0)),
        game_w=int(ctx.get("game_w", 256)),
        game_h=int(ctx.get("game_h", 224)),
    )


def project_points(
    points: Sequence[GuidePoint | tuple[int, int] | Sequence[int]],
    transform: OverlayTransform,
) -> list[tuple[float, float]]:
    """Project world waypoints to screen coordinates."""
    out: list[tuple[float, float]] = []
    for p in points:
        if isinstance(p, GuidePoint):
            wx, wy = p.x, p.y
        else:
            wx, wy = int(p[0]), int(p[1])
        out.append(transform.world_to_screen(wx, wy))
    return out


def nearest_waypoint_index(
    points: Sequence[GuidePoint | tuple[int, int] | Sequence[int]],
    wx: int,
    wy: int,
) -> int | None:
    """Index of the closest guide point to a world position (or None if empty)."""
    if not points:
        return None
    best_i = 0
    best_d = 10**18
    for i, p in enumerate(points):
        if isinstance(p, GuidePoint):
            px, py = p.x, p.y
        else:
            px, py = int(p[0]), int(p[1])
        d = (px - wx) * (px - wx) + (py - wy) * (py - wy)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def draw_guide_path(
    pg: Any,
    surface: Any,
    points: Sequence[GuidePoint | tuple[int, int] | Sequence[int]],
    transform: OverlayTransform,
    *,
    color: tuple[int, int, int] = (80, 255, 120),
    width: int = 2,
    radius: int = 5,
    highlight_index: int | None = None,
    highlight_color: tuple[int, int, int] = (255, 220, 40),
    dim_passed: bool = True,
    font: Any | None = None,
    draw_labels: bool = True,
) -> None:
    """Draw a world-space guide polyline onto ``surface``.

    Points behind the nearest waypoint (when ``highlight_index`` is set and
    ``dim_passed`` is true) are drawn darker so the remaining route stands out.
    """
    if len(points) < 1:
        return

    screen_pts = project_points(points, transform)
    hi = highlight_index if highlight_index is not None else 0

    # Segments: passed (dim) then remaining (bright).
    if dim_passed and hi > 0:
        passed = [screen_pts[i] for i in range(0, min(hi + 1, len(screen_pts)))]
        if len(passed) >= 2:
            dim = tuple(max(0, c // 3) for c in color)
            pg.draw.lines(surface, dim, False, passed, max(1, width - 1))
    remain_start = max(0, hi)
    remain = [screen_pts[i] for i in range(remain_start, len(screen_pts))]
    if len(remain) >= 2:
        pg.draw.lines(surface, color, False, remain, width)

    for i, (sp, wp) in enumerate(zip(screen_pts, points)):
        if not transform.on_screen(*sp):
            continue
        is_hi = i == hi
        is_goal = i == len(screen_pts) - 1
        col = highlight_color if is_hi else ((255, 90, 90) if is_goal else color)
        r = radius + (2 if is_hi or is_goal else 0)
        pg.draw.circle(surface, col, (int(sp[0]), int(sp[1])), r, 0 if is_hi else 2)
        if draw_labels and font is not None:
            label = wp.label if isinstance(wp, GuidePoint) else ""
            if not label and (is_hi or is_goal or i == 0):
                label = f"{i}"
            if label:
                text = font.render(str(label), True, col)
                surface.blit(text, (int(sp[0]) + 6, int(sp[1]) - 10))


def draw_player_marker(
    pg: Any,
    surface: Any,
    wx: int,
    wy: int,
    transform: OverlayTransform,
    *,
    color: tuple[int, int, int] = (255, 80, 80),
    radius: int = 6,
) -> None:
    """Dot at the player's current world position."""
    sx, sy = transform.world_to_screen(wx, wy)
    if not transform.on_screen(sx, sy):
        return
    pg.draw.circle(surface, color, (int(sx), int(sy)), radius, 2)
    pg.draw.line(surface, color, (int(sx) - radius - 2, int(sy)), (int(sx) + radius + 2, int(sy)), 1)
    pg.draw.line(surface, color, (int(sx), int(sy) - radius - 2), (int(sx), int(sy) + radius + 2), 1)
