"""Pure/scripted pixel-lane helpers for barn cow care and feed.

These builders take explicit pixel positions (no task object) and return
``make_action`` arrays or ``None`` when the player is already on target.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from harvest.tasks.cow_geometry import (
    COW_EXIT_PREP_PX,
    COW_RIGHT_AISLE_X,
    COW_UPPER_RIGHT_ROUTE_MAX_Y,
    FODDER_STAND,
    LEFT_BARN_SHIP_LANE_X,
    LEFT_BARN_SHIP_LOWER_Y,
    LEFT_COW_LOWER_LANE_X,
    LEFT_TROUGH_LANE_Y,
    LEFT_TROUGH_RETURN_X,
    MILK_SHIP_PIXEL_ROUTE,
    UPPER_BARN_RIGHT_AISLE_X,
    UPPER_BARN_SHIP_AISLE_X,
    UPPER_BARN_SHIP_CROSS_Y,
    UPPER_BARN_SHIP_ESCAPE_Y,
    UPPER_BARN_SHIP_LOWER_LANE_Y,
    CowFeedSpot,
    left_cow_lane_x,
)
from harvest.tasks.nav import make_action


Pixel = Tuple[int, int]


def run_to_pixel_axis(
    current: Pixel,
    target: Pixel,
    *,
    tolerance: int = 2,
    x_first: bool = False,
    y_first: bool = False,
) -> Optional[np.ndarray]:
    """Run toward ``target`` on one axis (optionally forced x/y first)."""
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    if abs(dx) <= tolerance and abs(dy) <= tolerance:
        return None
    if x_first and abs(dx) > tolerance:
        return make_action(right=dx > 0, left=dx < 0, b=True)
    if y_first and abs(dy) > tolerance:
        return make_action(down=dy > 0, up=dy < 0, b=True)
    if abs(dx) >= abs(dy) and abs(dx) > tolerance:
        return make_action(right=dx > 0, left=dx < 0, b=True)
    return make_action(down=dy > 0, up=dy < 0, b=True)


def left_lower_lane_from_right_action(x: int, y: int) -> Optional[np.ndarray]:
    """From the right/upper barn, reach the left lower shipping corridor."""
    if x <= LEFT_COW_LOWER_LANE_X + 2 and y >= LEFT_BARN_SHIP_LOWER_Y - 2:
        return None
    if y < UPPER_BARN_SHIP_ESCAPE_Y - 2:
        if x >= 120 and x < COW_RIGHT_AISLE_X - 2:
            return make_action(right=True, b=True)
        return make_action(down=True, b=True)
    if y <= UPPER_BARN_SHIP_ESCAPE_Y + 2 and x < UPPER_BARN_RIGHT_AISLE_X - 2:
        return make_action(right=True, b=True)
    if x >= UPPER_BARN_RIGHT_AISLE_X - 3 and y < UPPER_BARN_SHIP_CROSS_Y - 2:
        return make_action(down=True, b=True)
    if abs(y - UPPER_BARN_SHIP_CROSS_Y) <= 2 and x > UPPER_BARN_SHIP_AISLE_X:
        return make_action(left=True, b=True)
    if abs(y - UPPER_BARN_SHIP_LOWER_LANE_Y) <= 2 and x > MILK_SHIP_PIXEL_ROUTE[0][0]:
        return make_action(left=True, b=True)
    if x >= MILK_SHIP_PIXEL_ROUTE[0][0] + 6 and y < UPPER_BARN_SHIP_LOWER_LANE_Y - 2:
        return make_action(down=True, b=True)

    route = MILK_SHIP_PIXEL_ROUTE[:4]
    start_index = 0
    if y >= route[0][1] - 1 and x <= route[0][0] + 1:
        start_index = 1
    if y >= route[1][1] - 1 and x <= route[1][0] + 1:
        start_index = 2
    if y >= route[2][1] - 1 and x <= route[2][0] + 1:
        start_index = 3
    for index, target in enumerate(route[start_index:], start=start_index):
        if abs(x - target[0]) <= 1 and abs(y - target[1]) <= 1:
            continue
        if index in (0, 1, 3):
            return run_to_pixel_axis((x, y), target, x_first=True)
        return run_to_pixel_axis((x, y), target, y_first=True)
    return None


def left_side_vertical_nav_action(
    x: int,
    y: int,
    tx: int,
    ty: int,
    *,
    going_down: bool,
) -> Optional[np.ndarray]:
    """Reach wall-side interact pixels via the recorded left vertical lane.

    Climb/descend on lane x first while far from the target row. Only settle
    onto the interact column (~27) near the target Y — cutting left early at
    the lower corridor (x=54,y=345) dead-ends against the bottom wall.
    """
    if abs(x - tx) <= 2:
        if abs(y - ty) <= 1:
            return None
        return make_action(down=going_down, up=not going_down, b=abs(y - ty) > 8)
    if x > 120:
        action = left_lower_lane_from_right_action(x, y)
        if action is not None:
            return action
    if abs(y - ty) > 16:
        lane_x = left_cow_lane_x(y)
        if abs(x - lane_x) > 1:
            return make_action(
                right=x < lane_x,
                left=x > lane_x,
                b=abs(x - lane_x) > 8,
            )
        return make_action(down=going_down, up=not going_down, b=True)
    if abs(x - tx) > 1:
        return make_action(right=x < tx, left=x > tx, b=abs(x - tx) > 8)
    return make_action(down=going_down, up=not going_down, b=True)


def exit_prep_escape_action(x: int, y: int) -> Optional[np.ndarray]:
    """Pixel route out of left/upper dead-ends toward the lower-aisle prep stand."""
    tx, ty = COW_EXIT_PREP_PX
    if abs(x - tx) <= 3 and abs(y - ty) <= 3:
        return None
    # Upper-left stalls are a dead-end: go south before trying to cross east.
    if x < 120:
        if y < UPPER_BARN_SHIP_LOWER_LANE_Y - 2:
            return make_action(down=True, b=True)
        lane_x = left_cow_lane_x(y)
        if abs(x - lane_x) > 2:
            return make_action(right=x < lane_x, left=x > lane_x, b=True)
        if abs(y - ty) > 3:
            return make_action(down=y < ty, up=y > ty, b=True)
        return make_action(right=True, b=True)
    if y < UPPER_BARN_SHIP_LOWER_LANE_Y - 2:
        if abs(x - COW_RIGHT_AISLE_X) > 3:
            return make_action(
                right=x < COW_RIGHT_AISLE_X,
                left=x > COW_RIGHT_AISLE_X,
                b=True,
            )
        return make_action(down=True, b=True)
    if abs(y - ty) > 3:
        return make_action(down=y < ty, up=y > ty, b=True)
    if abs(x - tx) > 3:
        return make_action(right=x < tx, left=x > tx, b=True)
    return None


def left_feed_spot_action(
    spot: CowFeedSpot,
    current_x: int,
    current_y: int,
) -> Optional[np.ndarray]:
    """Pixel align for left-aisle trough stands (stand.x <= 7)."""
    if not (spot.stand[0] <= 7):
        return None
    target_x, target_y = spot.interact_px
    if spot.face == "left":
        if current_x > target_x + 2:
            if abs(current_y - LEFT_TROUGH_LANE_Y) > 2:
                return make_action(
                    up=current_y > LEFT_TROUGH_LANE_Y,
                    down=current_y < LEFT_TROUGH_LANE_Y,
                    b=True,
                )
            return make_action(left=True, b=True)
        if abs(current_y - target_y) > 2:
            return make_action(up=current_y > target_y, down=current_y < target_y, b=True)
        if abs(current_x - target_x) > 2:
            return make_action(left=current_x > target_x, right=current_x < target_x, b=True)
        return None
    if abs(current_x - target_x) > 2:
        return make_action(left=current_x > target_x, right=current_x < target_x, b=True)
    if abs(current_y - target_y) > 2:
        return make_action(up=current_y > target_y, down=current_y < target_y, b=True)
    return None


def left_cow_to_fodder_action(current_x: int, current_y: int) -> Optional[np.ndarray]:
    """From left-wall care area up to the fodder dispenser approach corridor."""
    fodder_x = FODDER_STAND[0] * 16 + 8
    if current_x <= 90 and current_y >= 240:
        if current_y < 327:
            if current_y < 300 and current_x > 22:
                return make_action(left=True, b=True)
            if current_x < 22:
                return make_action(right=True, b=True)
            return make_action(down=True, b=True)
        return make_action(right=True, b=True)
    if current_y >= 326 and current_x < 239:
        if current_x > 90 and current_y > 335:
            return None
        return make_action(right=True, b=True)
    if current_x >= 239 and current_y > 312:
        return make_action(up=True, b=True)
    if current_y <= 312 and current_y > LEFT_TROUGH_LANE_Y + 2 and current_x > 203:
        return make_action(left=True, b=True)
    if 196 <= current_x <= 205 and current_y > LEFT_TROUGH_LANE_Y + 2:
        return make_action(up=True, b=True)
    if 196 <= current_x <= fodder_x and abs(current_y - LEFT_TROUGH_LANE_Y) <= 2:
        if current_x < fodder_x - 2:
            return make_action(right=True, b=True)
    return None


def left_trough_return_action(current_x: int, current_y: int) -> Optional[np.ndarray]:
    """Return from a left trough interact pixel toward the fodder aisle."""
    fodder_x = FODDER_STAND[0] * 16 + 8
    if current_x > LEFT_TROUGH_RETURN_X:
        return None
    if current_y > LEFT_TROUGH_LANE_Y + 2:
        return make_action(up=True, b=True)
    if current_y < LEFT_TROUGH_LANE_Y - 2:
        return make_action(down=True, b=True)
    if current_x < fodder_x - 2:
        return make_action(right=True, b=True)
    return None


def milk_ship_escape_prefix_action(
    x: int,
    y: int,
    *,
    ship_route_index: int,
) -> Optional[np.ndarray]:
    """Upper/left barn escapes before following ``MILK_SHIP_PIXEL_ROUTE``.

    Returns an action for pre-route positioning, or ``None`` when the caller
    should advance along ``MILK_SHIP_PIXEL_ROUTE[ship_route_index]``.
    """
    if x < 120 and y < LEFT_BARN_SHIP_LOWER_Y - 2:
        if y >= UPPER_BARN_SHIP_LOWER_LANE_Y and x < LEFT_BARN_SHIP_LANE_X - 2:
            return make_action(right=True, b=True)
        return make_action(down=True, b=True)
    if x < 120 and x > LEFT_BARN_SHIP_LANE_X + 2:
        return make_action(left=True, b=True)
    if x < 120 and y < MILK_SHIP_PIXEL_ROUTE[-1][1] - 2:
        return make_action(down=True, b=True)
    if x < 120 and abs(x - MILK_SHIP_PIXEL_ROUTE[-1][0]) > 2:
        return make_action(
            right=x < MILK_SHIP_PIXEL_ROUTE[-1][0],
            left=x > MILK_SHIP_PIXEL_ROUTE[-1][0],
            b=True,
        )
    if y < UPPER_BARN_SHIP_ESCAPE_Y - 2:
        if x >= 120 and x < COW_RIGHT_AISLE_X - 2:
            return make_action(right=True, b=True)
        return make_action(down=True, b=True)
    if y <= UPPER_BARN_SHIP_ESCAPE_Y + 2 and x < UPPER_BARN_RIGHT_AISLE_X - 2:
        return make_action(right=True, b=True)
    if ship_route_index == 0 and x >= UPPER_BARN_RIGHT_AISLE_X - 3 and y < UPPER_BARN_SHIP_CROSS_Y - 2:
        return make_action(down=True, b=True)
    if (
        ship_route_index == 0
        and abs(y - UPPER_BARN_SHIP_CROSS_Y) <= 2
        and x > UPPER_BARN_SHIP_AISLE_X
    ):
        return make_action(left=True, b=True)
    if (
        ship_route_index == 0
        and abs(y - UPPER_BARN_SHIP_LOWER_LANE_Y) <= 2
        and x > MILK_SHIP_PIXEL_ROUTE[0][0]
    ):
        return make_action(left=True, b=True)
    if (
        ship_route_index == 0
        and x >= MILK_SHIP_PIXEL_ROUTE[0][0] + 6
        and y < UPPER_BARN_SHIP_LOWER_LANE_Y - 2
    ):
        return make_action(down=True, b=True)
    return None


def milk_ship_route_step_action(
    x: int,
    y: int,
    index: int,
) -> Optional[np.ndarray]:
    """One step toward ``MILK_SHIP_PIXEL_ROUTE[index]`` (or None if arrived)."""
    index = min(index, len(MILK_SHIP_PIXEL_ROUTE) - 1)
    target = MILK_SHIP_PIXEL_ROUTE[index]
    if abs(x - target[0]) <= 2 and abs(y - target[1]) <= 2:
        return None
    if index == 0:
        return run_to_pixel_axis((x, y), target, x_first=True)
    if index in (1, 3, 5):
        return run_to_pixel_axis((x, y), target, x_first=True)
    return run_to_pixel_axis((x, y), target, y_first=True)


def recorded_interact_lane_action(
    x: int,
    y: int,
    tx: int,
    ty: int,
    *,
    face: str,
) -> Optional[np.ndarray]:
    """Recorded right/left aisle approach toward an interact pixel.

    Does not handle adjacency short-circuit or exact-on-target checks —
    callers should return early for those cases.
    """
    if face not in ("left", "right"):
        return None

    # The barn stall rows are much faster and more reliable when entered
    # through the recorded right-side lane instead of BFS-chasing moving cows.
    if face == "left" and y < ty - 10:
        if tx >= 120:
            if ty <= COW_UPPER_RIGHT_ROUTE_MAX_Y and abs(x - COW_RIGHT_AISLE_X) > 2:
                return make_action(right=x < COW_RIGHT_AISLE_X, left=x > COW_RIGHT_AISLE_X, b=True)
            if x < 192:
                return make_action(right=True, b=True)
            if y < ty:
                return make_action(down=True, b=True)
        else:
            return left_side_vertical_nav_action(x, y, tx, ty, going_down=True)

    if face == "right" and y < ty - 10:
        if tx < 100:
            return left_side_vertical_nav_action(x, y, tx, ty, going_down=True)
        if x > 96:
            return make_action(left=True, b=True)
        if y < ty:
            return make_action(down=True, b=True)

    if face == "left" and y > ty + 10:
        if tx < 100:
            return left_side_vertical_nav_action(x, y, tx, ty, going_down=False)
        if ty <= COW_UPPER_RIGHT_ROUTE_MAX_Y:
            if abs(x - COW_RIGHT_AISLE_X) > 2:
                return make_action(right=x < COW_RIGHT_AISLE_X, left=x > COW_RIGHT_AISLE_X, b=True)
            return make_action(up=True, b=True)
        if ty <= 255:
            if x < 102:
                return make_action(right=True, b=True)
            if y > 342:
                return make_action(up=True, b=True)
            if x < 174:
                return make_action(right=True, b=True)
            if y > 315:
                return make_action(up=True, b=True)
            if x < 192:
                return make_action(right=True, b=True)
        else:
            if x < 160 and y > 330:
                return make_action(right=True, b=True)
            if y > 326:
                return make_action(up=True, b=True)
            if x < 192:
                return make_action(right=True, b=True)
        return make_action(up=True, b=True)

    if face == "right" and y > ty + 10:
        if tx < 100:
            return left_side_vertical_nav_action(x, y, tx, ty, going_down=False)
        if x < 102:
            return make_action(right=True, b=True)
        if y > 342:
            return make_action(up=True, b=True)
        if x < min(tx, 174):
            return make_action(right=True, b=True)
        return make_action(up=True, b=True)

    if abs(x - tx) > 1:
        return make_action(right=x < tx, left=x > tx, b=abs(x - tx) > 8)
    if abs(y - ty) > 1:
        return make_action(down=y < ty, up=y > ty, b=abs(y - ty) > 8)
    return None
