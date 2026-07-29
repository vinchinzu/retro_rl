"""Shared overworld movement helpers for Zelda I controllers.

Both the Level 1 phase controller and the Level 2 hop controller use the same
stuck tracking, periodic sword swing, edge recovery, and align-and-push
primitives. Keep route-specific geometry in the owning module.
"""

from __future__ import annotations

from typing import Callable

from retro_harness.nes import nes_action, nes_idle_action
from snes_oneshot.primitives import FrameAction
from zelda_i.ram import ZeldaSnapshot

DEFAULT_SWING_PERIOD = 12
DEFAULT_SWING_FRAMES = 3
DEFAULT_STUCK_THRESHOLD = 50

# Screen-edge thresholds (overworld playfield)
EDGE_SOUTH_Y = 212
EDGE_NORTH_Y = 62
EDGE_EAST_X = 232
EDGE_WEST_X = 14
ARRIVAL_EAST_X = 220
ARRIVAL_WEST_X = 30
ARRIVAL_NORTH_Y = 70
ARRIVAL_SOUTH_Y = 200


def swing_action(
    phase_frames: int,
    direction: str,
    reason: str,
    *,
    period: int = DEFAULT_SWING_PERIOD,
    hold: int = DEFAULT_SWING_FRAMES,
) -> FrameAction:
    """Walk in ``direction``, pulsing A for a few frames each period."""
    if period > 0 and phase_frames % period < hold:
        return FrameAction(nes_action(direction, "A"), f"{reason}_slash")
    return FrameAction(nes_action(direction), reason)


def track_stuck(
    snap: ZeldaSnapshot,
    *,
    last_x: int,
    last_y: int,
    last_screen: int,
    stuck: int,
) -> tuple[int, int, int, int]:
    """Return updated (stuck, last_x, last_y, last_screen)."""
    if (
        snap.link_x == last_x
        and snap.link_y == last_y
        and snap.screen == last_screen
        and not snap.transitioning
    ):
        stuck += 1
    else:
        stuck = 0
    return stuck, snap.link_x, snap.link_y, snap.screen


def on_arrival_edge(direction: str, snap: ZeldaSnapshot) -> bool:
    """True while Link is still on the edge that produced this hop's arrival."""
    if direction == "RIGHT":
        return snap.link_x > ARRIVAL_EAST_X
    if direction == "LEFT":
        return snap.link_x < ARRIVAL_WEST_X
    if direction == "UP":
        return snap.link_y < ARRIVAL_NORTH_Y
    if direction == "DOWN":
        return snap.link_y > ARRIVAL_SOUTH_Y
    return False


def recover_off_edge(
    snap: ZeldaSnapshot,
    travel_direction: str,
    *,
    swing: Callable[[str, str], FrameAction],
) -> FrameAction | None:
    """Nudge inward if Link is scraping the wrong screen edge."""
    if snap.link_y >= EDGE_SOUTH_Y and travel_direction != "DOWN":
        return swing("UP", "off_south")
    if snap.link_y <= EDGE_NORTH_Y and travel_direction != "UP":
        return swing("DOWN", "off_north")
    if snap.link_x >= EDGE_EAST_X and travel_direction != "RIGHT":
        return swing("LEFT", "off_east")
    if snap.link_x <= EDGE_WEST_X and travel_direction != "LEFT":
        return swing("RIGHT", "off_west")
    return None


def unstick_wiggle(
    stuck: int,
    *,
    reason: str = "unstick",
    reset_after: int = 140,
) -> tuple[FrameAction, int]:
    """Cycle cardinal directions with A when stuck. Returns (action, new_stuck)."""
    wiggle = ("UP", "DOWN", "LEFT", "RIGHT")[stuck % 4]
    new_stuck = 0 if stuck > reset_after else stuck
    return FrameAction(nes_action(wiggle, "A"), reason), new_stuck


def align_and_push(
    snap: ZeldaSnapshot,
    *,
    direction: str,
    reason: str,
    align_x: int | None = None,
    align_y: int | None = None,
    y_band: tuple[int, int] | None = None,
    stuck: int = 0,
    stuck_threshold: int = DEFAULT_STUCK_THRESHOLD,
    x_tol: int = 5,
    y_tol: int = 5,
    swing: Callable[[str, str], FrameAction] | None = None,
    swing_period: int = DEFAULT_SWING_PERIOD,
    swing_hold: int = DEFAULT_SWING_FRAMES,
    phase_frames: int = 0,
) -> FrameAction:
    """Align to optional x/y or y-band, then push in ``direction``."""

    def _swing(dir_: str, why: str) -> FrameAction:
        if swing is not None:
            return swing(dir_, why)
        return swing_action(
            phase_frames, dir_, why, period=swing_period, hold=swing_hold
        )

    if stuck > stuck_threshold:
        action, _ = unstick_wiggle(stuck, reason=f"{reason}_unstick")
        return action

    if y_band is not None:
        lo, hi = y_band
        if snap.link_y < lo:
            return _swing("DOWN", "band_down")
        if snap.link_y > hi:
            return _swing("UP", "band_up")
        return _swing(direction, reason)

    if (
        align_x is not None
        and abs(snap.link_x - align_x) > x_tol
        and 80 < snap.link_y < 205
    ):
        btn = "LEFT" if snap.link_x > align_x else "RIGHT"
        return _swing(btn, f"{reason}_ax")

    if (
        align_y is not None
        and abs(snap.link_y - align_y) > y_tol
        and 25 < snap.link_x < 230
    ):
        # Force the travel direction near the entry edge so corridor alignment
        # does not scrape rocks after a screen scroll.
        if direction == "RIGHT" and snap.link_x <= 18:
            return _swing("RIGHT", "enter_corridor")
        if direction == "RIGHT" and snap.link_y > 200:
            return _swing("UP", "climb_entry")
        if direction == "RIGHT" and snap.link_y < 70:
            return _swing("DOWN", "drop_entry")
        btn = "UP" if snap.link_y > align_y else "DOWN"
        return _swing(btn, f"{reason}_ay")

    return _swing(direction, reason)


def wake_or_wait_mode(phase_frames: int, mode: int) -> FrameAction:
    """Brief A pulse, then idle, while waiting out non-play modes."""
    if phase_frames % 30 < 3:
        return FrameAction(nes_action("A"), f"wake_mode_{mode}")
    return FrameAction(nes_idle_action(), f"wait_mode_{mode}")
