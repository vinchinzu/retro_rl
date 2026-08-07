"""Shared overworld movement helpers for Zelda I controllers.

Both the Level 1 phase controller and the Level 2 hop controller use the same
stuck tracking, periodic sword swing, edge recovery, and align-and-push
primitives. Keep route-specific geometry in the owning module.
"""

from __future__ import annotations

from typing import Callable

from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.input_script import FrameAction
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


# --- Dungeon diamond-block door approach (L2 0x7d east, 0x6e east, …) ---
# Mid-room diamond solids block a straight y≈141 corridor near x≈128–176.
# Correct policy (verified 2026-08-06): reach east wall on an open y-band, then
# cycle LEFT+vertical to free y≈141 *at the wall*, then RIGHT through the door.
# Do NOT fully retreat west of ~x=180 on y=141 (re-enters the solid).
# Do NOT y-align only with micro-LEFT at x≥200 without longer LEFT pulses.

DOOR_Y_DEFAULT = 141
DIAMOND_WALL_X = 200
DIAMOND_BAND_7D = 157  # open band for entry-east 0x7d → 0x7e
DIAMOND_BAND_6E = 113  # open band for 0x6e RIGHT key door → 0x6f


def diamond_east_phase(
    snap: ZeldaSnapshot,
    *,
    phase: str,
    band_y: int = DIAMOND_BAND_7D,
    door_y: int = DOOR_Y_DEFAULT,
    wall_x: int = DIAMOND_WALL_X,
    cycle: int = 0,
) -> tuple[FrameAction, str]:
    """One-frame policy for diamond-blocked east doors.

    Phases (caller advances when geometry matches):
      free     — leave west/east/south/north alcoves toward mid-room
      band     — align ``band_y`` at mid-x
      wall     — RIGHT on band until ``link_x >= wall_x``
      door_y   — at wall: LEFT×6 → vertical to door_y → RIGHT×10 cycles
      push     — hold RIGHT on door_y (re-nudge if y drifts)

    Returns (action, next_phase_hint). Caller may keep phase until transition.
    """
    x, y = snap.link_x, snap.link_y

    if phase == "free":
        if 70 <= x <= 180 and 110 <= y <= 175:
            return FrameAction(nes_action("UP" if y > band_y else "DOWN"), "band"), "band"
        if x >= 200:
            if not (120 <= y <= 170):
                return FrameAction(nes_action("UP" if y > 170 else "DOWN"), "free_ey"), "free"
            return FrameAction(nes_action("LEFT"), "free_ex"), "free"
        if x <= 48:
            if not (120 <= y <= 170):
                return FrameAction(nes_action("DOWN" if y < 120 else "UP"), "free_wy"), "free"
            return FrameAction(nes_action("RIGHT"), "free_wx"), "free"
        if y >= 195:
            if abs(x - 120) > 10:
                return FrameAction(nes_action("RIGHT" if x < 120 else "LEFT"), "free_sx"), "free"
            return FrameAction(nes_action("UP"), "free_sy"), "free"
        if y <= 95:
            if abs(x - 120) > 10:
                return FrameAction(nes_action("RIGHT" if x < 120 else "LEFT"), "free_nx"), "free"
            return FrameAction(nes_action("DOWN"), "free_ny"), "free"
        if abs(x - 120) >= abs(y - 141):
            return FrameAction(nes_action("RIGHT" if x < 120 else "LEFT"), "free_cx"), "free"
        return FrameAction(nes_action("DOWN" if y < 141 else "UP"), "free_cy"), "free"

    if phase == "band":
        if abs(y - band_y) <= 4 and 90 <= x <= 160:
            return FrameAction(nes_action("RIGHT"), "to_wall"), "wall"
        if abs(y - band_y) > 4:
            return FrameAction(nes_action("DOWN" if y < band_y else "UP"), "band_y"), "band"
        if x < 90:
            return FrameAction(nes_action("RIGHT"), "band_x"), "band"
        if x > 160:
            return FrameAction(nes_action("LEFT"), "band_x"), "band"
        return FrameAction(nes_action("RIGHT"), "band"), "band"

    if phase == "wall":
        if x >= wall_x:
            return FrameAction(nes_action("LEFT"), "at_wall"), "door_y"
        if abs(y - band_y) > 8:
            return FrameAction(nes_action("DOWN" if y < band_y else "UP"), "wall_y"), "wall"
        return FrameAction(nes_action("RIGHT"), "wall_r"), "wall"

    if phase == "door_y":
        # S2 cycle: LEFT block → vertical to door_y → RIGHT block.
        # Longer LEFT when still off door_y (vertical is solid at x≈200).
        step_in_cycle = cycle % 28
        if abs(y - door_y) <= 2 and x >= wall_x - 6:
            return FrameAction(nes_action("RIGHT"), "door_ready"), "push"
        left_hold = 10 if abs(y - door_y) > 2 else 6
        if step_in_cycle < left_hold:
            return FrameAction(nes_action("LEFT"), "door_left"), "door_y"
        if step_in_cycle < left_hold + 12:
            if abs(y - door_y) <= 2:
                return FrameAction(nes_action("RIGHT"), "door_r_early"), "door_y"
            return FrameAction(
                nes_action("UP" if y > door_y else "DOWN"), "door_vert"
            ), "door_y"
        return FrameAction(nes_action("RIGHT"), "door_right"), "door_y"

    # push — pure y-align + RIGHT. Do NOT LEFT-nudge here: that re-enters the
    # mid-room diamond solid on door_y (observed fail (208,149)→(176,141)).
    if abs(y - door_y) > 4:
        return FrameAction(nes_action("UP" if y > door_y else "DOWN"), "push_y"), "push"
    return FrameAction(nes_action("RIGHT"), "push_r"), "push"
