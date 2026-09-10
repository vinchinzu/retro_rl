"""One-frame Level 9 door policies for the backward endgame route."""

from __future__ import annotations

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.door_hop import DOOR_TOL, door_band_goal
from zelda_i.dungeon.hop_controller import dungeon_align_then_push
from zelda_i.level9.ganon import LEVEL9, ROOM_BEFORE_GANON, ROOM_GANON
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

NORTH_DOOR_X = 0x78
NORTH_DOOR_X_TOL = 4
NORTH_DOOR_GOAL = (NORTH_DOOR_X, 77)
ZELDA_DOOR_GOAL = (NORTH_DOOR_X, 77)


def leftover_door_step(
    snap: ZeldaSnapshot,
    leftover: tuple[int, int],
    hold_dir: str,
    default_goal: tuple[int, int],
    *,
    reason: str,
    tol: int = DOOR_TOL,
) -> FrameAction:
    """Align to leftover+hold_dir door-band, then hold. Off-column uses door x."""
    gx, gy = door_band_goal(hold_dir, leftover, default_goal, tol=tol)
    if hold_dir in ("UP", "DOWN"):
        return dungeon_align_then_push(
            snap, push_dir=hold_dir, target_x=gx, x_tol=tol, reason=reason
        )
    return dungeon_align_then_push(
        snap, push_dir=hold_dir, target_y=gy, y_tol=tol, reason=reason
    )


def final_patra_to_ganon_step(
    snap: ZeldaSnapshot,
    leftover: tuple[int, int] | None = None,
) -> FrameAction:
    """One frame of naturally cleared ``0x52`` → Ganon ``0x42``.

    Door column is leftover-relative: in-band leftover keeps its x; off-column
    leftover (Patra south-stand can finish at x≈112) uses the north mouth.
    """
    if snap.level != LEVEL9:
        return FrameAction(nes_idle_action(), "wait_level9")
    if snap.transitioning:
        return FrameAction(nes_action("UP"), "ganon_scroll")
    if snap.mode != PLAY_MODE:
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
    if snap.screen == ROOM_GANON:
        return FrameAction(nes_idle_action(), "ganon_arrived")
    if snap.screen != ROOM_BEFORE_GANON:
        return FrameAction(
            nes_idle_action(),
            f"unexpected_room_0x{snap.screen:02x}",
        )
    pose = leftover if leftover is not None else (int(snap.link_x), int(snap.link_y))
    return leftover_door_step(
        snap, pose, "UP", NORTH_DOOR_GOAL, reason="ganon", tol=NORTH_DOOR_X_TOL
    )


__all__ = [
    "NORTH_DOOR_GOAL",
    "NORTH_DOOR_X",
    "NORTH_DOOR_X_TOL",
    "ZELDA_DOOR_GOAL",
    "final_patra_to_ganon_step",
    "leftover_door_step",
]
