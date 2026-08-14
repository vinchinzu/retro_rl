"""One-frame Level 9 door policies for the backward endgame route."""

from __future__ import annotations

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level9_ganon import LEVEL9, ROOM_BEFORE_GANON, ROOM_GANON
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

NORTH_DOOR_X = 0x78
NORTH_DOOR_X_TOL = 4


def final_patra_to_ganon_step(snap: ZeldaSnapshot) -> FrameAction:
    """One frame of naturally cleared ``0x52`` → Ganon ``0x42``.

    The Patra south-stand fight can finish at x≈112, which sticks against the
    north wall.  Recenter to x≈120 before holding UP through the earned door.
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
    if abs(int(snap.link_x) - NORTH_DOOR_X) > NORTH_DOOR_X_TOL:
        direction = "LEFT" if snap.link_x > NORTH_DOOR_X else "RIGHT"
        return FrameAction(nes_action(direction), "ganon_align_x")
    return FrameAction(nes_action("UP"), "ganon_push_north")


__all__ = [
    "NORTH_DOOR_X",
    "NORTH_DOOR_X_TOL",
    "final_patra_to_ganon_step",
]
