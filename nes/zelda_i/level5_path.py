"""Level 5 multi-room path policy (0x66 → 0x76 east key door → 0x77).

Room specs and stop predicates remain in ``level5_dungeon``.
"""

from __future__ import annotations

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level5_dungeon import (
    LEVEL_5,
    ROOM_L5_ENTRY,
    ROOM_L5_GIBDO_66,
    ROOM_L5_POLS_77,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot

EAST_DOOR_APPROACH_Y = 157
EAST_DOOR_CHANNEL_Y = 141
EAST_DOOR_WALL_X = 200

_ISOLATED_ROOM_77 = "L5_Room_77"


def should_force_keys_zero(
    start_state: str,
    *,
    keep_keys: bool = False,
    force_keys_zero: bool = False,
) -> bool:
    """True only for isolated ``L5_Room_77`` combat, unless explicitly overridden."""
    if keep_keys:
        return False
    if force_keys_zero:
        return True
    return start_state == _ISOLATED_ROOM_77


def level5_east_key_step(snap: ZeldaSnapshot) -> FrameAction:
    """Deterministic 0x66→0x76→key door→0x77 navigation policy.

    Room 0x66 supplies the key.  Return through its south door, leave the
    0x76 north/south mouth, approach the east wall on y≈157, then move to the
    door channel y≈141 without stepping back into the center statues.
    """
    if snap.level != LEVEL_5:
        return FrameAction(nes_idle_action(), "east_key_wait_level5")
    if snap.screen == ROOM_L5_POLS_77 and snap.mode == PLAY_MODE:
        return FrameAction(nes_idle_action(), "east_key_arrived")

    if snap.screen == ROOM_L5_GIBDO_66:
        if snap.transitioning:
            return FrameAction(nes_action("DOWN"), "east_key_south_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), "east_key_wait_66")
        # The fixed key can be collected while Link is still standing on the
        # Stepladder across the horizontal river. Finish the crossing before
        # horizontal alignment; sideways input is locked on the ladder tile.
        if snap.link_y < 141:
            return FrameAction(nes_action("DOWN"), "east_key_finish_ladder")
        if abs(snap.link_x - 120) > 4:
            direction = "LEFT" if snap.link_x > 120 else "RIGHT"
            return FrameAction(nes_action(direction), "east_key_align_south_x")
        return FrameAction(nes_action("DOWN"), "east_key_return_76")

    if snap.screen == ROOM_L5_ENTRY:
        if snap.transitioning:
            return FrameAction(nes_action("RIGHT"), "east_key_east_scroll")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), "east_key_wait_76")
        if snap.link_y > 185:
            return FrameAction(nes_action("UP"), "east_key_leave_south_mouth")
        if snap.link_x < EAST_DOOR_WALL_X - 4:
            if abs(snap.link_y - EAST_DOOR_APPROACH_Y) > 3:
                direction = "UP" if snap.link_y > EAST_DOOR_APPROACH_Y else "DOWN"
                return FrameAction(nes_action(direction), "east_key_align_approach_y")
            return FrameAction(nes_action("RIGHT"), "east_key_approach_wall")
        if abs(snap.link_y - EAST_DOOR_CHANNEL_Y) > 3:
            direction = "UP" if snap.link_y > EAST_DOOR_CHANNEL_Y else "DOWN"
            return FrameAction(nes_action(direction), "east_key_align_channel_y")
        return FrameAction(nes_action("RIGHT"), "east_key_unlock_77")

    return FrameAction(
        nes_idle_action(), f"east_key_unexpected_room_0x{snap.screen:02x}"
    )


__all__ = [
    "EAST_DOOR_APPROACH_Y",
    "EAST_DOOR_CHANNEL_Y",
    "EAST_DOOR_WALL_X",
    "level5_east_key_step",
    "should_force_keys_zero",
]
