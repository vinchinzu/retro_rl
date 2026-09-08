"""Level 8 Magical Key cellar 0x0F: east-drop then west-ladder return.

Leftover is mode-9 (136,141) tile 36, Magic Key 1. y=141 LEFT is pit
tile 250. F1 cardinal DOWN at the pad did not move (south is brick).
Follow-up is L1/L7: RIGHT to the east column, LEFT+DOWN, floor LEFT,
UP (48,93). Live dest is play 0x1F (96,157), not Gleeok 0x3C.
OccupancyWalker is banned. No RAM writes. Not on L8_THROUGH.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.door_hop import RoomHopController, RoomHopSpec
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot

__all__ = [
    "ALIGN",
    "CELLAR_RETURN_GATE",
    "CELLAR_RETURN_DEST",
    "CELLAR_RETURN_MAX_FRAMES",
    "CELLAR_RETURN_POSE",
    "CELLAR_ROOM",
    "EAST_X",
    "EXIT_STAIRS",
    "FLOOR_Y",
    "LEVEL8",
    "PAD",
    "PIT_TILE",
    "STAIRS_TILES",
    "WEST_X",
    "Level8MagicKeyCellarReturnController",
    "magic_key_cellar_return_step",
    "make_magic_key_cellar_return_controller",
]

LEVEL8 = 8
CELLAR_ROOM = 0x0F
ALIGN = 2
WEST_X = 48
EAST_X = 176  # inbound CELLAR_EAST_X / L7 ROOM_4A_EAST_COL
FLOOR_Y = 189
EXIT_STAIRS = (48, 93)
PAD = (136, 141)
PIT_TILE = 250
STAIRS_TILES = range(0x70, 0x74)
CELLAR_RETURN_MAX_FRAMES = 4000
# Live F3/F4: west-ladder CheckWarp -> play 0x1F leftover (96,157).
CELLAR_RETURN_DEST = 0x1F
CELLAR_RETURN_POSE = (96, 157)
_SAMPLE_PERIOD = 12


def magic_key_cellar_return_step(snap: ZeldaSnapshot) -> FrameAction:
    """RIGHT to east, LEFT+DOWN, floor LEFT, UP west. Never LEFT at y=141."""
    x, y = int(snap.link_x), int(snap.link_y)
    tile = int(snap.colliding_tile)
    on_west = abs(x - WEST_X) <= ALIGN
    on_floor = y >= FLOOR_Y - ALIGN
    if on_floor:
        if x > WEST_X + ALIGN:
            return FrameAction(nes_action("LEFT"), "cellar_floor_west")
        if x < WEST_X - ALIGN:
            return FrameAction(nes_action("RIGHT"), "cellar_floor_east")
        return FrameAction(nes_action("UP"), "cellar_west_climb")
    if on_west:
        if y > EXIT_STAIRS[1] + ALIGN:
            return FrameAction(nes_action("UP"), "cellar_west_up")
        if tile in STAIRS_TILES:
            return FrameAction(nes_idle_action(), "cellar_exit_warp")
        # Tile 0x6F at (48,93) does not CheckWarp.
        return FrameAction(nes_action("UP"), "cellar_west_lip")
    # F1: cardinal DOWN at (136,141) tile 36 did not move (south brick).
    # F2: LEFT+DOWN at x=160 y=141 is still the pit (tile 250). Stay RIGHT
    # until the east column; inbound climbed this ladder at x=176.
    if x >= EAST_X - ALIGN:
        return FrameAction(nes_action("LEFT", "DOWN"), "cellar_east_drop")
    return FrameAction(nes_action("RIGHT"), "cellar_to_east")


def _cellar_return_policy(ctl: RoomHopController, snap: ZeldaSnapshot) -> FrameAction:
    """Novel: the walk runs in mode-9, and the L1 east-face pit tile is a fail."""
    x = int(snap.link_x)
    # L1 east-face colliding tile can be 250 during LEFT+DOWN clip.
    # Fail only if we are still west of the east column (F2 at x=160).
    if int(snap.colliding_tile) == PIT_TILE and x < EAST_X - ALIGN:
        return ctl.mark_fail("pit_tile_250")
    if snap.mode == PASSAGE_MODE:
        if snap.screen != CELLAR_ROOM:
            return ctl.mark_fail("unexpected_cellar")
        return magic_key_cellar_return_step(snap)
    if snap.mode == PLAY_MODE:
        if ctl.dest is not None and snap.screen != ctl.dest:
            return ctl.mark_fail(f"unexpected_play_0x{snap.screen:02x}")
        return FrameAction(nes_idle_action(), "wait_dest")
    return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")


CELLAR_RETURN_GATE = RoomHopSpec(
    spec_id="level8_magic_key_cellar_return",
    origin=CELLAR_ROOM,
    door="STAIRS",
    done_reason="left_0x0f_stairs",
    policy_fn=_cellar_return_policy,
    level=LEVEL8,
    max_frames=CELLAR_RETURN_MAX_FRAMES,
    sample_period=_SAMPLE_PERIOD,
    # No room fails: the dest mismatch is caught inside the policy, and any
    # settled play screen counts when the caller passes ``dest=None``.
    arrive_any=True,
    unexpected_note="",
)


@dataclass(kw_only=True)
class Level8MagicKeyCellarReturnController(RoomHopController):
    """0x0F pad leftover, east-drop, floor LEFT, west-ladder UP. Fixture-live."""

    spec: RoomHopSpec = CELLAR_RETURN_GATE


def make_magic_key_cellar_return_controller(
    *, dest: int | None = CELLAR_RETURN_DEST
) -> Level8MagicKeyCellarReturnController:
    return Level8MagicKeyCellarReturnController(dest=dest)
