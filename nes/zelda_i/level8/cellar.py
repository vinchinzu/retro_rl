"""Level 8 Magical Key cellar 0x0F: east-drop then west-ladder return.

Leftover is mode-9 (136,141) tile 36, Magic Key 1. y=141 LEFT is pit
tile 250. F1 cardinal DOWN at the pad did not move (south is brick).
Follow-up is L1/L7: RIGHT to the east column, LEFT+DOWN, floor LEFT,
UP (48,93). Live dest is play 0x1F (96,157), not Gleeok 0x3C.
OccupancyWalker is banned. No RAM writes. Not on L8_THROUGH.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot

__all__ = [
    "ALIGN",
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


def _leftover(snap: ZeldaSnapshot) -> dict[str, Any]:
    return {
        "x": int(snap.link_x),
        "y": int(snap.link_y),
        "mode": int(snap.mode),
        "screen": int(snap.screen),
        "tile": int(snap.colliding_tile),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "magic_key": int(getattr(snap, "magic_key", 0)),
        "triforce": int(snap.triforce),
    }


@dataclass(kw_only=True)
class Level8MagicKeyCellarReturnController(HopController):
    """0x0F pad leftover, east-drop, floor LEFT, west-ladder UP. Fixture-live."""

    spec_id: str = "level8_magic_key_cellar_return"
    max_frames: int = CELLAR_RETURN_MAX_FRAMES
    require_level: int = LEVEL8
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "left_0x0f_stairs"
    dest: int | None = None
    route_eligible: bool = False
    leftover: dict[str, Any] = field(default_factory=dict)
    writes: int = 0

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if snap.mode != PLAY_MODE or snap.transitioning:
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return True

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or not self.leftover or self.frames % _SAMPLE_PERIOD == 0:
            self.leftover = _leftover(snap)
        return action

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        x = int(snap.link_x)
        # L1 east-face colliding tile can be 250 during LEFT+DOWN clip.
        # Fail only if we are still west of the east column (F2 at x=160).
        if int(snap.colliding_tile) == PIT_TILE and x < EAST_X - ALIGN:
            return self.mark_fail("pit_tile_250")
        if snap.mode == PASSAGE_MODE:
            if snap.screen != CELLAR_ROOM:
                return self.mark_fail("unexpected_cellar")
            return magic_key_cellar_return_step(snap)
        if snap.mode == PLAY_MODE:
            if self.dest is not None and snap.screen != self.dest:
                return self.mark_fail(f"unexpected_play_0x{snap.screen:02x}")
            return FrameAction(nes_idle_action(), "wait_dest")
        return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "dest_screen": self.dest,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "door": "STAIRS",
            "leftover": dict(self.leftover),
        }


def make_magic_key_cellar_return_controller(
    *, dest: int | None = CELLAR_RETURN_DEST
) -> Level8MagicKeyCellarReturnController:
    return Level8MagicKeyCellarReturnController(dest=dest)
