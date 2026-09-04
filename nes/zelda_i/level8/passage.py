"""L8 Gleeok-suffix cellar 0x2F: east-spawn, floor-cross west.

Settled leftover is mode-9 (192,93) tile 0x6F on the east/source ladder.
DOWN to floor y=189, LEFT to x=48, UP west. Never UP on the east ladder
(CheckSubroom returns play 0x3F). Inverse of L8 MK 0x0F (which climbed
east then west). OccupancyWalker banned. No RAM writes. Not on L8_THROUGH.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.level8.path import GLEEOK_HYP
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot

__all__ = [
    "ALIGN",
    "CELLAR_ROOM",
    "CHECKSUBROOM_SPLIT_X",
    "DEST",
    "DEST_HYP",
    "DEST_POSE",
    "EAST_X",
    "EXIT_STAIRS",
    "FLOOR_Y",
    "MOUTH_Y",
    "PIT_TILE",
    "RAM_CLAIM",
    "SOURCE_ROOM",
    "SPAWN_XY",
    "STAIRS_TILES",
    "WEST_X",
    "Level8Passage2FController",
    "make_passage_2f_controller",
    "passage_2f_step",
]

LEVEL8 = 8
CELLAR_ROOM = 0x2F
SOURCE_ROOM = 0x3F
DEST_HYP = 0x4C  # confirmed live P1; not 0x3C / 0x3F
DEST = 0x4C  # live $EB from 0x2F west-ladder
DEST_POSE = (112, 125)  # live P1 arrival
EAST_X = 192  # settled S1/S2 leftover; L7 0x7B B-side analog
WEST_X = 48
FLOOR_Y = 189
MOUTH_Y = 93
EXIT_STAIRS = (WEST_X, MOUTH_Y)
SPAWN_XY = (EAST_X, MOUTH_Y)
PIT_TILE = 250
PIT_LEDGE_Y = 141
STAIRS_TILES = range(0x70, 0x74)
ALIGN = 4
CHECKSUBROOM_SPLIT_X = 0x80
CELLAR_CROSS_MAX_FRAMES = 4000
_SAMPLE_PERIOD = 12

RAM_CLAIM = (
    "From settled mode-9 cellar 0x2F leftover (192,93) east/source ladder, "
    "DOWN to floor y=189, LEFT to west x=48, UP west ladder. Never UP on "
    "the east source ladder (returns to play 0x3F). First settled play $EB "
    "is RAM (hyp 0x4C, NOT source 0x3F, NOT Gleeok 0x3C). Keys 8->8 bombs "
    "6->6 MK 1 TF 0x7F. One gate."
)


def passage_2f_step(snap: ZeldaSnapshot) -> FrameAction:
    """DOWN east column, floor LEFT to x=48, UP west. Never source UP."""
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
        if y > MOUTH_Y + ALIGN:
            return FrameAction(nes_action("UP"), "cellar_west_up")
        if tile in STAIRS_TILES:
            return FrameAction(nes_idle_action(), "cellar_exit_warp")
        return FrameAction(nes_action("UP"), "cellar_west_lip")
    if tile == PIT_TILE and x < EAST_X - ALIGN:
        return FrameAction(nes_action("RIGHT"), "cellar_pit_to_east")
    if abs(x - EAST_X) > ALIGN:
        btn = "RIGHT" if x < EAST_X else "LEFT"
        return FrameAction(nes_action(btn), "cellar_to_east")
    return FrameAction(nes_action("DOWN"), "cellar_east_drop")


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
class Level8Passage2FController(HopController):
    """0x2F east-spawn leftover, DOWN, floor LEFT, west-ladder UP. Fixture-live."""

    spec_id: str = "level8_passage_2f"
    max_frames: int = CELLAR_CROSS_MAX_FRAMES
    require_level: int = LEVEL8
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "left_0x2f_stairs"
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
        if snap.screen in (SOURCE_ROOM, GLEEOK_HYP):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        return snap.screen != CELLAR_ROOM

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or not self.leftover or self.frames % _SAMPLE_PERIOD == 0:
            self.leftover = _leftover(snap)
        return action

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        if snap.screen == GLEEOK_HYP:
            return self.mark_fail("gleeok_0x3c")
        if snap.mode == PLAY_MODE and snap.screen == SOURCE_ROOM:
            return self.mark_fail("returned_source_0x3f")
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != CELLAR_ROOM
            and self.dest is not None
            and snap.screen != self.dest
        ):
            return self.mark_fail(f"unexpected_play_0x{snap.screen:02x}")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        x = int(snap.link_x)
        if snap.mode == PLAY_MODE:
            return FrameAction(nes_idle_action(), "wait_dest")
        if snap.mode != PASSAGE_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != CELLAR_ROOM:
            return self.mark_fail(f"unexpected_cellar_0x{snap.screen:02x}")
        act = passage_2f_step(snap)
        if act.reason.endswith(("_up", "_climb", "_lip")) and x >= CHECKSUBROOM_SPLIT_X:
            return self.mark_fail("up_on_source_ladder")
        return act

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "dest_screen": self.dest,
            "policy": RAM_CLAIM,
            "evidence": "fixture-live",
            "route_eligible": False,
            "natural_entry": False,
            "writes": int(self.writes),
            "door": "STAIRS",
            "leftover": dict(self.leftover),
        }


def make_passage_2f_controller(
    *, dest: int | None = DEST,
) -> Level8Passage2FController:
    return Level8Passage2FController(dest=dest)
