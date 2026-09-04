"""Play 0x3F leftover → walk-on of the east diamond-platform stairs.

One gate. Dest is RAM (hyp mode-9 cellar 0x2F). Fail Gleeok 0x3C and
Magical Key cellar 0x0F. Do not chain the cellar-cross. OccupancyWalker
banned. CheckWarp is exact: stand on tiles 0x70-0x73 and idle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.level8.cellar import CELLAR_ROOM
from zelda_i.level8.path import EAST_3E_DEST, EAST_3E_DEST_POSE, GLEEOK_HYP
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot

__all__ = [
    "STAIRS_3F_DEST",
    "STAIRS_3F_DEST_HYP",
    "STAIRS_3F_DEST_MODE",
    "STAIRS_3F_DEST_POSE",
    "STAIRS_3F_HYP_XY",
    "STAIRS_3F_ORIGIN",
    "STAIRS_3F_ORIGIN_POSE",
    "STAIRS_TILES",
    "Level8Stairs3FController",
    "make_stairs_3f_controller",
    "stairs_3f_step",
]

LEVEL8 = 8
STAIRS_3F_ORIGIN = EAST_3E_DEST  # 0x3F
STAIRS_3F_ORIGIN_POSE = EAST_3E_DEST_POSE  # (32, 141) west mouth
STAIRS_3F_DEST_HYP = 0x2F  # confirmed live K3; not 0x3C / 0x0F
STAIRS_3F_DEST = 0x2F  # live $EB from 0x3F stairs; mode-9 cellar
STAIRS_3F_DEST_POSE = (208, 141)  # live K3 arrival
STAIRS_3F_DEST_MODE = 9
# K1/K2: visual stairs are tile 0x77 (decorative hole). Live CheckWarp
# is tile 0x71 at (193,141). x-first RIGHT along y=141 toward (208,93)
# crosses it and idles. Dest is RAM.
STAIRS_3F_HYP_XY = (208, 93)
STAIRS_TILES = range(0x70, 0x74)
_SAMPLE_PERIOD = 12
_MAX_FRAMES = 4000


def stairs_3f_step(snap: ZeldaSnapshot) -> FrameAction:
    """Walk to (208,93), idle on 0x70-0x73. Exact CheckWarp. No occupancy."""
    x, y = int(snap.link_x), int(snap.link_y)
    tile = int(snap.colliding_tile)
    if tile in STAIRS_TILES:
        return FrameAction(nes_idle_action(), "stairs_stand")
    gx, gy = STAIRS_3F_HYP_XY
    if x != gx:
        btn = "RIGHT" if x < gx else "LEFT"
        return FrameAction(nes_action(btn), "stairs_x")
    if y != gy:
        btn = "DOWN" if y < gy else "UP"
        return FrameAction(nes_action(btn), "stairs_y")
    return FrameAction(nes_idle_action(), "stairs_exact")


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
class Level8Stairs3FController(HopController):
    """0x3F leftover → east stairs walk-on. Dest is RAM; fail 0x0F / 0x3C."""

    spec_id: str = "level8_stairs_3f"
    max_frames: int = _MAX_FRAMES
    require_level: int = LEVEL8
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    done_reason: str = "left_0x3f_stairs"
    dest: int | None = None
    route_eligible: bool = False
    leftover: dict[str, Any] = field(default_factory=dict)
    writes: int = 0

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        if snap.transitioning:
            return False
        if snap.screen in (CELLAR_ROOM, GLEEOK_HYP):
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        if snap.mode == PLAY_MODE and snap.screen == STAIRS_3F_ORIGIN:
            return False
        if snap.mode == PASSAGE_MODE:
            return True
        if snap.mode == PLAY_MODE and snap.screen != STAIRS_3F_ORIGIN:
            return True
        return False

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"m{snap.mode}_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        return FrameAction(nes_idle_action(), "stairs_scroll")

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
        if snap.screen == CELLAR_ROOM:
            return self.mark_fail(f"cellar_0x{snap.screen:02x}")
        if (
            not snap.transitioning
            and snap.screen != STAIRS_3F_ORIGIN
            and self.dest is not None
            and snap.screen != self.dest
        ):
            return self.mark_fail(f"unexpected_0x{snap.screen:02x}")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode == PASSAGE_MODE:
            return FrameAction(nes_idle_action(), "cellar_hold")
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != STAIRS_3F_ORIGIN:
            return FrameAction(nes_idle_action(), "dest_settle")
        return stairs_3f_step(snap)

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


def make_stairs_3f_controller(
    *, dest: int | None = STAIRS_3F_DEST
) -> Level8Stairs3FController:
    return Level8Stairs3FController(dest=dest)
