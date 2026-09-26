"""Play 0x3F leftover → walk-on of the east diamond-platform stairs.

One gate. Dest is RAM (hyp mode-9 cellar 0x2F). Fail Gleeok 0x3C and
Magical Key cellar 0x0F. Do not chain the cellar-cross. OccupancyWalker
banned. CheckWarp is exact: stand on tiles 0x70-0x73 and idle.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.door_hop import HopFail, RoomHopController, RoomHopSpec
from zelda_i.level8.cellar import CELLAR_ROOM
from zelda_i.level8.path import EAST_3E_DEST, EAST_3E_DEST_POSE, GLEEOK_HYP
from zelda_i.ram import ZeldaSnapshot

__all__ = [
    "STAIRS_3F_GATE",
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
STAIRS_3F_HYP_XY = (208, 141)
STAIRS_TILES = range(0x70, 0x74)
_SAMPLE_PERIOD = 12
_MAX_FRAMES = 4000


from zelda_i.overworld.common import walk_or_swing


def stairs_3f_step(snap: ZeldaSnapshot, frames: int = 0) -> FrameAction:
    """Walk to (208,141), idle on 0x70-0x73. Exact CheckWarp. No occupancy."""
    x, y = int(snap.link_x), int(snap.link_y)
    tile = int(snap.colliding_tile)
    if tile in STAIRS_TILES:
        return FrameAction(nes_idle_action(), "stairs_stand")
    gx, gy = STAIRS_3F_HYP_XY
    if x != gx:
        btn = "RIGHT" if x < gx else "LEFT"
        return walk_or_swing(frames, btn, "stairs_x", snap)
    if y != gy:
        btn = "DOWN" if y < gy else "UP"
        return walk_or_swing(frames, btn, "stairs_y", snap)
    return FrameAction(nes_idle_action(), "stairs_exact")


STAIRS_3F_GATE = RoomHopSpec(
    spec_id="level8_stairs_3f",
    origin=STAIRS_3F_ORIGIN,
    door="STAIRS",
    done_reason="left_0x3f_stairs",
    step=stairs_3f_step,
    level=LEVEL8,
    max_frames=_MAX_FRAMES,
    sample_period=_SAMPLE_PERIOD,
    fails=(
        HopFail((GLEEOK_HYP,), "gleeok_0x3c"),
        HopFail((CELLAR_ROOM,), "cellar_0x{screen:02x}"),
    ),
    # The walk-on lands in mode-9; a passage arrival counts, and the mode is
    # part of the arrival note.
    require_play_arrival=False,
    passage_arrival=True,
    arrive_note="m{mode}_0x{screen:02x}_{x}_{y}",
    unexpected_note="unexpected_0x{screen:02x}",
    unexpected_play_only=False,
    scroll_reason="stairs_scroll",
    settle_reason="dest_settle",
    passage_hold_reason="cellar_hold",
)


@dataclass(kw_only=True)
class Level8Stairs3FController(RoomHopController):
    """0x3F leftover → east stairs walk-on. Dest is RAM; fail 0x0F / 0x3C."""

    spec: RoomHopSpec = STAIRS_3F_GATE

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        return stairs_3f_step(snap, self.frames)


def make_stairs_3f_controller(
    *, dest: int | None = STAIRS_3F_DEST
) -> Level8Stairs3FController:
    return Level8Stairs3FController(dest=dest)
