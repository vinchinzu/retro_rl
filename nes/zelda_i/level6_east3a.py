"""Historical Level 6 0x3A east-wall diagnostic.

The dated A-side cellar return was play 0x3A (96,157). Center hole is not CheckWarp
(center3a v1 timeout (112,141) tile 118). The visual east mouth is a ROM
wall: the south-around path reaches (208,141), but RIGHT cannot transition.
This controller preserves that exact diagnostic path. Occupancy grades every
walk. Do not poke bow/arrows/doors/keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.hop_controller import HopController, WAIT_SCROLL_B
from zelda_i.level6_occupancy import (
    l6_play_dest_success,
    record_l6_walk,
)
from zelda_i.level6_overworld import LEVEL6, LEVEL6_BLOCK_3A_ROOM
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.screen_glance import CLEAR_3A, GlanceLeftover, grade_controller
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "EAST3A_MAX_FRAMES",
    "EAST_DOOR",
    "Level6East3AController",
    "level6_east3a_glance",
    "level6_east3a_stages",
    "level6_east3a_success",
    "make_east3a_controller",
]

EAST3A_MAX_FRAMES = 4000
EAST3A_SAMPLE_PERIOD = 8
EAST_DOOR = (208, 141)
# The cellar spits Link immediately southwest of the revealed center hole.
# x=144 is the first conservative column east of its collision footprint.
SOUTH_AROUND_X = 144
SOUTH_LANE_Y = 157
# v2 (96,143) tile 119 counted as y-band (tol=4) then RIGHT 0px.
DOOR_TOL = 1
DATED_SPIT = (96, 157)


@dataclass
class Level6East3AController(HopController):
    """From cellar spit, walk south-around the hole to the east door."""

    spec_id: str = "level6_east_0x3a"
    room: int = LEVEL6_BLOCK_3A_ROOM
    max_frames: int = EAST3A_MAX_FRAMES
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = record_l6_walk(
            self.samples,
            snap,
            reason=action.reason,
            frames=self.frames,
            period=EAST3A_SAMPLE_PERIOD,
            misses=self.walker.misses,
            force=force,
        )
        return action

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        self.walker.last_dir = None
        return FrameAction(nes_idle_action(), "wait_scroll")

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        return l6_play_dest_success(snap, not_room=self.room)

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return f"dest_{snap.mode}_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"

    def mark_done(self, snap: ZeldaSnapshot, note: str | None = None) -> FrameAction:
        self.walker.last_dir = None
        self.done_reason = f"dest_{snap.mode}"
        return super().mark_done(snap, note)

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        if snap.mode != PLAY_MODE:
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            return self.mark_fail(f"left_level_{snap.level}")
        if snap.screen != self.room:
            return self.mark_fail(f"left_0x{self.room:02x}_to_0x{snap.screen:02x}")

        xy = (int(snap.link_x), int(snap.link_y))
        gx, gy = EAST_DOOR
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before:
            self.notes.append(
                f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}"
                f"_tile_{snap.colliding_tile}"
            )

        # The direct y-align at x=96 is the dated BLOCKED path: it puts Link
        # against the west face of the revealed center hole. Keep the cellar
        # spit on its live south lane until the full sprite is east of the hole.
        if xy[0] < SOUTH_AROUND_X:
            goal, reason = (SOUTH_AROUND_X, SOUTH_LANE_Y), "south_around"
        elif abs(xy[1] - gy) > DOOR_TOL:
            goal, reason = (xy[0], gy), "east_side"
        elif xy[0] < gx - DOOR_TOL:
            goal, reason = EAST_DOOR, "door"
        else:
            return self.mark_fail(
                f"rom_wall_east_{xy[0]}_{xy[1]}_tile_{snap.colliding_tile}"
            )

        if goal != self.walker.goal:
            self.walker.path = None
            self.walker.goal = goal
        direction = self.walker.next_dir(xy, goal)
        if direction is None:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"{reason}_stand")
        return FrameAction(nes_action(direction), f"{reason}_path")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                f"cellar08 spit {DATED_SPIT} → y={SOUTH_LANE_Y} RIGHT to "
                f"x={SOUTH_AROUND_X} → y=141 RIGHT to {EAST_DOOR}; "
                "occupancy miss-block/replan on every walk; no path stands; "
                "ROM 0x3A east=wall diagnostic"
            ),
            "leftover": dict(self.leftover),
            "spec_id": self.spec_id,
            "room": self.room,
            "misses": self.walker.misses,
        }


def make_east3a_controller() -> Level6East3AController:
    """Walk 0x3A east door from the cellar08 spit. Dest is RAM."""
    return Level6East3AController()


def level6_east3a_stages():
    """Run only the dated 0x3A wall diagnostic from the cleared-room boundary."""
    return (
        ("level6_east_0x3a", make_east3a_controller(), EAST3A_MAX_FRAMES),
    )


def level6_east3a_glance(controller: Any) -> GlanceLeftover:
    """Diagnostic leftover; it is not the route-eligible cellar leave."""
    return grade_controller(controller, CLEAR_3A)


def level6_east3a_success(snap: ZeldaSnapshot) -> bool:
    """Mode 9 or play ≠ 0x3A. Rod and TF 0x1F stay."""
    return l6_play_dest_success(snap, not_room=LEVEL6_BLOCK_3A_ROOM)
