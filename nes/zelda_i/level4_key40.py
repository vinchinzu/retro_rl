"""L4 0x40 key hunt for the dest_6b-clear leftover.

``level4_maze_path.Level4Key40Controller`` hunts ``KEY_40_PICKUP_XY=(120,117)``.
Live ``l6_dest_clear5b_v1`` timed out in the south pocket ``(120,149)`` with
the key on the floor ~8px north. Do not grow ``level4_maze_path`` (1128).
"""

from __future__ import annotations

from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level4_dungeon import (
    KEY_40_PICKUP_XY,
    ROOM_ITEM_SMALL_KEY,
    ROOM_L4_ZOLS_40,
)
from zelda_i.level4_maze_path import Key40Phase, Level4Key40Controller
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

# PNG leftover: key just north of Link at (120,149).
SOUTH_POCKET_KEY_XY: tuple[int, int] = (120, 141)
SOUTH_POCKET_X_TOL = 8
SOUTH_POCKET_Y_MIN = 125
SOUTH_POCKET_Y_MAX = 165


def live_key_xy(snap: ZeldaSnapshot) -> tuple[int, int] | None:
    """Room-item / drop slot whose type is the small key."""
    for obj in snap.objects:
        if obj.slot == 0:
            continue
        if obj.type_id == ROOM_ITEM_SMALL_KEY and obj.x and obj.y:
            return (int(obj.x), int(obj.y))
    return None


def south_pocket_key_xy(snap: ZeldaSnapshot) -> tuple[int, int] | None:
    if abs(snap.link_x - SOUTH_POCKET_KEY_XY[0]) > SOUTH_POCKET_X_TOL:
        return None
    if not SOUTH_POCKET_Y_MIN <= snap.link_y <= SOUTH_POCKET_Y_MAX:
        return None
    return SOUTH_POCKET_KEY_XY


class Level4Key40SpineController(Level4Key40Controller):
    """Same maze path, but HUNT occupancy-chases a live key / south pocket."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.walker = OccupancyWalker()

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.phase is Key40Phase.HUNT:
            target = live_key_xy(snap) or south_pocket_key_xy(snap)
            if target is not None:
                return self._chase_key(snap, target)
        return super().step(snap)

    def _chase_key(
        self, snap: ZeldaSnapshot, target: tuple[int, int]
    ) -> FrameAction:
        self.frames += 1
        self.phase_frames += 1
        if snap.mode == 17:
            return self._fail("link_death")
        if (
            snap.screen == ROOM_L4_ZOLS_40
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and self.keys_before is not None
            and snap.keys > self.keys_before
        ):
            self.success = True
            self._set_phase(Key40Phase.DONE, "key_collected")
            return FrameAction(nes_idle_action(), "done")
        tx, ty = target
        xy = (int(snap.link_x), int(snap.link_y))
        if abs(xy[0] - tx) <= 4 and abs(xy[1] - ty) <= 4:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), "key_stand")
        self.walker.observe(xy)
        # v2 leftover (128,149): 8 misses blocked the dest tile and stood.
        dest = (tx, ty)
        if dest in self.walker.grid.blocked:
            self.walker.grid.blocked.discard(dest)
            self.walker.path = None
        direction = self.walker.next_dir(xy, dest)
        if direction is None and dest != KEY_40_PICKUP_XY:
            self.walker.path = None
            direction = self.walker.next_dir(xy, KEY_40_PICKUP_XY)
        if direction is None:
            return FrameAction(nes_idle_action(), "key_no_path")
        return FrameAction(nes_action(direction), "key_chase")

    def report(self) -> dict[str, Any]:
        payload = super().report()
        payload["hunt"] = "occupancy_live_key_or_south_pocket"
        payload["occupancy_misses"] = int(self.walker.misses)
        return payload


def make_room_40_key_controller() -> Level4Key40SpineController:
    """Spine factory. Isolated maze_path factory stays the scripted hunt."""
    return Level4Key40SpineController()
