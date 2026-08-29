"""L4 0x40 key hunt for the dest_6b-clear leftover.

``level4_maze_path.Level4Key40Controller`` hunts ``KEY_40_PICKUP_XY=(120,117)``.
Live ``l6_dest_clear5b_v1`` timed out in the south pocket ``(120,149)`` with
the key on the floor ~8px north. Do not grow ``level4_maze_path`` (1128).
"""

from __future__ import annotations

from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level4.dungeon import (
    KEY_40_PICKUP_XY,
    ROOM_ITEM_SMALL_KEY,
    ROOM_40_SPEC,
    ROOM_L4_ZOLS_40,
)
from zelda_i.level4.maze_path import (
    KEY_40_PATH_ANCHOR,
    Key40Phase,
    Level4Key40Controller,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk.physics import OccupancyWalker

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
    """Exact-normalized maze path, then conservative occupancy fallback."""

    path_start: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.walker = OccupancyWalker()

    def policy(
        self, snap: ZeldaSnapshot, xy: tuple[int, int]
    ) -> FrameAction:
        if self.phase is Key40Phase.FIGHT:
            live = ROOM_40_SPEC.live_enemies(snap)
            if (
                not live
                and self._clear.max_live_enemies >= ROOM_40_SPEC.expected_enemy_count
            ):
                self._set_phase(Key40Phase.ALIGN, "room_cleared")
                return FrameAction(nes_idle_action(), "align_settle")
            return self._clear.step(snap)

        if self.phase is Key40Phase.ALIGN:
            ax, ay = KEY_40_PATH_ANCHOR
            if xy == (ax, ay):
                self.path_start = xy
                self._set_phase(Key40Phase.PATH, "aligned_exact_path_anchor")
                return FrameAction(nes_idle_action(), "anchor_exact")
            if xy[0] != ax:
                direction = "RIGHT" if xy[0] < ax else "LEFT"
            else:
                direction = "DOWN" if xy[1] < ay else "UP"
            return FrameAction(nes_action(direction), f"align_exact_{direction}")

        if self.phase is Key40Phase.HUNT:
            target = live_key_xy(snap) or south_pocket_key_xy(snap)
            if target is not None:
                return self._chase_key(snap, target)
        return super().policy(snap, xy)

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
        dest = (tx, ty)
        direction = self.walker.next_dir(xy, dest)
        if direction is None:
            return FrameAction(nes_idle_action(), "key_no_path")
        return FrameAction(nes_action(direction), "key_chase")

    def report(self) -> dict[str, Any]:
        payload = super().report()
        payload["alignment"] = "exact_xy_before_open_loop"
        payload["path_start"] = list(self.path_start) if self.path_start else None
        payload["hunt"] = "occupancy_live_key_or_south_pocket_no_unblock"
        payload["occupancy_misses"] = int(self.walker.misses)
        return payload


def make_room_40_key_controller() -> Level4Key40SpineController:
    """Spine factory. Isolated maze_path factory stays the scripted hunt."""
    return Level4Key40SpineController()
