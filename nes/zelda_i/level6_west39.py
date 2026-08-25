"""Level 6 west of 0x39 after clear3a leftover.

Reuse north39 prefix: occupancy 0x3A LEFT, replan leftover miss (v1
halted f2 0px tile 119). v2 timed out occupancy-stand at west mouth
(32,93) tile 200 — north39 west_align DOWN then west_push. Reclear
0x39 Vires. Then WEST, not north to 0x29. Halt at first occupancy miss
after leaving 0x3A. Clip only after a new live miss. Isolated BFS
banned. Ignore 0x2B. Do not KEY-UP 0x09 / 0x29. Do not CheckWarp 0x3A
stairs. Do not bomb. Dest is RAM (may be 0x38 from the east). Do not
invent Gohma.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_dungeon import ROOM_39_SPEC, make_clear_39_controller
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_BLOCK_3A_ROOM,
    LEVEL6_DARK_29_ROOM,
    LEVEL6_DARK_39_ROOM,
    LEVEL6_ROD_WIZZ_ROOM,
)
from zelda_i.level6_path import NORTH_BAND_Y, NORTH_DOOR_X, WEST_CLIP_X
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyGrid, OccupancyWalker

__all__ = [
    "WEST39_MAX_FRAMES",
    "WEST_DOOR",
    "WEST_SPAWN_XMIN",
    "Level6West39Controller",
    "level6_west39_stages",
    "level6_west39_success",
    "make_west39_controller",
]

WEST_DOOR = (32, 141)
WEST_DOOR_TOL = 4
WEST_SPAWN_XMIN = 16
WEST39_MAX_FRAMES = 20000
WEST39_SAMPLE_PERIOD = 12
CELLAR_MODE = 9
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)
START_ROOM = LEVEL6_BLOCK_3A_ROOM
VIA_ROOM = LEVEL6_DARK_39_ROOM


def _new_walker() -> OccupancyWalker:
    return OccupancyWalker(grid=OccupancyGrid(xmin=WEST_SPAWN_XMIN))


@dataclass
class Level6West39Controller:
    """0x3A occupancy LEFT, reclear 0x39, occupancy west. Dest is RAM."""

    spec_id: str = "level6_west_0x39"
    room: int = START_ROOM
    max_frames: int = WEST39_MAX_FRAMES
    frames: int = 0
    keys: int = -1
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=_new_walker)
    fighter: Any = None

    def _rod(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "rod", 0))

    def _bow(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "bow", 0))

    def _arrows(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "arrows", 0))

    def _goal(self) -> tuple[int, int]:
        if self.room == START_ROOM:
            return (WEST_CLIP_X, WEST_DOOR[1])
        return WEST_DOOR

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            "x": int(snap.link_x),
            "y": int(snap.link_y),
            "mode": int(snap.mode),
            "screen": int(snap.screen),
            "keys": int(snap.keys),
            "bombs": int(snap.bombs),
            "map": int(snap.map),
            "triforce": int(snap.triforce),
            "rod": self._rod(snap),
            "bow": self._bow(snap),
            "arrows": self._arrows(snap),
            "tile": int(snap.colliding_tile),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
            "room": int(self.room),
        }
        if force or self.frames <= 2 or self.frames % WEST39_SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "reason": action.reason,
                    "rod": self._rod(snap),
                    "keys": int(snap.keys),
                    "bombs": int(snap.bombs),
                    "tile": int(snap.colliding_tile),
                    "misses": self.walker.misses,
                    "room": int(self.room),
                }
            )
        return action

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        if note not in self.notes:
            self.notes.append(note)
        return self._emit(snap, FrameAction(nes_idle_action(), note), force=True)

    def _mark_success(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.keys >= 0 and int(snap.keys) < self.keys:
            self.notes.append(
                f"key_spent_{self.room:02x}_to_{snap.screen:02x}"
                f"_{self.keys}->{int(snap.keys)}"
            )
        self.keys = int(snap.keys)
        note = (
            f"arrived_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_rod={self._rod(snap)}_tf={snap.triforce:02x}"
            f"_keys={int(snap.keys)}"
        )
        self.success = True
        self.notes.append(note)
        self.walker.last_dir = None
        return self._emit(
            snap,
            FrameAction(nes_idle_action(), f"arrived_{snap.screen:02x}"),
            force=True,
        )

    def _arrive_39(self, snap: ZeldaSnapshot) -> None:
        self.notes.append(
            f"arrived_39_{snap.link_x}_{snap.link_y}_keys={int(snap.keys)}"
        )
        self.room = VIA_ROOM
        self.keys = int(snap.keys)
        self.walker = _new_walker()
        self.fighter = None

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.keys < 0:
            self.keys = int(snap.keys)
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if "timeout" not in self.notes:
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                    f"_mode={snap.mode}_rod={self._rod(snap)}_keys={int(snap.keys)}"
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == 17:
            return self._fail(snap, "link_death")
        if snap.mode == CELLAR_MODE:
            return self._fail(
                snap,
                f"warped_cellar_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
            )
        if snap.level == 0:
            return self._fail(
                snap, f"ow_early_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )
        if snap.level != LEVEL6:
            return self._fail(snap, f"left_level_{snap.level}_{snap.screen:02x}")
        if (
            snap.screen != self.room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and self._rod(snap) != 0
        ):
            if snap.screen == LEVEL6_ROD_WIZZ_ROOM:
                return self._fail(
                    snap,
                    f"key_up_09_{snap.link_x}_{snap.link_y}_keys={int(snap.keys)}",
                )
            if snap.screen == LEVEL6_DARK_29_ROOM:
                return self._fail(
                    snap,
                    f"north_29_{snap.link_x}_{snap.link_y}_keys={int(snap.keys)}",
                )
            if snap.screen == START_ROOM:
                return self._fail(
                    snap,
                    f"backtrack_3a_{snap.link_x}_{snap.link_y}",
                )
            if snap.screen == VIA_ROOM:
                self._arrive_39(snap)
                return self._emit(
                    snap, FrameAction(nes_action("LEFT"), "room_settle"), force=True
                )
            if not level6_west39_success(snap):
                return self._fail(
                    snap,
                    f"wrong_room_{snap.screen:02x}_{snap.link_x}_{snap.link_y}",
                )
            return self._mark_success(snap)
        if snap.transitioning or snap.mode in WAIT_MODES:
            self.walker.last_dir = None
            return FrameAction(nes_action("LEFT"), "west_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.room:
            self.walker.last_dir = None
            return FrameAction(nes_action("LEFT"), "west_settle")

        xy = (int(snap.link_x), int(snap.link_y))
        if snap.screen == VIA_ROOM and ROOM_39_SPEC.live_enemies(snap):
            if self.fighter is None:
                self.fighter = make_clear_39_controller()
                self.notes.append(f"reclear_39_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            return self._emit(snap, self.fighter.step(snap))
        self.fighter = None

        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before:
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")
            # v1 leftover LEFT (144,141) 0px tile 119. north39 replans this
            # same miss then enters 0x39. Halt only after leaving 0x3A.
            if self.room != START_ROOM:
                return self._fail(snap, f"occupancy_halt_{xy[0]}_{xy[1]}")

        gx, gy = self._goal()
        # v2 boxed (32,93) tile 200 occupancy_stand. north39 LEFT hop
        # west_align when x<=goal then west_push. Do not KEY-UP north.
        if xy[0] <= gx:
            self.walker.last_dir = None
            if abs(xy[1] - gy) > WEST_DOOR_TOL:
                btn = "UP" if xy[1] > gy else "DOWN"
                return self._emit(snap, FrameAction(nes_action(btn), "west_align"))
            return self._emit(snap, FrameAction(nes_action("LEFT"), "west_push"))

        dest = self._goal()
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)
        if direction is None:
            if self.room == START_ROOM:
                self.walker.last_dir = None
                return self._emit(
                    snap, FrameAction(nes_idle_action(), "occupancy_stand")
                )
            return self._fail(snap, f"occupancy_halt_{xy[0]}_{xy[1]}")
        if (
            direction == "UP"
            and snap.screen == VIA_ROOM
            and xy[1] <= NORTH_BAND_Y
            and abs(xy[0] - NORTH_DOOR_X) <= WEST_DOOR_TOL
        ):
            return self._fail(snap, f"north_key_halt_{xy[0]}_{xy[1]}")
        reason = "left_path" if self.room == START_ROOM else "west_path"
        return self._emit(snap, FrameAction(nes_action(direction), reason))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "occupancy 0x3A LEFT replan leftover miss like north39; "
                "west_align then west_push (v2 stand (32,93) tile 200); "
                "reclear 0x39 Vires → occupancy west; halt first miss after "
                "leaving 0x3A; no north 0x29; no KEY-UP 0x09; no stairs3a "
                "CheckWarp; dest is RAM; no bomb; ignore 0x2B"
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "room": self.room,
            "goal": self._goal(),
            "keys": self.keys,
        }


def make_west39_controller() -> Level6West39Controller:
    """Occupancy west of 0x39 after 0x3A leftover. Do not poke doors/bombs."""
    return Level6West39Controller()


def level6_west39_stages():
    """Play 0x3A leftover (144,141) → 0x39 reclear → west. Dest is RAM."""
    ctl = make_west39_controller()
    return (
        ("level6_west_0x39", ctl, ctl.max_frames),
    )


def level6_west39_success(snap: ZeldaSnapshot) -> bool:
    """Play dest ≠ 0x3A and ≠ 0x29 (and not via 0x39). Dest is RAM."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen != START_ROOM
        and snap.screen != LEVEL6_DARK_29_ROOM
        and snap.screen != VIA_ROOM
        and snap.screen != LEVEL6_ROD_WIZZ_ROOM
        and snap.triforce == 0x1F
        and int(getattr(snap, "rod", 0)) != 0
    )
