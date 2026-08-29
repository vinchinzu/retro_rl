"""Level 6 play 0x29 inland north after north39 enter-stop.

Leftover (120,205) south mouth. Dated miss: occupancy UP @ x=120 boxes
(120,157) tile 244 (exit-ow v3). LEFT+UP clip off the center column,
then OccupancyWalker to the north door. Reclear wizzrobes if live
(kill-door; do not hold-UP). Dest is RAM (likely 0x19). Do not poke
bow/arrows/doors/keys. Do not invent Gohma. Isolated BFS banned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon import GenericDungeonRoomController
from zelda_i.level6_dungeon import ROOM_29_SPEC
from zelda_i.level6_occupancy import l6_leftover, l6_play_dest_success
from zelda_i.level6_overworld import LEVEL6, LEVEL6_DARK_29_ROOM, LEVEL6_DARK_39_ROOM
from zelda_i.level6_path import NORTH_BAND_Y, NORTH_DOOR_X, NORTH_DOOR_Y
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyGrid, OccupancyWalker

__all__ = [
    "INLAND29_MAX_FRAMES",
    "Level6Inland29Controller",
    "level6_inland29_stages",
    "level6_inland29_success",
    "make_inland29_controller",
]

DOOR_TOL = 4
# south29 aisle; occupancy UP @ x=120 never reaches it (v3 y=157 tile 244).
CLIP_Y = 141
WEST_SPAWN_XMIN = 16
INLAND29_MAX_FRAMES = 12000
INLAND29_SAMPLE_PERIOD = 12
CELLAR_MODE = 9
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)


def _new_walker() -> OccupancyWalker:
    return OccupancyWalker(grid=OccupancyGrid(xmin=WEST_SPAWN_XMIN))


@dataclass
class Level6Inland29Controller:
    """LEFT+UP off tile-244 column, occupancy north. Dest is RAM."""

    spec_id: str = "level6_inland_0x29"
    room: int = LEVEL6_DARK_29_ROOM
    goal: tuple[int, int] = (NORTH_DOOR_X, NORTH_DOOR_Y)
    max_frames: int = INLAND29_MAX_FRAMES
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
        return int(snap.rod)

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            **l6_leftover(snap),
            "map": int(snap.map),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
        }
        if force or self.frames <= 2 or self.frames % INLAND29_SAMPLE_PERIOD == 0:
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
                    "tile": int(snap.colliding_tile),
                    "misses": self.walker.misses,
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
                f"key_spent_29_to_{snap.screen:02x}_{self.keys}->{int(snap.keys)}"
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
            snap, FrameAction(nes_idle_action(), f"arrived_{snap.screen:02x}"), force=True
        )

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
            if snap.screen == LEVEL6_DARK_39_ROOM:
                return self._fail(
                    snap,
                    f"backtrack_39_{snap.link_x}_{snap.link_y}",
                )
            return self._mark_success(snap)
        if snap.transitioning or snap.mode in WAIT_MODES:
            self.walker.last_dir = None
            return FrameAction(nes_action("UP"), "north_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.screen != self.room:
            self.walker.last_dir = None
            return FrameAction(nes_action("UP"), "north_settle")

        xy = (int(snap.link_x), int(snap.link_y))
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        # Dated miss: occupancy UP @ x=120 from (120,205) boxes (120,157)
        # tile 244. Clip off that column; do not occupancy-UP it again.
        if xy[1] > CLIP_Y:
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_action("LEFT", "UP"), "inland_clip")
            )

        if ROOM_29_SPEC.live_enemies(snap):
            if self.fighter is None:
                self.fighter = GenericDungeonRoomController(ROOM_29_SPEC)
                self.notes.append(f"reclear_29_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            return self._emit(snap, self.fighter.step(snap))
        self.fighter = None

        if xy[1] <= NORTH_BAND_Y:
            # v1 leftover (48,109) cardinal RIGHT boxed tile 244 (0x28 v6).
            self.walker.last_dir = None
            if abs(xy[0] - NORTH_DOOR_X) > DOOR_TOL:
                if xy[0] < NORTH_DOOR_X:
                    return self._emit(
                        snap, FrameAction(nes_action("RIGHT", "UP"), "door_clip")
                    )
                return self._emit(
                    snap, FrameAction(nes_action("LEFT", "UP"), "door_clip")
                )
            return self._emit(snap, FrameAction(nes_action("UP"), "north_push"))

        dest = self.goal
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)
        if direction is None:
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(f"stand_f{self.frames}_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "occupancy_stand")
            )
        return self._emit(
            snap, FrameAction(nes_action(direction), "north_path")
        )

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "LEFT+UP clip off occupancy-UP @ x=120 tile 244, occupancy "
                "north; RIGHT+UP on north band (v1 (48,109)); reclear if live"
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "room": self.room,
            "goal": self.goal,
            "keys": self.keys,
        }


def make_inland29_controller() -> Level6Inland29Controller:
    """Clip inland 0x29 then occupancy north. Do not poke bow/arrows/doors."""
    return Level6Inland29Controller()


def level6_inland29_stages():
    """Play 0x29 leftover (120,205) → clip past tile 244 → dest RAM."""
    ctl = make_inland29_controller()
    return (
        ("level6_inland_0x29", ctl, ctl.max_frames),
    )


def level6_inland29_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 room other than 0x29/0x39 with ADDR_ROD. Dest is RAM."""
    return l6_play_dest_success(
        snap,
        not_room=LEVEL6_DARK_29_ROOM,
        passage_ok=False,
        forbid=(LEVEL6_DARK_39_ROOM,),
    )
