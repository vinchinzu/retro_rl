"""Level 6 play 0x09 south door after Rod cellar return.

Leftover (192,141) NE spawn; left 0x68 gone; remaining right block;
south mouth open on the leftover PNG. Occupancy to (120,189) then DOWN.
Dest is RAM after the transition — do not invent Gohma. Do not poke
bow/arrows/doors/keys. Halt north of leftover (stairs 0x71).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_overworld import LEVEL6, LEVEL6_ROD_WIZZ_ROOM
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyWalker

__all__ = [
    "SOUTH09_MAX_FRAMES",
    "SOUTH_DOOR_X",
    "SOUTH_DOOR_Y",
    "Level6South09Controller",
    "level6_south09_stages",
    "level6_south09_success",
    "make_south09_controller",
]

SOUTH_DOOR_X = 120
SOUTH_DOOR_Y = 189
SOUTH_BAND_Y = 181
SOUTH_DOOR_TOL = 4
NORTH_HALT_Y = 109
SOUTH09_MAX_FRAMES = 4000
SOUTH09_SAMPLE_PERIOD = 12
CELLAR_MODE = 9


@dataclass
class Level6South09Controller:
    """Occupancy south mouth of cleared 0x09. Dest is RAM, not Gohma."""

    spec_id: str = "level6_south_0x09"
    room: int = LEVEL6_ROD_WIZZ_ROOM
    goal: tuple[int, int] = (SOUTH_DOOR_X, SOUTH_DOOR_Y)
    max_frames: int = SOUTH09_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(default_factory=OccupancyWalker)

    def _rod(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "rod", 0))

    def _bow(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "bow", 0))

    def _arrows(self, snap: ZeldaSnapshot) -> int:
        return int(getattr(snap, "arrows", 0))

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
        }
        if force or self.frames <= 2 or self.frames % SOUTH09_SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "reason": action.reason,
                    "rod": self._rod(snap),
                    "bow": self._bow(snap),
                    "arrows": self._arrows(snap),
                    "keys": int(snap.keys),
                    "tile": int(snap.colliding_tile),
                    "misses": self.walker.misses,
                }
            )
        return action

    def _mark_success(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.success = True
        self.notes.append(note)
        self.walker.last_dir = None
        return self._emit(
            snap,
            FrameAction(nes_idle_action(), f"arrived_{snap.screen:02x}"),
            force=True,
        )

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            if "timeout" not in self.notes:
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                    f"_mode={snap.mode}_rod={self._rod(snap)}"
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == 17:
            self.failed = True
            self.notes.append("link_death")
            return self._emit(
                snap, FrameAction(nes_idle_action(), "link_death"), force=True
            )
        if snap.mode == CELLAR_MODE:
            self.failed = True
            self.notes.append(
                f"warped_cellar_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "warped_cellar"), force=True
            )
        if (
            snap.level == LEVEL6
            and snap.screen != self.room
            and snap.mode == PLAY_MODE
            and not snap.transitioning
            and self._rod(snap) != 0
        ):
            return self._mark_success(
                snap,
                f"arrived_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                f"_rod={self._rod(snap)}",
            )
        if snap.transitioning or snap.mode in (2, 3, 4, 6, 7):
            self.walker.last_dir = None
            return FrameAction(nes_action("DOWN"), "south_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if snap.level != LEVEL6:
            self.failed = True
            self.notes.append(f"left_level_{snap.level}")
            return self._emit(
                snap, FrameAction(nes_idle_action(), "left_level"), force=True
            )
        if snap.screen != self.room:
            self.walker.last_dir = None
            return FrameAction(nes_action("DOWN"), "south_settle")

        xy = (int(snap.link_x), int(snap.link_y))
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")

        # Stairs 0x71 is north of leftover. Occupancy must not walk UP into it.
        if xy[1] <= NORTH_HALT_Y:
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(f"north_f{self.frames}_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "south_north_halt")
            )

        if xy[1] >= SOUTH_BAND_Y:
            self.walker.last_dir = None
            if abs(xy[0] - SOUTH_DOOR_X) > SOUTH_DOOR_TOL:
                btn = "LEFT" if xy[0] > SOUTH_DOOR_X else "RIGHT"
                return self._emit(
                    snap, FrameAction(nes_action(btn), "south_align")
                )
            return self._emit(snap, FrameAction(nes_action("DOWN"), "south_push"))

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
                snap, FrameAction(nes_idle_action(), "south_stand")
            )
        return self._emit(
            snap, FrameAction(nes_action(direction), "south_path")
        )

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "occupancy to (120,189) then DOWN; halt y<=109; dest is RAM"
            ),
            "leftover": dict(self.leftover),
            "misses": self.walker.misses,
            "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id,
            "room": self.room,
            "goal": self.goal,
        }


def make_south09_controller() -> Level6South09Controller:
    """Occupancy south of cleared 0x09. Do not poke bow/arrows/doors."""
    return Level6South09Controller()


def level6_south09_stages():
    """Play 0x09 leftover (192,141) → occupancy south door. Dest is RAM."""
    ctl = make_south09_controller()
    return (
        ("level6_south_0x09", ctl, ctl.max_frames),
    )


def level6_south09_success(snap: ZeldaSnapshot) -> bool:
    """Play-ready L6 room other than 0x09 with ADDR_ROD. Dest is RAM."""
    return (
        snap.level == LEVEL6
        and snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen != LEVEL6_ROD_WIZZ_ROOM
        and snap.triforce == 0x1F
        and int(getattr(snap, "rod", 0)) != 0
    )
