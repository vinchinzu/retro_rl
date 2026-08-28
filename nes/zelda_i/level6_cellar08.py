"""Level 6 cellar 0x08 after the 0x3A position-write warp.

PRG1 ROM + ``CheckWarps``/``CheckSubroom`` decode: cellar 0x08 has endpoint
A=0x3A and endpoint B=0x1D. Entering from A initializes Link on the left
ladder (x=48), regardless of the pre-init warp-trigger pose (208,93). The
route must descend to the tunnel floor, cross RIGHT to x=192, and climb the
B-side ladder. Climbing the left ladder naturally returns to 0x3A.

Dest is exact RAM play 0x1D. Do not invent Gohma; ROM doors continue
0x1D DOWN -> 0x2D LEFT -> 0x2C KEY-UP -> boss 0x1C. Do not poke
bow/arrows/doors/keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_occupancy import l6_leftover, l6_play_dest_success
from zelda_i.level6_overworld import LEVEL6
from zelda_i.level6_stairs3a_warp import (
    STAIRS_3A_WARP_MAX_FRAMES,
    make_stairs_3a_warp_controller,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.screen_glance import CELLAR08_LEAVE, GlanceLeftover, grade_controller

__all__ = [
    "CELLAR_08_MAX_FRAMES",
    "CELLAR_08_ROOM",
    "CELLAR_08_DEST_ROOM",
    "Level6Cellar08Controller",
    "level6_cellar08_glance",
    "level6_cellar08_stages",
    "level6_cellar08_success",
    "make_cellar08_controller",
]

CELLAR_08_ROOM = 0x08
CELLAR_08_SOURCE_ROOM = 0x3A
CELLAR_08_DEST_ROOM = 0x1D
CELLAR_08_MAX_FRAMES = 4000
CELLAR_08_SAMPLE_PERIOD = 8
# Authorized CheckWarp trigger. InitMode9 replaces this with the A-side spawn.
EAST_MOUTH = (208, 93)
LEFT_LADDER_X = 48
RIGHT_LADDER_X = 192
FLOOR_Y = 189
MOUTH_Y = 93
ALIGN_TOL = 4
WAIT_MODES = (2, 3, 4, 6, 7, 10, 16)
CELLAR_PLAY_MODES = (9, 11)


@dataclass
class Level6Cellar08Controller:
    """Wait for A-side spawn, then floor RIGHT → B-side UP → 0x1D."""

    spec_id: str = "level6_cellar_0x08"
    room: int = CELLAR_08_ROOM
    max_frames: int = CELLAR_08_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, Any] = field(default_factory=dict)
    arrival_seen: bool = False
    on_floor: bool = False

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            **l6_leftover(snap),
            "health": int(snap.health),
            "map": int(snap.map),
        }
        if force or self.frames <= 2 or self.frames % CELLAR_08_SAMPLE_PERIOD == 0:
            self.samples.append(
                {
                    "frame": self.frames,
                    "x": int(snap.link_x),
                    "y": int(snap.link_y),
                    "mode": int(snap.mode),
                    "screen": int(snap.screen),
                    "reason": action.reason,
                    "tile": int(snap.colliding_tile),
                    "on_floor": self.on_floor,
                }
            )
        return action

    def _fail(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.failed = True
        if note not in self.notes:
            self.notes.append(note)
        return self._emit(snap, FrameAction(nes_idle_action(), note), force=True)

    def _done(self, snap: ZeldaSnapshot, note: str) -> FrameAction:
        self.success = True
        self.notes.append(note)
        return self._emit(snap, FrameAction(nes_idle_action(), "emerged"), force=True)

    def _at_dest(self, snap: ZeldaSnapshot) -> bool:
        return l6_play_dest_success(
            snap,
            not_room=self.room,
            dest_room=CELLAR_08_DEST_ROOM,
            passage_ok=False,
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
                    f"_mode={snap.mode}"
                )
            return self._emit(snap, FrameAction(nes_idle_action(), "timeout"), force=True)
        if snap.mode == 17:
            return self._fail(snap, "link_death")
        if self._at_dest(snap):
            return self._done(
                snap, f"play_0x{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            )
        if (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen != self.room
        ):
            if snap.screen == CELLAR_08_SOURCE_ROOM:
                return self._fail(snap, "returned_source_0x3a")
            return self._fail(snap, f"wrong_play_0x{snap.screen:02x}")
        if snap.transitioning or snap.mode in WAIT_MODES:
            return self._emit(snap, FrameAction(nes_idle_action(), "wait_scroll"))
        if snap.mode not in CELLAR_PLAY_MODES and snap.mode != PLAY_MODE:
            return self._emit(
                snap, FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
            )
        if snap.level != LEVEL6:
            return self._fail(snap, f"left_level_{snap.level}")

        xy = (int(snap.link_x), int(snap.link_y))
        if not self.arrival_seen:
            if (
                abs(xy[0] - LEFT_LADDER_X) <= ALIGN_TOL
                and xy[1] >= MOUTH_Y - ALIGN_TOL
            ):
                self.arrival_seen = True
                self.notes.append(f"a_side_spawn_{xy[0]}_{xy[1]}")
            else:
                return self._emit(
                    snap,
                    FrameAction(nes_idle_action(), "passage_init_wait"),
                )
        if xy[1] >= FLOOR_Y - ALIGN_TOL:
            self.on_floor = True
        if not self.on_floor:
            if xy[1] < FLOOR_Y - ALIGN_TOL:
                return self._emit(snap, FrameAction(nes_action("DOWN"), "drop_y"))
        if abs(xy[0] - RIGHT_LADDER_X) > ALIGN_TOL:
            btn = "LEFT" if xy[0] > RIGHT_LADDER_X else "RIGHT"
            return self._emit(snap, FrameAction(nes_action(btn), "cross_x"))
        if xy[1] > MOUTH_Y:
            return self._emit(snap, FrameAction(nes_action("UP"), "climb_y"))
        return self._emit(snap, FrameAction(nes_action("UP"), "mouth_up"))

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "samples": list(self.samples),
            "policy": (
                "mode-9 0x08: wait for A-side x=48 spawn, DOWN to y=189, "
                "RIGHT to x=192, UP B-side ladder; exact dest play 0x1d"
            ),
            "leftover": dict(self.leftover),
            "spec_id": self.spec_id,
            "room": self.room,
            "dest": CELLAR_08_DEST_ROOM,
            "arrival_seen": self.arrival_seen,
            "on_floor": self.on_floor,
        }


def make_cellar08_controller() -> Level6Cellar08Controller:
    """Cross cellar 0x08 from the 0x3A arrival mouth. Do not return."""
    return Level6Cellar08Controller()


def level6_cellar08_stages():
    """Warp 0x3A (dedicated predecessor), then cross cellar 0x08 A → B."""
    warp = make_stairs_3a_warp_controller()
    cellar = make_cellar08_controller()
    return (
        ("level6_stairs_0x3a_warp", warp, STAIRS_3A_WARP_MAX_FRAMES),
        ("level6_cellar_0x08", cellar, CELLAR_08_MAX_FRAMES),
    )


def level6_cellar08_glance(controller: Any) -> GlanceLeftover:
    """Exact 0x1D leftover glance after crossing cellar 0x08."""
    return grade_controller(controller, CELLAR08_LEAVE)


def level6_cellar08_success(snap: ZeldaSnapshot) -> bool:
    """Exact B endpoint 0x1D. A-side return 0x3A is a failure."""
    return l6_play_dest_success(
        snap,
        not_room=CELLAR_08_ROOM,
        dest_room=CELLAR_08_DEST_ROOM,
        passage_ok=False,
    )
