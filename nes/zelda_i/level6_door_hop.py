"""Shared L6 occupancy dest-hop controller.

Hops own leftover geometry via ``DoorHopSpec``. Observe + replan + stand;
do not fail the hop on occupancy miss (east3a only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.level6_occupancy import l6_play_dest_success, record_l6_walk
from zelda_i.level6_overworld import (
    LEVEL6,
    LEVEL6_DARK_29_ROOM,
    LEVEL6_DARK_39_ROOM,
    LEVEL6_GLEEOK_ROOM,
    LEVEL6_GOHMA_ROOM,
    LEVEL6_GOHMA_WING_1D_ROOM,
    LEVEL6_GOHMA_WING_2C_ROOM,
    LEVEL6_GOHMA_WING_2D_ROOM,
    LEVEL6_MAP_ROOM,
    LEVEL6_ROD_WIZZ_ROOM,
    LEVEL6_WIZZROBE_28_ROOM,
)
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk_physics import OccupancyGrid, OccupancyWalker

SOUTH_DOOR_X, SOUTH_DOOR_Y, SOUTH_BAND_Y, SOUTH_DOOR_TOL = 120, 189, 181, 4
EAST_DOOR_X, EAST_DOOR_Y, EAST_DOOR_TOL = 208, 141, 4
WEST_DOOR_X, WEST_DOOR_Y, WEST_SPAWN_XMIN = 32, 141, 16
NORTH_DOOR_X, NORTH_DOOR_Y, EAST_SPAWN_XMAX = 120, 93, 232
NORTH_HALT_Y, CLIP_Y, DOOR_TOL = 109, 141, 4
DOOR_HOP_MAX_FRAMES, SAMPLE_PERIOD = 4000, 12
CELLAR_MODE, DEATH_MODE = 9, 17
WAIT_SCROLL = (2, 3, 4, 6, 7)
WAIT_SCROLL_B = (2, 3, 4, 6, 7, 10, 16)
SOUTH09_MAX_FRAMES = SOUTH19_MAX_FRAMES = SOUTH29_MAX_FRAMES = DOOR_HOP_MAX_FRAMES
EAST29_MAX_FRAMES = EAST39_MAX_FRAMES = DOOR_HOP_MAX_FRAMES
WEST19_MAX_FRAMES = SOUTH18_MAX_FRAMES = SOUTH1D_MAX_FRAMES = WEST2D_MAX_FRAMES = NORTH2C_MAX_FRAMES = DOOR_HOP_MAX_FRAMES
_TAG = {"DOWN": "south", "RIGHT": "east", "LEFT": "west", "UP": "north"}

__all__ = [
    "CLIP_Y", "DOOR_HOP_MAX_FRAMES", "EAST29_MAX_FRAMES", "EAST39_MAX_FRAMES",
    "EAST29_SPEC", "EAST39_SPEC", "EAST_DOOR_TOL", "EAST_DOOR_X", "EAST_DOOR_Y",
    "EAST_SPAWN_XMAX", "NORTH2C_MAX_FRAMES", "NORTH2C_SPEC", "NORTH_DOOR_X",
    "NORTH_DOOR_Y", "NORTH_HALT_Y", "SOUTH09_MAX_FRAMES", "SOUTH09_SPEC",
    "SOUTH18_MAX_FRAMES", "SOUTH18_SPEC", "SOUTH19_MAX_FRAMES", "SOUTH19_SPEC",
    "SOUTH1D_MAX_FRAMES", "SOUTH1D_SPEC", "SOUTH29_MAX_FRAMES", "SOUTH29_SPEC",
    "SOUTH_BAND_Y", "SOUTH_DOOR_TOL", "SOUTH_DOOR_X", "SOUTH_DOOR_Y",
    "WEST19_MAX_FRAMES", "WEST19_SPEC", "WEST2D_MAX_FRAMES", "WEST2D_SPEC",
    "WEST_DOOR_X", "WEST_DOOR_Y", "WEST_SPAWN_XMIN", "DoorHopSpec",
    "Level6DoorHopController", "door_hop_stages", "door_hop_success",
]


@dataclass(frozen=True)
class DoorHopSpec:
    """Per-hop leftover geometry. Buttons stay on the spec, not the walker."""

    spec_id: str
    room: int
    goal: tuple[int, int]
    hold_dir: str
    policy: str
    dest_room: int | None = None
    wait_modes: tuple[int, ...] = WAIT_SCROLL
    max_frames: int = DOOR_HOP_MAX_FRAMES
    sample_period: int = SAMPLE_PERIOD
    door_tol: int = DOOR_TOL
    grid_xmin: int | None = None
    grid_xmax: int | None = None
    grid_ymin: int | None = None
    clip_y: int | None = None
    clip_buttons: tuple[str, ...] | None = None
    clip_side: str | None = None
    clip_reason: str = ""
    south_band: bool = False
    south_face: bool = False
    push_at_goal: bool = False
    cardinal_hold: bool = False
    align: str | None = None
    align_at: int | None = None
    north_halt_y: int | None = None
    north_halt_reason: str = ""
    forbid_up: bool = False
    forbid_up_y: int | None = None
    forbid_up_reason: str = "south_up_halt"
    forbid_down: bool = False
    stand_reason: str = ""
    fail_key_up: int | None = None
    fail_backtrack: int | None = None
    track_keys: bool = False
    fail_ow: bool = False
    key_from: str = ""


SOUTH09_SPEC = DoorHopSpec(
    "level6_south_0x09", LEVEL6_ROD_WIZZ_ROOM, (SOUTH_DOOR_X, SOUTH_DOOR_Y),
    "DOWN", "occupancy to (120,189) then DOWN; halt y<=109; dest is RAM",
    north_halt_y=NORTH_HALT_Y, north_halt_reason="south_north_halt",
    south_band=True,
)
SOUTH19_SPEC = DoorHopSpec(
    "level6_south_0x19", LEVEL6_MAP_ROOM, (SOUTH_DOOR_X, SOUTH_DOOR_Y),
    "DOWN", "occupancy to (120,189) then DOWN; never UP; dest is RAM",
    south_band=True, forbid_up=True,
)
SOUTH29_SPEC = DoorHopSpec(
    "level6_south_0x29", LEVEL6_DARK_29_ROOM, (SOUTH_DOOR_X, SOUTH_DOOR_Y),
    "DOWN", "RIGHT+DOWN clip off (55,133), occupancy x=120 @ y=141, then DOWN",
    clip_y=CLIP_Y, clip_buttons=("RIGHT", "DOWN"), clip_side="below",
    clip_reason="south_clip", south_band=True, south_face=True, align="x",
    align_at=CLIP_Y, forbid_up=True,
)
EAST29_SPEC = DoorHopSpec(
    "level6_east_0x29", LEVEL6_DARK_29_ROOM, (EAST_DOOR_X, EAST_DOOR_Y),
    "RIGHT", "RIGHT+DOWN clip off (55,133), occupancy y=141 RIGHT; dest is RAM",
    clip_y=CLIP_Y, clip_buttons=("RIGHT", "DOWN"), clip_side="below",
    clip_reason="east_clip", push_at_goal=True, align="y",
)
EAST39_SPEC = DoorHopSpec(
    "level6_east_0x39", LEVEL6_DARK_39_ROOM, (EAST_DOOR_X, EAST_DOOR_Y),
    "RIGHT", "RIGHT+UP clip off (136,173), cardinal RIGHT on y=141; dest is RAM",
    clip_y=CLIP_Y, clip_buttons=("RIGHT", "UP"), clip_side="above",
    clip_reason="east_clip", push_at_goal=True, cardinal_hold=True,
)
WEST19_SPEC = DoorHopSpec(
    "level6_west_0x19", LEVEL6_MAP_ROOM, (WEST_DOOR_X, WEST_DOOR_Y), "LEFT",
    "y=141 first, occupancy to (32,141), LEFT; halt y<=109; no KEY-UP 0x09; skip Map",
    dest_room=LEVEL6_GLEEOK_ROOM, wait_modes=WAIT_SCROLL_B,
    grid_xmin=WEST_SPAWN_XMIN, grid_ymin=NORTH_HALT_Y, push_at_goal=True,
    align="y", north_halt_y=NORTH_HALT_Y, north_halt_reason="north_key_halt",
    forbid_up=True, forbid_up_y=WEST_DOOR_Y, forbid_up_reason="north_key_halt",
    stand_reason="occupancy_stand", fail_key_up=LEVEL6_ROD_WIZZ_ROOM,
    fail_backtrack=LEVEL6_DARK_29_ROOM, track_keys=True, fail_ow=True,
    key_from="19",
)
SOUTH18_SPEC = DoorHopSpec(
    "level6_south_0x18", LEVEL6_GLEEOK_ROOM, (SOUTH_DOOR_X, SOUTH_DOOR_Y),
    "DOWN",
    "x-align occupancy (120,y) then DOWN (120,189); halt y<=109; "
    "no KEY-UP 0x09; no CheckWarp north hole; south clip LIVE-TBD",
    dest_room=LEVEL6_WIZZROBE_28_ROOM, wait_modes=WAIT_SCROLL_B,
    grid_ymin=NORTH_HALT_Y, south_band=True, align="x",
    north_halt_y=NORTH_HALT_Y, north_halt_reason="north_hole_halt",
    forbid_up=True, forbid_up_reason="north_hole_halt",
    stand_reason="occupancy_stand", fail_key_up=LEVEL6_ROD_WIZZ_ROOM,
    fail_backtrack=LEVEL6_MAP_ROOM, track_keys=True, fail_ow=True, key_from="18",
)
SOUTH1D_SPEC = DoorHopSpec(
    "level6_south_0x1d", LEVEL6_GOHMA_WING_1D_ROOM, (SOUTH_DOOR_X, SOUTH_DOOR_Y),
    "DOWN",
    "occupancy to (120,189) then DOWN from leftover (96,157); dest play 0x2d "
    "(120,77); keys stay 4; N/W/E wall; do not batch 0x2C/Gohma",
    dest_room=LEVEL6_GOHMA_WING_2D_ROOM, wait_modes=WAIT_SCROLL_B,
    grid_ymin=NORTH_HALT_Y, south_band=True,
    north_halt_y=NORTH_HALT_Y, north_halt_reason="south_north_halt",
    forbid_up=True, forbid_up_reason="south_north_halt",
    stand_reason="occupancy_stand", track_keys=True, fail_ow=True, key_from="1d",
)
WEST2D_SPEC = DoorHopSpec(
    "level6_west_0x2d", LEVEL6_GOHMA_WING_2D_ROOM, (WEST_DOOR_X, WEST_DOOR_Y),
    "LEFT",
    "y=141 first from leftover (120,77), occupancy to (32,141), LEFT; dest "
    "play 0x2c; keys stay 4; west is open; fail 0x1D/Gohma 0x1C",
    dest_room=LEVEL6_GOHMA_WING_2C_ROOM, wait_modes=WAIT_SCROLL_B,
    grid_xmin=WEST_SPAWN_XMIN, push_at_goal=True, align="y",
    forbid_up=True, forbid_up_y=NORTH_HALT_Y, forbid_up_reason="north_back_halt",
    stand_reason="occupancy_stand", fail_backtrack=LEVEL6_GOHMA_WING_1D_ROOM,
    track_keys=True, fail_ow=True, key_from="2d",
)
NORTH2C_SPEC = DoorHopSpec(
    "level6_north_0x2c", LEVEL6_GOHMA_WING_2C_ROOM, (NORTH_DOOR_X, NORTH_DOOR_Y),
    "UP",
    "x-align leftover (224,141) occupancy KEY-UP (120,93); dest play 0x1c; "
    "keys 4->3; fail 0x2D / south 0x3C; do not fight Gohma",
    dest_room=LEVEL6_GOHMA_ROOM, wait_modes=WAIT_SCROLL_B,
    grid_xmax=EAST_SPAWN_XMAX, push_at_goal=True, align="x",
    forbid_down=True, stand_reason="occupancy_stand",
    fail_backtrack=LEVEL6_GOHMA_WING_2D_ROOM, track_keys=True, fail_ow=True,
    key_from="2c",
)


def _walker(spec: DoorHopSpec) -> OccupancyWalker:
    kw: dict[str, int] = {}
    if spec.grid_xmin is not None:
        kw["xmin"] = spec.grid_xmin
    if spec.grid_xmax is not None:
        kw["xmax"] = spec.grid_xmax
    if spec.grid_ymin is not None:
        kw["ymin"] = spec.grid_ymin
    return OccupancyWalker(grid=OccupancyGrid(**kw)) if kw else OccupancyWalker()


def door_hop_stages(spec: DoorHopSpec):
    ctl = Level6DoorHopController(spec)
    return ((spec.spec_id, ctl, ctl.max_frames),)


def door_hop_success(spec: DoorHopSpec, snap: ZeldaSnapshot) -> bool:
    return l6_play_dest_success(
        snap, not_room=spec.room, dest_room=spec.dest_room, passage_ok=False
    )


@dataclass
class Level6DoorHopController:
    """Occupancy dest hop. Unique leftover geometry lives on ``spec``."""

    spec: DoorHopSpec
    frames: int = 0
    keys: int = -1
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(init=False)
    spec_id: str = field(init=False)
    room: int = field(init=False)
    dest: int | None = field(init=False)
    goal: tuple[int, int] = field(init=False)
    max_frames: int = field(init=False)

    def __post_init__(self) -> None:
        spec = self.spec
        self.spec_id = spec.spec_id
        self.room = spec.room
        self.dest = spec.dest_room
        self.goal = spec.goal
        self.max_frames = spec.max_frames
        self.walker = _walker(spec)

    def _tag(self) -> str:
        return _TAG[self.spec.hold_dir]

    def _emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            **record_l6_walk(
                self.samples, snap, reason=action.reason, frames=self.frames,
                period=self.spec.sample_period, misses=self.walker.misses,
                force=force,
            ),
            "map": int(snap.map),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
        }
        return action

    def _fail(
        self, snap: ZeldaSnapshot, note: str, reason: str | None = None
    ) -> FrameAction:
        self.failed = True
        if note not in self.notes:
            self.notes.append(note)
        return self._emit(
            snap, FrameAction(nes_idle_action(), reason or note), force=True
        )

    def _mark_success(self, snap: ZeldaSnapshot) -> FrameAction:
        spec = self.spec
        if spec.track_keys:
            if self.keys >= 0 and int(snap.keys) < self.keys:
                self.notes.append(
                    f"key_spent_{spec.key_from}_to_{snap.screen:02x}"
                    f"_{self.keys}->{int(snap.keys)}"
                )
            self.keys = int(snap.keys)
            note = (
                f"arrived_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                f"_rod={int(snap.rod)}_tf={snap.triforce:02x}_keys={int(snap.keys)}"
            )
        else:
            note = (
                f"arrived_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                f"_rod={int(snap.rod)}"
            )
        self.success = True
        self.notes.append(note)
        self.walker.last_dir = None
        return self._emit(
            snap, FrameAction(nes_idle_action(), f"arrived_{snap.screen:02x}"),
            force=True,
        )

    def _dest(self, snap: ZeldaSnapshot) -> FrameAction | None:
        spec = self.spec
        if snap.screen == spec.room:
            return None
        if snap.mode != PLAY_MODE or snap.transitioning or snap.rod == 0:
            return None
        xy = f"{snap.link_x}_{snap.link_y}"
        if spec.fail_key_up is not None and snap.screen == spec.fail_key_up:
            return self._fail(snap, f"key_up_09_{xy}_keys={int(snap.keys)}")
        if spec.fail_backtrack is not None and snap.screen == spec.fail_backtrack:
            return self._fail(
                snap, f"backtrack_{spec.fail_backtrack:02x}_{xy}"
            )
        if spec.dest_room is not None and snap.screen != spec.dest_room:
            return self._fail(snap, f"wrong_room_{snap.screen:02x}_{xy}")
        if l6_play_dest_success(
            snap, not_room=spec.room, dest_room=spec.dest_room, passage_ok=False
        ):
            return self._mark_success(snap)
        return None

    def _path_dest(self, xy: tuple[int, int]) -> tuple[int, int]:
        spec = self.spec
        gx, gy = spec.goal
        x, y = xy
        if spec.align == "x" and abs(x - gx) > spec.door_tol:
            return (gx, spec.align_at if spec.align_at is not None else y)
        if spec.align == "y" and abs(y - gy) > spec.door_tol:
            return (x, gy)
        return spec.goal

    def _walk(self, snap: ZeldaSnapshot) -> FrameAction:
        spec = self.spec
        xy = (int(snap.link_x), int(snap.link_y))
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")
        tag = self._tag()
        gx, gy = spec.goal
        tol = spec.door_tol
        if spec.north_halt_y is not None and xy[1] <= spec.north_halt_y:
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), spec.north_halt_reason)
            )
        at_push = False
        if spec.push_at_goal:
            if spec.hold_dir == "RIGHT" and abs(snap.link_y - gy) <= tol:
                at_push = snap.link_x >= gx - tol
            elif spec.hold_dir == "LEFT" and abs(snap.link_y - gy) <= tol:
                at_push = snap.link_x <= gx + tol
            elif spec.hold_dir == "UP" and abs(snap.link_x - gx) <= tol:
                at_push = snap.link_y <= gy + tol
        if at_push:
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_action(spec.hold_dir), f"{tag}_push")
            )
        clipping = False
        if spec.clip_buttons is not None and spec.clip_y is not None:
            if spec.clip_side == "below":
                clipping = xy[1] < spec.clip_y - tol
            elif spec.clip_side == "above":
                clipping = xy[1] > spec.clip_y + tol
        if clipping:
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_action(*spec.clip_buttons), spec.clip_reason)
            )
        if spec.south_band and xy[1] >= SOUTH_BAND_Y:
            self.walker.last_dir = None
            if abs(xy[0] - gx) > tol:
                horiz = "LEFT" if xy[0] > gx else "RIGHT"
                if spec.south_face:
                    return self._emit(
                        snap, FrameAction(nes_action(horiz, "UP"), "south_face")
                    )
                return self._emit(
                    snap, FrameAction(nes_action(horiz), "south_align")
                )
            return self._emit(snap, FrameAction(nes_action("DOWN"), "south_push"))
        if spec.hold_dir == "UP" and xy[1] <= NORTH_HALT_Y:
            self.walker.last_dir = None
            if abs(xy[0] - gx) > tol:
                horiz = "LEFT" if xy[0] > gx else "RIGHT"
                return self._emit(snap, FrameAction(nes_action(horiz), "north_align"))
            return self._emit(snap, FrameAction(nes_action("UP"), "north_push"))
        if spec.cardinal_hold:
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_action(spec.hold_dir), f"{tag}_hold")
            )
        dest = self._path_dest(xy)
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)
        if direction == "UP" and spec.forbid_up and (
            spec.forbid_up_y is None or xy[1] <= spec.forbid_up_y
        ):
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), spec.forbid_up_reason)
            )
        if direction == "DOWN" and spec.forbid_down:
            self.walker.last_dir = None
            return self._emit(
                snap, FrameAction(nes_idle_action(), "south_open_halt")
            )
        if direction is None:
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(f"stand_f{self.frames}_{xy[0]}_{xy[1]}")
            self.walker.last_dir = None
            reason = spec.stand_reason or f"{tag}_stand"
            return self._emit(snap, FrameAction(nes_idle_action(), reason))
        return self._emit(snap, FrameAction(nes_action(direction), f"{tag}_path"))

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        spec = self.spec
        self.frames += 1
        if spec.track_keys and self.keys < 0:
            self.keys = int(snap.keys)
        if self.success:
            return FrameAction(nes_idle_action(), "done")
        if self.failed or self.frames >= self.max_frames:
            self.failed = True
            extra = f"_keys={int(snap.keys)}" if spec.track_keys else ""
            if not any(n.startswith("timeout") for n in self.notes):
                self.notes.append(
                    f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                    f"_mode={snap.mode}_rod={int(snap.rod)}{extra}"
                )
            return self._emit(
                snap, FrameAction(nes_idle_action(), "timeout"), force=True
            )
        if snap.mode == DEATH_MODE:
            return self._fail(snap, "link_death")
        if snap.mode == CELLAR_MODE:
            note = f"warped_cellar_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            return self._fail(snap, note, None if spec.fail_ow else "warped_cellar")
        if spec.fail_ow:
            if snap.level == 0:
                return self._fail(
                    snap, f"ow_early_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                )
            if snap.level != LEVEL6:
                return self._fail(
                    snap, f"left_level_{snap.level}_{snap.screen:02x}"
                )
        arrived = self._dest(snap)
        if arrived is not None:
            return arrived
        if snap.transitioning or snap.mode in spec.wait_modes:
            self.walker.last_dir = None
            return FrameAction(nes_action(spec.hold_dir), f"{self._tag()}_scroll")
        if snap.mode != PLAY_MODE:
            self.walker.last_dir = None
            return FrameAction(nes_idle_action(), f"wait_mode_{snap.mode}")
        if not spec.fail_ow and snap.level != LEVEL6:
            return self._fail(snap, f"left_level_{snap.level}", "left_level")
        if snap.screen != spec.room:
            self.walker.last_dir = None
            return FrameAction(nes_action(spec.hold_dir), f"{self._tag()}_settle")
        return self._walk(snap)

    def report(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "success": self.success, "failed": self.failed, "frames": self.frames,
            "notes": list(self.notes), "samples": list(self.samples),
            "policy": self.spec.policy, "leftover": dict(self.leftover),
            "misses": self.walker.misses, "blocked": len(self.walker.grid.blocked),
            "spec_id": self.spec_id, "room": self.room, "goal": self.goal,
        }
        if self.spec.dest_room is not None or self.spec.track_keys:
            out["dest"] = self.dest
            out["keys"] = self.keys
        return out
