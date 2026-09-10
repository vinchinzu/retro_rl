"""Shared dungeon door-hop row engines.

Two row tables, both riding ``HopController``; neither holds a room number.

* ``DoorHopSpec`` / ``DoorHopController`` -- occupancy-BFS dest hops.  L6's
  ten generic door hops are rows.  The occupancy success predicate, the walk
  recorder and the level number are injected, so the engine is not L6's.
* ``RoomHopSpec`` / ``RoomHopController`` -- one-frame cardinal step hops.
  A row owns ``(origin, dest, door, step, done_reason)`` plus its fail rooms;
  L8's interior gates are rows.  A room whose gate needs a whole novel policy
  (the two cellar crosses) keeps that policy verbatim and hands it over as
  ``policy_fn``; the lifecycle still comes from here.

Geometry stays in the level module that measured it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i.dungeon.hop_controller import (
    CELLAR_MODE,
    HopController,
    WAIT_SCROLL,
    WAIT_SCROLL_B,
)
from zelda_i.ram import PASSAGE_MODE, PLAY_MODE, ZeldaSnapshot
from zelda_i.walk.physics import OccupancyGrid, OccupancyWalker

SOUTH_BAND_Y = 181
NORTH_HALT_Y = 109
DOOR_TOL = 4
DOOR_HOP_MAX_FRAMES = 4000
SAMPLE_PERIOD = 12
_TAG = {"DOWN": "south", "RIGHT": "east", "LEFT": "west", "UP": "north"}

__all__ = [
    "DOOR_HOP_MAX_FRAMES",
    "DOOR_TOL",
    "NORTH_HALT_Y",
    "SAMPLE_PERIOD",
    "SOUTH_BAND_Y",
    "DoorHopController",
    "DoorHopSpec",
    "HopFail",
    "RoomHopController",
    "RoomHopSpec",
    "door_band_goal",
    "door_hop_stages",
    "door_hop_success",
    "hop_leftover",
    "play_dest_success",
    "record_walk",
]


# --------------------------------------------------------------------------
# Occupancy dest hop (L6's ten rows)
# --------------------------------------------------------------------------


def door_band_goal(
    hold_dir: str,
    leftover: tuple[int, int],
    default_goal: tuple[int, int],
    *,
    tol: int = DOOR_TOL,
    north_band_y: int = NORTH_HALT_Y,
    south_band_y: int = SOUTH_BAND_Y,
) -> tuple[int, int]:
    """Occupancy dest from leftover + hold_dir; ``default_goal`` is the door mouth.

    Off-column leftover uses the door column, not leftover x (UP into a wall at x=208).
    """
    gx, gy = default_goal
    x, y = leftover
    if hold_dir == "UP":
        dest_x = x if abs(x - gx) <= tol else gx
        return (dest_x, north_band_y)
    if hold_dir == "DOWN":
        dest_x = x if abs(x - gx) <= tol else gx
        return (dest_x, south_band_y)
    dest_y = y if abs(y - gy) <= tol else gy
    return (gx, dest_y)


def hop_leftover(snap: ZeldaSnapshot) -> dict[str, Any]:
    """x/y/mode/screen/tile/keys/bombs/magic_key/triforce from a snap."""
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


def play_dest_success(
    snap: ZeldaSnapshot,
    *,
    not_room: int,
    passage_ok: bool = True,
    dest_room: int | None = None,
) -> bool:
    """Settled play on ``dest_room`` (or any room but ``not_room``)."""
    if dest_room is not None:
        return (
            snap.mode == PLAY_MODE
            and not snap.transitioning
            and snap.screen == dest_room
        )
    if passage_ok and snap.mode == PASSAGE_MODE:
        return True
    return (
        snap.mode == PLAY_MODE
        and not snap.transitioning
        and snap.screen != not_room
    )


def record_walk(
    samples: list[dict[str, Any]],
    snap: ZeldaSnapshot,
    *,
    reason: str,
    frames: int,
    period: int,
    misses: int,
    force: bool = False,
) -> dict[str, int]:
    """Default walk recorder: leftover plus a sample on the cadence."""
    if force or frames <= 2 or frames % period == 0:
        samples.append(
            {
                "frame": frames,
                "x": int(snap.link_x),
                "y": int(snap.link_y),
                "mode": int(snap.mode),
                "screen": int(snap.screen),
                "reason": reason,
                "tile": int(snap.colliding_tile),
                "misses": misses,
            }
        )
    return hop_leftover(snap)


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
    # Injected level bindings: the engine holds no level constants.
    level: int | None = None
    success_fn: Callable[..., bool] = play_dest_success
    record_fn: Callable[..., dict[str, int]] = record_walk
    south_band_y: int = SOUTH_BAND_Y
    north_band_y: int = NORTH_HALT_Y


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
    ctl = DoorHopController(spec)
    return ((spec.spec_id, ctl, ctl.max_frames),)


def door_hop_success(spec: DoorHopSpec, snap: ZeldaSnapshot) -> bool:
    return spec.success_fn(
        snap, not_room=spec.room, dest_room=spec.dest_room, passage_ok=False
    )


@dataclass
class DoorHopController(HopController):
    """Occupancy dest hop. Unique leftover geometry lives on ``spec``."""

    spec: DoorHopSpec
    keys: int = -1
    samples: list[dict[str, Any]] = field(default_factory=list)
    leftover: dict[str, int] = field(default_factory=dict)
    walker: OccupancyWalker = field(init=False)
    room: int = field(init=False)
    dest: int | None = field(init=False)
    goal: tuple[int, int] = field(init=False)
    _goal_bound: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        spec = self.spec
        self.spec_id = spec.spec_id
        self.room = spec.room
        self.dest = spec.dest_room
        self.goal = spec.goal
        self.max_frames = spec.max_frames
        self.wait_modes = spec.wait_modes
        self.walker = _walker(spec)

    def _tag(self) -> str:
        return _TAG[self.spec.hold_dir]

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        self.leftover = {
            **self.spec.record_fn(
                self.samples, snap, reason=action.reason, frames=self.frames,
                period=self.spec.sample_period, misses=self.walker.misses,
                force=force,
            ),
            "map": int(snap.map),
            "cur_opened_doors": int(snap.cur_opened_doors),
            "open_doorway_mask": int(snap.open_doorway_mask),
        }
        return action

    def timeout_note(self, snap: ZeldaSnapshot) -> str:
        extra = f"_keys={int(snap.keys)}" if self.spec.track_keys else ""
        return (
            f"timeout_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            f"_mode={snap.mode}_rod={int(snap.rod)}{extra}"
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        self.walker.last_dir = None
        return FrameAction(nes_action(self.spec.hold_dir), f"{self._tag()}_scroll")

    def _fail(
        self, snap: ZeldaSnapshot, note: str, reason: str | None = None
    ) -> FrameAction:
        del snap
        return self.mark_fail(note, reason)

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
        self.walker.last_dir = None
        self.done_reason = f"arrived_{snap.screen:02x}"
        return self.mark_done(snap, note)

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
        if spec.success_fn(
            snap, not_room=spec.room, dest_room=spec.dest_room, passage_ok=False
        ):
            return self._mark_success(snap)
        return None

    def _bind_goal(self, xy: tuple[int, int]) -> None:
        if self._goal_bound:
            return
        spec = self.spec
        self.goal = door_band_goal(
            spec.hold_dir,
            xy,
            spec.goal,
            tol=spec.door_tol,
            north_band_y=spec.north_band_y,
            south_band_y=spec.south_band_y,
        )
        self._goal_bound = True

    def _path_dest(self, xy: tuple[int, int]) -> tuple[int, int]:
        spec = self.spec
        gx, gy = self.goal
        x, y = xy
        align = spec.align
        if align is None and spec.hold_dir in ("UP", "DOWN"):
            align = "x"
        if align is None and spec.hold_dir in ("LEFT", "RIGHT"):
            align = "y"
        if align == "x" and abs(x - gx) > spec.door_tol:
            return (gx, spec.align_at if spec.align_at is not None else y)
        if align == "y" and abs(y - gy) > spec.door_tol:
            return (x, gy)
        return self.goal

    def _idle(self, reason: str) -> FrameAction:
        self.walker.last_dir = None
        return FrameAction(nes_idle_action(), reason)

    def _halt(self, snap: ZeldaSnapshot, xy: tuple[int, int]) -> FrameAction | None:
        spec = self.spec
        if spec.north_halt_y is not None and xy[1] <= spec.north_halt_y:
            return self._idle(spec.north_halt_reason)
        return None

    def _clip(self, snap: ZeldaSnapshot, xy: tuple[int, int]) -> FrameAction | None:
        spec = self.spec
        if spec.clip_buttons is None or spec.clip_y is None:
            return None
        tol = spec.door_tol
        if spec.clip_side == "below":
            clipping = xy[1] < spec.clip_y - tol
        elif spec.clip_side == "above":
            clipping = xy[1] > spec.clip_y + tol
        else:
            clipping = False
        if not clipping:
            return None
        self.walker.last_dir = None
        return FrameAction(nes_action(*spec.clip_buttons), spec.clip_reason)

    def _south_band(
        self, snap: ZeldaSnapshot, xy: tuple[int, int]
    ) -> FrameAction | None:
        spec = self.spec
        if not spec.south_band or xy[1] < spec.south_band_y:
            return None
        self.walker.last_dir = None
        gx, tol = spec.goal[0], spec.door_tol
        if abs(xy[0] - gx) > tol:
            horiz = "LEFT" if xy[0] > gx else "RIGHT"
            if spec.south_face:
                return FrameAction(nes_action(horiz, "UP"), "south_face")
            return FrameAction(nes_action(horiz), "south_align")
        return FrameAction(nes_action("DOWN"), "south_push")

    def _hold(self, snap: ZeldaSnapshot, xy: tuple[int, int]) -> FrameAction | None:
        spec = self.spec
        gx, gy = spec.goal
        tol = spec.door_tol
        tag = self._tag()
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
            return FrameAction(nes_action(spec.hold_dir), f"{tag}_push")
        if spec.hold_dir == "UP" and xy[1] <= spec.north_band_y:
            self.walker.last_dir = None
            if abs(xy[0] - gx) > tol:
                horiz = "LEFT" if xy[0] > gx else "RIGHT"
                return FrameAction(nes_action(horiz), "north_align")
            return FrameAction(nes_action("UP"), "north_push")
        if spec.cardinal_hold:
            self.walker.last_dir = None
            if spec.align == "y" and abs(xy[1] - gy) > tol:
                vert = "DOWN" if xy[1] < gy else "UP"
                return FrameAction(nes_action(vert), f"{tag}_align")
            if spec.align == "x" and abs(xy[0] - gx) > tol:
                horiz = "LEFT" if xy[0] > gx else "RIGHT"
                return FrameAction(nes_action(horiz), f"{tag}_align")
            return FrameAction(nes_action(spec.hold_dir), f"{tag}_hold")
        return None

    def _occupancy(
        self, snap: ZeldaSnapshot, xy: tuple[int, int]
    ) -> FrameAction | None:
        spec = self.spec
        dest = self._path_dest(xy)
        if dest != self.walker.goal:
            self.walker.path = None
            self.walker.goal = dest
        direction = self.walker.next_dir(xy, dest)
        if direction == "UP" and spec.forbid_up and (
            spec.forbid_up_y is None or xy[1] <= spec.forbid_up_y
        ):
            return self._idle(spec.forbid_up_reason)
        if direction == "DOWN" and spec.forbid_down:
            return self._idle("south_open_halt")
        if direction is None:
            if self.frames <= 8 or self.frames % 60 == 0:
                self.notes.append(f"stand_f{self.frames}_{xy[0]}_{xy[1]}")
            return self._idle(spec.stand_reason or f"{self._tag()}_stand")
        return FrameAction(nes_action(direction), f"{self._tag()}_path")

    def _walk(self, snap: ZeldaSnapshot) -> FrameAction:
        xy = (int(snap.link_x), int(snap.link_y))
        self._bind_goal(xy)
        prev_dir = self.walker.last_dir
        misses_before = self.walker.misses
        self.walker.observe(xy)
        if self.walker.misses > misses_before and (
            self.walker.misses <= 8 or self.frames % 60 == 0
        ):
            self.notes.append(f"miss_f{self.frames}_{prev_dir}_{xy[0]}_{xy[1]}")
        for fn in (self._halt, self._clip, self._south_band, self._hold, self._occupancy):
            action = fn(snap, xy)
            if action is not None:
                return action
        return self._idle(f"{self._tag()}_stand")

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        spec = self.spec
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        if snap.mode == CELLAR_MODE:
            note = f"warped_cellar_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
            return self._fail(snap, note, None if spec.fail_ow else "warped_cellar")
        if spec.fail_ow:
            if snap.level == 0:
                return self._fail(
                    snap, f"ow_early_{snap.screen:02x}_{snap.link_x}_{snap.link_y}"
                )
            if spec.level is not None and snap.level != spec.level:
                return self._fail(
                    snap, f"left_level_{snap.level}_{snap.screen:02x}"
                )
        arrived = self._dest(snap)
        if arrived is not None:
            return arrived
        waited = self.wait_not_play(snap)
        if waited is not None:
            self.walker.last_dir = None
            return waited
        if not spec.fail_ow and spec.level is not None and snap.level != spec.level:
            return self._fail(snap, f"left_level_{snap.level}", "left_level")
        if snap.screen != spec.room:
            self.walker.last_dir = None
            return FrameAction(nes_action(spec.hold_dir), f"{self._tag()}_settle")
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        return self._walk(snap)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        if self.spec.track_keys and self.keys < 0:
            self.keys = int(snap.keys)
        return super().step(snap)

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


# --------------------------------------------------------------------------
# One-frame cardinal room hop (L8's interior gates)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HopFail:
    """One guard rule. ``note`` may format ``{screen}`` (e.g. ``0x{screen:02x}``)."""

    rooms: tuple[int, ...]
    note: str
    on_passage: bool = False
    play_only: bool = False


@dataclass(frozen=True)
class RoomHopSpec:
    """One interior gate: origin, door, the measured one-frame step, fails."""

    spec_id: str
    origin: int
    door: str
    done_reason: str
    step: Callable[[ZeldaSnapshot], FrameAction] | None = None
    policy_fn: Callable[[Any, ZeldaSnapshot], FrameAction] | None = None
    level: int | None = None
    max_frames: int = DOOR_HOP_MAX_FRAMES
    wait_modes: tuple[int, ...] = WAIT_SCROLL_B
    sample_period: int = SAMPLE_PERIOD
    leftover_fn: Callable[[ZeldaSnapshot], dict[str, Any]] = hop_leftover
    fails: tuple[HopFail, ...] = ()
    require_play_arrival: bool = True
    passage_arrival: bool = False
    arrive_any: bool = False
    arrive_note: str = "play_0x{screen:02x}_{x}_{y}"
    unexpected_note: str = "unexpected_play_0x{screen:02x}"
    unexpected_play_only: bool = True
    scroll_button: str | None = None
    scroll_reason: str = "wait_scroll"
    settle_button: str | None = None
    settle_reason: str = ""
    passage_hold_reason: str = ""
    report_extra: tuple[tuple[str, Any], ...] = ()

    @property
    def blocked_rooms(self) -> frozenset[int]:
        """Rooms that never count as an arrival: the union of the fail rooms."""
        return frozenset(room for rule in self.fails for room in rule.rooms)


@dataclass(kw_only=True)
class RoomHopController(HopController):
    """Leftover-sampling dest hop driven by one ``RoomHopSpec`` row."""

    spec: RoomHopSpec
    dest: int | None = None
    route_eligible: bool = False
    leftover: dict[str, Any] = field(default_factory=dict)
    writes: int = 0

    def __post_init__(self) -> None:
        spec = self.spec
        self.spec_id = spec.spec_id
        self.max_frames = spec.max_frames
        self.wait_modes = spec.wait_modes
        self.done_reason = spec.done_reason
        self.require_level = spec.level

    @property
    def stage_id(self) -> str:
        return self.spec_id

    def arrived(self, snap: ZeldaSnapshot) -> bool:
        spec = self.spec
        if snap.transitioning:
            return False
        if spec.require_play_arrival and snap.mode != PLAY_MODE:
            return False
        if snap.screen in spec.blocked_rooms:
            return False
        if self.dest is not None:
            return snap.screen == self.dest
        if spec.passage_arrival:
            if snap.mode == PLAY_MODE and snap.screen == spec.origin:
                return False
            if snap.mode == PASSAGE_MODE:
                return True
            if snap.mode == PLAY_MODE and snap.screen != spec.origin:
                return True
            return False
        if spec.arrive_any:
            return True
        return snap.screen != spec.origin

    def on_arrive(self, snap: ZeldaSnapshot) -> str:
        return self.spec.arrive_note.format(
            screen=int(snap.screen),
            x=int(snap.link_x),
            y=int(snap.link_y),
            mode=int(snap.mode),
        )

    def scroll_action(self, snap: ZeldaSnapshot) -> FrameAction:
        del snap
        spec = self.spec
        if spec.scroll_button is None:
            return FrameAction(nes_idle_action(), spec.scroll_reason)
        return FrameAction(nes_action(spec.scroll_button), spec.scroll_reason)

    def emit(
        self, snap: ZeldaSnapshot, action: FrameAction, *, force: bool = False
    ) -> FrameAction:
        if force or not self.leftover or self.frames % self.spec.sample_period == 0:
            self.leftover = self.spec.leftover_fn(snap)
        return action

    def guard(self, snap: ZeldaSnapshot) -> FrameAction | None:
        spec = self.spec
        blocked = HopController.guard(self, snap)
        if blocked is not None:
            return blocked
        screen = int(snap.screen)
        for rule in spec.fails:
            if rule.play_only and snap.mode != PLAY_MODE:
                continue
            hit = screen in rule.rooms or (
                rule.on_passage and snap.mode == PASSAGE_MODE
            )
            if hit:
                return self.mark_fail(rule.note.format(screen=screen))
        if (
            spec.unexpected_note
            and self.dest is not None
            and (not spec.unexpected_play_only or snap.mode == PLAY_MODE)
            and not snap.transitioning
            and screen != spec.origin
            and screen != self.dest
        ):
            return self.mark_fail(spec.unexpected_note.format(screen=screen))
        return None

    def policy(self, snap: ZeldaSnapshot) -> FrameAction:
        spec = self.spec
        if spec.policy_fn is not None:
            return spec.policy_fn(self, snap)
        if spec.passage_hold_reason and snap.mode == PASSAGE_MODE:
            return FrameAction(nes_idle_action(), spec.passage_hold_reason)
        waited = self.wait_not_play(snap)
        if waited is not None:
            return waited
        if snap.screen != spec.origin:
            if spec.settle_button is None:
                return FrameAction(nes_idle_action(), spec.settle_reason)
            return FrameAction(
                nes_action(spec.settle_button), spec.settle_reason
            )
        assert spec.step is not None
        return spec.step(snap)

    def report(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "success": self.success,
            "failed": self.failed,
            "frames": self.frames,
            "notes": list(self.notes),
            "spec_id": self.spec_id,
            "dest_screen": self.dest,
            "evidence": "fixture-live",
            "route_eligible": self.route_eligible,
            "natural_entry": False,
            "writes": int(self.writes),
            "door": self.spec.door,
            "leftover": dict(self.leftover),
        }
        out.update(self.spec.report_extra)
        return out
