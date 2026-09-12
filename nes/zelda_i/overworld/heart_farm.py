"""Clean overworld heart farming (no RAM health writes).

Patrol, swing, pick up drops. Overworld foes do not respawn until Link
leaves; a restock neighbor (kwargs or ``farm_at``) scrolls out and back.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from retro_harness.controls import NES_BUTTON_NAME_TO_INDEX
from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action
from zelda_i import combat as _combat
from zelda_i.combat import overworld_threat_objects
from zelda_i.dungeon.ids import (
    CLOCK_DROP_STATE,
    FAIRY_DROP_STATE,
    FIVE_RUPEE_DROP_STATE,
    RUPEE_DROP_OBJECT_TYPE,
    RUPEE_DROP_STATE,
)
from zelda_i.overworld.common import (
    track_stuck,
    wake_or_wait_mode,
    walk_or_swing,
)
from zelda_i.overworld.locations import farm_at
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot
from zelda_i.walk.physics import OccupancyGrid, OccupancyWalker

# Default patrol on 0x4A — mid horizontal corridor (y≈140) is open; south
# wall blocks y≳160 and north pockets need the channel at x≈16–64.
DEFAULT_4A_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (48, 141),
    (96, 141),
    (144, 141),
    (192, 141),
    (208, 125),
    (176, 109),
    (128, 125),
    (80, 141),
    (40, 125),
)

# Screen-agnostic sweep for the low-heart hook: the open mid band plus one
# north lane. Any overworld screen the spine walks has a live y≈141 corridor
# (that is how the hop crossed it), so this patrols where enemies already are.
BAND_SWEEP_WAYPOINTS: tuple[tuple[int, int], ...] = (
    (64, 141),
    (120, 141),
    (176, 141),
    (176, 109),
    (120, 109),
    (64, 109),
)

DEFAULT_MAX_FRAMES = 3600
DEFAULT_EMPTY_WAIT_FRAMES = 90
FARM_SWING_PERIOD = 8
FARM_SWING_HOLD = 3
WAYPOINT_TOL = 6

_OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}
_CARDINALS = frozenset(_OPPOSITE)
# Dungeon OccupancyGrid xmax=216 traps OW x≈240. True no-move is a miss;
# 1px OccupancyWalker.observe would block a 2px OW slide.
_OW_OCC_BOUNDS = (0, 255, 0, 239)


def _ow_farm_grid() -> OccupancyGrid:
    xmin, xmax, ymin, ymax = _OW_OCC_BOUNDS
    return OccupancyGrid(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


def _cardinal_from_action(action: Sequence[int]) -> str | None:
    """Return the single cardinal direction pressed in ``action``, or None."""
    pressed = [
        c
        for c in ("UP", "DOWN", "LEFT", "RIGHT")
        if NES_BUTTON_NAME_TO_INDEX.get(c) is not None
        and NES_BUTTON_NAME_TO_INDEX[c] < len(action)
        and action[NES_BUTTON_NAME_TO_INDEX[c]]
    ]
    return pressed[0] if len(pressed) == 1 else None


class HeartFarmPhase(Enum):
    FARM = auto()
    DONE = auto()
    FAILED = auto()


def _in_drop_bounds(obj) -> bool:
    return obj.slot >= 1 and 40 < obj.y < 220 and 8 < obj.x < 248


_RUPEE_STATES = frozenset(
    {RUPEE_DROP_STATE, FIVE_RUPEE_DROP_STATE, CLOCK_DROP_STATE}
)


def _advanced(last_xy: tuple[int, int], xy: tuple[int, int], last_dir: str) -> bool:
    """True if Link made forward progress along the requested axis."""
    if last_dir == "RIGHT":
        return xy[0] > last_xy[0]
    if last_dir == "LEFT":
        return xy[0] < last_xy[0]
    if last_dir == "DOWN":
        return xy[1] > last_xy[1]
    if last_dir == "UP":
        return xy[1] < last_xy[1]
    return False


def _rupee_drops(snap: ZeldaSnapshot) -> tuple:
    """1-rupee / 5-rupee floor drops. Type 0x60 is shared with hearts."""
    return tuple(
        obj
        for obj in snap.objects
        if _in_drop_bounds(obj)
        and int(obj.type_id) == RUPEE_DROP_OBJECT_TYPE
        and int(obj.state) in _RUPEE_STATES
    )


def _heart_drops(snap: ZeldaSnapshot) -> tuple:
    """Heart/fairy floor drops. Item identity is ObjState, not ObjType."""
    heart_fn = getattr(_combat, "heart_or_fairy_drops", None)
    if callable(heart_fn):
        return tuple(heart_fn(snap))
    pred = getattr(_combat, "is_heart_or_fairy_drop", None)
    if callable(pred):
        return tuple(obj for obj in snap.objects if pred(obj))
    states = getattr(_combat, "HEART_OR_FAIRY_STATES", None)
    if states:
        return tuple(
            obj
            for obj in snap.objects
            if _in_drop_bounds(obj)
            and int(obj.type_id) == RUPEE_DROP_OBJECT_TYPE
            and int(obj.state) in states
        )
    return ()


def _pickup_reason(obj) -> str:
    if int(obj.state) == FAIRY_DROP_STATE:
        return "farm_fairy"
    return "farm_heart"


def _hold_for_forced_fairy(snap: ZeldaSnapshot) -> bool:
    """Stay for the $0627 16-kill fairy when the counter is 14..15."""
    count = getattr(snap, "world_kill_count", None)
    if count is None:
        return False
    return 14 <= int(count) <= 15


@dataclass
class HeartFarmController:
    """Patrol a screen until ``filled_hearts >= min_filled`` (Clean combat only).

    Heart/fairy contact scoops before chase, then rupees, then restock/patrol.
    Occupancy miss (true no-move) → block that cell → replan; no path →
    stand. ``min_filled<=0`` is inert (the path-layer ``farm_below_hearts=0``
    analog). Never writes health.
    """

    min_filled: int = 3
    max_frames: int = DEFAULT_MAX_FRAMES
    farm_screen: int = 0x4A
    waypoints: tuple[tuple[int, int], ...] = DEFAULT_4A_WAYPOINTS
    restock_neighbor_screen: int | None = None
    restock_direction: str | None = None  # from farm_screen toward neighbor
    empty_wait_frames: int = DEFAULT_EMPTY_WAIT_FRAMES
    phase: HeartFarmPhase = HeartFarmPhase.FARM
    frames: int = 0
    waypoint_index: int = 0
    stuck: int = 0
    empty_frames: int = 0
    last_x: int = -1
    last_y: int = -1
    last_screen: int = -1
    success: bool = False
    notes: list[str] = field(default_factory=list)
    start_filled: int = -1
    peak_filled: int = 0
    _walker: OccupancyWalker | None = field(default=None, repr=False)
    _walker_screen: int = -1
    _walk_frame: int = field(default=-1, repr=False)

    def __post_init__(self) -> None:
        if self.restock_neighbor_screen is not None and self.restock_direction is not None:
            return
        spot = farm_at(self.farm_screen)
        if spot is None:
            return
        if self.restock_neighbor_screen is None:
            self.restock_neighbor_screen = spot.restock_neighbor
        if self.restock_direction is None:
            self.restock_direction = spot.restock_direction

    def reset(self) -> None:
        self.phase = HeartFarmPhase.FARM
        self.frames = 0
        self.waypoint_index = 0
        self.stuck = 0
        self.empty_frames = 0
        self.last_x = -1
        self.last_y = -1
        self.last_screen = -1
        self.success = False
        self.notes.clear()
        self.start_filled = -1
        self.peak_filled = 0
        self._walker = None
        self._walker_screen = -1
        self._walk_frame = -1

    def _has_restock(self) -> bool:
        return self.restock_neighbor_screen is not None and self.restock_direction is not None

    def _set_done(self, note: str) -> FrameAction:
        self.success = True
        self.phase = HeartFarmPhase.DONE
        if note and (not self.notes or self.notes[-1] != note):
            self.notes.append(note)
        return FrameAction(nes_idle_action(), "farm_done")

    def _set_failed(self, note: str) -> FrameAction:
        self.success = False
        self.phase = HeartFarmPhase.FAILED
        self.notes.append(note)
        return FrameAction(nes_idle_action(), note)

    def already_satisfied(self, snap: ZeldaSnapshot) -> bool:
        return snap.filled_hearts >= self.min_filled and self.min_filled > 0

    def _transition_direction(self, snap: ZeldaSnapshot) -> str:
        direction = self.restock_direction or "LEFT"
        returning_to_farm = (
            snap.screen == self.restock_neighbor_screen
            or snap.next_screen == self.farm_screen
        )
        if returning_to_farm:
            return _OPPOSITE[direction]
        return direction

    def _hearts_met(self, snap: ZeldaSnapshot) -> bool:
        return snap.filled_hearts >= self.min_filled

    def _is_entering_screen(self, snap: ZeldaSnapshot) -> str | None:
        """If Link is still on the transition edge boundary, return inward cardinal."""
        if snap.screen != self.farm_screen:
            return None
        direction = self.restock_direction or ("LEFT" if self.farm_screen == 0x4A else None)
        if direction == "LEFT" and snap.link_x < 36:
            return "RIGHT"
        if direction == "RIGHT" and snap.link_x > 220:
            return "LEFT"
        if direction == "UP" and snap.link_y < 70:
            return "DOWN"
        if direction == "DOWN" and snap.link_y > 200:
            return "UP"
        return None

    def _grade_occupancy(self, snap: ZeldaSnapshot) -> OccupancyWalker:
        if self._walker is None or self._walker_screen != int(snap.screen):
            self._walker = OccupancyWalker(grid=_ow_farm_grid())
            self._walker_screen = int(snap.screen)
            self._walk_frame = -1
        walker = self._walker
        xy = (int(snap.link_x), int(snap.link_y))
        if self._walk_frame != self.frames - 1:
            walker.last_dir = None
            walker.last_xy = None
        link_state = (
            snap.objects[0].state
            if snap.objects and snap.objects[0].slot == 0
            else 0
        )
        is_swinging = link_state != 0 or snap.mode != PLAY_MODE
        if is_swinging:
            walker.last_dir = None
            walker.last_xy = None
        elif walker.last_dir in _CARDINALS and walker.last_xy is not None:
            if not _advanced(walker.last_xy, xy, walker.last_dir):
                walker.grid.mark_blocked_ahead(*walker.last_xy, walker.last_dir)
                walker.path = None
                walker.misses += 1
        walker.last_xy = xy
        walker._graded = True
        return walker

    def _walk_to(
        self,
        snap: ZeldaSnapshot,
        goal: tuple[int, int],
        reason: str,
        *,
        slash: bool = True,
    ) -> FrameAction:
        """Occupancy to ``goal``. Miss → block → replan; no path → stand."""
        walker = self._grade_occupancy(snap)
        xy = (int(snap.link_x), int(snap.link_y))
        dest = (int(goal[0]), int(goal[1]))
        if walker.goal != dest:
            walker.path = None
            walker.goal = dest
        path = walker.grid.shortest_path(xy, dest)
        if path is None or xy == dest:
            walker.path = None
            walker.last_dir = None
            self._walk_frame = self.frames
            return FrameAction(nes_idle_action(), "occupancy_stand")
        walker.path = path
        direction = walker.next_dir(xy, dest)
        if direction is None:
            walker.path = None
            walker.last_dir = None
            self._walk_frame = self.frames
            return FrameAction(nes_idle_action(), "occupancy_stand")
        if slash:
            act = walk_or_swing(
                self.frames,
                direction,
                reason,
                snap,
                period=FARM_SWING_PERIOD,
                hold=FARM_SWING_HOLD,
            )
            walker.last_dir = _cardinal_from_action(act.action)
        else:
            act = FrameAction(nes_action(direction), reason)
            walker.last_dir = direction
        self._walk_frame = self.frames
        return act

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.start_filled < 0:
            self.start_filled = snap.filled_hearts
            self.peak_filled = snap.filled_hearts
        self.peak_filled = max(self.peak_filled, snap.filled_hearts)

        self.stuck, self.last_x, self.last_y, self.last_screen = track_stuck(
            snap,
            last_x=self.last_x,
            last_y=self.last_y,
            last_screen=self.last_screen,
            stuck=self.stuck,
        )

        if self.min_filled <= 0:
            return self._set_done("farm_skipped")

        if snap.mode == 17:
            return self._set_failed("link_death")

        if self.frames >= self.max_frames:
            if snap.filled_hearts >= self.min_filled:
                return self._set_done("farm_timeout_ok")
            if snap.filled_hearts >= 2 and snap.filled_hearts >= self.start_filled:
                self.notes.append(
                    f"farm_soft_ok hearts={snap.filled_hearts}<{self.min_filled}"
                )
                return self._set_done("farm_soft_ok")
            self.notes.append(
                f"farm_timeout hearts={snap.filled_hearts}/{self.min_filled}"
            )
            self.phase = HeartFarmPhase.FAILED
            self.success = False
            return FrameAction(nes_idle_action(), "farm_timeout")

        restock = self._has_restock()
        recovered = self._hearts_met(snap)
        if recovered and not restock:
            return self._set_done(
                f"farm_ok_{self.start_filled}_to_{snap.filled_hearts}"
            )

        if snap.transitioning or snap.mode not in (PLAY_MODE, 8):
            if restock and snap.transitioning:
                reason = "farm_return_scroll" if recovered else "farm_scroll"
                return FrameAction(nes_action(self._transition_direction(snap)), reason)
            if snap.mode not in (PLAY_MODE, 8, 6, 7, 16):
                return wake_or_wait_mode(self.frames, snap.mode)
            return FrameAction(nes_idle_action(), "farm_wait_mode")

        if snap.level != 0:
            return self._set_failed("left_overworld")

        # Restock: stay FARM on the neighbor and walk back. Any other screen
        # still soft-fails. Do not invent a neighbor when the pair is missing.
        if restock and snap.screen == self.restock_neighbor_screen:
            return FrameAction(
                nes_action(_OPPOSITE[self.restock_direction or "LEFT"]),
                "farm_respawn",
            )
        if snap.screen != self.farm_screen:
            self.notes.append(f"left_screen_{snap.screen:02x}")
            self.phase = HeartFarmPhase.FAILED
            self.success = False
            return FrameAction(nes_idle_action(), "left_farm_screen")

        if recovered:
            return self._set_done(
                f"farm_ok_{self.start_filled}_to_{snap.filled_hearts}"
            )

        hearts = _heart_drops(snap)
        if hearts:
            self.empty_frames = 0
            nearest = min(
                hearts,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            return self._walk_to(
                snap,
                (int(nearest.x), int(nearest.y)),
                _pickup_reason(nearest),
                slash=False,
            )

        enter_dir = self._is_entering_screen(snap)
        if enter_dir is not None:
            return FrameAction(nes_action(enter_dir), "farm_enter")

        enemies = list(overworld_threat_objects(snap))
        if enemies:
            self.empty_frames = 0
            nearest = min(
                enemies,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            return self._walk_to(snap, (int(nearest.x), int(nearest.y)), "farm_chase")

        drops = _rupee_drops(snap)
        if drops:
            self.empty_frames = 0
            nearest = min(
                drops,
                key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
            )
            return self._walk_to(snap, (int(nearest.x), int(nearest.y)), "farm_rupee")

        if restock:
            self.empty_frames += 1
            if self.empty_frames < self.empty_wait_frames or _hold_for_forced_fairy(snap):
                direction = "RIGHT" if snap.link_x < 160 else "LEFT"
                return FrameAction(nes_action(direction), "farm_wait")
            self.empty_frames = 0
            return FrameAction(nes_action(self.restock_direction or "LEFT"), "farm_leave")

        if not self.waypoints:
            return self._walk_to(
                snap, (min(240, int(snap.link_x) + 48), int(snap.link_y)), "farm_patrol"
            )

        tx, ty = self.waypoints[self.waypoint_index % len(self.waypoints)]
        if abs(snap.link_x - tx) <= WAYPOINT_TOL and abs(snap.link_y - ty) <= WAYPOINT_TOL:
            self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
            self.stuck = 0
            tx, ty = self.waypoints[self.waypoint_index % len(self.waypoints)]
        return self._walk_to(snap, (tx, ty), "farm")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "phase": self.phase.name,
            "frames": self.frames,
            "min_filled": self.min_filled,
            "farm_screen": self.farm_screen,
            "restock_neighbor_screen": self.restock_neighbor_screen,
            "restock_direction": self.restock_direction,
            "start_filled": self.start_filled,
            "peak_filled": self.peak_filled,
            "waypoint_index": self.waypoint_index,
            "occupancy_misses": 0 if self._walker is None else self._walker.misses,
            "notes": list(self.notes),
        }
