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
from zelda_i.combat import (
    BOMB_DROP_OBJECT_TYPE,
    BOMB_DROP_STATES,
    overworld_threat_objects,
)
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
from zelda_i.walk.physics import OPPOSITE, OccupancyGrid, OccupancyWalker

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
# Restock scroll-out budget. Mid-screen to an edge is ~110f at 2px/f;
# 300 leaves room for one occupancy replan around a tree pocket.
LEAVE_MAX_FRAMES = 300
# Lane the leave walks to before it pushes the edge. Every overworld screen
# the spine crosses has a live y=141 corridor (that is how the hop crossed
# it), so this is an open cell to aim at, not a guess about 0x4A.
RESTOCK_LANE_Y = 141
RESTOCK_LANE_X = 120
# Edge cell per restock direction, on that lane.
LEAVE_GOALS: dict[str, tuple[int, int]] = {
    "LEFT": (8, RESTOCK_LANE_Y),
    "RIGHT": (248, RESTOCK_LANE_Y),
    "UP": (RESTOCK_LANE_X, 64),
    "DOWN": (RESTOCK_LANE_X, 216),
}
FARM_SWING_PERIOD = 8
# Occupancy arrival: a goal within this many px on both axes is reached.
ARRIVE_PX = 1
# Bomb top-up bar, and the budget one bomb slot may hold the farm for. A
# reachable contact drop is banked in well under a second; a slot still
# sitting there after this is the phantom, not a bomb.
BOMB_TOPUP_BELOW = 8
BOMB_CHASE_MAX_FRAMES = 120
FARM_SWING_HOLD = 3
WAYPOINT_TOL = 6

_CARDINALS = frozenset(OPPOSITE)
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


def owns_bombs(snap: ZeldaSnapshot) -> bool:
    """True when Link owns bombs at all — the gate on any bomb-drop branch.

    ``ZeldaSnapshot`` has no ``max_bombs`` field: ``ADDR_MAX_BOMBS`` (``$067C``)
    is defined in ``ram.py`` but never read into the snapshot, so ownership is
    proxied by a live count — and picked up from the real capacity the moment
    ram.py exposes it. Pre-L1 Link owns no bombs, which is exactly the leg
    (0x77 -> 0x4A) where every ``0x60``/state-``0x00`` slot is a phantom.
    """
    cap = getattr(snap, "max_bombs", None)
    if cap is not None and int(cap) > 0:
        return True
    return int(snap.bombs) > 0


def _can_take_bombs(snap: ZeldaSnapshot) -> bool:
    """Link owns bombs and is under the farm's top-up bar."""
    return owns_bombs(snap) and int(snap.bombs) < BOMB_TOPUP_BELOW


def _bomb_drops(snap: ZeldaSnapshot) -> tuple:
    """Bomb floor drops, or ``()`` when Link cannot bank one.

    Bomb is ROM item code ``0x00`` (``ids.BOMB_DROP_STATE``), so a bomb drop is
    the one drop whose ObjState is *identical* to a cleared object slot — every
    other drop helper in this package filters on a non-zero state for that
    reason. There is no second witness in the snapshot (type is ``0x60`` for
    every drop, hp is 0 for every drop, x/y go stale on pickup), so the
    ownership gate is what separates a real bomb from a phantom slot.
    """
    if not _can_take_bombs(snap):
        return ()
    return tuple(
        obj
        for obj in snap.objects
        if _in_drop_bounds(obj)
        and int(obj.type_id) == BOMB_DROP_OBJECT_TYPE
        and int(obj.state) in BOMB_DROP_STATES
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
class FarmOccupancy:
    """Bump-learning occupancy walk for one overworld screen.

    The overworld has no ``$6530`` tile map to measure walls from, so the grid
    starts empty and learns a bush or a rock by failing to advance into it
    (:func:`_advanced`): block that cell, drop the path, replan. No path at all
    means stand — a farm never wiggles.

    Held by both farms: the heart farm's patrol / chase / pickup walks, and the
    restock leave in either farm (a bare directional hold has no answer to a
    bush between Link and the screen edge).
    """

    walker: OccupancyWalker | None = None
    screen: int = -1
    frame: int = -1

    @property
    def misses(self) -> int:
        return 0 if self.walker is None else self.walker.misses

    @property
    def retargets(self) -> int:
        return 0 if self.walker is None else self.walker.retargets

    def reset(self) -> None:
        self.walker = None
        self.screen = -1
        self.frame = -1

    def grade(self, snap: ZeldaSnapshot, frames: int) -> OccupancyWalker:
        """Grade last frame's held button, then hand back the walker."""
        if self.walker is None or self.screen != int(snap.screen):
            self.walker = OccupancyWalker(grid=_ow_farm_grid())
            self.screen = int(snap.screen)
            self.frame = -1
        walker = self.walker
        xy = (int(snap.link_x), int(snap.link_y))
        if self.frame != frames - 1:
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

    def walk(
        self,
        snap: ZeldaSnapshot,
        frames: int,
        goal: tuple[int, int],
        reason: str,
        *,
        slash: bool = True,
        period: int = FARM_SWING_PERIOD,
        hold: int = FARM_SWING_HOLD,
    ) -> FrameAction:
        """Occupancy to ``goal``. Miss -> block -> replan; no path -> stand."""
        walker = self.grade(snap, frames)
        xy = (int(snap.link_x), int(snap.link_y))
        dest = (int(goal[0]), int(goal[1]))
        if walker.goal != dest:
            walker.path = None
            walker.goal = dest
        path = walker.grid.shortest_path(xy, dest)
        if path is None and not walker.grid.passable(*dest):
            # The goal cell itself is blocked, so ``shortest_path`` refuses it
            # (an in-bounds blocked goal returns None — it no longer hands back
            # a bogus path that walks into the wall). On this grid a block is
            # always *inferred*: one bump on the drop's own cell would strand
            # the farm on ``occupancy_stand`` until ``farm_screen_dead``, on a
            # drop Link could have walked up to. Aim at the nearest open cell —
            # contact pickup does not need the exact pixel.
            #
            # Not the ``next_dir`` forget path: that clears every inferred
            # block, and a walker on an empty overworld grid just re-learns the
            # same wall (``measured_walker``'s docstring; ``enter_6f_key``
            # burned 4,000f that way). A start that is fenced in — ``path is
            # None`` with a passable goal — still stands, which is what the
            # frozen-chase tests pin.
            open_dest = walker.grid.nearest_open(*dest)
            if open_dest is not None and open_dest != dest:
                walker.retargets += 1
                dest = open_dest
                walker.goal = dest
                walker.path = None
                path = walker.grid.shortest_path(xy, dest)
        # Link steps 1-2 px, so an exact-pixel stop on an odd goal overshoots
        # forever (live 0x79: y 121<->123 for ~1300 frames of flutter).
        arrived = max(abs(xy[0] - dest[0]), abs(xy[1] - dest[1])) <= ARRIVE_PX
        if path is None or arrived:
            walker.path = None
            walker.last_dir = None
            self.frame = frames
            return FrameAction(nes_idle_action(), "occupancy_stand")
        walker.path = path
        direction = walker.next_dir(xy, dest)
        if direction is None:
            walker.path = None
            walker.last_dir = None
            self.frame = frames
            return FrameAction(nes_idle_action(), "occupancy_stand")
        if slash:
            act = walk_or_swing(
                frames,
                direction,
                reason,
                snap,
                period=period,
                hold=hold,
            )
            walker.last_dir = _cardinal_from_action(act.action)
        else:
            act = FrameAction(nes_action(direction), reason)
            walker.last_dir = direction
        self.frame = frames
        return act


def leave_goal(direction: str) -> tuple[int, int] | None:
    """Edge cell on the open lane for a restock leave in ``direction``."""
    return LEAVE_GOALS.get(direction)


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
    bomb_chase_max_frames: int = BOMB_CHASE_MAX_FRAMES
    phase: HeartFarmPhase = HeartFarmPhase.FARM
    frames: int = 0
    waypoint_index: int = 0
    stuck: int = 0
    empty_frames: int = 0
    leaving: bool = False
    saw_prey: bool = False
    leave_frames: int = 0
    restocks: int = 0
    last_x: int = -1
    last_y: int = -1
    last_screen: int = -1
    success: bool = False
    notes: list[str] = field(default_factory=list)
    start_filled: int = -1
    peak_filled: int = 0
    bomb_chase: int = 0
    bomb_chase_bombs: int = -1
    _occ: FarmOccupancy = field(default_factory=FarmOccupancy, repr=False)

    @property
    def _walker(self) -> OccupancyWalker | None:
        return self._occ.walker

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
        self.leaving = False
        self.saw_prey = False
        self.leave_frames = 0
        self.restocks = 0
        self.last_x = -1
        self.last_y = -1
        self.last_screen = -1
        self.success = False
        self.notes.clear()
        self.start_filled = -1
        self.peak_filled = 0
        self.bomb_chase = 0
        self.bomb_chase_bombs = -1
        self._occ.reset()

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
            return OPPOSITE[direction]
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

    def _walk_to(
        self,
        snap: ZeldaSnapshot,
        goal: tuple[int, int],
        reason: str,
        *,
        slash: bool = True,
    ) -> FrameAction:
        """Occupancy to ``goal``. Miss -> block -> replan; no path -> stand."""
        return self._occ.walk(snap, self.frames, goal, reason, slash=slash)

    def _screen_dead(self, snap: ZeldaSnapshot) -> FrameAction:
        """Give up: a restock returned an empty screen.

        Measured on 0x4A (2026-09-12): once the opening wave is killed the
        screen stays empty through a depth-1 (0x49) *and* a depth-2
        (0x49-0x59-0x49) round trip. Looping the restock only burns the
        budget, so end on the same policy as the timeout.
        """
        self.notes.append(
            f"farm_screen_dead hearts={snap.filled_hearts}/{self.min_filled}"
        )
        if snap.filled_hearts >= self.min_filled:
            return self._set_done("farm_ok_dead")
        if snap.filled_hearts >= 2 and snap.filled_hearts >= self.start_filled:
            return self._set_done("farm_soft_ok")
        self.phase = HeartFarmPhase.FAILED
        self.success = False
        return FrameAction(nes_idle_action(), "farm_screen_dead")

    def _bomb_step(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """Scoop a bomb drop — but never let one hold the farm open.

        Bomb is ROM item code ``0x00``, indistinguishable from a cleared object
        slot (see :func:`_bomb_drops`), and this branch used to run *ahead* of
        the rupee scoop and reset ``empty_frames`` on every match: one phantom
        slot pinned the farm for the whole ``max_frames``. Three guards now:
        the branch is last, it never touches ``empty_frames`` (so the
        wait / leave / restock / give-up ladder keeps its schedule), and the
        chase is dropped once it stops resolving — a reachable contact drop is
        banked in a handful of frames, a phantom never is.
        """
        bombs = _bomb_drops(snap)
        if not bombs:
            self.bomb_chase = 0
            self.bomb_chase_bombs = -1
            return None
        if int(snap.bombs) != self.bomb_chase_bombs:
            self.bomb_chase = 0
            self.bomb_chase_bombs = int(snap.bombs)
        self.bomb_chase += 1
        if self.bomb_chase > self.bomb_chase_max_frames:
            return None
        nearest = min(
            bombs,
            key=lambda o: abs(o.x - snap.link_x) + abs(o.y - snap.link_y),
        )
        return self._walk_to(
            snap, (int(nearest.x), int(nearest.y)), "farm_bomb", slash=False
        )

    def _leave_step(self, snap: ZeldaSnapshot) -> FrameAction:
        """Hold the restock walk until the screen actually flips.

        One-frame ``farm_leave`` never scrolled: ``empty_frames`` reset the
        moment it fired, so the wait oscillation undid the single step and the
        screen never restocked (At4A min_filled=4: 1348 farm_wait / 15
        farm_leave, peak 3, zero respawns).
        """
        direction = self.restock_direction or "LEFT"
        self.leave_frames += 1
        if self.leave_frames > LEAVE_MAX_FRAMES:
            self.leaving = False
            self.leave_frames = 0
            self.notes.append(f"farm_leave_stalled_{snap.link_x}_{snap.link_y}")
            return FrameAction(nes_idle_action(), "farm_leave_stalled")
        goal = LEAVE_GOALS.get(direction)
        if goal is None:
            return FrameAction(nes_action(direction), "farm_leave")
        gx, gy = goal
        at_edge = (
            (direction == "LEFT" and snap.link_x <= gx)
            or (direction == "RIGHT" and snap.link_x >= gx)
            or (direction == "UP" and snap.link_y <= gy)
            or (direction == "DOWN" and snap.link_y >= gy)
        )
        if at_edge:
            return FrameAction(nes_action(direction), "farm_leave_push")
        return self._walk_to(snap, goal, "farm_leave", slash=False)

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
            if self.leaving:
                self.leaving = False
                self.leave_frames = 0
                self.restocks += 1
                self.notes.append(f"farm_restock_{self.restocks}")
            return FrameAction(
                nes_action(OPPOSITE[self.restock_direction or "LEFT"]),
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

        # Not while leaving: the enter guard pushes inward at x<36, which is
        # exactly where the restock walk has to finish.
        enter_dir = None if self.leaving else self._is_entering_screen(snap)
        if enter_dir is not None:
            return FrameAction(nes_action(enter_dir), "farm_enter")

        enemies = list(overworld_threat_objects(snap))
        if enemies:
            self.empty_frames = 0
            self.saw_prey = True
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

        # Last, and never resetting ``empty_frames``: see ``_bomb_step``.
        bomb = self._bomb_step(snap)
        if bomb is not None:
            return bomb

        if restock:
            if self.leaving:
                return self._leave_step(snap)
            self.empty_frames += 1
            if self.empty_frames < self.empty_wait_frames or _hold_for_forced_fairy(snap):
                direction = "RIGHT" if snap.link_x < 160 else "LEFT"
                return FrameAction(nes_action(direction), "farm_wait")
            self.empty_frames = 0
            if self.restocks >= 1 and not self.saw_prey:
                return self._screen_dead(snap)
            self.saw_prey = False
            self.leaving = True
            self.leave_frames = 0
            return self._leave_step(snap)

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
            "restocks": self.restocks,
            "saw_prey": self.saw_prey,
            "leaving": self.leaving,
            "occupancy_misses": self._occ.misses,
            "occupancy_retargets": self._occ.retargets,
            "notes": list(self.notes),
        }


# Pond fairies (``Z_04.asm`` ``UpdatePondFairy``). The OW room table holds
# object list 0x2F on two first-quest screens only, 0x39 and 0x43
# (``LevelBlockAttrsC``/``D``, read live). The fairy wakes when Link's Y is
# exactly $AD and his X is $70..$80, then fills every heart; nothing else on
# the screen matters. It is a full refill with no RAM write.
POND_FAIRY_TYPE = 0x2F
POND_SCREENS = frozenset({0x39, 0x43})
POND_EDGE_Y = 0xAD
POND_EDGE_X = 120  # inside $70..$80
POND_MAX_FRAMES = 900
# ``World_IsFillingHearts`` ran 1 -> 3 hearts in 95 frames (``OW_39``), then
# the fairy holds Link for $50 more.
POND_SETTLE_FRAMES = 16


@dataclass
class PondFairyController:
    """Stand on the pond edge until the fairy has filled every heart.

    Starts on a pond screen, south of the basin, in play. Lines up on
    ``POND_EDGE_X``, walks UP until Link's Y is ``POND_EDGE_Y``, then idles
    while the ROM fills. ``success`` is full health, read from RAM.
    """

    max_frames: int = POND_MAX_FRAMES
    frames: int = 0
    success: bool = False
    failed: bool = False
    notes: list[str] = field(default_factory=list)
    start_hearts: float = -1.0
    settle: int = 0
    _last_y: int = -1
    _still: int = 0

    def _fail(self, note: str) -> FrameAction:
        self.failed = True
        self.notes.append(note)
        return FrameAction(nes_idle_action(), note)

    def step(self, snap: ZeldaSnapshot) -> FrameAction:
        self.frames += 1
        if self.frames > self.max_frames:
            return self._fail("pond_timeout")
        if snap.mode != PLAY_MODE or snap.transitioning:
            return FrameAction(nes_idle_action(), "pond_wait_mode")
        if int(snap.screen) not in POND_SCREENS:
            return self._fail(f"pond_off_screen_{int(snap.screen):02x}")
        if self.start_hearts < 0:
            self.start_hearts = float(snap.whole_hearts)
        lx, ly = int(snap.link_x), int(snap.link_y)
        on_edge = ly == POND_EDGE_Y and 0x70 <= lx <= 0x80
        if on_edge or self.settle:
            self.settle += 1
            if snap.health_is_full and int(snap.heart_partial) == 0xFF:
                # The fairy halts Link (slot 0 ObjState $40) through the
                # fill and $50 frames after; done is the ROM letting go.
                if self.settle >= POND_SETTLE_FRAMES and int(snap.objects[0].state) == 0:
                    self.success = True
                    self.notes.append("pond_full")
                    return FrameAction(nes_idle_action(), "pond_done")
            elif self.settle > POND_MAX_FRAMES // 2:
                return self._fail("pond_no_fill")
            return FrameAction(nes_idle_action(), "pond_fill")
        if abs(lx - POND_EDGE_X) > 1:
            return FrameAction(
                nes_action("RIGHT" if lx < POND_EDGE_X else "LEFT"), "pond_align"
            )
        if ly > POND_EDGE_Y:
            return FrameAction(nes_action("UP"), "pond_up")
        # North of the edge row (the basin blocks this, but a knockback or
        # a grid snap can land there): step back down onto it.
        return FrameAction(nes_action("DOWN"), "pond_down")

    def report(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "frames": self.frames,
            "start_hearts": self.start_hearts,
            "settle": self.settle,
            "notes": list(self.notes),
        }
