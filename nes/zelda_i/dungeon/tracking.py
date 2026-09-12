"""Frame-to-frame object tracking for Zelda I combat policies.

Room policies only ever saw a single frame of `snap.objects`, so every
controller that needed motion re-derived it locally (``level5/path.py``,
``level5/dungeon.py``, ``level6/wizzrobe.py``, ``level6/gohma.py``,
``level8/magic_key.py`` each keep their own ``prev_xy`` / ``gx_hist``).
This module is the one place that turns slots into tracks with velocity,
so ``dungeon.threat`` can reason about *when* something arrives instead of
where it is right now.

Hazard class is motion-first on purpose. The L6 wizzrobe beam is type
``0x59``, which is in no ``dungeon.ids`` projectile table; a small HP-0
slot moving 3 px/frame is a shot no matter what its type byte says.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from zelda_i.combat import FLOOR_DROP_TYPES
from zelda_i.dungeon.behaviors import (
    EnemyKind,
    is_projectile,
    kind_for_type,
    shield_blocks,
    uses_type_only_liveness,
)
from zelda_i.ram import ZeldaObject, ZeldaSnapshot

__all__ = (
    "HazardClass",
    "TrackedObject",
    "ObjectTracker",
    "PROJECTILE_SPEED",
    "RESPAWN_JUMP",
    "TRACK_HISTORY",
)

# Samples kept per slot. Zelda sprites move on a 2-4 frame cadence, so a
# 2-sample difference is mostly zeros; 6 samples average that out without
# lagging a direction change by more than ~3 frames.
TRACK_HISTORY = 6
# Above this speed an HP-0 slot is a shot even when its type is unknown.
PROJECTILE_SPEED = 1.6
# A slot that teleports this far in one frame is a new object, not motion.
RESPAWN_JUMP = 32
_EMPTY_TYPES = frozenset({0, 0xFF})


class HazardClass(str, Enum):
    """What a slot does to Link on contact."""

    BODY = "body"          # walks / hops; contact damage, sword-killable
    PROJECTILE = "shot"    # travels in a line; contact damage, not killable
    DROP = "drop"          # floor pickup; never damages
    NONE = "none"          # empty or dead slot


@dataclass(frozen=True)
class TrackedObject:
    """One slot plus the motion the tracker measured for it."""

    slot: int
    type_id: int
    x: int
    y: int
    vx: float
    vy: float
    hp: int
    state: int
    facing: int
    age: int
    kind: EnemyKind
    hazard: HazardClass
    blockable: bool

    @property
    def speed(self) -> float:
        return max(abs(self.vx), abs(self.vy))

    @property
    def moving(self) -> bool:
        return self.vx != 0.0 or self.vy != 0.0

    @property
    def is_hazard(self) -> bool:
        return self.hazard in (HazardClass.BODY, HazardClass.PROJECTILE)

    def at(self, frames: float) -> tuple[float, float]:
        """Linear extrapolation of this track ``frames`` ahead."""
        return (self.x + self.vx * frames, self.y + self.vy * frames)

    def closing_on(self, x: int, y: int) -> bool:
        """True when the track's velocity shortens the gap to ``(x, y)``."""
        if not self.moving:
            return False
        now = abs(self.x - x) + abs(self.y - y)
        nx, ny = self.at(4.0)
        return abs(nx - x) + abs(ny - y) < now

    def approach_side(self, x: int, y: int) -> str:
        """Compass side this track came at ``(x, y)`` from.

        Velocity first: a shot that already reached Link sits on top of him,
        so its position says nothing while its heading still names the side
        it was fired from.
        """
        if self.moving:
            if abs(self.vx) >= abs(self.vy):
                return "W" if self.vx > 0 else "E"
            return "N" if self.vy > 0 else "S"
        dx = self.x - int(x)
        dy = self.y - int(y)
        if abs(dx) >= abs(dy):
            return "E" if dx > 0 else "W"
        return "S" if dy > 0 else "N"


def _is_empty(type_id: int) -> bool:
    return (int(type_id) & 0xFF) in _EMPTY_TYPES


def _hazard_class(
    obj: ZeldaObject,
    speed: float,
    kind: EnemyKind,
    *,
    alive_hp: bool,
    age: int,
) -> HazardClass:
    """Classify a slot by type table first, then by motion.

    Motion is the fallback because shot types are discovered per lane and
    never make it into a shared table in time. The live L6 census had
    ``0x59`` beams carrying ``hp=128``, so HP cannot be the test — an
    unrecognised type travelling at shot speed is a shot. Age-1 unknown
    slots have speed 0; calling them bodies is how a point-blank 0x59
    spawn still reports ``body 0x59``. Motion confirms next frame.
    """
    type_id = int(obj.type_id) & 0xFF
    if _is_empty(type_id):
        return HazardClass.NONE
    if type_id in FLOOR_DROP_TYPES:
        return HazardClass.DROP
    if is_projectile(obj):
        return HazardClass.PROJECTILE
    dead = int(obj.hp) <= 0 and not alive_hp
    untyped = kind is EnemyKind.UNKNOWN
    # No samples yet: do not default an unknown slot to BODY. Next frame
    # the speed either confirms a shot or reclassifies a walker.
    if untyped and (speed >= PROJECTILE_SPEED or age <= 1):
        return HazardClass.PROJECTILE
    if speed >= PROJECTILE_SPEED and dead:
        return HazardClass.PROJECTILE
    if dead:
        return HazardClass.NONE  # corpse
    return HazardClass.BODY


class ObjectTracker:
    """Slot-keyed position history → per-frame ``TrackedObject`` tuple.

    Identity is ``(slot, type_id)``. A type change or a teleport bigger than
    ``RESPAWN_JUMP`` restarts the track so a respawn never reads as a
    200 px/frame missile. A ``(level, screen)`` change also restarts every
    track: two different rooms reuse the same slot numbers for unrelated
    objects, and a same-type object landing within ``RESPAWN_JUMP`` of the
    previous room's last position would otherwise read as continuous motion
    across the door.
    """

    def __init__(self, history: int = TRACK_HISTORY) -> None:
        self.history = max(2, int(history))
        self.frames = 0
        self._xs: dict[int, list[tuple[int, int]]] = {}
        self._types: dict[int, int] = {}
        self._ages: dict[int, int] = {}
        self._link: list[tuple[int, int]] = []
        self._last_snap: ZeldaSnapshot | None = None
        self._last: tuple[TrackedObject, ...] = ()
        self._room: tuple[int, int] | None = None

    # --- observation ---------------------------------------------------

    def observe(self, snap: ZeldaSnapshot) -> tuple[TrackedObject, ...]:
        """Ingest one frame. Re-observing the same snapshot is a no-op.

        Controllers nest (``super().step`` inside an override), so the same
        frame can reach the tracker twice; the identity guard keeps velocity
        from being sampled at double rate.
        """
        if snap is self._last_snap:
            return self._last
        self._last_snap = snap
        self.frames += 1
        room = (int(snap.level), int(snap.screen))
        if self._room is not None and room != self._room:
            # A new room reuses slot numbers for unrelated objects. Without
            # this, a same-type object that happens to land within
            # RESPAWN_JUMP of the previous room's last position in the same
            # slot reads as one continuous track across the door.
            self._xs.clear()
            self._types.clear()
            self._ages.clear()
        self._room = room
        self._push(self._link, (int(snap.link_x), int(snap.link_y)))

        seen: set[int] = set()
        tracked: list[TrackedObject] = []
        for obj in snap.objects:
            slot = int(obj.slot)
            if slot < 1:
                continue
            seen.add(slot)
            tracked.append(self._track(obj))
        for stale in set(self._xs) - seen:
            self._drop(stale)
        self._last = tuple(tracked)
        return self._last

    def _track(self, obj: ZeldaObject) -> TrackedObject:
        slot = int(obj.slot)
        type_id = int(obj.type_id) & 0xFF
        xy = (int(obj.x), int(obj.y))
        hist = self._xs.get(slot)
        if (
            hist is None
            or self._types.get(slot) != type_id
            or max(abs(xy[0] - hist[-1][0]), abs(xy[1] - hist[-1][1]))
            > RESPAWN_JUMP
        ):
            hist = []
            self._xs[slot] = hist
            self._ages[slot] = 0
        self._types[slot] = type_id
        self._push(hist, xy)
        self._ages[slot] += 1
        vx, vy = _velocity(hist)
        speed = max(abs(vx), abs(vy))
        kind = kind_for_type(type_id)
        hazard = _hazard_class(
            obj,
            speed,
            kind,
            alive_hp=uses_type_only_liveness(obj),
            age=self._ages[slot],
        )
        return TrackedObject(
            slot=slot,
            type_id=type_id,
            x=xy[0],
            y=xy[1],
            vx=vx,
            vy=vy,
            hp=int(obj.hp),
            state=int(obj.state),
            facing=int(obj.facing),
            age=self._ages[slot],
            kind=kind,
            hazard=hazard,
            blockable=hazard is HazardClass.PROJECTILE and shield_blocks(obj),
        )

    def _push(self, hist: list[tuple[int, int]], xy: tuple[int, int]) -> None:
        hist.append(xy)
        del hist[: -self.history]

    def _drop(self, slot: int) -> None:
        self._xs.pop(slot, None)
        self._types.pop(slot, None)
        self._ages.pop(slot, None)

    # --- queries -------------------------------------------------------

    @property
    def link_velocity(self) -> tuple[float, float]:
        return _velocity(self._link)

    def hazards(
        self,
        tracked: tuple[TrackedObject, ...] | None = None,
    ) -> tuple[TrackedObject, ...]:
        source = self._last if tracked is None else tracked
        return tuple(t for t in source if t.is_hazard)

    def projectiles(
        self,
        tracked: tuple[TrackedObject, ...] | None = None,
    ) -> tuple[TrackedObject, ...]:
        source = self._last if tracked is None else tracked
        return tuple(t for t in source if t.hazard is HazardClass.PROJECTILE)

    def by_slot(self, slot: int) -> TrackedObject | None:
        return next((t for t in self._last if t.slot == int(slot)), None)


def _velocity(hist: list[tuple[int, int]]) -> tuple[float, float]:
    """Mean px/frame across the kept samples (0 with fewer than two)."""
    if len(hist) < 2:
        return (0.0, 0.0)
    span = len(hist) - 1
    return (
        (hist[-1][0] - hist[0][0]) / span,
        (hist[-1][1] - hist[0][1]) / span,
    )
