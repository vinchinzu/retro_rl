"""Damage attribution: which object took the heart, not which tile Link died on.

Every blocked residual in ``docs/tasks`` records the death *pose* — room,
``(128,181)``, mode 17, third time — and none records the killer. Without a
cause, the next sitting can only guess a new position rule, which is how a
stand line gets tuned three times and still dies.

``DamageLog`` watches the heart value and names the hazard responsible using
the previous frame's ``dungeon.tracking`` tracks, because the shot that
landed is often already despawned by the time the hearts change.

A hit is a heart drop *or* the rising edge of Link's invincibility timer
``$04F0``. Under the Survival refill the hearts are topped up before the
next snapshot, so a drop alone never shows: every engine room reported
``hits_by_cause {}`` while losing 10-13 hearts (Blue Ring power-on 9).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

from zelda_i.dungeon.tracking import HazardClass, TrackedObject
from zelda_i.ram import ZeldaSnapshot

__all__ = ("HitEvent", "DamageLog", "heart_value", "HIT_DEBOUNCE")

# Hurt-freeze plus invulnerability; two decrements closer than this are one hit.
HIT_DEBOUNCE = 8
DEATH_MODE = 17


def heart_value(snap: ZeldaSnapshot) -> int:
    """Filled hearts and the partial byte as one comparable number.

    ``$066F`` low nibble is whole hearts; a half-heart hit only moves the
    ``$0670`` partial, so whole hearts alone misses half the damage.
    """
    return ((int(snap.health) & 0x0F) << 8) + (int(snap.heart_partial) & 0xFF)


@dataclass(frozen=True)
class HitEvent:
    """One attributed loss of health."""

    frame: int
    link_xy: tuple[int, int]
    value_before: int
    value_after: int
    phase: str
    action: str
    slot: int | None = None
    type_id: int | None = None
    hazard: str = HazardClass.NONE.value
    kind: str = "unknown"
    cause_xy: tuple[int, int] | None = None
    cause_v: tuple[float, float] | None = None
    bearing: str = "?"
    distance: int | None = None
    closing: bool = False
    fatal: bool = False

    @property
    def label(self) -> str:
        """One line a residual can quote instead of a tile."""
        if self.type_id is None:
            return f"f{self.frame} unattributed while action={self.action}"
        vx, vy = self.cause_v or (0.0, 0.0)
        return (
            f"f{self.frame} {self.hazard} 0x{self.type_id:02x} ({self.kind}) "
            f"from {self.bearing} v=({vx:+.1f},{vy:+.1f}) d={self.distance} "
            f"while action={self.action} phase={self.phase}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "link_xy": list(self.link_xy),
            "phase": self.phase,
            "action": self.action,
            "slot": self.slot,
            "type_id": self.type_id,
            "hazard": self.hazard,
            "kind": self.kind,
            "cause_xy": list(self.cause_xy) if self.cause_xy else None,
            "cause_v": list(self.cause_v) if self.cause_v else None,
            "bearing": self.bearing,
            "distance": self.distance,
            "closing": self.closing,
            "fatal": self.fatal,
            "label": self.label,
        }


@dataclass
class DamageLog:
    """Attribute each heart loss to a tracked hazard.

    Feed it every frame alongside the tracker output. ``observe`` returns the
    ``HitEvent`` on the frame a hit lands, and ``None`` otherwise.
    """

    debounce: int = HIT_DEBOUNCE
    hits: list[HitEvent] = field(default_factory=list)
    death: HitEvent | None = None
    frames: int = 0
    _value: int | None = field(default=None, init=False, repr=False)
    _prev: tuple[TrackedObject, ...] = field(
        default=(), init=False, repr=False
    )
    _prev_xy: tuple[int, int] | None = field(
        default=None, init=False, repr=False
    )
    _last_hit: int = field(default=-10**6, init=False, repr=False)
    _last_snap: ZeldaSnapshot | None = field(
        default=None, init=False, repr=False
    )
    _room: tuple[int, int] | None = field(
        default=None, init=False, repr=False
    )
    _iframes: int = field(default=0, init=False, repr=False)

    def observe(
        self,
        snap: ZeldaSnapshot,
        tracked: tuple[TrackedObject, ...],
        *,
        action: str = "",
        phase: str = "",
    ) -> HitEvent | None:
        if snap is self._last_snap:
            return None
        self._last_snap = snap
        self.frames += 1
        room = (int(snap.level), int(snap.screen))
        if self._room is not None and room != self._room:
            # Same slot numbers, different objects. ObjectTracker already
            # drops history here; keeping `_prev` would blame the old room's
            # nearest hazard (advanced 1f) for a first-frame hit in the new
            # one — a heart drop in 0x79 reading as the 0x78 keese.
            self._prev = ()
            self._prev_xy = None
        self._room = room
        value = heart_value(snap)
        iframes = int(getattr(snap, "link_iframes", 0))
        armed = iframes > 0 and self._iframes == 0
        self._iframes = iframes
        event: HitEvent | None = None
        if (
            self._value is not None
            and (value < self._value or armed)
            and self.frames - self._last_hit > self.debounce
        ):
            event = self._attribute(snap, value, action=action, phase=phase)
            self.hits.append(event)
            self._last_hit = self.frames
        if int(snap.mode) == DEATH_MODE and self.death is None:
            self.death = self._fatal(snap, value, event, action=action, phase=phase)
        self._value = value
        self._prev = tracked
        self._prev_xy = (int(snap.link_x), int(snap.link_y))
        return event

    # --- internals -----------------------------------------------------

    def _attribute(
        self,
        snap: ZeldaSnapshot,
        value: int,
        *,
        action: str,
        phase: str,
    ) -> HitEvent:
        """Blame the previous-frame hazard closest to where Link now stands.

        Previous-frame tracks are used because a shot despawns on contact;
        each is advanced one frame so a fast shot is judged where it landed.
        """
        link = (int(snap.link_x), int(snap.link_y))
        best: TrackedObject | None = None
        best_rank = 10**9
        best_d = 0
        for track in self._prev:
            if not track.is_hazard:
                continue
            hx, hy = track.at(1.0)
            d = int(max(abs(hx - link[0]), abs(hy - link[1])))
            # A shot that reached Link outranks a body idling at equal range.
            # Receding shots do not get the nudge: flying away at d=8 would
            # otherwise rank as 4 and beat a body overlapping at 6. Ranking
            # only — the reported distance below stays the true gap.
            rank = (
                d - 4
                if track.hazard is HazardClass.PROJECTILE
                and track.closing_on(*link)
                else d
            )
            if rank < best_rank:
                best_rank, best_d, best = rank, d, track
        common = {
            "frame": self.frames,
            "link_xy": link,
            "value_before": self._value or value,
            "value_after": value,
            "phase": phase,
            "action": action,
        }
        if best is None:
            return HitEvent(**common)
        return HitEvent(
            **common,
            slot=best.slot,
            type_id=best.type_id,
            hazard=best.hazard.value,
            kind=best.kind.value,
            cause_xy=(best.x, best.y),
            cause_v=(best.vx, best.vy),
            bearing=best.approach_side(*link),
            distance=max(0, best_d),
            closing=best.closing_on(*link),
        )

    def _fatal(
        self,
        snap: ZeldaSnapshot,
        value: int,
        event: HitEvent | None,
        *,
        action: str,
        phase: str,
    ) -> HitEvent:
        """Mark the killing hit fatal, or record an honest unattributed one.

        Death always has *a* frame, even when no tracked hazard ever moved
        the heart value (health was already at the floor, or the game's own
        instant-kill path never showed up as a decrement). Silently leaving
        ``death`` as ``None`` here would hide a real death from ``report()``;
        ``_attribute`` already returns an unattributed ``HitEvent`` in the
        equivalent no-source case, so mirror that instead of a bare ``None``.
        """
        last = event or (self.hits[-1] if self.hits else None)
        if last is None:
            last = HitEvent(
                frame=self.frames,
                link_xy=(int(snap.link_x), int(snap.link_y)),
                value_before=self._value if self._value is not None else value,
                value_after=value,
                phase=phase,
                action=action,
            )
        fatal = replace(last, fatal=True)
        if self.hits and self.hits[-1] is last:
            self.hits[-1] = fatal
        return fatal

    # --- reporting -----------------------------------------------------

    def report(self) -> dict[str, Any]:
        causes: Counter[str] = Counter()
        for hit in self.hits:
            key = (
                f"0x{hit.type_id:02x}_{hit.bearing}"
                if hit.type_id is not None
                else "unattributed"
            )
            causes[key] += 1
        return {
            "hits": len(self.hits),
            "hits_by_cause": dict(causes.most_common()),
            "death_cause": self.death.label if self.death else None,
            "events": [hit.to_dict() for hit in self.hits],
        }
