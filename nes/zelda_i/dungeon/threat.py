"""Time-to-contact threat model and the reactive evade tactic.

Every blocked Clean room in ``docs/tasks/rr-npv.*`` failed the same way: the
policy knew *where* things were and answered with a per-room position table
(``x==116 -> LEFT``, ``STAND_Y=181``, ``y>=173 -> UP``). Those tables have no
notion of when a hazard arrives, so they oscillate on the stand line
(L8 ``0x1E`` 128<->112) or walk back under a landing body (L5 ``0x77``).

This module answers the other question: given ``dungeon.tracking`` velocity,
how many frames until something touches Link if he stands, and which single
button buys the most frames. It has no room constants — geometry comes in as
``bounds`` / ``blocked_dirs`` from the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from zelda_i.walk.physics import OPPOSITE
from zelda_i.combat import direction_to_facing
from zelda_i.dungeon.tracking import HazardClass, TrackedObject
from zelda_i.ram import ZeldaSnapshot

__all__ = (
    "Impact",
    "EvadeDecision",
    "ReactiveEvader",
    "DEFAULT_HORIZON",
    "TRIGGER_TTC",
    "FIRING_BAND",
    "MIN_DODGE_BODY",
    "MIN_DODGE_SHOT",
    "MIN_ESCAPE_GAIN",
    "CONTACT_NOW",
    "dodgeable",
    "contact_frames",
    "assess",
    "firing_axis",
    "in_firing_line",
    "off_line_step",
    "STEPS",
)

# Link walks ~1 px/frame; a step button therefore buys ~1 px of separation
# per frame of horizon. Anything shorter than ~8 frames of warning cannot be
# out-walked, which is why the trigger sits above it.
LINK_SPEED = 1.0
LINK_HALF = 8
BODY_HALF = 8
SHOT_HALF = 4
# A sidestep only clears a hitbox once Link has walked its full pad, so a
# dodge needs at least this many frames of warning. Below it the honest
# answers are sword, shield, or having stood somewhere else — which is why
# the L8 0x1E stand-line peel could not have worked at any tuning.
MIN_DODGE_BODY = LINK_HALF + BODY_HALF
MIN_DODGE_SHOT = LINK_HALF + SHOT_HALF
DEFAULT_HORIZON = 32
# Just above the body threshold: react while a sidestep can still finish.
TRIGGER_TTC = MIN_DODGE_BODY + 2
# Candidates within this many frames of the best are treated as equal, so
# hysteresis and goal progress decide instead of ±1 frame of arithmetic noise.
TTC_TIE = 2
COMMIT_FRAMES = 10
# A step that only postpones the hit is oscillation fuel, not an escape.
MIN_ESCAPE_GAIN = 4
# At or below this the hit is already landing; peel to shorten contact
# instead of standing in it.
CONTACT_NOW = 1
# Facing a blockable shot only helps if there is room to turn first.
SHIELD_MIN_DISTANCE = 24

# A shooter hits anything sharing its axis within roughly a tile.
FIRING_BAND = 12
_FACING_AXIS = {0x08: "col", 0x04: "col", 0x01: "row", 0x02: "row"}
_FACING_SIGN = {0x08: -1, 0x04: 1, 0x01: 1, 0x02: -1}

STAND = "STAND"
STEPS: dict[str, tuple[int, int]] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


@dataclass(frozen=True)
class Impact:
    """When the first hazard reaches Link, and which one."""

    frames: int
    horizon: int
    source: TrackedObject | None = None

    @property
    def safe(self) -> bool:
        return self.source is None or self.frames > self.horizon

    @property
    def imminent(self) -> bool:
        return not self.safe

    def within(self, ttc: int) -> bool:
        return not self.safe and self.frames <= int(ttc)


@dataclass(frozen=True)
class EvadeDecision:
    """One frame of advice. ``direction`` None means hold still."""

    direction: str | None
    reason: str
    ttc: int
    stand_ttc: int
    source_slot: int | None = None
    source_type: int | None = None
    committed: bool = False
    shield: bool = False

    @property
    def stands(self) -> bool:
        return self.direction is None


def _half(hazard: TrackedObject) -> int:
    return SHOT_HALF if hazard.hazard is HazardClass.PROJECTILE else BODY_HALF


def contact_frames(
    link_xy: tuple[int, int],
    hazard: TrackedObject,
    *,
    horizon: int = DEFAULT_HORIZON,
    step: tuple[int, int] = (0, 0),
    speed: float = LINK_SPEED,
    bounds: tuple[int, int, int, int] | None = None,
) -> int:
    """Frames until ``hazard`` overlaps Link, or ``horizon + 1`` if never.

    Both bodies travel in straight lines: Link along ``step`` at ``speed``,
    the hazard along its tracked velocity. A body that is already touching
    Link returns 0.
    """
    lx, ly = float(link_xy[0]), float(link_xy[1])
    pad = LINK_HALF + _half(hazard)
    sx, sy = step
    for t in range(horizon + 1):
        px = lx + sx * speed * t
        py = ly + sy * speed * t
        if bounds is not None:
            x_lo, x_hi, y_lo, y_hi = bounds
            px = min(max(px, x_lo), x_hi)
            py = min(max(py, y_lo), y_hi)
        hx, hy = hazard.at(float(t))
        if abs(hx - px) < pad and abs(hy - py) < pad:
            return t
    return horizon + 1


def assess(
    link_xy: tuple[int, int],
    hazards: tuple[TrackedObject, ...],
    *,
    horizon: int = DEFAULT_HORIZON,
    step: tuple[int, int] = (0, 0),
    speed: float = LINK_SPEED,
    bounds: tuple[int, int, int, int] | None = None,
) -> Impact:
    """Earliest contact across ``hazards`` for one candidate step."""
    best = horizon + 1
    source: TrackedObject | None = None
    for hazard in hazards:
        if not hazard.is_hazard:
            continue
        t = contact_frames(
            link_xy,
            hazard,
            horizon=horizon,
            step=step,
            speed=speed,
            bounds=bounds,
        )
        if t < best:
            best, source = t, hazard
            if best == 0:
                break
    return Impact(frames=best, horizon=horizon, source=source)


def dodgeable(impact: Impact, *, speed: float = LINK_SPEED) -> bool:
    """True when a sidestep can still clear the hitbox in time.

    ``False`` means no amount of position tuning saves this frame: the room
    policy has to answer with the sword, the shield, or a stand cell that was
    never on the line.
    """
    if impact.safe or impact.source is None:
        return True
    pad = LINK_HALF + _half(impact.source)
    return impact.frames * speed >= pad


def firing_axis(body: TrackedObject) -> str | None:
    """``"row"`` / ``"col"``: the line ``body`` shoots along, from its facing.

    Wizzrobes fire the way they face; Gohma fires down its column. Both are
    known one frame *before* the shot slot exists, which is the only window
    where a 1 px/frame walker can still get out of the way.
    """
    return _FACING_AXIS.get(int(body.facing) & 0x0F)


def in_firing_line(
    link_xy: tuple[int, int],
    body: TrackedObject,
    *,
    band: int = FIRING_BAND,
) -> bool:
    """True when Link shares ``body``'s firing axis on the side it faces."""
    axis = firing_axis(body)
    if axis is None:
        return False
    sign = _FACING_SIGN.get(int(body.facing) & 0x0F, 0)
    if axis == "row":
        if abs(body.y - link_xy[1]) > band:
            return False
        return (link_xy[0] - body.x) * sign > 0
    if abs(body.x - link_xy[0]) > band:
        return False
    return (link_xy[1] - body.y) * sign > 0


def off_line_step(
    link_xy: tuple[int, int],
    bodies: tuple[TrackedObject, ...],
    *,
    band: int = FIRING_BAND,
    bounds: tuple[int, int, int, int] | None = None,
    blocked_dirs: frozenset[str] | set[str] | tuple[str, ...] = (),
) -> str | None:
    """Step that leaves every shooter's firing line, or ``None`` if clear.

    This is the pre-emptive half of the policy: dodging a live shot at
    1 px/frame rarely gains a frame (see ``EvadeDecision`` ``evade_no_gain``),
    so the win is standing off the axis before it is fired.
    """
    shooters = tuple(
        b for b in bodies if b.is_hazard and in_firing_line(link_xy, b, band=band)
    )
    if not shooters:
        return None
    banned = frozenset(d.upper() for d in blocked_dirs)
    ranked: list[tuple[int, int, str]] = []
    for name, (sx, sy) in STEPS.items():
        if name in banned:
            continue
        x = link_xy[0] + sx * (band + 1)
        y = link_xy[1] + sy * (band + 1)
        if bounds is not None:
            x_lo, x_hi, y_lo, y_hi = bounds
            if not (x_lo <= x <= x_hi and y_lo <= y <= y_hi):
                continue
        clear = sum(
            1 for b in shooters if not in_firing_line((x, y), b, band=band)
        )
        if not clear:
            continue
        # Clearing the most lines is worth nothing in a corner: the L6 0x78
        # ROM trial stepped east off one wizzrobe row into (189,149) where
        # four 0x59 beams converged. Break the tie toward open floor.
        ranked.append((-clear, -_interior(x, y, bounds), name))
    if not ranked:
        return None
    return min(ranked)[2]


def _interior(x: int, y: int, bounds: tuple[int, int, int, int] | None) -> int:
    """Distance to the nearest bound: how much room the step leaves."""
    if bounds is None:
        return 0
    x_lo, x_hi, y_lo, y_hi = bounds
    return min(x - x_lo, x_hi - x, y - y_lo, y_hi - y)


@dataclass
class ReactiveEvader:
    """Pick the button that buys the most frames before the next hit.

    The caller owns the fight: ``decide`` returns ``None`` whenever standing
    still is already safe, so the room policy keeps driving. It only speaks
    when something is inbound inside ``trigger_ttc``.

    Commitment is the anti-oscillation half. A chosen escape is held for
    ``commit_frames`` while it stays within ``TTC_TIE`` of the best option,
    and the reverse of the last step is only taken when it is *clearly*
    better. Without those two rules the search reproduces the L8 ``0x1E``
    128<->112 stand-line loop exactly.
    """

    horizon: int = DEFAULT_HORIZON
    trigger_ttc: int = TRIGGER_TTC
    commit_frames: int = COMMIT_FRAMES
    bounds: tuple[int, int, int, int] | None = None
    speed: float = LINK_SPEED
    shield: bool = True
    # Pre-emptive: step off a shooter's axis while nothing is in flight yet.
    avoid_firing_lines: bool = False
    firing_band: int = FIRING_BAND
    min_gain: int = MIN_ESCAPE_GAIN
    evades: int = 0
    off_line_steps: int = 0
    _commit_dir: str | None = field(default=None, init=False, repr=False)
    _commit_left: int = field(default=0, init=False, repr=False)
    _last_dir: str | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        self._commit_dir = None
        self._commit_left = 0
        self._last_dir = None

    def decide(
        self,
        snap: ZeldaSnapshot,
        tracked: tuple[TrackedObject, ...],
        *,
        goal: tuple[int, int] | None = None,
        blocked_dirs: frozenset[str] | set[str] | tuple[str, ...] = (),
        allow: tuple[str, ...] = ("UP", "DOWN", "LEFT", "RIGHT"),
    ) -> EvadeDecision | None:
        """Advice for this frame, or ``None`` when standing is safe."""
        hazards = tuple(t for t in tracked if t.is_hazard)
        link = (int(snap.link_x), int(snap.link_y))
        stand = self._impact(link, hazards, (0, 0))
        if not stand.within(self.trigger_ttc):
            self._decay()
            return self._off_line(link, hazards, blocked_dirs)

        block = self._shield_face(snap, link, stand)
        if block is not None:
            return block

        banned = frozenset(d.upper() for d in blocked_dirs)
        options: dict[str, tuple[int, Impact]] = {}
        for name in allow:
            if name in banned or name not in STEPS:
                continue
            if not self._can_move(link, STEPS[name]):
                continue
            impact = self._impact(link, hazards, STEPS[name])
            options[name] = (impact.frames, impact)
        if not options:
            self._commit(None)
            return self._stand_decision(stand, "evade_boxed_in")

        best = max(v[0] for v in options.values())
        decisive = best > self.horizon or best - stand.frames >= self.min_gain
        if not decisive and stand.frames <= CONTACT_NOW:
            peel = self._peel(link, stand, options)
            if peel is not None:
                return peel
        if not decisive:
            # Either nothing buys time, or the best step only postpones the
            # hit by a frame or two — which is how a stand line starts
            # shuffling. Hold, and let the room policy answer with the sword,
            # the shield, or a stand cell that was never on the line.
            self._commit(None)
            return self._stand_decision(stand, "evade_no_gain")

        choice = self._choose(options, best, link, goal)
        self._commit(choice)
        self.evades += 1
        frames, impact = options[choice]
        return EvadeDecision(
            direction=choice,
            reason="evade_commit" if self._commit_left > 1 else "evade",
            ttc=frames,
            stand_ttc=stand.frames,
            source_slot=stand.source.slot if stand.source else None,
            source_type=stand.source.type_id if stand.source else None,
            committed=choice == self._commit_dir,
        )

    # --- internals -----------------------------------------------------

    def _impact(
        self,
        link: tuple[int, int],
        hazards: tuple[TrackedObject, ...],
        step: tuple[int, int],
    ) -> Impact:
        return assess(
            link,
            hazards,
            horizon=self.horizon,
            step=step,
            speed=self.speed,
            bounds=self.bounds,
        )

    def _can_move(self, link: tuple[int, int], step: tuple[int, int]) -> bool:
        """False when the step only presses Link into the room bound."""
        if self.bounds is None:
            return True
        x_lo, x_hi, y_lo, y_hi = self.bounds
        x = link[0] + step[0]
        y = link[1] + step[1]
        return x_lo <= x <= x_hi and y_lo <= y <= y_hi

    def _choose(
        self,
        options: dict[str, tuple[int, Impact]],
        best: int,
        link: tuple[int, int],
        goal: tuple[int, int] | None,
    ) -> str:
        near_best = [
            name for name, (frames, _) in options.items() if frames >= best - TTC_TIE
        ]
        if (
            self._commit_left > 0
            and self._commit_dir in near_best
        ):
            return str(self._commit_dir)
        pool = near_best
        reverse = OPPOSITE.get(self._last_dir or "")
        if reverse in pool:
            others = [name for name in pool if name != reverse]
            # Turning around is how a stand line oscillates. Only do it when
            # the reverse clearly beats every other option, not on a tie.
            if others and options[reverse][0] <= max(
                options[name][0] for name in others
            ) + TTC_TIE:
                pool = others
        return min(pool, key=lambda name: self._rank(name, options, link, goal))

    def _rank(
        self,
        name: str,
        options: dict[str, tuple[int, Impact]],
        link: tuple[int, int],
        goal: tuple[int, int] | None,
    ) -> tuple[int, int, str]:
        frames = options[name][0]
        step = STEPS[name]
        if goal is None:
            progress = 0
        else:
            here = abs(goal[0] - link[0]) + abs(goal[1] - link[1])
            there = abs(goal[0] - (link[0] + step[0] * 8)) + abs(
                goal[1] - (link[1] + step[1] * 8)
            )
            progress = there - here
        return (-frames, progress, name)

    def _shield_face(
        self,
        snap: ZeldaSnapshot,
        link: tuple[int, int],
        stand: Impact,
    ) -> EvadeDecision | None:
        """Turn into a blockable shot instead of out-running it."""
        source = stand.source
        if not self.shield or source is None or not source.blockable:
            return None
        if abs(source.x - link[0]) + abs(source.y - link[1]) < SHIELD_MIN_DISTANCE:
            return None
        # ``approach_side`` names where the shot came from; block by facing it.
        toward = {"E": "RIGHT", "W": "LEFT", "N": "UP", "S": "DOWN"}[
            source.approach_side(link[0], link[1])
        ]
        if int(snap.facing) == direction_to_facing(toward):
            return EvadeDecision(
                direction=None,
                reason="shield_hold",
                ttc=stand.frames,
                stand_ttc=stand.frames,
                source_slot=source.slot,
                source_type=source.type_id,
                shield=True,
            )
        return EvadeDecision(
            direction=toward,
            reason="shield_turn",
            ttc=stand.frames,
            stand_ttc=stand.frames,
            source_slot=source.slot,
            source_type=source.type_id,
            shield=True,
        )

    def _off_line(
        self,
        link: tuple[int, int],
        hazards: tuple[TrackedObject, ...],
        blocked_dirs: frozenset[str] | set[str] | tuple[str, ...],
    ) -> EvadeDecision | None:
        """Leave a shooter's axis while there is still time to walk."""
        if not self.avoid_firing_lines:
            return None
        bodies = tuple(
            h for h in hazards if h.hazard is not HazardClass.PROJECTILE
        )
        step = off_line_step(
            link,
            bodies,
            band=self.firing_band,
            bounds=self.bounds,
            blocked_dirs=blocked_dirs,
        )
        if step is None:
            return None
        self._commit(step)
        self.off_line_steps += 1
        return EvadeDecision(
            direction=step,
            reason="off_firing_line",
            ttc=self.horizon + 1,
            stand_ttc=self.horizon + 1,
        )

    def _peel(
        self,
        link: tuple[int, int],
        stand: Impact,
        options: dict[str, tuple[int, Impact]],
    ) -> EvadeDecision | None:
        """Break the axis Link shares with the thing touching him.

        Sliding along a shared row keeps Link in the line of fire; the axis
        to open is the one that is already nearly zero. Among the steps that
        open it, take the one with more room to the bound.
        """
        source = stand.source
        if source is None or not options:
            return None
        dx = source.x - link[0]
        dy = source.y - link[1]
        vertical = abs(dx) > abs(dy)
        preferred = ("UP", "DOWN") if vertical else ("LEFT", "RIGHT")
        pool = [name for name in preferred if name in options]
        if not pool:
            pool = list(options)
        choice = min(pool, key=lambda name: (-self._room_ahead(link, name), name))
        self._commit(choice)
        self.evades += 1
        return EvadeDecision(
            direction=choice,
            reason="evade_peel",
            ttc=options[choice][0],
            stand_ttc=stand.frames,
            source_slot=source.slot,
            source_type=source.type_id,
        )

    def _room_ahead(self, link: tuple[int, int], name: str) -> int:
        """Free pixels between Link and the bound in ``name``."""
        if self.bounds is None:
            return 0
        x_lo, x_hi, y_lo, y_hi = self.bounds
        return {
            "UP": link[1] - y_lo,
            "DOWN": y_hi - link[1],
            "LEFT": link[0] - x_lo,
            "RIGHT": x_hi - link[0],
        }[name]

    def _stand_decision(self, stand: Impact, reason: str) -> EvadeDecision:
        return EvadeDecision(
            direction=None,
            reason=reason,
            ttc=stand.frames,
            stand_ttc=stand.frames,
            source_slot=stand.source.slot if stand.source else None,
            source_type=stand.source.type_id if stand.source else None,
        )

    def _commit(self, direction: str | None) -> None:
        if direction is None:
            self._commit_dir = None
            self._commit_left = 0
            return
        if direction != self._commit_dir:
            self._commit_dir = direction
            self._commit_left = self.commit_frames
        else:
            self._commit_left = max(0, self._commit_left - 1)
        self._last_dir = direction

    def _decay(self) -> None:
        if self._commit_left > 0:
            self._commit_left -= 1
        if self._commit_left == 0:
            self._commit_dir = None
