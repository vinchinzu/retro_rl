"""Ground-truth forward prediction: the ROM is the enemy model.

Every combat layer in this tree predicts the future by extrapolating a
measured velocity in a straight line (``dungeon.tracking._velocity`` feeds
``TrackedObject.at`` feeds ``dungeon.threat.contact_frames``). That model is
honest for about eight frames and then stops being one. Measured against the
ROM by ``scratch/probe_rollout.py`` -- 288 sample points, 1582 hazard
predictions, over two live overworld screens:

    ``At4A``, six blue tektites (the pre-L1 bowl):

        horizon   mean err    p90    >= 16 px wrong
            8 f     2.7 px    12 px         4 %
           16 f     7.5 px    34 px        22 %
           32 f    19.3 px    71 px        34 %
           48 f    31.3 px   114 px        42 %

    ``At78``, four octoroks -- straight-line walkers, the easiest case in
    the game, and still:

        horizon   mean err    p90    >= 16 px wrong
            8 f     0.4 px     2 px         0 %
           16 f     2.0 px     7 px         1 %
           32 f     6.3 px    20 px        14 %
           48 f    11.4 px    33 px        29 %

The error is bimodal, not noisy: the median is 0 px because a tektite
between hops has not moved at all, and the p90 is 34 px because the frame it
jumps is the frame the straight line is a fiction. Averaging cannot see that
and neither can a tolerance.

``threat.MIN_DODGE_BODY`` is 16: a sidestep does not clear a body until Link
has walked the full 16 px pad, so every dodge decision is a claim about frame
16 and beyond. At frame 16 roughly one hazard prediction in four is already
wrong by a whole hitbox, and at ``threat.DEFAULT_HORIZON`` (32) it is one in
three. **The dodge horizon and the model's validity horizon do not overlap.**
No amount of tuning ``TRIGGER_TTC`` fixes that; the model is the bug.

The fix is not to port the enemy AI out of the disassembly. The ROM is
already here, it is already exact, and it is cheap to ask. Measured on this
machine (stable-retro / fceumm, Zen-class CPU):

    em.get_state()   1.4 us      em.set_state()   4.4 us
    em.step()      226.3 us      (4418 fps, 73x real time)
    5 candidate directions x 30 frames      33.7 ms

and a restore-plus-replay is **bit-exact**: the PRNG, the animation phase and
the frame cadence all live inside the state blob, so a rollout is not an
approximation of the future, it *is* the future. That is strictly more than a
re-implementation could offer, for none of the archaeology.

What the disassembly is still worth is the *constants* a rollout cannot hand
back cheaply -- hitbox extents, per-type speeds, damage tables, spawn tables.
Those turn a wide search into a narrow one. They are not the engine.

Usage -- the caller owns the emulator and gets it back untouched::

    rollout = Rollout(env)
    outcomes = rollout.fan({"UP": ..., "DOWN": ..., "STAND": ()}, frames=24)
    best = min(outcomes, key=lambda o: (o.contact_frame is not None, -o.moved))

Traps this module exists to respect:

- **One emulator per process** (``AGENTS.md``). :class:`Rollout` never makes
  one; it borrows the live env and restores it in a ``finally``.
- **A turn and a swing cannot share a frame.** A plan is a frame-by-frame
  button script, not a direction, so a caller can express "hold LEFT three
  frames, then press A" -- the thing ``nes_action(face, "A")`` cannot say.
- **ButtonsPressed is an edge.** Held A does not re-swing, so a plan that
  wants two swings has to release between them. The scripts are literal.
- **``set_button_mask`` wants a ``uint8`` array.** Handed a Python int it
  reads a one-element array -- the B button -- so every direction in a fan
  lands on the same pixel and the rollout looks deterministic because it is
  doing nothing. ``press`` is the only constructor for a plan frame.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from retro_harness.nes import nes_action
from zelda_i.combat import chebyshev, heart_value, live_enemies
from zelda_i.ram import PLAY_MODE, ZeldaSnapshot, read_snapshot

__all__ = [
    "ROLLOUT_EVADE_DIRECTIONS",
    "ROLLOUT_EVADE_FRAMES",
    "ROLLOUT_EVADE_MIN_GAIN",
    "ROLLOUT_EVADE_REPLAN_FRAMES",
    "ROLLOUT_EVADE_SCREEN_BUDGET",
    "ROLLOUT_EVADE_TRIGGER_RADIUS",
    "Branch",
    "EvadeStep",
    "Frame",
    "Outcome",
    "Plan",
    "Rollout",
    "RolloutEvader",
    "hold",
    "in_play",
    "press",
    "swing_after",
]

# One frame of a plan is an ordinary NES action vector -- the same currency
# ``FrameAction.action`` carries -- so a plan can be spliced straight out of a
# controller's own output. ``em.set_button_mask`` wants it as ``uint8``; it
# rejects a Python int outright above 0xFF and silently reads a small one as a
# one-button array, which is a no-op that looks like a working rollout.
Frame = tuple[int, ...]
Plan = tuple[Frame, ...]


def press(*names: str) -> Frame:
    """One plan frame. ``press()`` is idle. Raises on an unknown button."""
    return tuple(int(v) for v in nes_action(*names))


def hold(direction: str | None, frames: int) -> Plan:
    """A plan that holds one direction (or nothing) for ``frames``."""
    frame = press() if direction in (None, "STAND") else press(direction)
    return (frame,) * max(int(frames), 0)


def swing_after(direction: str, turn_frames: int, frames: int) -> Plan:
    """Turn for ``turn_frames``, press A for one frame, then hold the rest.

    ``turn_frames`` exists because ``$0098`` does not agree with a direction
    on the frame it is first held: holding the direction alone turns Link in
    1-4 frames, and A pressed before that goes out along the *old* axis
    (``AGENTS.md``, ``scratch/probe_turn_swing.py``). A is a single frame
    because ButtonsPressed is an edge.
    """
    turn = press(direction)
    blade = press(direction, "A")
    tail = max(int(frames) - int(turn_frames) - 1, 0)
    return (turn,) * max(int(turn_frames), 0) + (blade,) + (turn,) * tail


@dataclass(frozen=True)
class Outcome:
    """What one plan actually did, read off the ROM rather than modelled.

    ``contact_frame`` is the frame ``$04F0`` armed, which is Link_BeHarmed
    firing -- the collision that survives a Survival refill, and the only
    damage signal that sees a *bumped* Link at all. ``hp_lost`` is in 1/256
    of a heart (``combat.heart_value``) so a wooden-sword chip reads 128 and
    not zero.
    """

    label: str
    frames: int
    snap: ZeldaSnapshot
    hp_lost: int
    contact_frame: int | None
    kills: int
    rupees: int
    moved: int
    left_screen: bool

    @property
    def safe(self) -> bool:
        return self.contact_frame is None

    @property
    def hits(self) -> int:
        """Whole hits, for a caller that wants an event count not a depth."""
        return 0 if self.contact_frame is None else 1

    def report(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "frames": self.frames,
            "safe": self.safe,
            "contact_frame": self.contact_frame,
            "hp_lost": self.hp_lost,
            "hearts_lost": round(self.hp_lost / 256.0, 3),
            "kills": self.kills,
            "rupees": self.rupees,
            "moved": self.moved,
            "left_screen": self.left_screen,
            "xy": [int(self.snap.link_x), int(self.snap.link_y)],
        }


class Rollout:
    """Branch the live emulator over candidate plans and restore it.

    The env handed in is the one the walk is running on. Every public method
    saves before it branches and restores in a ``finally``, so a caller can
    ask this question on any frame without owning a second emulator or
    perturbing the tape it is recording. ``state_restores`` counts the
    restores for a run report -- a rollout is not a ``set_state`` of the kind
    ``STATUS.md`` means by "no state load", but it is still a restore and a
    report that hides it is lying about the walk.
    """

    def __init__(self, env: Any) -> None:
        self.env = env
        self.em = env.em
        self.rollouts = 0
        self.frames_rolled = 0

    # --- the kernel ---------------------------------------------------- #

    def run(self, label: str, plan: Sequence[Frame]) -> Outcome:
        """Replay one button script from the live frame. Restores after."""
        return self.fan({label: plan})[0]

    def fan(
        self,
        plans: Mapping[str, Sequence[Frame]],
        *,
        frames: int | None = None,
    ) -> tuple[Outcome, ...]:
        """Replay every plan from this one frame. Order follows ``plans``.

        ``frames`` pads or truncates every plan to a common length, so a
        caller comparing directions does not have to build equal-length
        scripts by hand. A plan shorter than ``frames`` holds its last frame
        (idle if it is empty), which is what "keep walking that way" means.
        """
        if not plans:
            return ()
        before = read_snapshot(self.env.get_ram())
        state = self.em.get_state()
        outcomes: list[Outcome] = []
        try:
            for label, plan in plans.items():
                script = _pad(tuple(tuple(int(v) for v in f) for f in plan), frames)
                self.em.set_state(state)
                outcomes.append(self._replay(label, script, before))
        finally:
            self.em.set_state(state)
        return tuple(outcomes)

    def _replay(
        self, label: str, script: Sequence[Frame], before: ZeldaSnapshot
    ) -> Outcome:
        self.rollouts += 1
        self.frames_rolled += len(script)
        hp0 = heart_value(before)
        alive0 = len(live_enemies(before))
        rupees0 = int(before.rupees)
        screen0, level0 = int(before.screen), int(before.level)
        iframes = int(before.link_iframes)
        contact: int | None = None
        snap = before
        for index, frame in enumerate(script):
            self.em.set_button_mask(np.asarray(frame, dtype=np.uint8), 0)
            self.em.step()
            snap = read_snapshot(self.env.get_ram())
            now = int(snap.link_iframes)
            # 0 -> nonzero is the Link_BeHarmed edge. A decrementing timer is
            # the *same* hit still cooling down, and counting it again would
            # rank a plan that stands in one body worse than one that walks
            # through three.
            if contact is None and now > 0 and iframes == 0:
                contact = index + 1
            iframes = now
        left = int(snap.screen) != screen0 or int(snap.level) != level0
        # Kills and money only mean anything while the screen is the same
        # one: a scroll renumbers every slot and hands out a fresh census.
        killed = max(alive0 - len(live_enemies(snap)), 0) if not left else 0
        return Outcome(
            label=label,
            frames=len(script),
            snap=snap,
            hp_lost=max(hp0 - heart_value(snap), 0),
            contact_frame=contact,
            kills=killed,
            rupees=max(int(snap.rupees) - rupees0, 0) if not left else 0,
            moved=abs(int(snap.link_x) - int(before.link_x))
            + abs(int(snap.link_y) - int(before.link_y)),
            left_screen=left,
        )

    # --- the question the evader actually asks -------------------------- #

    def walk(
        self,
        waypoints: Sequence[tuple[int, int]],
        *,
        prefix: Sequence[Frame] = (),
        frames: int = 300,
        tol: int = 2,
    ) -> tuple[Plan, Outcome, bool]:
        """Roll ``prefix`` then a lattice walk through ``waypoints``; restore.

        The walk's presses are ``room_step``'s on the rolled frames, and the
        emulator is deterministic, so replaying the returned script live
        walks the same path into the same outcome. ``reached`` is whether the
        last waypoint was met inside ``frames``.
        """
        from zelda_i.dungeon.hop_controller import room_step

        state = self.em.get_state()
        script = [tuple(int(v) for v in f) for f in prefix]
        i = 0
        try:
            for frame in script:
                self.em.set_button_mask(np.asarray(frame, dtype=np.uint8), 0)
                self.em.step()
            for _ in range(int(frames)):
                snap = read_snapshot(self.env.get_ram())
                while i < len(waypoints) and chebyshev(
                    snap.link_x, snap.link_y, *waypoints[i]
                ) <= tol:
                    i += 1
                if i >= len(waypoints):
                    break
                step = room_step(snap, waypoints[i], tol=tol, env=self.env)
                frame = press(step) if step else press()
                script.append(frame)
                self.em.set_button_mask(np.asarray(frame, dtype=np.uint8), 0)
                self.em.step()
        finally:
            self.em.set_state(state)
        plan = tuple(script)
        return plan, self.run("walk", plan), i >= len(waypoints)

    def first_contact(self, plan: Sequence[Frame]) -> int | None:
        """Frames until Link is harmed under ``plan``, or None inside it.

        The ROM twin of ``dungeon.threat.contact_frames``, which answers the
        same question from two straight lines and a pair of hitbox halves.
        """
        return self.run("probe", plan).contact_frame

    def best_step(
        self,
        *,
        frames: int = 24,
        directions: Iterable[str] = ("UP", "DOWN", "LEFT", "RIGHT", "STAND"),
    ) -> Outcome:
        """The single held direction that survives longest, then travels most.

        Deliberately one ply of one held button: that is the shape of the
        decision ``threat.ReactiveEvader`` already makes, so this is a drop-in
        answer to it and an A/B against it is a fair one. It is also the whole
        reason the evader is wall-blind -- ``_can_move`` tests a rectangle,
        and a rollout tests the ROM, walls and rocks and bushes included.
        """
        plans = {d: hold(d, frames) for d in directions}
        outcomes = self.fan(plans)
        return max(outcomes, key=_survival_key)

    # --- more than one ply ---------------------------------------------- #

    @contextmanager
    def branching(self) -> Iterator["Branch"]:
        """Borrow the emulator for a *multi-ply* search; restore it after.

        :meth:`fan` replays every plan from the same live frame, which is one
        ply by construction. A room is not one ply (``solver.py``): a search
        over sequences has to keep going from where a segment *ended*, and
        that means handing a core state token back to the caller instead of
        throwing it away. This is the only supported way to do that, because
        the discipline the rest of the module exists for -- save once at the
        top, restore in a ``finally`` -- has to hold across the whole search
        and not once per segment.

        The ledger is unchanged: every :meth:`Branch.play` goes through the
        same ``_replay``, so ``rollouts`` and ``frames_rolled`` price a search
        exactly as they price a fan, and a caller charging a budget can read
        ``frames_rolled`` before and after.
        """
        state = self.em.get_state()
        try:
            yield Branch(self, state)
        finally:
            self.em.set_state(state)

    def report(self) -> dict[str, int]:
        return {"rollouts": self.rollouts, "frames_rolled": self.frames_rolled}

    def reset(self) -> None:
        """Zero the ledger, keep the env. A new walk, the same emulator."""
        self.rollouts = 0
        self.frames_rolled = 0


@dataclass(frozen=True)
class Branch:
    """A search's handle on the borrowed emulator, plus the frame it started on.

    ``root`` is the opaque core state of the live frame :meth:`Rollout.branching`
    was entered on. Every other token is one :meth:`play` handed back, so a
    search can extend a sequence for the price of one ``set_state`` instead of
    replaying the whole prefix.

    Deliberately not a mini-emulator: it owns no state of its own, it cannot
    be constructed without the context manager that restores the machine, and
    the deltas in each :class:`Outcome` are measured from the token the
    segment started at -- so ``hp_lost`` is that segment's damage, while the
    absolute reading a scorer wants is ``Outcome.snap``.
    """

    rollout: "Rollout"
    root: bytes

    def play(
        self,
        label: str,
        plan: Sequence[Frame],
        token: bytes | None = None,
    ) -> tuple[bytes, Outcome]:
        """Replay ``plan`` from ``token`` (the root by default). No restore.

        The restore is the context manager's job and happens once, at the end
        of the search, which is the whole reason this is not a public method
        on :class:`Rollout`.
        """
        em = self.rollout.em
        em.set_state(self.root if token is None else token)
        before = read_snapshot(self.rollout.env.get_ram())
        script = tuple(tuple(int(v) for v in f) for f in plan)
        outcome = self.rollout._replay(label, script, before)
        return em.get_state(), outcome


def _survival_key(outcome: Outcome) -> tuple[int, int, int]:
    """Later contact first, then less damage, then more ground covered.

    A plan with no contact at all ranks above every plan that has one, which
    is what ``frames + 1`` buys: it is strictly greater than any in-window
    contact frame, the same convention ``threat.contact_frames`` uses for
    "never" within a horizon.
    """
    return (_reach(outcome), -outcome.hp_lost, outcome.moved)


def _pad(script: Plan, frames: int | None) -> Plan:
    if frames is None:
        return script
    want = max(int(frames), 0)
    if len(script) >= want:
        return script[:want]
    tail = script[-1] if script else press()
    return script + (tail,) * (want - len(script))


def in_play(snap: ZeldaSnapshot) -> bool:
    """True when a rollout is meaningful: no scroll, no cave, no menu."""
    return int(snap.mode) == PLAY_MODE and not snap.transitioning


# ------------------------------------------------ the evader on a budget ---
# A rollout is 226 us a frame and a fan is five of them, so the thing that
# decides whether this kernel can sit on a 200k-frame walk at all is *how
# often it is asked*, not how good the answer is. ``AGENTS.md`` Traps: "every
# rung needs a budget. The contact strike had none and one body owned 24877
# frames." These four numbers are that budget, and each one is a test.
#
# The horizon. ``threat.MIN_DODGE_BODY`` is 16, so a sidestep is not finished
# until frame 16; 24 leaves eight frames of proof that the step still holds
# once it has cleared the pad. Past ~30 the fan stops being cheap and the
# question stops being a dodge.
ROLLOUT_EVADE_FRAMES = 24
# Re-plan cadence. A fan every frame is 28 ms per frame of walk — three
# orders of magnitude over budget — and it is also the two-pixel tug-of-war
# this tree keeps re-learning (``path._OCCUPIED_LANE_NO_GAIN_CAP``,
# ``path._SPIT_DUCK_COMMIT``): a bearing that rotates one pixel reverses a
# per-frame decision. One fan then eight frames of holding it is 3.5 ms per
# in-range frame, and a commit besides.
ROLLOUT_EVADE_REPLAN_FRAMES = 8
# The trigger gate. Nothing outside this Chebyshev radius can reach Link
# inside the horizon: the fastest overworld hazard is the ``0x55`` spit at
# ~1.5 px/frame (24 f = 36 px), Link closes at most 24 px of his own, and
# ``MIN_DODGE_BODY`` is the 16 px pad that counts as touching. The gate is
# re-read every frame (it is a subtraction, not a rollout), so a hazard that
# crosses the line still gets ~20 frames of window.
ROLLOUT_EVADE_TRIGGER_RADIUS = 56
# A step that only postpones the hit is oscillation fuel, not an escape --
# the same rule and the same number as ``threat.MIN_ESCAPE_GAIN``. Below it
# the rung declines and the ladder below it (the ``ReactiveEvader``, then the
# hop) drives the frame exactly as it does today.
ROLLOUT_EVADE_MIN_GAIN = 4
# Fans one screen visit may spend. The cadence alone bounds the cost per
# frame; this bounds it per *screen*, which is the failure mode a cadence
# cannot see -- a wave that parks a body inside the trigger radius for a
# 4000-frame grind (``path.DEFAULT_HOP_SCREEN_MAX_FRAMES``) would otherwise
# buy 500 fans on one screen. 120 fans is ~3.4 s of wall clock and covers
# ~960 in-range frames, which is longer than any measured overworld fight.
ROLLOUT_EVADE_SCREEN_BUDGET = 120
# ``STAND`` is in the fan because the gain is measured *against* it: without
# the standing outcome there is no way to tell an escape from a shuffle.
STAND_LABEL = "STAND"
ROLLOUT_EVADE_DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT", STAND_LABEL)


@dataclass(frozen=True)
class EvadeStep:
    """One frame of ROM-truth advice. ``direction`` None means "not mine".

    The twin of ``threat.EvadeDecision``, with the model's fields replaced by
    measurements: ``contact_frame`` is the frame ``$04F0`` armed under the
    chosen plan rather than a straight-line time-to-contact, and ``gain`` is
    how many frames of life that plan bought over standing still -- both read
    off the ROM, both in the same units.
    """

    direction: str | None
    reason: str
    contact_frame: int | None
    stand_contact: int | None
    gain: int
    moved: int
    replanned: bool

    @property
    def claims(self) -> bool:
        return self.direction is not None


@dataclass
class RolloutEvader:
    """``threat.ReactiveEvader``, asked of the ROM instead of a velocity.

    Same shape of answer -- one held direction, this frame -- so it drops in
    where ``ReactiveEvader.decide`` sits and an A/B between the two changes
    nothing else. Three things it can see that the model cannot:

    * **Walls.** ``ReactiveEvader._can_move`` tests a *rectangle*
      (``path._EVADE_BOUNDS`` is the scroll box), so it dodges into rocks and
      bushes; screen-wide poisoning of 0x7B for eight hits came from exactly
      that. A blocked direction rolls out 0 px of ``moved`` and loses on the
      tie-break, with no occupancy grid anywhere.
    * **The muzzle hold.** The ``0x55`` spit is born two frames into
      ``ObjState 0x03`` and then sits on the Zora's mouth ~17 frames, so
      ``ObjectTracker`` measures zero velocity and ``assess`` calls it safe
      for the whole window in which it could still be walked away from. A
      rollout sees the hit land.
    * **Census edge cases.** Dormant leevers, corpse slots and type-byte
      oddities stop mattering: contact is the ``$04F0`` arming, not a verdict
      about which slots are enemies.

    What it is *not*: a second dispatcher. It answers one question and the
    caller places it on a ladder (``path.THREAT_RUNG_ROLLOUT``). It declines
    -- returns ``None`` -- on every frame it has nothing measured to say, and
    the rung below it runs unchanged.
    """

    rollout: Rollout
    frames: int = ROLLOUT_EVADE_FRAMES
    replan_frames: int = ROLLOUT_EVADE_REPLAN_FRAMES
    trigger_radius: int = ROLLOUT_EVADE_TRIGGER_RADIUS
    min_gain: int = ROLLOUT_EVADE_MIN_GAIN
    screen_budget: int = ROLLOUT_EVADE_SCREEN_BUDGET
    directions: tuple[str, ...] = ROLLOUT_EVADE_DIRECTIONS
    # Accounting. ``replans`` is the number of *fans*; the number of rollouts
    # and the frames rolled are the kernel's own ledger and are surfaced with
    # these, because a report that hides a ``set_state`` is lying about the walk.
    replans: int = 0
    claims: int = 0
    held_frames: int = 0
    declines: dict[str, int] = field(default_factory=dict)
    _dir: str | None = field(default=None, init=False, repr=False)
    _cooldown: int = field(default=0, init=False, repr=False)
    _room: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _room_replans: int = field(default=0, init=False, repr=False)

    # --- the question -------------------------------------------------- #

    def decide(
        self,
        snap: ZeldaSnapshot,
        hazards: Iterable[Any] = (),
        *,
        blocked: Iterable[str] = (),
    ) -> EvadeStep | None:
        """Advice for this frame, or ``None`` when this rung wants nothing.

        ``hazards`` is anything with ``.x`` / ``.y`` (``TrackedObject`` is the
        live caller) -- only the *nearest* one is read, and only to decide
        whether the fan is worth 28 ms. The dodge itself asks the ROM.

        ``blocked`` names directions the caller will not accept, which on the
        overworld is the scroll line (``path._evade_blocked_dirs``): a dodge
        that scrolls the screen has not dodged, it has changed hops. Dropping
        them here rather than after the fan also buys back their rollouts.
        """
        room = (int(snap.level), int(snap.screen))
        if room != self._room:
            # Slots renumber across a scroll and the budget is per visit.
            self._room = room
            self._room_replans = 0
            self._release()
        if not in_play(snap):
            # A scroll, a cave or a menu: the ROM is not simulating a walk,
            # so a rollout of one means nothing. The ``ReactiveEvader`` below
            # keeps the frame.
            self._release()
            return self._decline("not_in_play")
        gap = self._nearest(snap, hazards)
        if gap is None or gap > self.trigger_radius:
            self._release()
            return self._decline("no_hazard")
        if self._cooldown > 0:
            self._cooldown -= 1
            if self._dir is None:
                # The last fan said standing was as good as walking. Hold
                # that verdict rather than re-buying it every frame.
                return self._decline("hold_stand")
            self.held_frames += 1
            self.claims += 1
            return EvadeStep(
                direction=self._dir,
                reason="rollout_evade_hold",
                contact_frame=None,
                stand_contact=None,
                gain=0,
                moved=0,
                replanned=False,
            )
        if self._room_replans >= self.screen_budget:
            self._release()
            return self._decline("screen_budget")
        return self._replan(snap, blocked)

    # --- internals ----------------------------------------------------- #

    def _replan(self, snap: ZeldaSnapshot, blocked: Iterable[str]) -> EvadeStep | None:
        banned = {str(d).upper() for d in blocked}
        # ``STAND`` is never banned: it is the baseline the gain is measured
        # against, and a caller banning it would silently make every dodge
        # look decisive.
        wanted = tuple(
            d for d in self.directions if d == STAND_LABEL or d not in banned
        )
        outcomes = self.rollout.fan({d: hold(d, self.frames) for d in wanted})
        self.replans += 1
        self._room_replans += 1
        # The cooldown is charged whatever the verdict. A fan that ends in a
        # decline costs the same 28 ms as one that ends in a dodge, so a
        # quiet frame inside the trigger radius must not re-buy it.
        self._cooldown = max(self.replan_frames - 1, 0)
        by_label = {o.label: o for o in outcomes}
        stand = by_label.get(STAND_LABEL)
        stand_reach = _reach(stand) if stand is not None else 0
        # A step that scrolls the screen has left the hop, not the hazard.
        moves = [
            o for o in outcomes if o.label != STAND_LABEL and not o.left_screen
        ]
        if not moves:
            self._dir = None
            return self._decline("boxed_in")
        best = max(moves, key=_survival_key)
        gain = _reach(best) - stand_reach
        if gain < self.min_gain:
            self._dir = None
            return self._decline("no_gain")
        self._dir = best.label
        self.claims += 1
        return EvadeStep(
            direction=best.label,
            reason="rollout_evade",
            contact_frame=best.contact_frame,
            stand_contact=stand.contact_frame if stand is not None else None,
            gain=gain,
            moved=best.moved,
            replanned=True,
        )

    def _nearest(self, snap: ZeldaSnapshot, hazards: Iterable[Any]) -> int | None:
        """Chebyshev gap to the closest hazard, or ``None`` if there is none.

        Deliberately the cheapest possible gate: a subtraction per slot, no
        velocity, no census verdict. It decides whether to *spend* a rollout,
        and a gate that costs what it guards is not a gate.
        """
        lx, ly = int(snap.link_x), int(snap.link_y)
        best: int | None = None
        for hazard in hazards:
            if not getattr(hazard, "is_hazard", True):
                continue
            gap = chebyshev(lx, ly, int(hazard.x), int(hazard.y))
            if best is None or gap < best:
                best = gap
        return best

    def _decline(self, reason: str) -> None:
        self.declines[reason] = self.declines.get(reason, 0) + 1
        return None

    def _release(self) -> None:
        self._dir = None
        self._cooldown = 0

    def reset(self) -> None:
        """New walk, same emulator: drop the commit and the ledger."""
        self._release()
        self._room = None
        self._room_replans = 0
        self.replans = self.claims = self.held_frames = 0
        self.declines = {}
        self.rollout.reset()

    def report(self) -> dict[str, Any]:
        """The ledger a run report must not hide, plus the budget it ran on."""
        return {
            "replans": self.replans,
            "claims": self.claims,
            "held_frames": self.held_frames,
            "declines": dict(sorted(self.declines.items(), key=lambda kv: -kv[1])),
            "budget": {
                "frames": self.frames,
                "replan_frames": self.replan_frames,
                "trigger_radius": self.trigger_radius,
                "min_gain": self.min_gain,
                "screen_budget": self.screen_budget,
            },
            **self.rollout.report(),
        }


def _reach(outcome: Outcome) -> int:
    """Frames this plan survives: the horizon plus one when it never hits."""
    if outcome.contact_frame is None:
        return outcome.frames + 1
    return outcome.contact_frame
