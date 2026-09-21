"""A room is not one ply: budget-bounded ROM-truth search over held segments.

``rollout.Rollout.best_step`` is deliberately **one ply of one held button**.
That shape exists so the ROM-truth evader is a fair A/B against
``dungeon.threat.ReactiveEvader``, which also decides one held direction per
frame. It is the right shape for a dodge and the wrong shape for a room: a
clear is "drift DOWN twelve frames, turn LEFT four, swing, hold UP twenty",
and no amount of re-asking a one-ply question composes into that. Every room
in this tree that needs such a sequence currently carries it as a **position
table** -- a hand-written waypoint list plus a fixed swing cadence
(``level7.cellar.Room1ACandleController._aggressive_fight`` is
``frames % 8 < 4``) -- and a position table is a plan that was searched once,
by a human, against a build of the ROM that has since moved.

This module searches that plan instead, from the live frame, against the ROM.

The design is lifted from the other game in this monorepo:
``snes/super_metroid/room_adapter.py`` has a working, unit-tested version of
exactly this search. Nothing is imported across games (they share no RAM, no
action vector and no notion of a room) -- five *ideas* are reimplemented here:

1. **A held pulse is the search unit** (:class:`TimedAction`), not a frame.
   Pulse lengths are short, medium, long, and the buttons come from
   ``rollout.press`` because ``em.set_button_mask`` wants a ``uint8`` array
   and silently misreads an int.
2. **The search core is generic** (:func:`beam_search`) over ``(token,
   state)`` with injected ``expand`` / ``score`` / ``reached`` / ``state_key``.
   That is what makes a solver whose real inputs are a ROM and a savestate
   testable with no emulator at all: ``tests/test_solver.py`` drives it over
   plain integers.
3. **Cost is a penalty, not a cliff** (:data:`SOLVER_FRAME_PENALTY`). A longer
   plan is worse than a short one by a continuous amount, so the search
   prefers the fast clear without being forbidden the slow one.
4. **Quantized ``state_key`` de-duplication is the pruning.** Two branches
   that put Link in the same 4 px cell facing the same way with the same
   census are the same branch; collapsing them is what stops the beam filling
   with held-direction variants of one idea. It is also why this module does
   not need ``dungeon.species`` to bound the search -- C5 is used in exactly
   one place, and only where it genuinely narrows a branch (see
   :func:`body_cost`).
5. **A bounded partial plan is a result.** When the goal is not met the search
   returns its best prefix with ``reached=False``, which is precisely the
   card's "a solver gets a per-room frame budget and drops to the current
   controller past it".

**Budgets.** ``AGENTS.md``: *"Every rung needs a budget. The contact strike
had none and one body owned 24877 frames."* There are three here and each one
is a named constant with a test:

* :data:`SOLVER_PLAN_FRAMES` bounds the *plan* -- no returned sequence is
  longer, so a commit cannot outlive the state it was planned from.
* :data:`SOLVER_EXPANSION_BUDGET` bounds one *search* -- a hard stop on
  ``expand`` calls, which is what actually prices a search in emulator frames.
* :data:`SOLVER_ROOM_ROLLOUT_FRAMES` bounds a *room visit*. Past it
  :meth:`RoomSolver.act` declines every frame and the controller underneath
  drives the room exactly as it does today. Local caps do not compose, so the
  room budget is charged from ``Rollout.frames_rolled`` -- the kernel's own
  ledger -- and not from a counter this module keeps by hand.

**Traps honoured.** A turn and a swing cannot share a frame, so a swing is an
*atomic* segment (:func:`swing_action`) built by ``rollout.swing_after`` with
at least one turn frame in front of the A; that also guarantees the release
between two swings that ``ButtonsPressed`` being an edge requires. Health is
read through ``combat.heart_value`` (``$066F`` low nibble is whole hearts
minus one and a wooden chip never touches it). One emulator per process: the
live env is borrowed through ``Rollout.branching`` and restored in a
``finally``.

**What this module does not do.** It does not dispatch. :meth:`RoomSolver.act`
answers one question -- "the frame I would play, or nothing" -- and a caller
places it on an ``overworld.arbiter.Arbiter`` ladder behind a flag that is off
by default (``level7.cellar.Room1ACandleController.attach_solver``).

**The wired room is L7 ``0x1A``** and it is the only one. What is proven is
the machinery: the search returns a multi-segment plan, the budgets hold, and
the emulator comes back where it was. What is *not* proven is that the search
clears that room faster, safer or at all -- that is a measurement, it needs
the ROM and a probe with a report attached, and nothing here has run one.

Three more rooms are shaped for this and are deliberately left undone:

* **L6 Gohma ``0x1C``** (and L8 ``0x1E``). ``dungeon/gohma.py`` already has
  the ``$03C7`` eye clock, and the shot has to go on the open-eye *rising
  edge* -- any fixed cadence aliases onto the 17-frame blink (20+ arrows, 0
  connects). The alphabet is not this one: it is the firing column plus a
  ``press("B")`` segment, and the objective is Gohma's HP, carried from
  ``species`` and never from a colour (``ids.py`` labels ``0x33`` red with
  ROM HP 96, which is three wooden arrows, so the labels look swapped).
* **The ``0x68`` block-push rooms** (L8 ``0x1F``, L9 ``0x55``). The objective
  is the block's ``y``/``x``, not a census, and the alphabet is one direction
  -- the *open* side, which is north (hold DOWN) in both measured rooms, not
  a fixed south face. ``dungeon/tilemap.py`` has to be dumped first: a search
  that does not know the walls will plan a push from inside one.
* **L7 ``0x1A``'s own perimeter hunt**, the half of that room this wiring does
  not touch. The solver rung is gated on a live wave and the hunt waypoints
  stay scripted, so the last (NE) goriya is still walked to by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Generic, Sequence, TypeVar

from retro_harness.input_script import FrameAction
from zelda_i.combat import chebyshev, heart_value, live_enemies
from zelda_i.dungeon.species import species_of
from zelda_i.ram import ZeldaObject, ZeldaSnapshot
from zelda_i.rollout import Frame, Plan, Rollout, in_play, press, swing_after

__all__ = [
    "LEFT_ROOM_COST",
    "SOLVER_APPROACH_CLAMP",
    "SOLVER_APPROACH_WEIGHT",
    "SOLVER_BEAM_WIDTH",
    "SOLVER_DAMAGE_WEIGHT",
    "SOLVER_DIRECTIONS",
    "SOLVER_EXPANSION_BUDGET",
    "SOLVER_FRAME_PENALTY",
    "SOLVER_HP_WEIGHT",
    "SOLVER_KILL_WEIGHT",
    "SOLVER_MAX_DEPTH",
    "SOLVER_PLAN_FRAMES",
    "SOLVER_PULSE_FRAMES",
    "SOLVER_ROOM_ROLLOUT_FRAMES",
    "SOLVER_SWING_FRAMES",
    "SOLVER_TURN_FRAMES",
    "ClearObjective",
    "RoomSolver",
    "SearchConfig",
    "SolvedPlan",
    "TimedAction",
    "beam_search",
    "body_cost",
    "hold_action",
    "room_state_key",
    "segment_library",
    "stand_action",
    "swing_action",
]

StateT = TypeVar("StateT")
TokenT = TypeVar("TokenT")

# --- The segment alphabet ------------------------------------------------
# Three pulse lengths, the same short / medium / long shape the Super Metroid
# adapter uses. 4 is "nudge and re-look", 12 is roughly the 16 px dodge pad at
# Link's ~1.3 px/frame, 24 is the ``ROLLOUT_EVADE_FRAMES`` horizon -- past
# which the ROM's own answer stops being about the decision being made.
SOLVER_PULSE_FRAMES = (4, 12, 24)
SOLVER_DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
# Frames of the direction held *before* A. ``AGENTS.md``: holding a direction
# alone turns Link in 1-4 frames, always, and A pressed before ``$0098``
# agrees sends the blade out along the old axis -- a miss plus 13 pinned
# frames. 4 is the measured worst case, not a guess.
SOLVER_TURN_FRAMES = 4
# A whole swing segment: the turn, the one-frame A (the press is an edge), and
# the tail that holds the facing while the 13-frame animation plays out and
# the hit registers.
SOLVER_SWING_FRAMES = 20

# --- The search shape ----------------------------------------------------
SOLVER_BEAM_WIDTH = 4
SOLVER_MAX_DEPTH = 3
# No plan is longer than this. Three long segments plus a swing is 92; the cap
# is what stops a deeper config from committing a sequence that outlives the
# frame it was planned from.
SOLVER_PLAN_FRAMES = 96
# Score units per frame of plan. The cheapest real progress is a chip of one
# body's HP (:data:`SOLVER_HP_WEIGHT` / max HP, order 1), so at 0.02 a 96-frame
# plan carries 1.92 of penalty: enough to break a tie between two plans that
# achieve the same thing, never enough to outweigh a kill.
SOLVER_FRAME_PENALTY = 0.02
# Hard stop on ``expand`` calls in one search. With the default library (17
# segments, mean 15 frames) a full beam is 1 + 4 + 4 = 9 expanding nodes and
# 153 expansions, so 200 is a ceiling rather than a working limit -- it exists
# so a caller who widens the beam cannot accidentally buy an unbounded search.
SOLVER_EXPANSION_BUDGET = 200
# Emulator frames one *room visit* may spend on searching. ~226 us a frame
# measured (``rollout.py``), so 40k frames is ~9 s of wall clock and ~13 full
# searches. Past it :meth:`RoomSolver.act` declines for the rest of the visit
# and the controller below drives the room unchanged.
SOLVER_ROOM_ROLLOUT_FRAMES = 40_000

# --- Objective weights ---------------------------------------------------
# A dead body is worth two thirds of a full-HP live one; the remaining third
# is the HP term, which is what lets the beam see a *chip* and keep swinging
# at the same target instead of wandering to a fresh one.
SOLVER_KILL_WEIGHT = 64.0
SOLVER_HP_WEIGHT = 32.0
# Per 1/256 of a heart, so a whole heart is 128 -- two kills. Link trading one
# heart for two goriyas is a wash; for one goriya it is a loss.
SOLVER_DAMAGE_WEIGHT = 0.5
# Per pixel of Chebyshev gap to the nearest reachable body, clamped.
#
# **A flat score is a blind beam.** Without this term nothing the search can
# do to a room changes its cost until a blade actually lands, so every branch
# at depth 1 scores identically, the beam keeps the four cheapest (the
# shortest pulses, on the frame penalty alone) and the one segment that walks
# far enough to be *in range* at depth 2 is pruned before it is ever swung
# from. The clamp is there so approach can never outbid a kill: 128 px at
# 0.25 is 32, half of :data:`SOLVER_KILL_WEIGHT`.
SOLVER_APPROACH_WEIGHT = 0.25
SOLVER_APPROACH_CLAMP = 128
# Scrolling out, taking the stairs or opening a menu ends the room the search
# was reasoning about: the census renumbers and every comparison after it is
# between two different rooms. Not a hard reject -- a cost, so the search
# still returns a partial plan when every branch leaves.
LEFT_ROOM_COST = 1_000_000.0

_ACTION_WIDTH = len(press())


# --- The search unit -----------------------------------------------------


@dataclass(frozen=True)
class TimedAction:
    """One held pulse: a button vector, held ``frames`` frames.

    ``buttons`` must come from ``rollout.press`` -- an NES action *vector*,
    not a bitmask. ``em.set_button_mask`` handed a Python int reads a
    one-element array (the B button), so every direction in a fan lands on the
    same pixel and the search looks deterministic because it is doing nothing.

    ``script`` carries a segment whose frames are not all the same, which in
    this game is exactly one thing: a swing. A turn and a swing cannot share a
    frame, so "face LEFT and cut" is a *sequence* and has to be atomic, or the
    search is free to schedule the A on the frame it changed direction.
    :func:`swing_action` is the only constructor that sets it.
    """

    buttons: Frame
    frames: int
    label: str = ""
    script: Plan | None = None

    def __post_init__(self) -> None:
        _check_frame(self.buttons)
        if self.frames <= 0:
            raise ValueError("TimedAction.frames must be positive")
        if self.script is not None:
            if len(self.script) != self.frames:
                raise ValueError("TimedAction.script must be `frames` long")
            for frame in self.script:
                _check_frame(frame)

    @property
    def plan(self) -> Plan:
        """The frame-by-frame button script this segment plays."""
        if self.script is not None:
            return self.script
        return (self.buttons,) * self.frames


def _check_frame(frame: Any) -> None:
    if not isinstance(frame, tuple) or len(frame) != _ACTION_WIDTH:
        raise ValueError(
            f"a plan frame is a {_ACTION_WIDTH}-wide action vector from press()"
        )
    if any(int(v) not in (0, 1) for v in frame):
        raise ValueError("a plan frame is binary; press() is the constructor")


def hold_action(direction: str, frames: int) -> TimedAction:
    """Hold one direction. The ordinary travel segment."""
    return TimedAction(
        buttons=press(direction),
        frames=int(frames),
        label=f"hold_{direction.lower()}{int(frames)}",
    )


def stand_action(frames: int) -> TimedAction:
    """Hold nothing. Standing is a move: it is what a gain is measured against."""
    return TimedAction(buttons=press(), frames=int(frames), label=f"stand{int(frames)}")


def swing_action(
    direction: str,
    turn_frames: int = SOLVER_TURN_FRAMES,
    frames: int = SOLVER_SWING_FRAMES,
) -> TimedAction:
    """Turn, then cut, then hold the facing. Atomic on purpose.

    ``turn_frames`` must be at least 1, and that single rule buys both traps
    at once: the A never shares a frame with a direction change, and any two
    swing segments played back to back have a non-A frame between them, which
    is what ``ButtonsPressed`` being an *edge* requires for the second swing to
    exist at all.
    """
    turn = int(turn_frames)
    if turn < 1:
        raise ValueError("a turn and a swing cannot share a frame: turn_frames >= 1")
    if int(frames) <= turn:
        raise ValueError("a swing segment needs a frame for A after the turn")
    return TimedAction(
        buttons=press(direction),
        frames=int(frames),
        label=f"swing_{direction.lower()}",
        script=swing_after(direction, turn, int(frames)),
    )


def segment_library(
    directions: Sequence[str] = SOLVER_DIRECTIONS,
    pulses: Sequence[int] = SOLVER_PULSE_FRAMES,
    *,
    swings: bool = True,
    stand: bool = True,
    turn_frames: int = SOLVER_TURN_FRAMES,
    swing_frames: int = SOLVER_SWING_FRAMES,
) -> tuple[TimedAction, ...]:
    """The default alphabet: every direction at every pulse, plus the swings.

    17 segments with the defaults (4 directions x 3 pulses, 4 swings, 1 stand).
    A caller narrows it -- a block-push room wants one direction and no swing,
    a boss room wants the shot and no travel -- which is the cheapest way to
    make a search smaller and the only one that does not cost accuracy.
    """
    out: list[TimedAction] = []
    for direction in directions:
        for pulse in pulses:
            out.append(hold_action(direction, pulse))
    if swings:
        out.extend(
            swing_action(d, turn_frames, swing_frames) for d in directions
        )
    if stand:
        out.append(stand_action(max(int(pulses[0]), 1) if pulses else 4))
    return tuple(out)


# --- The generic core ----------------------------------------------------


@dataclass(frozen=True)
class SearchConfig:
    """Shape and budget of one search. Every field is a bound."""

    beam_width: int = SOLVER_BEAM_WIDTH
    max_depth: int = SOLVER_MAX_DEPTH
    frame_penalty: float = SOLVER_FRAME_PENALTY
    frame_budget: int = SOLVER_PLAN_FRAMES
    expansion_budget: int = SOLVER_EXPANSION_BUDGET


@dataclass(frozen=True)
class SolvedPlan:
    """The best sequence found, whether or not it is the goal.

    ``reached=False`` with a non-empty ``plan`` is the normal, useful answer:
    the best bounded prefix, which the caller commits and then re-searches
    from. ``expanded`` is the pruning number -- the count of segments actually
    rolled -- and is what a de-duplication test asserts on, because the beam
    width alone cannot show that anything was collapsed.
    """

    plan: Plan
    segments: tuple[tuple[str, int], ...]
    score_before: float
    score_after: float
    expanded: int
    reached: bool
    exhausted: bool = False

    @property
    def frame_count(self) -> int:
        return len(self.plan)

    @property
    def labels(self) -> tuple[str, ...]:
        """One label per *segment*, in order. The plan's shape, readably."""
        return tuple(label for label, _ in self.segments)

    def frame_labels(self) -> tuple[str, ...]:
        """One label per *frame*, so a run log can price a segment exactly."""
        out: list[str] = []
        for label, frames in self.segments:
            out.extend([label] * int(frames))
        return tuple(out)

    def report(self) -> dict[str, Any]:
        return {
            "frames": self.frame_count,
            "labels": list(self.labels),
            "score_before": round(self.score_before, 3),
            "score_after": round(self.score_after, 3),
            "expanded": self.expanded,
            "reached": self.reached,
            "exhausted": self.exhausted,
        }


@dataclass
class _Node(Generic[StateT, TokenT]):
    token: TokenT
    state: StateT
    plan: Plan
    segments: tuple[tuple[str, int], ...]
    score: float


def beam_search(
    initial_token: TokenT,
    initial_state: StateT,
    *,
    actions: Sequence[TimedAction],
    expand: Callable[[TokenT, TimedAction], tuple[TokenT, StateT]],
    score: Callable[[StateT], float],
    reached: Callable[[StateT, float], bool],
    state_key: Callable[[StateT], tuple[Any, ...]],
    config: SearchConfig = SearchConfig(),
) -> SolvedPlan:
    """Deterministic beam search over segment sequences. No emulator here.

    The four callbacks are the whole seam. ``expand`` is the only one that
    costs anything (live, it is a ``set_state`` plus N ``step``s); the other
    three are pure. Injecting them is what lets the core be driven over plain
    integers in an offline test while the live driver hands it savestates.

    Budgets: ``config.frame_budget`` refuses to build a plan longer than
    itself, and ``config.expansion_budget`` stops the search outright. Both
    are surfaced -- a search that stopped because it ran out of budget comes
    back ``exhausted=True``, so a caller can tell "nothing better exists" from
    "I was not allowed to look".
    """
    if not actions:
        raise ValueError("a segment search needs at least one TimedAction")
    score_before = float(score(initial_state))
    if reached(initial_state, score_before):
        return SolvedPlan((), (), score_before, score_before, 0, True)

    root: _Node[StateT, TokenT] = _Node(
        initial_token, initial_state, (), (), score_before
    )
    beam = [root]
    best = root
    goal: _Node[StateT, TokenT] | None = None
    expanded = 0
    exhausted = False

    for _depth in range(max(1, int(config.max_depth))):
        children: list[_Node[StateT, TokenT]] = []
        for node in beam:
            for action in actions:
                if len(node.plan) + action.frames > config.frame_budget:
                    continue
                if expanded >= config.expansion_budget:
                    exhausted = True
                    break
                token, state = expand(node.token, action)
                expanded += 1
                child = _Node(
                    token,
                    state,
                    node.plan + tuple(action.plan),
                    node.segments + ((action.label, action.frames),),
                    float(score(state)),
                )
                children.append(child)
                if (child.score, len(child.plan)) < (best.score, len(best.plan)):
                    best = child
                if reached(state, child.score):
                    if goal is None or (len(child.plan), child.score) < (
                        len(goal.plan),
                        goal.score,
                    ):
                        goal = child
            if exhausted:
                break

        if goal is not None:
            best = goal
            break
        if exhausted or not children:
            break

        # Quantized state de-duplication prevents identical idle / held
        # branches from consuming the whole beam. This is the pruning: without
        # it the beam fills with pulse-length variants of one idea, and the
        # search spends its expansion budget re-proving that four frames of
        # LEFT and twelve frames of LEFT go the same way.
        unique: dict[tuple[Any, ...], _Node[StateT, TokenT]] = {}
        for child in children:
            key = state_key(child.state)
            incumbent = unique.get(key)
            rank = child.score + config.frame_penalty * len(child.plan)
            if incumbent is None or rank < (
                incumbent.score + config.frame_penalty * len(incumbent.plan)
            ):
                unique[key] = child
        beam = sorted(
            unique.values(),
            key=lambda node: (
                node.score + config.frame_penalty * len(node.plan),
                len(node.plan),
                node.segments,
            ),
        )[: max(1, int(config.beam_width))]
        if not beam:
            break

    return SolvedPlan(
        plan=best.plan,
        segments=best.segments,
        score_before=score_before,
        score_after=best.score,
        expanded=expanded,
        reached=goal is not None,
        exhausted=exhausted,
    )


# --- The zelda objective -------------------------------------------------


def body_cost(obj: ZeldaObject) -> float:
    """Cost of one live body: the kill, plus the HP still on it.

    The HP term is *normalised by the type's ROM maximum*
    (``dungeon.species``), and that is the one place this module needs C5.
    Raw ``$03xx`` HP is not comparable across types -- Gohma is 96 and a
    goriya is 32 -- so an un-normalised sum makes one boss outweigh three
    bodies the sword can actually reach, and the beam chases the number
    instead of the room. An unknown id, or a type the ROM array does not
    reach, scores as full: never a discount for ignorance.
    """
    top = species_of(int(obj.type_id)).hp
    if not top:
        return SOLVER_KILL_WEIGHT + SOLVER_HP_WEIGHT
    share = min(max(int(obj.hp) / float(top), 0.0), 1.0)
    return SOLVER_KILL_WEIGHT + SOLVER_HP_WEIGHT * share


@dataclass(frozen=True)
class ClearObjective:
    """Score a room clear: fewer bodies, less HP on them, Link unharmed.

    Lower is better, like the Super Metroid adapter and unlike every ``max``
    in ``rollout.py`` -- a search minimises a cost so that "nothing left to
    do" is zero and a caller can read the number.

    ``room`` pins the ``(level, screen)`` the search is about. A branch that
    scrolls out, takes the stairs or opens a menu is not cheap, it is
    *incomparable*: the slot census renumbers and the score would read as a
    total clear. :data:`LEFT_ROOM_COST` prices that, rather than rejecting it,
    so a boxed-in search still returns something.

    ``unreachable`` is the room's own geometry -- ``0x1A``'s sealed centre
    cross traps a goriya where no sword can go -- and a body it names is
    dropped from both the cost and the goal. That is a deliberate, documented
    lie about the room: the alternative is a search that spends its whole
    budget walking at something it cannot hit.
    """

    room: tuple[int, int] | None = None
    unreachable: Callable[[ZeldaObject], bool] | None = None
    damage_weight: float = SOLVER_DAMAGE_WEIGHT
    approach_weight: float = SOLVER_APPROACH_WEIGHT
    approach_clamp: int = SOLVER_APPROACH_CLAMP

    def bodies(self, snap: ZeldaSnapshot) -> tuple[ZeldaObject, ...]:
        """Live, killable, reachable slots. The set the goal is about."""
        live = live_enemies(snap)
        if self.unreachable is None:
            return tuple(live)
        return tuple(o for o in live if not self.unreachable(o))

    def elsewhere(self, snap: ZeldaSnapshot) -> bool:
        """True when this snapshot is not the room the search is about."""
        if not in_play(snap):
            return True
        if self.room is None:
            return False
        return (int(snap.level), int(snap.screen)) != self.room

    def approach(self, snap: ZeldaSnapshot) -> int:
        """Clamped Chebyshev gap to the nearest reachable body; 0 when none."""
        bodies = self.bodies(snap)
        if not bodies:
            return 0
        gap = min(
            chebyshev(int(snap.link_x), int(snap.link_y), int(o.x), int(o.y))
            for o in bodies
        )
        return min(max(gap, 0), int(self.approach_clamp))

    def score(self, snap: ZeldaSnapshot) -> float:
        if self.elsewhere(snap):
            return LEFT_ROOM_COST
        cost = sum(body_cost(o) for o in self.bodies(snap))
        cost += self.approach_weight * self.approach(snap)
        # Health enters as a *level*, not a delta: every candidate shares one
        # baseline, so the constant cancels and the objective stays a pure
        # function of the snapshot -- which is what lets a test score a
        # hand-built snapshot without staging a search around it.
        return cost - self.damage_weight * float(heart_value(snap))

    def reached(self, snap: ZeldaSnapshot, _score: float = 0.0) -> bool:
        return not self.elsewhere(snap) and not self.bodies(snap)


def room_state_key(snap: ZeldaSnapshot) -> tuple[int, ...]:
    """Quantized identity of a room state, for de-duplication.

    Link's position at 4 px, his facing, and the census (how many bodies and
    how much HP is left on them). Two branches that agree on all of it are the
    same position in the room reached two ways, and keeping both costs a beam
    slot for nothing. The 4 px quantum is the Super Metroid adapter's, and it
    is under Link's ~1.3 px/frame step, so a segment that actually moved him
    never collapses into the one that did not.
    """
    bodies = live_enemies(snap)
    return (
        int(snap.level),
        int(snap.screen),
        int(snap.link_x) // 4,
        int(snap.link_y) // 4,
        int(snap.facing),
        len(bodies),
        sum(int(o.hp) for o in bodies),
        int(snap.health),
    )


# --- The live driver -----------------------------------------------------


@dataclass
class RoomSolver:
    """Plan a room from the live frame, commit the plan, drop out on budget.

    Borrows the env through :class:`~zelda_i.rollout.Rollout`, which restores
    it in a ``finally`` -- **one emulator per process**, and this never makes
    one. :meth:`act` is the whole per-frame surface: a ``FrameAction`` to
    claim the frame, or ``None`` to decline it, which is the exact shape an
    ``overworld.arbiter.Rung`` wants.

    A plan is **committed**, not re-decided. Re-searching every frame would
    cost ~9 s of wall clock per second of walk and would also re-learn the
    two-pixel tug-of-war this tree keeps re-learning: a bearing that rotates
    one pixel reverses a per-frame decision. The queue is dropped on a room
    change and on any frame the ROM is not simulating a walk.

    The room budget is charged from ``Rollout.frames_rolled``, the kernel's
    own ledger, so it cannot drift from what was actually rolled -- and
    :meth:`report` surfaces it, because a report that hides a ``set_state`` is
    lying about the walk.
    """

    rollout: Rollout
    objective: ClearObjective = field(default_factory=ClearObjective)
    actions: tuple[TimedAction, ...] = field(default_factory=segment_library)
    config: SearchConfig = field(default_factory=SearchConfig)
    room_budget: int = SOLVER_ROOM_ROLLOUT_FRAMES
    reason_prefix: str = "solver"
    # Accounting.
    searches: int = 0
    claims: int = 0
    committed_frames: int = 0
    goals: int = 0
    declines: dict[str, int] = field(default_factory=dict)
    last_plan: SolvedPlan | None = field(default=None, repr=False)
    _room: tuple[int, int] | None = field(default=None, init=False, repr=False)
    _room_frames: int = field(default=0, init=False, repr=False)
    _queue: list[tuple[Frame, str]] = field(
        default_factory=list, init=False, repr=False
    )

    # --- the question -------------------------------------------------- #

    def act(self, snap: ZeldaSnapshot) -> FrameAction | None:
        """This frame's advice, or ``None`` when this rung wants nothing."""
        room = (int(snap.level), int(snap.screen))
        if room != self._room:
            # Slots renumber across a scroll and the budget is per visit.
            self._room = room
            self._room_frames = 0
            self._queue.clear()
        if not in_play(snap):
            # A scroll, a cellar drop or a menu: the ROM is not simulating a
            # walk, so neither a plan made here nor one made before it means
            # anything. The controller below keeps the frame.
            self._queue.clear()
            return self._decline("not_in_play")
        if self._queue:
            return self._play()
        objective = replace(self.objective, room=room)
        if objective.reached(snap):
            return self._decline("clear")
        if self._room_frames >= self.room_budget:
            # The drop-out. Past the room budget this rung is silent for the
            # rest of the visit and the position table underneath drives.
            return self._decline("room_budget")
        plan = self.solve(snap, objective=objective)
        if not plan.plan:
            return self._decline("no_plan")
        self._queue.extend(zip(plan.plan, plan.frame_labels()))
        self.committed_frames += len(plan.plan)
        return self._play()

    def solve(
        self,
        snap: ZeldaSnapshot,
        *,
        objective: ClearObjective | None = None,
    ) -> SolvedPlan:
        """One search from the live frame. Restores the emulator, always."""
        goal = objective if objective is not None else replace(
            self.objective, room=(int(snap.level), int(snap.screen))
        )
        spent_before = self.rollout.frames_rolled
        with self.rollout.branching() as branch:

            def expand(token: Any, action: TimedAction) -> tuple[Any, ZeldaSnapshot]:
                child, outcome = branch.play(action.label, action.plan, token)
                return child, outcome.snap

            plan = beam_search(
                branch.root,
                snap,
                actions=self.actions,
                expand=expand,
                score=goal.score,
                reached=lambda state, value: goal.reached(state, value),
                state_key=room_state_key,
                config=self.config,
            )
        self.searches += 1
        self._room_frames += self.rollout.frames_rolled - spent_before
        self.goals += int(plan.reached)
        self.last_plan = plan
        return plan

    # --- internals ----------------------------------------------------- #

    def _play(self) -> FrameAction:
        frame, label = self._queue.pop(0)
        self.claims += 1
        return FrameAction(list(frame), f"{self.reason_prefix}_{label}")

    def _decline(self, reason: str) -> None:
        self.declines[reason] = self.declines.get(reason, 0) + 1
        return None

    def release(self) -> None:
        """Drop the committed plan, keep the ledger (the room changed under us)."""
        self._queue.clear()

    def reset(self) -> None:
        """New walk, same emulator: drop the commit, the budget and the ledger."""
        self._queue.clear()
        self._room = None
        self._room_frames = 0
        self.searches = self.claims = self.committed_frames = self.goals = 0
        self.declines = {}
        self.last_plan = None
        self.rollout.reset()

    @property
    def queued(self) -> int:
        """Frames of committed plan still to play."""
        return len(self._queue)

    def report(self) -> dict[str, Any]:
        """The ledger a run report must not hide, plus the budget it ran on."""
        return {
            "searches": self.searches,
            "claims": self.claims,
            "committed_frames": self.committed_frames,
            "goals": self.goals,
            "room_frames": self._room_frames,
            "declines": dict(sorted(self.declines.items(), key=lambda kv: -kv[1])),
            "last_plan": self.last_plan.report() if self.last_plan else None,
            "budget": {
                "beam_width": self.config.beam_width,
                "max_depth": self.config.max_depth,
                "frame_budget": self.config.frame_budget,
                "expansion_budget": self.config.expansion_budget,
                "frame_penalty": self.config.frame_penalty,
                "room_budget": self.room_budget,
            },
            **self.rollout.report(),
        }
