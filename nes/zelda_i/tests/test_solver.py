"""Offline cover for ``zelda_i.solver`` — the multi-segment room search.

Three tiers, and the split matters when reading a failure:

* **Generic core.** :func:`~zelda_i.solver.beam_search` is driven over plain
  integers with injected callbacks and no emulator at all — the same trick
  ``snes/super_metroid/tests/test_room_adapter.py`` uses. Every budget, the
  partial-plan contract and the de-duplication are proved here, where there
  is nothing but the search to blame.
* **The alphabet.** The segment builders, including the two ROM traps that
  are structural properties of a plan rather than facts about a run: a turn
  and a swing cannot share a frame, and ``ButtonsPressed`` is an edge.
* **The live driver**, against a ``_FakeCore`` that branches: it owns a real
  state token, it plays a button script, and it hands a token back — enough
  for a three-segment search with a goal in it, and enough to assert that the
  caller's emulator comes back exactly where it was.

Marked in each docstring: *behavioural* (it would fail if the behaviour
changed) or *structural* (it pins a name, a shape or a constant).
"""

from __future__ import annotations

from itertools import count

import numpy as np
import pytest

from zelda_i.combat import live_enemies
from zelda_i.dungeon.ids import GORIYA_OBJECT_TYPE
from zelda_i.ram import (
    ADDR_LINK_FACING,
    ADDR_LINK_X,
    ADDR_LINK_Y,
    ADDR_OBJ_HP,
    ADDR_OBJ_TYPE,
    PLAY_MODE,
    read_snapshot,
)
from zelda_i.rollout import Rollout, press, swing_after
from zelda_i.solver import (
    SOLVER_APPROACH_CLAMP,
    SOLVER_EXPANSION_BUDGET,
    SOLVER_KILL_WEIGHT,
    SOLVER_PLAN_FRAMES,
    SOLVER_ROOM_ROLLOUT_FRAMES,
    SOLVER_TURN_FRAMES,
    ClearObjective,
    RoomSolver,
    SearchConfig,
    SolvedPlan,
    TimedAction,
    beam_search,
    body_cost,
    hold_action,
    room_state_key,
    segment_library,
    stand_action,
    swing_action,
)
from zelda_i.tests.ram_helpers import make_ram

IDLE = press()
A_INDEX = press("A").index(1)
ROOM = (7, 0x1A)

# --- the integer world the generic core is tested in ---------------------
# One action, one state, one goal: the search has to walk an integer to a
# target. Nothing about zelda is in scope here, which is the point.
ONE = TimedAction(IDLE, 1, "one")
TWO = TimedAction(IDLE, 2, "two")
FOUR = TimedAction(IDLE, 4, "four")


def _walk(target: int, *, actions=(ONE, TWO, FOUR), config=None) -> SolvedPlan:
    return beam_search(
        0,
        0,
        actions=actions,
        expand=lambda token, action: (
            token + action.frames,
            token + action.frames,
        ),
        score=lambda state: float(abs(target - state)),
        reached=lambda state, score: score == 0.0,
        state_key=lambda state: (state,),
        config=config or SearchConfig(beam_width=8, max_depth=4, frame_penalty=1.0),
    )


# --- generic core --------------------------------------------------------


def test_beam_search_reaches_the_goal_in_the_fewest_frames() -> None:
    """Behavioural. Five frames of goal, found as 4+1 and not as 1+1+1+1+1."""
    plan = _walk(5)
    assert plan.reached
    assert plan.frame_count == 5
    assert plan.score_before == 5.0
    assert plan.score_after == 0.0
    assert plan.labels in (("four", "one"), ("one", "four"), ("two", "two", "one"))


def test_beam_search_returns_the_best_bounded_partial_plan() -> None:
    """Behavioural. The card's drop-out: not the goal, still the best prefix."""
    plan = beam_search(
        0,
        0,
        actions=(TWO,),
        expand=lambda token, action: (
            token + action.frames,
            token + action.frames,
        ),
        score=lambda state: float(abs(10 - state)),
        reached=lambda _state, score: score == 0.0,
        state_key=lambda state: (state,),
        config=SearchConfig(beam_width=1, max_depth=2, frame_penalty=0.0),
    )
    assert not plan.reached
    assert plan.frame_count == 4
    assert plan.score_after == 6.0


def test_a_goal_already_met_costs_no_expansions() -> None:
    """Behavioural. A cleared room is not a search."""
    plan = _walk(0)
    assert plan.reached
    assert plan.plan == ()
    assert plan.expanded == 0


def test_the_expansion_budget_stops_the_search_and_says_so() -> None:
    """Behavioural. The budget, not the depth, is what ends this search."""
    plan = _walk(
        1000,
        config=SearchConfig(
            beam_width=8, max_depth=9, frame_penalty=0.0,
            frame_budget=10_000, expansion_budget=7,
        ),
    )
    assert plan.expanded <= 7
    assert plan.exhausted
    assert not plan.reached


def test_the_frame_budget_refuses_to_build_a_longer_plan() -> None:
    """Behavioural. No returned plan outlives the state it was planned from."""
    plan = _walk(
        1000,
        config=SearchConfig(
            beam_width=4, max_depth=9, frame_penalty=0.0,
            frame_budget=9, expansion_budget=10_000,
        ),
    )
    assert plan.frame_count <= 9
    assert not plan.reached


def test_the_state_key_dedup_collapses_identical_branches() -> None:
    """Behavioural, and the number is the pruning itself.

    Four actions that all advance the integer by one are four *identical*
    branches. With a key on the state they collapse to one beam slot and the
    second depth expands 4 nodes; with a key that never collides they each
    keep a slot and the second depth expands 16. Asserting the *expanded*
    count is the only way to see that — the plan is the same either way.
    """
    clones = tuple(TimedAction(IDLE, 1, f"clone{i}") for i in range(4))
    kwargs = dict(
        actions=clones,
        expand=lambda token, action: (token + 1, token + 1),
        score=lambda state: float(abs(9 - state)),
        reached=lambda _state, score: score == 0.0,
        config=SearchConfig(
            beam_width=4, max_depth=2, frame_penalty=0.0,
            expansion_budget=10_000,
        ),
    )
    collapsed = beam_search(0, 0, state_key=lambda state: (state,), **kwargs)
    tick = count()
    spread = beam_search(0, 0, state_key=lambda state: (state, next(tick)), **kwargs)
    assert collapsed.expanded == 8
    assert spread.expanded == 20
    assert collapsed.expanded < spread.expanded


def test_the_frame_penalty_breaks_a_tie_toward_the_shorter_plan() -> None:
    """Behavioural. Cost is a penalty, not a cliff: both branches are legal and
    both reach the same state, so only the frames they cost separate them —
    and with the penalty off, the survivor is whichever was expanded first."""
    short = TimedAction(IDLE, 1, "short")
    long_ = TimedAction(IDLE, 6, "long")

    def run(penalty: float) -> SolvedPlan:
        return beam_search(
            0,
            0,
            actions=(long_, short),
            # Both advance the state by one; only the frames differ.
            expand=lambda token, action: (token + 1, token + 1),
            score=lambda state: float(abs(3 - state)),
            reached=lambda _state, score: score == 0.0,
            state_key=lambda state: (state,),
            config=SearchConfig(beam_width=1, max_depth=2, frame_penalty=penalty),
        )

    assert run(1.0).labels == ("short", "short")
    assert run(0.0).labels[0] == "long"


def test_beam_search_without_an_action_raises() -> None:
    """Structural. An empty alphabet is a caller bug, not an empty plan."""
    with pytest.raises(ValueError, match="at least one TimedAction"):
        beam_search(
            0, 0, actions=(), expand=lambda t, a: (t, t), score=lambda s: 0.0,
            reached=lambda s, v: True, state_key=lambda s: (s,),
        )


# --- the segment alphabet ------------------------------------------------


def test_timed_action_rejects_a_non_positive_pulse() -> None:
    """Structural. A zero-frame segment is a node the search cannot leave."""
    with pytest.raises(ValueError, match="frames must be positive"):
        TimedAction(IDLE, 0, "nothing")


def test_timed_action_rejects_a_bitmask() -> None:
    """Behavioural. ``set_button_mask`` misreads an int as the B button, so a
    search built from ints looks deterministic because it is doing nothing."""
    with pytest.raises(ValueError, match="action vector from press"):
        TimedAction(0x40, 4, "bitmask")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="binary"):
        TimedAction((0, 0, 0, 0, 0, 0, 2, 0, 0), 4, "not_binary")


def test_a_held_segment_is_one_frame_repeated() -> None:
    """Behavioural. A pulse is held, not pressed once."""
    action = hold_action("LEFT", 5)
    assert action.plan == (press("LEFT"),) * 5
    assert stand_action(3).plan == (IDLE,) * 3


def test_a_swing_never_puts_the_turn_and_the_a_on_one_frame() -> None:
    """Behavioural, and the trap ``AGENTS.md`` names outright.

    ``nes_action(face, "A")`` from a walking Link keeps the *old* facing 22
    times in 64 and the blade goes out along the axis the body is not on.
    Every frame before the A holds the new direction alone.
    """
    for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
        plan = swing_action(direction).plan
        a_frames = [i for i, f in enumerate(plan) if f[A_INDEX] == 1]
        assert a_frames == [SOLVER_TURN_FRAMES], direction
        assert all(f == press(direction) for f in plan[:SOLVER_TURN_FRAMES])
        assert plan[SOLVER_TURN_FRAMES] == press(direction, "A")


def test_two_swings_back_to_back_have_a_release_between_them() -> None:
    """Behavioural. ``ButtonsPressed`` is an edge: held A does not re-swing."""
    plan = swing_action("LEFT").plan + swing_action("UP").plan
    a_frames = [i for i, f in enumerate(plan) if f[A_INDEX] == 1]
    assert len(a_frames) == 2
    first, second = a_frames
    assert second > first + 1
    assert all(plan[i][A_INDEX] == 0 for i in range(first + 1, second))


def test_a_swing_with_no_turn_frame_is_refused() -> None:
    """Structural. The one rule that buys both traps at once."""
    with pytest.raises(ValueError, match="cannot share a frame"):
        swing_action("LEFT", turn_frames=0)
    with pytest.raises(ValueError, match="needs a frame for A"):
        swing_action("LEFT", turn_frames=4, frames=4)


def test_a_swing_segment_is_exactly_the_rollout_builder() -> None:
    """Structural. One builder for the shape, not a second copy of it."""
    assert swing_action("UP", 4, 20).plan == swing_after("UP", 4, 20)


def test_the_default_library_is_every_direction_plus_the_swings() -> None:
    """Structural. 4 directions x 3 pulses + 4 swings + stand."""
    library = segment_library()
    assert len(library) == 17
    assert sum(1 for a in library if a.label.startswith("swing_")) == 4
    assert len(segment_library(swings=False, stand=False)) == 12


# --- the zelda objective -------------------------------------------------

_DEFAULTS = {"mode": PLAY_MODE, "level": 7, "screen": 0x1A, "x": 100, "y": 100,
             "health": 0x22, "sword": 1}


def _ram(bodies=(), **fields) -> np.ndarray:
    ram = make_ram(_DEFAULTS, **fields)
    ram[0x0670] = 0xFF  # heart partial full
    for slot, (x, y, hp, type_id) in enumerate(bodies, start=1):
        ram[ADDR_LINK_X + slot] = x
        ram[ADDR_LINK_Y + slot] = y
        ram[ADDR_OBJ_HP + slot] = hp
        ram[ADDR_OBJ_TYPE + slot] = type_id
    return ram


def _snap(bodies=(), **fields):
    return read_snapshot(_ram(bodies, **fields))


GORIYA = GORIYA_OBJECT_TYPE


def test_a_dead_room_scores_below_a_live_one() -> None:
    """Behavioural. Lower is better, and a kill is the biggest single step."""
    objective = ClearObjective(room=ROOM)
    live = objective.score(_snap(((108, 100, 48, GORIYA),)))
    dead = objective.score(_snap())
    assert dead < live - SOLVER_KILL_WEIGHT


def test_chipping_a_body_scores_below_leaving_it_full() -> None:
    """Behavioural. The HP term is what lets the beam keep one target."""
    objective = ClearObjective(room=ROOM)
    full = objective.score(_snap(((108, 100, 48, GORIYA),)))
    chipped = objective.score(_snap(((108, 100, 24, GORIYA),)))
    assert chipped < full


def test_walking_closer_scores_below_standing_far_away() -> None:
    """Behavioural. A flat score is a blind beam: approach has to be progress."""
    objective = ClearObjective(room=ROOM)
    far = objective.score(_snap(((200, 100, 48, GORIYA),)))
    near = objective.score(_snap(((120, 100, 48, GORIYA),)))
    assert near < far


def test_approach_is_clamped_below_the_price_of_a_kill() -> None:
    """Behavioural. Approach must never outbid a kill."""
    objective = ClearObjective(room=ROOM)
    assert objective.approach(_snap(((255, 255, 48, GORIYA),))) == SOLVER_APPROACH_CLAMP
    live_far = objective.score(_snap(((255, 255, 48, GORIYA),)))
    dead = objective.score(_snap())
    assert dead < live_far


def test_losing_a_heart_costs_more_than_one_kill() -> None:
    """Behavioural. ``$066F`` low nibble is whole hearts minus one; the score
    reads ``combat.heart_value``, which a wooden chip actually moves."""
    objective = ClearObjective(room=ROOM)
    healthy = objective.score(_snap(((108, 100, 48, GORIYA),)))
    hurt = objective.score(_snap(((108, 100, 48, GORIYA),), health=0x21))
    assert hurt > healthy + SOLVER_KILL_WEIGHT


def test_an_unreachable_body_is_dropped_from_the_goal() -> None:
    """Behavioural. 0x1A's sealed cross: a body no sword can reach is not the
    room's fault, and a goal that waits for it never fires."""
    sealed = _snap(((128, 144, 48, GORIYA),))
    assert not ClearObjective(room=ROOM).reached(sealed)
    trapped = ClearObjective(room=ROOM, unreachable=lambda o: int(o.x) == 128)
    assert trapped.reached(sealed)
    assert trapped.bodies(sealed) == ()


def test_leaving_the_room_is_not_a_clear() -> None:
    """Behavioural. A scroll renumbers every slot and hands out a fresh census,
    which an unguarded score reads as a total clear."""
    objective = ClearObjective(room=ROOM)
    gone = _snap(screen=0x1B)
    assert not objective.reached(gone)
    assert objective.score(gone) > objective.score(_snap(((108, 100, 48, GORIYA),)))


def test_a_transition_is_not_a_clear_either() -> None:
    """Behavioural. Mode 6/7/16 is a scroll: the ROM is not simulating a walk."""
    assert not ClearObjective(room=ROOM).reached(_snap(mode=6))


def test_body_cost_normalises_hp_by_the_rom_maximum() -> None:
    """Behavioural, and the one place ``dungeon.species`` is used.

    Raw ``$03xx`` HP is not comparable across types. A half-HP goriya must
    cost less than a full one, and a type the ROM array does not reach scores
    as full rather than free.
    """
    full = _snap(((108, 100, 48, GORIYA),)).objects[1]
    half = _snap(((108, 100, 24, GORIYA),)).objects[1]
    assert body_cost(half) < body_cost(full)
    unknown = _snap(((108, 100, 40, 0xF0),)).objects[1]
    assert body_cost(unknown) >= body_cost(full)


def test_the_state_key_collapses_a_two_pixel_step_and_not_a_four_pixel_one() -> None:
    """Behavioural. The 4 px quantum is under Link's own step, so a segment
    that moved him never collapses into the one that did not."""
    base = _snap(((108, 100, 48, GORIYA),))
    assert room_state_key(base) == room_state_key(
        _snap(((108, 100, 48, GORIYA),), x=101)
    )
    assert room_state_key(base) != room_state_key(
        _snap(((108, 100, 48, GORIYA),), x=104)
    )
    assert room_state_key(base) != room_state_key(((_snap(()))))


# --- the live driver -----------------------------------------------------


class _FakeCore:
    """A tiny branching 'ROM': Link walks, bodies block him, A cuts.

    Enough of a machine to make a multi-segment search mean something — a
    token really is a state, a token really can be returned to, and the goal
    is only reachable by a *sequence* (walk into range, then cut twice). It
    ledgers every save and restore, because the contract this fake exists to
    police is that the caller's emulator comes back where it was.
    """

    STEP = 1
    REACH = (8, 24)
    DAMAGE = 24

    def __init__(self, link=(100, 100), bodies=((60, 100, 48),), screen=0x1A) -> None:
        self.screen = screen
        self.link = link
        self.bodies = [list(b) for b in bodies]
        self.mask = IDLE
        self.saves = 0
        self.restores = 0
        self.steps = 0
        self._dirs = {press(d): d for d in ("UP", "DOWN", "LEFT", "RIGHT")}
        self._swings = {press(d, "A"): d for d in ("UP", "DOWN", "LEFT", "RIGHT")}

    # stable-retro surface
    def get_state(self):
        self.saves += 1
        return (self.link, tuple(tuple(b) for b in self.bodies))

    def set_state(self, state) -> None:
        self.restores += 1
        link, bodies = state
        self.link = tuple(link)
        self.bodies = [list(b) for b in bodies]

    def set_button_mask(self, mask, _player: int = 0) -> None:
        assert isinstance(mask, np.ndarray) and mask.dtype == np.uint8, mask
        self.mask = tuple(int(v) for v in mask)

    def step(self) -> None:
        self.steps += 1
        direction = self._dirs.get(self.mask)
        swing = self._swings.get(self.mask)
        if direction is not None:
            self._walk(direction)
        if swing is not None:
            self._cut(swing)

    _DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

    def _walk(self, direction: str) -> None:
        dx, dy = self._DELTA[direction]
        x = self.link[0] + dx * self.STEP
        y = self.link[1] + dy * self.STEP
        for bx, by, hp in self.bodies:
            if hp > 0 and abs(x - bx) < self.REACH[0] and abs(y - by) < self.REACH[0]:
                return
        self.link = (max(0, min(255, x)), max(0, min(255, y)))

    def _cut(self, direction: str) -> None:
        dx, dy = self._DELTA[direction]
        lo, hi = self.REACH
        for body in self.bodies:
            if body[2] <= 0:
                continue
            fwd = (body[0] - self.link[0]) * dx + (body[1] - self.link[1]) * dy
            lat = abs((body[0] - self.link[0]) * dy + (body[1] - self.link[1]) * dx)
            if lo <= fwd <= hi and lat <= 8:
                body[2] = max(body[2] - self.DAMAGE, 0)
                return

    def get_ram(self) -> np.ndarray:
        bodies = tuple((b[0], b[1], b[2], GORIYA) for b in self.bodies if b[2] > 0)
        return _ram(
            bodies, x=self.link[0], y=self.link[1], screen=self.screen
        )


class _FakeEnv:
    def __init__(self, core: _FakeCore) -> None:
        self.em = core

    def get_ram(self) -> np.ndarray:
        return self.em.get_ram()


def _solver(core: _FakeCore | None = None, **kwargs) -> tuple[RoomSolver, _FakeCore]:
    core = core or _FakeCore()
    kwargs.setdefault("actions", segment_library(swings=True, stand=False))
    kwargs.setdefault(
        "config", SearchConfig(beam_width=4, max_depth=3, expansion_budget=400)
    )
    return RoomSolver(Rollout(_FakeEnv(core)), **kwargs), core


def test_the_search_finds_a_sequence_a_single_ply_cannot() -> None:
    """Behavioural, and the card. The body is 40 px away and needs two cuts:
    no one held button reaches it, so the goal is only met by a *sequence*."""
    solver, core = _solver()
    plan = solver.solve(read_snapshot(core.get_ram()))
    assert plan.reached, plan.report()
    assert len(plan.labels) >= 2
    assert plan.labels[-1].startswith("swing_")
    assert plan.frame_count <= SOLVER_PLAN_FRAMES


def test_a_searched_plan_still_turns_before_it_swings() -> None:
    """Behavioural. The trap survives composition: every A in the committed
    plan is preceded by the turn frames of its own segment."""
    solver, core = _solver()
    plan = solver.solve(read_snapshot(core.get_ram()))
    frames = plan.plan
    for index, frame in enumerate(frames):
        if frame[A_INDEX] != 1:
            continue
        assert index >= SOLVER_TURN_FRAMES
        held = frames[index - SOLVER_TURN_FRAMES : index]
        facing = tuple(v for i, v in enumerate(frame) if i != A_INDEX)
        for earlier in held:
            assert earlier[A_INDEX] == 0
            assert tuple(
                v for i, v in enumerate(earlier) if i != A_INDEX
            ) == facing


def test_a_search_leaves_the_emulator_exactly_where_it_was() -> None:
    """Behavioural. The whole contract: a walk can ask this on any frame."""
    solver, core = _solver()
    before = core.get_state()
    solver.solve(read_snapshot(core.get_ram()))
    assert core.get_state() == before
    assert core.restores > 0
    assert solver.report()["frames_rolled"] > 0


def test_a_raising_score_still_restores_the_emulator() -> None:
    """Behavioural. The restore is a ``finally``, not a happy-path line."""
    solver, core = _solver()
    before = core.get_state()
    boom = ClearObjective(room=ROOM)
    object.__setattr__(boom, "unreachable", _raise)
    with pytest.raises(RuntimeError):
        solver.solve(read_snapshot(core.get_ram()), objective=boom)
    assert core.get_state() == before


def _raise(_obj):
    raise RuntimeError("scorer died mid-search")


def test_the_plan_is_committed_and_played_frame_by_frame() -> None:
    """Behavioural. A plan is committed, not re-decided: one search, then N
    frames of advice, and the emulator is not touched again in between."""
    solver, core = _solver()
    snap = read_snapshot(core.get_ram())
    first = solver.act(snap)
    assert first is not None
    assert first.reason.startswith("solver_")
    searches, rolled = solver.searches, solver.report()["frames_rolled"]
    played = [first]
    while solver.queued:
        act = solver.act(read_snapshot(core.get_ram()))
        assert act is not None
        played.append(act)
    assert solver.searches == searches
    assert solver.report()["frames_rolled"] == rolled
    assert len(played) == solver.last_plan.frame_count
    assert [a.action for a in played] == [list(f) for f in solver.last_plan.plan]


def test_the_room_budget_stops_the_solver_and_it_drops_to_the_caller() -> None:
    """Behavioural. Past the per-room budget every frame is declined, which is
    the card's "drops to the current controller"."""
    solver, core = _solver(room_budget=1)
    snap = read_snapshot(core.get_ram())
    assert solver.act(snap) is not None  # the first search is allowed
    while solver.queued:
        solver.act(read_snapshot(core.get_ram()))
    after = solver.searches
    for _ in range(50):
        assert solver.act(read_snapshot(core.get_ram())) is None
    assert solver.searches == after
    assert solver.declines["room_budget"] == 50


def test_the_room_budget_is_charged_from_the_kernels_own_ledger() -> None:
    """Behavioural. Local caps do not compose; a hand-kept counter drifts."""
    solver, core = _solver()
    solver.act(read_snapshot(core.get_ram()))
    assert solver.report()["room_frames"] == solver.report()["frames_rolled"]


def test_a_scroll_drops_the_committed_plan() -> None:
    """Behavioural. A plan made in one room is not advice in the next."""
    solver, core = _solver()
    solver.act(read_snapshot(core.get_ram()))
    assert solver.queued > 0
    assert solver.act(_snap(mode=6)) is None
    assert solver.queued == 0
    assert solver.declines["not_in_play"] == 1


def test_a_new_room_resets_the_budget() -> None:
    """Behavioural. The budget is per *visit*, like every other screen budget."""
    solver, core = _solver(room_budget=1)
    solver.act(read_snapshot(core.get_ram()))
    while solver.queued:
        solver.act(read_snapshot(core.get_ram()))
    assert solver.act(read_snapshot(core.get_ram())) is None
    # A scroll: a fresh wave, a fresh budget.
    core.screen = 0x1B
    core.link = (100, 100)
    core.bodies = [[60, 100, 48]]
    assert solver.act(read_snapshot(core.get_ram())) is not None


def test_a_cleared_room_is_declined_not_searched() -> None:
    """Behavioural. Nothing to do is not a plan."""
    solver, _ = _solver()
    assert solver.act(_snap()) is None
    assert solver.searches == 0
    assert solver.declines["clear"] == 1


def test_reset_drops_the_commit_the_budget_and_the_ledger() -> None:
    """Behavioural. A new walk on the same emulator."""
    solver, core = _solver()
    solver.act(read_snapshot(core.get_ram()))
    solver.reset()
    assert solver.queued == 0
    assert solver.searches == 0
    assert solver.report()["frames_rolled"] == 0
    assert solver.report()["room_frames"] == 0


def test_the_report_carries_the_budget_it_ran_on() -> None:
    """Structural. A report that hides a ``set_state`` is lying about the walk."""
    solver, core = _solver()
    solver.act(read_snapshot(core.get_ram()))
    report = solver.report()
    assert report["budget"]["room_budget"] == SOLVER_ROOM_ROLLOUT_FRAMES
    assert report["budget"]["frame_budget"] == SOLVER_PLAN_FRAMES
    assert set(report) >= {"searches", "claims", "rollouts", "frames_rolled"}


def test_the_default_budgets_are_the_named_constants() -> None:
    """Structural. Every rung needs a budget, and a budget needs a name."""
    config = SearchConfig()
    assert config.frame_budget == SOLVER_PLAN_FRAMES
    assert config.expansion_budget == SOLVER_EXPANSION_BUDGET
    assert RoomSolver(Rollout(_FakeEnv(_FakeCore()))).room_budget == (
        SOLVER_ROOM_ROLLOUT_FRAMES
    )


def test_a_default_search_stays_inside_its_expansion_budget() -> None:
    """Behavioural. The shipped shape does not need the hard stop; it has it."""
    solver, core = _solver(config=SearchConfig())
    plan = solver.solve(read_snapshot(core.get_ram()))
    assert plan.expanded <= SOLVER_EXPANSION_BUDGET


def test_live_enemies_is_what_the_objective_counts() -> None:
    """Structural. One census, shared with the rest of the tree."""
    snap = _snap(((108, 100, 48, GORIYA),))
    assert ClearObjective(room=ROOM).bodies(snap) == live_enemies(snap)


def test_link_facing_is_part_of_the_state_key() -> None:
    """Behavioural. Two branches that face different ways are different plans:
    the blade goes out along the facing, not along the travel."""
    a = _ram(((108, 100, 48, GORIYA),))
    b = a.copy()
    b[ADDR_LINK_FACING] = 0x08
    assert room_state_key(read_snapshot(a)) != room_state_key(read_snapshot(b))
