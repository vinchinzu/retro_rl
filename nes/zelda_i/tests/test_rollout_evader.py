"""Offline cover for the ROM-truth evader and the rung it sits on.

No emulator. A ``_FakeEmulator`` plays the part exactly as it does in
``test_rollout.py``: it owns an integer "state", steps a scripted RAM tape per
button mask, and ledgers every save/restore. That is enough to assert the two
things this module has to get right and cannot get right by accident --

* **the budget**, which is a *count of rollouts*, not a feeling. Every test
  below that names a cadence, a radius or a screen cap asserts
  ``rollout.report()["rollouts"]`` or ``replans``, because a rung whose cost
  is not a number is the contact strike that owned 24877 frames
  (``AGENTS.md`` Traps);
* **the fall-back**, which is that the ``ReactiveEvader`` below it keeps every
  frame the rollout declines.

What is *not* here is whether the rollout dodges better than the model. That
is a live-ROM question and it is answered by ``scratch/probe_evader_ab.py``,
not by this file. The live half of the kernel is
``tests/rom/test_rollout_truth.py`` and ``tests/rom/test_rollout_evader_live.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from zelda_i.dungeon.tracking import EnemyKind, HazardClass, TrackedObject
from zelda_i.overworld.path import (
    THREAT_RUNG_DUCK,
    THREAT_RUNG_REACTIVE,
    THREAT_RUNG_ROLLOUT,
    OverworldPathController,
    ScreenHop,
)
from zelda_i.ram import PLAY_MODE, read_snapshot
from zelda_i.rollout import (
    ROLLOUT_EVADE_DIRECTIONS,
    ROLLOUT_EVADE_REPLAN_FRAMES,
    ROLLOUT_EVADE_TRIGGER_RADIUS,
    Rollout,
    RolloutEvader,
    press,
)
from zelda_i.tests.ram_helpers import make_ram

_DEFAULTS = {"mode": PLAY_MODE, "level": 0, "screen": 0x7B, "x": 100, "y": 100,
             "health": 0x22, "sword": 1}
IDLE = press()
FRAMES = 24  # the evader's default horizon
FAN = len(ROLLOUT_EVADE_DIRECTIONS)


def _ram(**fields) -> np.ndarray:
    ram = make_ram(_DEFAULTS, **fields)
    ram[0x0670] = 0xFF  # heart partial full; heart_value reads it
    return ram


def _snap(**fields):
    return read_snapshot(_ram(**fields))


class _FakeEmulator:
    """Steps a per-mask RAM tape and ledgers every state save/restore."""

    def __init__(self, tapes: dict, base: np.ndarray) -> None:
        self.tapes = {tuple(k): v for k, v in tapes.items()}
        self.ram = base
        self.base = base
        self.mask = IDLE
        self.cursor = 0
        self.saves = 0
        self.restores = 0
        self.steps = 0

    def get_state(self) -> bytes:
        self.saves += 1
        return b"pin"

    def set_state(self, state: bytes) -> None:
        assert state == b"pin"
        self.restores += 1
        self.ram = self.base
        self.cursor = 0

    def set_button_mask(self, mask, _player: int = 0) -> None:
        # The real core misreads a Python int as a one-button array, which is
        # a no-op that looks like a working rollout. Hold the array contract.
        assert isinstance(mask, np.ndarray) and mask.dtype == np.uint8, mask
        self.mask = tuple(int(v) for v in mask)

    def step(self) -> None:
        self.steps += 1
        tape = self.tapes.get(self.mask, ())
        if self.cursor < len(tape):
            self.ram = tape[self.cursor]
        self.cursor += 1


class _FakeEnv:
    def __init__(self, em: _FakeEmulator) -> None:
        self.em = em

    def get_ram(self) -> np.ndarray:
        return self.em.ram


class _Hazard:
    """The duck-typed half of a ``TrackedObject`` the trigger gate reads."""

    def __init__(self, x: int, y: int, is_hazard: bool = True) -> None:
        self.x, self.y, self.is_hazard = x, y, is_hazard


def _tape(hit_at: int | None, n: int = FRAMES, **fields) -> list[np.ndarray]:
    """``n`` frames of RAM whose iframe timer arms at ``hit_at`` (1-based)."""
    return [
        _ram(link_iframes=24 if hit_at is not None and i + 1 >= hit_at else 0,
             **fields)
        for i in range(n)
    ]


def _dir_masks() -> dict[str, tuple[int, ...]]:
    return {d: (IDLE if d == "STAND" else press(d)) for d in ROLLOUT_EVADE_DIRECTIONS}


def _env(per_direction: dict[str, list[np.ndarray]], base=None):
    """An env whose tape depends on which direction the plan holds."""
    masks = _dir_masks()
    tapes = {masks[d]: tape for d, tape in per_direction.items()}
    em = _FakeEmulator(tapes, _ram() if base is None else base)
    return _FakeEnv(em), em


def _evader(per_direction: dict[str, list[np.ndarray]], base=None, **kwargs):
    env, em = _env(per_direction, base)
    return RolloutEvader(Rollout(env), **kwargs), em


# ``LEFT`` walks clear, everything else is hit on frame 2. The gain is then
# (24 + 1) - 2 = 23 frames, far over ``min_gain``.
_LEFT_IS_THE_ESCAPE = {
    "LEFT": _tape(None),
    "RIGHT": _tape(2),
    "UP": _tape(2),
    "DOWN": _tape(2),
    "STAND": _tape(2),
}
_NEAR = (_Hazard(100 + ROLLOUT_EVADE_TRIGGER_RADIUS - 4, 100),)
_FAR = (_Hazard(100 + ROLLOUT_EVADE_TRIGGER_RADIUS + 4, 100),)


def _body(x: int, y: int = 100) -> TrackedObject:
    """A real tracked body, for the rungs that read more than ``.x`` / ``.y``.

    The duck rung above the rollout one asks for ``hazard`` and
    ``blockable``, so a path-level test cannot hand the ladder the duck-typed
    stub the kernel is happy with.
    """
    return TrackedObject(
        slot=1, type_id=0x0E, x=x, y=y, vx=0.0, vy=0.0, hp=0x20, state=1,
        facing=0x02, age=8, kind=EnemyKind.LEEVER, hazard=HazardClass.BODY,
        blockable=False,
    )


_NEAR_BODY = (_body(100 + ROLLOUT_EVADE_TRIGGER_RADIUS - 4),)


# --- the trigger gate: only while something is in range ---------------- #

def test_nothing_in_range_never_costs_a_rollout() -> None:
    """BEHAVIOURAL. The gate is the budget: a fan is 28 ms and a walk is
    200k frames, so a rung that fans on quiet frames is not affordable."""
    evader, em = _evader(_LEFT_IS_THE_ESCAPE)
    for _ in range(40):
        assert evader.decide(_snap(), _FAR) is None
    assert (evader.replans, em.steps, em.saves) == (0, 0, 0)
    assert evader.declines == {"no_hazard": 40}


def test_no_hazards_at_all_is_the_same_decline() -> None:
    """BEHAVIOURAL. An empty census is not a reason to roll the ROM."""
    evader, em = _evader(_LEFT_IS_THE_ESCAPE)
    assert evader.decide(_snap(), ()) is None
    assert (evader.replans, em.steps) == (0, 0)


def test_a_dormant_slot_is_not_a_trigger() -> None:
    """BEHAVIOURAL. ``is_hazard`` False (a leever under the sand, a corpse
    slot) must not open the gate — ``AGENTS.md``: a leever with ``ObjState``
    0 is not a body."""
    evader, em = _evader(_LEFT_IS_THE_ESCAPE)
    assert evader.decide(_snap(), (_Hazard(104, 100, is_hazard=False),)) is None
    assert (evader.replans, em.steps) == (0, 0)


def test_a_hazard_inside_the_radius_buys_exactly_one_fan() -> None:
    """BEHAVIOURAL. One fan is one rollout per candidate direction, and no
    more: the horizon and the direction count are the whole per-fan cost."""
    evader, em = _evader(_LEFT_IS_THE_ESCAPE)
    step = evader.decide(_snap(), _NEAR)
    assert step is not None and step.direction == "LEFT"
    assert evader.replans == 1
    assert evader.rollout.report() == {
        "rollouts": FAN,
        "frames_rolled": FAN * FRAMES,
    }
    assert em.steps == FAN * FRAMES


def test_the_radius_is_chebyshev_on_the_nearest_hazard_only() -> None:
    """BEHAVIOURAL. One body in range opens the gate however many are out."""
    evader, _ = _evader(_LEFT_IS_THE_ESCAPE)
    hazards = (_Hazard(250, 250), _Hazard(100, 140), _Hazard(20, 20))
    assert evader.decide(_snap(), hazards) is not None
    assert evader.replans == 1


# --- the re-plan interval: the other half of the budget ---------------- #

def test_the_replan_interval_is_the_budget_not_a_preference() -> None:
    """BEHAVIOURAL. 24 frames of continuous threat is 3 fans, not 24. This
    is the assertion the card asks for: the *number of rollouts* over N
    frames. At one fan per frame the rung would cost 28 ms of wall clock per
    frame of walk, which is three orders of magnitude over budget."""
    evader, em = _evader(_LEFT_IS_THE_ESCAPE)
    claimed = [evader.decide(_snap(), _NEAR) for _ in range(24)]
    assert evader.replans == 24 // ROLLOUT_EVADE_REPLAN_FRAMES == 3
    assert evader.rollout.rollouts == 3 * FAN
    assert em.steps == 3 * FAN * FRAMES
    # Every frame is still answered; the held ones just cost nothing.
    assert all(s is not None and s.direction == "LEFT" for s in claimed)
    assert [s.replanned for s in claimed[:9]] == [True] + [False] * 7 + [True]
    assert (evader.claims, evader.held_frames) == (24, 21)


def test_a_declining_fan_costs_the_same_cooldown_as_a_dodging_one() -> None:
    """BEHAVIOURAL. The failure this exists to stop: a hazard parked in range
    on a frame where standing is fine would otherwise re-buy the fan every
    single frame — the decline is the expensive branch, not the cheap one."""
    even = {d: _tape(None) for d in ROLLOUT_EVADE_DIRECTIONS}
    evader, _ = _evader(even)
    for _ in range(16):
        assert evader.decide(_snap(), _NEAR) is None
    assert evader.replans == 2
    assert evader.declines == {"no_gain": 2, "hold_stand": 14}


def test_a_step_that_only_postpones_the_hit_is_not_an_escape() -> None:
    """BEHAVIOURAL. ``min_gain`` is ``threat.MIN_ESCAPE_GAIN``'s rule: a
    two-frame reprieve is oscillation fuel. Standing is hit on frame 8 and
    the best walk on frame 10, so the rung declines and the ladder below it
    drives."""
    evader, _ = _evader({
        "STAND": _tape(8), "LEFT": _tape(10), "RIGHT": _tape(9),
        "UP": _tape(8), "DOWN": _tape(7),
    })
    assert evader.decide(_snap(), _NEAR) is None
    assert evader.declines == {"no_gain": 1}


def test_a_clear_escape_past_min_gain_is_taken() -> None:
    """BEHAVIOURAL. The mirror of the above at one frame more of gain."""
    evader, _ = _evader({
        "STAND": _tape(8), "LEFT": _tape(12), "RIGHT": _tape(9),
        "UP": _tape(8), "DOWN": _tape(7),
    })
    step = evader.decide(_snap(), _NEAR)
    assert step is not None and step.direction == "LEFT"
    assert (step.contact_frame, step.stand_contact, step.gain) == (12, 8, 4)


# --- what the ROM sees that the model cannot --------------------------- #

def test_a_wall_loses_the_safe_tie_without_any_occupancy_grid() -> None:
    """BEHAVIOURAL. ``ReactiveEvader._can_move`` tests the scroll rectangle,
    so it cannot see a rock and dodges into one — screen-wide poisoning of
    0x7B for eight hits. Here LEFT walks into rock (0 px of ``moved``) and
    RIGHT walks; both are safe, and the ROM breaks the tie."""
    evader, _ = _evader({
        "STAND": _tape(4),
        "LEFT": [_ram(x=100) for _ in range(FRAMES)],
        "RIGHT": [_ram(x=100 + i) for i in range(1, FRAMES + 1)],
        "UP": _tape(4), "DOWN": _tape(4),
    })
    step = evader.decide(_snap(), _NEAR)
    assert step is not None and step.direction == "RIGHT" and step.moved == FRAMES


def test_a_dodge_that_scrolls_the_screen_is_not_a_dodge() -> None:
    """BEHAVIOURAL. A step over the scroll line has changed hops, not
    dodged: the same rule ``path._evade_blocked_dirs`` keeps for the model,
    measured here instead of guessed from a margin."""
    gone = [_ram(screen=0x7C) for _ in range(FRAMES)]
    evader, _ = _evader({
        "STAND": _tape(4), "LEFT": gone, "RIGHT": _tape(4),
        "UP": _tape(4), "DOWN": _tape(4),
    })
    assert evader.decide(_snap(), _NEAR) is None
    assert evader.declines == {"no_gain": 1}


def test_every_direction_boxed_in_declines_rather_than_pressing_one() -> None:
    """BEHAVIOURAL. With nothing left to try the honest answer is the rung
    below, not a button."""
    gone = [_ram(screen=0x7C) for _ in range(FRAMES)]
    evader, _ = _evader({d: gone for d in ROLLOUT_EVADE_DIRECTIONS})
    assert evader.decide(_snap(), _NEAR) is None
    assert evader.declines == {"boxed_in": 1}


# --- the caller's blocked set ------------------------------------------ #

def test_a_blocked_direction_is_never_rolled_out() -> None:
    """BEHAVIOURAL. Dropping it before the fan buys the rollout back; doing
    it after would pay for an answer the caller had already refused."""
    evader, em = _evader(_LEFT_IS_THE_ESCAPE)
    step = evader.decide(_snap(), _NEAR, blocked=("LEFT", "UP"))
    assert step is None  # the only escape was refused
    assert evader.rollout.rollouts == FAN - 2
    assert em.steps == (FAN - 2) * FRAMES


def test_stand_survives_a_caller_that_blocks_everything() -> None:
    """BEHAVIOURAL. ``STAND`` is the baseline the gain is measured against;
    a caller banning it would make every shuffle look decisive."""
    evader, _ = _evader(_LEFT_IS_THE_ESCAPE)
    assert evader.decide(
        _snap(), _NEAR, blocked=("LEFT", "RIGHT", "UP", "DOWN", "STAND")
    ) is None
    assert evader.rollout.rollouts == 1
    assert evader.declines == {"boxed_in": 1}


# --- the guards: a rollout has to mean something ----------------------- #

def test_a_transition_falls_back_to_the_reactive_evader() -> None:
    """BEHAVIOURAL. ``in_play`` False is a scroll, a cave or a menu: the ROM
    is not simulating a walk, so a rollout of one is not a prediction."""
    evader, em = _evader(_LEFT_IS_THE_ESCAPE)
    assert evader.decide(_snap(mode=6), _NEAR) is None
    assert evader.decide(_snap(mode=11), _NEAR) is None  # cave
    assert (evader.replans, em.steps) == (0, 0)
    assert evader.declines == {"not_in_play": 2}


def test_the_screen_budget_retires_the_rung_on_one_screen() -> None:
    """BEHAVIOURAL. ``AGENTS.md``: every rung needs a budget, and local caps
    do not compose. The cadence bounds cost per frame; only this bounds it
    per screen, where a wave can park a body in range for 4000 frames."""
    evader, _ = _evader(_LEFT_IS_THE_ESCAPE, screen_budget=2)
    steps = [evader.decide(_snap(), _NEAR) for _ in range(80)]
    assert evader.replans == 2
    assert evader.declines.get("screen_budget") == 80 - 2 * ROLLOUT_EVADE_REPLAN_FRAMES
    assert steps[-1] is None


def test_a_scroll_refunds_the_screen_budget_and_drops_the_commit() -> None:
    """BEHAVIOURAL. The budget is per screen *visit*, and a commit made
    against slot 3 on 0x7B means nothing on 0x7C — slots renumber."""
    evader, _ = _evader(_LEFT_IS_THE_ESCAPE, screen_budget=1)
    for _ in range(40):
        evader.decide(_snap(), _NEAR)
    assert evader.replans == 1
    step = evader.decide(_snap(screen=0x7C), _NEAR)
    assert step is not None and step.replanned
    assert evader.replans == 2


# --- the contract the caller's tape depends on ------------------------- #

def test_the_evader_leaves_the_live_emulator_where_it_found_it() -> None:
    """BEHAVIOURAL. One save per fan, one restore before each plan plus one
    final restore back to the caller — the walk's own frame is untouched."""
    evader, em = _evader(_LEFT_IS_THE_ESCAPE)
    evader.decide(_snap(), _NEAR)
    assert (em.saves, em.restores) == (1, FAN + 1)
    assert em.ram is em.base


def test_report_ledgers_the_restores_and_the_budget_it_ran_on() -> None:
    """STRUCTURAL. A report that hides a ``set_state`` is lying about the
    walk, so the ledger and the numbers that bounded it travel together."""
    evader, _ = _evader(_LEFT_IS_THE_ESCAPE)
    for _ in range(9):
        evader.decide(_snap(), _NEAR)
    out = evader.report()
    assert out["replans"] == 2
    assert out["rollouts"] == 2 * FAN
    assert out["frames_rolled"] == 2 * FAN * FRAMES
    assert out["claims"] == 9 and out["held_frames"] == 7
    assert out["budget"]["replan_frames"] == ROLLOUT_EVADE_REPLAN_FRAMES
    assert out["budget"]["trigger_radius"] == ROLLOUT_EVADE_TRIGGER_RADIUS


def test_reset_drops_the_commit_and_the_ledger_but_keeps_the_env() -> None:
    """STRUCTURAL. A new walk on the same emulator: the kernel was handed in
    and is not the controller's to drop."""
    evader, em = _evader(_LEFT_IS_THE_ESCAPE)
    evader.decide(_snap(), _NEAR)
    env = evader.rollout.env
    evader.reset()
    assert evader.report()["rollouts"] == 0 and evader.replans == 0
    assert evader.declines == {}
    assert evader.rollout.env is env
    # The commit is gone, so the next in-range frame re-plans immediately.
    evader.decide(_snap(), _NEAR)
    assert evader.replans == 1 and em.saves == 2


# --- the rung on the ladder -------------------------------------------- #

def _controller(**kwargs):
    return OverworldPathController(
        hops=(ScreenHop(0x7C, "RIGHT"),), evade=True, **kwargs
    )


def test_the_threat_ladder_is_duck_then_rollout_then_reactive() -> None:
    """STRUCTURAL. Ordering is a number a test can read, which is the whole
    of ``overworld.arbiter``. The duck stays on top: it is the one rung that
    answers a ``0x55`` and it is live on the M5 Clean chain."""
    assert THREAT_RUNG_DUCK < THREAT_RUNG_ROLLOUT < THREAT_RUNG_REACTIVE
    names = [r.name for r in _controller().threat_arbiter.rungs]
    assert names == ["threat_duck", "threat_rollout", "threat_reactive"]


def test_attaching_the_rollout_does_not_change_the_ladder_shape() -> None:
    """STRUCTURAL. Both arms are the *same* ladder with one flag between
    them, so a census taken on either is directly comparable and
    ``threat_rollout: 0`` is a reading, not a missing row."""
    ctl = _controller()
    before = [(r.name, r.priority) for r in ctl.threat_arbiter.rungs]
    env, _ = _env(_LEFT_IS_THE_ESCAPE)
    ctl.attach_rollout(env)
    assert [(r.name, r.priority) for r in ctl.threat_arbiter.rungs] == before


def test_the_rung_is_off_until_it_is_attached() -> None:
    """BEHAVIOURAL. The default arm is the tree that was here before: no
    rollout, no ledger in the report, and ``ReactiveEvader`` driving."""
    ctl = _controller()
    assert ctl.rollout_evade is False
    snap = _snap()
    ctl._observe_threats(snap)
    assert ctl._rung_rollout_evade(snap) is None
    assert "rollout" not in ctl.report()
    assert ctl.report()["threat_census"]["threat_rollout"] == 0


def test_attach_rollout_selects_the_arm_and_the_rung_claims_the_frame() -> None:
    """BEHAVIOURAL. With the arm selected the rung owns the frame, the
    reason string names it, and the census credits it — which is what makes
    an A/B attributable rather than a rupee count."""
    ctl = _controller()
    env, _ = _env(_LEFT_IS_THE_ESCAPE)
    evader = ctl.attach_rollout(env)
    assert ctl.rollout_evade is True and ctl._rollout is evader
    snap = _snap()
    ctl._observe_threats(snap)
    ctl._tracked = _NEAR_BODY  # a real body the whole ladder can read
    act = ctl._threat_action(snap, None)
    assert act is not None and act.reason == "rollout_evade"
    assert ctl.threat_arbiter.last_winner == "threat_rollout"
    assert ctl.evade_reasons["rollout_evade"] == 1
    assert ctl.report()["threat_census"]["threat_rollout"] == 1


def test_a_sword_window_does_not_hide_a_rom_measured_escape() -> None:
    """BEHAVIOURAL. A stationary Zora muzzle is invisible to the velocity
    tracker, so ``_shot_first`` cannot exempt it from the model's sword-yield
    gate. The rollout needs no such gate: STAND is its control arm. If STAND
    is hit and LEFT is safe, the measured escape outranks the swing; if both
    are safe, ``no_gain`` hands the frame back to the hunter below."""
    ctl = _controller()
    env, _ = _env(_LEFT_IS_THE_ESCAPE)
    ctl.attach_rollout(env)
    ctl.hunter = SimpleNamespace(striking=lambda _snap: True)
    snap = _snap()
    ctl._tracked = _NEAR_BODY

    act = ctl._rung_rollout_evade(snap)

    assert act is not None and act.reason == "rollout_evade"
    assert ctl._rollout is not None and ctl._rollout.replans == 1
    assert "rollout_yield_to_sword" not in ctl.evade_reasons


def test_a_frame_the_rollout_declines_falls_through_to_the_rung_below() -> None:
    """BEHAVIOURAL. The fall-back is the point of a flagged swap: nothing is
    deleted, so a declined frame is decided exactly as it is today."""
    ctl = _controller()
    env, _ = _env({d: _tape(None) for d in ROLLOUT_EVADE_DIRECTIONS})
    ctl.attach_rollout(env)
    snap = _snap()
    ctl._observe_threats(snap)
    ctl._tracked = _NEAR_BODY
    assert ctl._threat_action(snap, None) is None
    assert ctl._rollout is not None and ctl._rollout.declines == {"no_gain": 1}
    census = ctl.report()["threat_census"]
    assert census["threat_rollout"] == 0 and census["yielded"] == 1


def test_the_run_report_surfaces_the_rollout_ledger_beside_the_census() -> None:
    """STRUCTURAL. Requirement of the card and of ``docs/STATUS.md``'s
    spirit: a rollout is a ``set_state`` and the report has to say so."""
    ctl = _controller()
    env, _ = _env(_LEFT_IS_THE_ESCAPE)
    ctl.attach_rollout(env)
    snap = _snap()
    ctl._observe_threats(snap)
    ctl._tracked = _NEAR_BODY
    ctl._threat_action(snap, None)
    out = ctl.report()
    assert out["rollout"]["rollouts"] == FAN
    assert out["rollout"]["frames_rolled"] == FAN * FRAMES
    assert "rung_census" in out and "threat_census" in out
    import json

    json.loads(json.dumps(out["rollout"]))  # a report has to be writable


def test_reset_keeps_the_bound_kernel_and_zeroes_its_ledger() -> None:
    """STRUCTURAL. The kernel holds the live env — dropping it would silence
    the arm mid-walk, the same rule the hunter gets."""
    ctl = _controller()
    env, _ = _env(_LEFT_IS_THE_ESCAPE)
    evader = ctl.attach_rollout(env)
    snap = _snap()
    ctl._observe_threats(snap)
    ctl._tracked = _NEAR_BODY
    ctl._threat_action(snap, None)
    ctl.reset()
    assert ctl._rollout is evader and ctl.rollout_evade is True
    assert evader.report()["rollouts"] == 0


def test_the_budget_knobs_are_constructor_arguments_not_attributes() -> None:
    """BEHAVIOURAL (the second half), and it is the ``AGENTS.md`` dataclass
    trap: a default
    written onto the class after it exists is a no-op on every instance, so
    every ablation before 2026-09-16 measured the unablated hunt.
    ``attach_rollout`` therefore passes the knobs to ``__init__``."""
    ctl = _controller()
    env, _ = _env(_LEFT_IS_THE_ESCAPE)
    evader = ctl.attach_rollout(env, replan_frames=4, trigger_radius=8)
    assert (evader.replan_frames, evader.trigger_radius) == (4, 8)
    # And the knob is real: at radius 8 the same hazard is out of range.
    assert evader.decide(_snap(), _NEAR) is None
    assert evader.replans == 0


def test_a_bad_knob_name_is_a_typeerror_not_a_silent_default() -> None:
    """STRUCTURAL. A probe that misspells a knob must fail loudly."""
    env, _ = _env(_LEFT_IS_THE_ESCAPE)
    with pytest.raises(TypeError):
        _controller().attach_rollout(env, replan_framez=4)
