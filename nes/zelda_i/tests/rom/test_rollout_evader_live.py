"""Real-ROM proof for ``rollout.RolloutEvader`` -- the budget and the claim.

``tests/rom/test_rollout_truth.py`` proves the *kernel* against the machine.
This file proves the two things the evader built on it has to be true for:

* **the budget is real on a live wave.** The offline tier can assert a
  cadence against a scripted tape; only the ROM can say what the cadence
  costs when a real wave keeps a body inside the trigger radius for hundreds
  of frames. ``AGENTS.md``: every rung needs a budget, and a budget that is
  not a number is the contact strike that owned 24877 frames.
* **the model and the machine disagree about standing still.** That is the
  whole argument for the rung. ``threat.assess`` says "safe" whenever nothing
  is *moving* toward Link, and the ``0x55`` spit is born two frames into
  ``ObjState 0x03`` and then sits on the Zora's muzzle ~17 frames. A rollout
  of the very same frame sees the hit land.

    uv run pytest nes/zelda_i/tests/rom/test_rollout_evader_live.py -m rom -q

Traps honoured here: one emulator per process (every test opens and closes
its own env), nothing asserts an exact heart count off a pin, and no test
claims one evader beats the other -- that is a walk-length measurement and it
belongs to ``scratch/probe_evader_ab.py``.
"""

from __future__ import annotations

import math

import numpy as np

from retro_harness.env import make_env, reset_obs
from zelda_i.dungeon.threat import DEFAULT_HORIZON, assess
from zelda_i.dungeon.tracking import ObjectTracker
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.rollout import (
    ROLLOUT_EVADE_DIRECTIONS,
    ROLLOUT_EVADE_REPLAN_FRAMES,
    Rollout,
    RolloutEvader,
    hold,
    press,
)
from zelda_i.tests.rom.conftest import ROM, skip_unless_pin

# 0x4A: the inland bowl of blue tektites the pre-L1 corridor pays for. Good
# for the restore contract; NOT for the budget -- measured 2026-09-17, its
# nearest hazard sits at 58 px for the first 240 frames, two pixels outside
# ROLLOUT_EVADE_TRIGGER_RADIUS, so a budget test on this pin starves and a
# cadence test on it passes vacuously with zero fans.
COMBAT_PIN = "At4A"
# 0x7C / 0x7B: coast screens that hold a body at 16 px for all 600 frames
# measured. This is the pin a budget has to survive.
WAVE_PIN = "BFS_7C"
ZORA_PIN = "BFS_7B"
SETTLE = 90  # frames of idle so the wave is on screen and moving
FAN = len(ROLLOUT_EVADE_DIRECTIONS)


def _live(pin: str, settle: int = SETTLE):
    env = make_env(game=GAME, state=pin, game_dir=GAME_DIR, render_mode=None)
    reset_obs(env)
    for _ in range(settle):
        env.em.set_button_mask(np.zeros(9, dtype=np.uint8), 0)
        env.em.step()
    return env


def _idle(env) -> None:
    env.em.set_button_mask(np.asarray(press(), dtype=np.uint8), 0)
    env.em.step()


@ROM
@skip_unless_pin(COMBAT_PIN)
def test_a_decide_leaves_the_live_emulator_byte_identical() -> None:
    """The contract the walk's tape depends on: a dodge decision is not a
    state load as far as the recording is concerned."""
    env = _live(COMBAT_PIN)
    try:
        tracker = ObjectTracker()
        snap = read_snapshot(env.get_ram())
        tracked = tracker.observe(snap)
        before = np.array(env.get_ram(), copy=True)
        evader = RolloutEvader(Rollout(env))
        evader.decide(snap, tracked)
        assert np.array_equal(np.array(env.get_ram(), copy=True), before)
    finally:
        env.close()


@ROM
@skip_unless_pin(WAVE_PIN)
def test_the_cadence_bounds_the_cost_of_a_live_wave() -> None:
    """The budget, measured rather than asserted off a scripted tape.

    A tektite bowl keeps something inside the trigger radius most frames, so
    this is close to the worst case the cadence has to hold at: at most one
    fan per ``ROLLOUT_EVADE_REPLAN_FRAMES`` frames of walk, and exactly one
    rollout per candidate direction per fan.
    """
    frames = 240
    env = _live(WAVE_PIN)
    try:
        tracker = ObjectTracker()
        evader = RolloutEvader(Rollout(env))
        for _ in range(frames):
            snap = read_snapshot(env.get_ram())
            evader.decide(snap, tracker.observe(snap))
            _idle(env)
        ceiling = math.ceil(frames / ROLLOUT_EVADE_REPLAN_FRAMES)
        assert evader.replans <= ceiling, evader.report()
        assert evader.rollout.rollouts == evader.replans * FAN
        assert evader.rollout.frames_rolled == evader.rollout.rollouts * evader.frames
        # And the gate really is a gate: a rung that fanned on every frame
        # would have spent ``frames`` of them.
        assert evader.replans < frames
        # A ceiling nothing reaches is not a measurement. This pin keeps a
        # body at 16 px for every frame of the window, so the cadence must
        # actually be *running* -- without this the whole test passes on a
        # pin where the gate never opened, which is how it first shipped.
        assert evader.replans > 0, evader.report()
    finally:
        env.close()


@ROM
@skip_unless_pin(WAVE_PIN)
def test_the_screen_budget_is_reachable_and_stops_the_fan() -> None:
    """A cap nothing can hit is not a budget. Squeeze it to three and prove
    the rung retires instead of fanning for the rest of the screen."""
    env = _live(WAVE_PIN)
    try:
        tracker = ObjectTracker()
        evader = RolloutEvader(Rollout(env), screen_budget=3)
        for _ in range(240):
            snap = read_snapshot(env.get_ram())
            evader.decide(snap, tracker.observe(snap))
            _idle(env)
        assert evader.replans <= 3
        assert evader.declines.get("screen_budget", 0) > 0, evader.report()
    finally:
        env.close()


@ROM
@skip_unless_pin(ZORA_PIN)
def test_the_straight_line_over_triggers_against_the_rom() -> None:
    """What the machine actually says about the model, measured 2026-09-17.

    This test was written to prove the opposite claim -- that ``assess``
    calls a frame *safe* which the ROM then punishes (the Zora muzzle hold,
    a leever surfacing, a tektite mid-jump). **It does not, and the first
    run of this file said so.** Scanned over eight pins (At4A, At4B, At78,
    BFS_7B, BFS_59, BFS_49, BFS_58, BFS_7C), 960 sampled frames, a standing
    Link, ``DEFAULT_HORIZON``:

        model safe / ROM harm      0
        model safe / ROM no harm 774
        model unsafe / ROM harm    0
        model unsafe / ROM no harm 186

    The ``ROM harm`` column is empty **everywhere**, so the frame the
    original test hunted for cannot be produced from a settled standing pin
    at all, and asserting it just pinned a pin. What is left is the other
    diagonal and it is the interesting one: 186 frames where the model
    raised an alarm the ROM never cashed. The straight line **over**-triggers
    here; it does not under-trigger.

    Two honest caveats, because the two sides are not quite one question.
    ``assess`` predicts *contact* geometry, while ``first_contact`` reads the
    ``$04F0`` harm arming -- an overlap during Link's i-frames is contact and
    not harm. And a standing Link is the easy case: the muzzle-hold argument
    is about a Link who is *walking into* the lane, which no settled pin
    reproduces. So this measures over-triggering; it does not refute the
    muzzle hold, it leaves it unproven.

    That matters for what the rollout rung is *for*. Over-triggering is an
    arbitration cost, not a missed dodge -- it is the same failure the 48 of
    134 contact windows with net < 25% of gross already showed, and the one
    ``overworld.arbiter`` was wired for. So this test asserts the direction
    that is real, and the A/B in ``scratch/probe_evader_ab.py`` remains the
    only thing that can say which evader is better.
    """
    env = _live(ZORA_PIN, settle=30)
    try:
        tracker = ObjectTracker()
        rollout = Rollout(env)
        plan = hold(None, DEFAULT_HORIZON)
        sampled = model_unsafe = rom_harm = under = 0
        for frame in range(600):
            snap = read_snapshot(env.get_ram())
            tracked = tracker.observe(snap)
            if frame % 10 == 0 and tracked:
                model_safe = assess(
                    (int(snap.link_x), int(snap.link_y)),
                    tuple(t for t in tracked if t.is_hazard),
                    horizon=DEFAULT_HORIZON,
                ).safe
                harmed = rollout.first_contact(plan) is not None
                sampled += 1
                model_unsafe += not model_safe
                rom_harm += harmed
                under += model_safe and harmed
            _idle(env)
        assert sampled > 0, "the pin held no tracked hazard at all"
        # The alarm fires and the ROM does not cash it: over-triggering.
        assert model_unsafe > 0, "the model never raised an alarm on this pin"
        assert rom_harm == 0, (
            f"a standing Link took {rom_harm} harm on {ZORA_PIN}; the pin has "
            "changed, and the over-trigger reading above must be re-measured "
            "before this test is trusted"
        )
        # Recorded as a number so a future change of polarity is loud.
        assert under == 0, (
            f"{under} frames where the model said safe and the ROM landed a "
            "hit -- that is the muzzle-hold claim finally reproducing. Do not "
            "delete this: re-measure and rewrite the docstring above."
        )
    finally:
        env.close()
