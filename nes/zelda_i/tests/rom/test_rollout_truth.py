"""Real-ROM proof that ``zelda_i.rollout`` predicts, rather than models.

The ROM tier was combat-free until this file. Every other tier can only check
that the *model* is self-consistent; these tests check the model against the
machine, which is the only place the enemy AI actually lives.

    uv run pytest nes/zelda_i/tests/rom/test_rollout_truth.py -m rom -q

Traps honoured here: one emulator per process (every test opens and closes
its own env), and nothing asserts an exact heart count off a pin.
"""

from __future__ import annotations

import numpy as np

from retro_harness.env import make_env, reset_obs
from zelda_i.dungeon.tracking import HazardClass, ObjectTracker
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.rollout import Rollout, hold, press
from zelda_i.tests.rom.conftest import ROM, skip_unless_pin

# 0x4A: the inland bowl of blue tektites the pre-L1 corridor pays for.
# 0x78: the row-7 octorok screen, straight-line walkers, the easy case.
COMBAT_PIN = "At4A"
WALKER_PIN = "At78"
SETTLE = 90  # frames of idle so the wave is on screen and moving


def _live(pin: str, settle: int = SETTLE):
    env = make_env(game=GAME, state=pin, game_dir=GAME_DIR, render_mode=None)
    reset_obs(env)
    for _ in range(settle):
        env.em.set_button_mask(np.zeros(9, dtype=np.uint8), 0)
        env.em.step()
    return env


@ROM
@skip_unless_pin(COMBAT_PIN)
def test_a_fan_leaves_the_live_emulator_byte_identical() -> None:
    """The whole contract. A walk can ask this on any frame and keep its tape."""
    env = _live(COMBAT_PIN)
    try:
        before = np.array(env.get_ram(), copy=True)
        rollout = Rollout(env)
        rollout.fan({d: hold(d, 40) for d in ("UP", "DOWN", "LEFT", "RIGHT")})
        assert np.array_equal(np.array(env.get_ram(), copy=True), before)
        assert rollout.report() == {"rollouts": 4, "frames_rolled": 160}
    finally:
        env.close()


@ROM
@skip_unless_pin(COMBAT_PIN)
def test_the_walk_continues_identically_after_a_fan() -> None:
    """RAM equality is necessary but not sufficient — the core's own hidden
    state (PPU phase, PRNG, APU) has to survive too, or the next 200 frames
    diverge and every recorded tape after a rollout is a different run."""
    env = _live(COMBAT_PIN)
    try:
        pin = env.em.get_state()
        clean = []
        for _ in range(200):
            env.em.set_button_mask(np.asarray(press("LEFT"), dtype=np.uint8), 0)
            env.em.step()
            clean.append(np.array(env.get_ram(), copy=True))

        env.em.set_state(pin)
        Rollout(env).fan({d: hold(d, 40) for d in ("UP", "DOWN", "RIGHT")})
        after = []
        for _ in range(200):
            env.em.set_button_mask(np.asarray(press("LEFT"), dtype=np.uint8), 0)
            env.em.step()
            after.append(np.array(env.get_ram(), copy=True))

        first_divergence = next(
            (i for i, (a, b) in enumerate(zip(clean, after)) if not np.array_equal(a, b)),
            None,
        )
        assert first_divergence is None, f"walk diverged at frame {first_divergence}"
    finally:
        env.close()


@ROM
@skip_unless_pin(COMBAT_PIN)
def test_a_rollout_is_bit_exact_not_approximate() -> None:
    """This is why the enemy AI does not need porting: replay is the future,
    PRNG and animation phase included, to the byte."""
    env = _live(COMBAT_PIN)
    try:
        rollout = Rollout(env)
        plan = hold("LEFT", 60)
        first = rollout.run("a", plan).snap
        second = rollout.run("a", plan).snap
        assert first == second
    finally:
        env.close()


@ROM
@skip_unless_pin(COMBAT_PIN)
def test_different_plans_reach_different_futures() -> None:
    """A guard against a silent no-op: if set_button_mask stopped reaching the
    core, every test above would still pass."""
    env = _live(COMBAT_PIN)
    try:
        outcomes = Rollout(env).fan({d: hold(d, 40) for d in ("UP", "DOWN", "LEFT", "RIGHT")})
        seats = {(int(o.snap.link_x), int(o.snap.link_y)) for o in outcomes}
        assert len(seats) > 1, f"every direction landed on {seats}"
    finally:
        env.close()


@ROM
@skip_unless_pin(WALKER_PIN)
def test_rollout_beats_linear_extrapolation_at_the_dodge_horizon() -> None:
    """The measurement this module was written for.

    ``threat.MIN_DODGE_BODY`` is 16, so every dodge is a claim about frame 16
    and beyond, and ``TrackedObject.at`` answers it by extending a 6-sample
    mean velocity in a straight line. Roll the ROM forward instead and score
    both against the truth. The rollout's error is zero by construction; the
    point of asserting it is that the *linear* error is not small, measured
    live, on the easiest screen in the game — straight-line walkers.
    """
    env = _live(WALKER_PIN)
    try:
        tracker = ObjectTracker()
        rollout = Rollout(env)
        horizon = 32
        errors: list[int] = []
        for frame in range(600):
            env.em.set_button_mask(np.zeros(9, dtype=np.uint8), 0)
            env.em.step()
            tracked = tracker.observe(read_snapshot(env.get_ram()))
            if frame < 60 or frame % 20:
                continue
            bodies = [t for t in tracked if t.hazard is HazardClass.BODY and t.hp > 0]
            if not bodies:
                continue
            truth = rollout.run("idle", hold(None, horizon)).snap
            real = {int(o.slot): (int(o.x), int(o.y), int(o.hp)) for o in truth.objects}
            for body in bodies:
                seat = real.get(body.slot)
                if seat is None or seat[2] <= 0:
                    continue  # died or left: not a prediction error
                px, py = body.at(float(horizon))
                errors.append(int(abs(px - seat[0]) + abs(py - seat[1])))

        assert len(errors) >= 30, f"too few samples to judge ({len(errors)})"
        whiffs = sum(1 for e in errors if e >= 16)
        # A hitbox is 16 px. Linear extrapolation misses by a whole one often
        # enough that a dodge chosen on it is a coin flip; the exact rate
        # drifts with the pin, so the bar is deliberately loose.
        assert whiffs / len(errors) >= 0.05, (
            f"linear extrapolation looked accurate ({whiffs}/{len(errors)} "
            "off by a hitbox) — re-measure before trusting this claim"
        )
    finally:
        env.close()


@ROM
@skip_unless_pin(COMBAT_PIN)
def test_best_step_is_wall_aware_without_any_occupancy_grid() -> None:
    """``ReactiveEvader._can_move`` tests a rectangle, so it dodges into rocks.
    A rollout asks the ROM, so a blocked direction reports the 0 px it walked.
    """
    env = _live(COMBAT_PIN)
    try:
        outcomes = Rollout(env).fan({d: hold(d, 24) for d in ("UP", "DOWN", "LEFT", "RIGHT")})
        moved = {o.label: o.moved for o in outcomes}
        assert any(v > 0 for v in moved.values()), moved
        best = Rollout(env).best_step(frames=24)
        assert best.label in {*moved, "STAND"}
        if best.label != "STAND":
            assert best.moved > 0 or not best.safe
    finally:
        env.close()
