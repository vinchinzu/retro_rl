"""Real-ROM proof that ``zelda_i.solver`` searches without spending the walk.

    uv run pytest nes/zelda_i/tests/rom/test_solver_live.py -m rom -q

Deliberately **contract** assertions, not outcome ones. What a multi-segment
search finds on a given pin is a measurement, and a measurement belongs in a
probe with a report attached to it -- not in a test that has to be green on
every machine and every ROM revision. What must hold on every pin, every
time, is exactly three things: the emulator comes back where it was, the
budgets are respected, and the ledger does not hide the ``set_state``.

The pin is ``At4A`` -- the inland bowl of blue tektites the pre-L1 corridor
pays for -- because it is the combat pin ``test_rollout_truth.py`` already
uses, so this file adds no new fixture. The objective is a plain
``ClearObjective`` over the overworld screen, which is the same search the L7
``0x1A`` wiring runs; the room it points at is whatever the pin is actually
on, read from RAM rather than asserted.

Traps honoured: one emulator per process (each test opens and closes its own
env) and nothing asserts an exact heart count off a pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from retro_harness.env import make_env, reset_obs
from zelda_i.combat import live_enemies
from zelda_i.paths import GAME, GAME_DIR
from zelda_i.ram import read_snapshot
from zelda_i.rollout import Rollout, in_play, press
from zelda_i.solver import (
    SOLVER_EXPANSION_BUDGET,
    ClearObjective,
    RoomSolver,
    SearchConfig,
    segment_library,
)
from zelda_i.tests.rom.conftest import ROM, skip_unless_pin

COMBAT_PIN = "At4A"
SETTLE = 90  # frames of idle so the wave is on screen and moving


def _live(pin: str, settle: int = SETTLE):
    env = make_env(game=GAME, state=pin, game_dir=GAME_DIR, render_mode=None)
    reset_obs(env)
    for _ in range(settle):
        env.em.set_button_mask(np.zeros(9, dtype=np.uint8), 0)
        env.em.step()
    return env


def _ready(env):
    """The live snapshot, or a skip. A search needs a walk and a wave.

    Not a hidden assertion: a pin that has drifted to a cave mouth or has had
    its wave cleared is a *fixture* problem, and a contract test that fails on
    it says the wrong thing about the solver.
    """
    snap = read_snapshot(env.get_ram())
    if not in_play(snap):
        pytest.skip(f"{COMBAT_PIN} is not in live play (mode {snap.mode})")
    if not live_enemies(snap):
        pytest.skip(f"{COMBAT_PIN} has no live wave to search against")
    return snap


def _solver(env, **kwargs) -> RoomSolver:
    kwargs.setdefault("actions", segment_library(swings=True, stand=False))
    kwargs.setdefault("config", SearchConfig(beam_width=2, max_depth=2))
    kwargs.setdefault("objective", ClearObjective())
    return RoomSolver(Rollout(env), **kwargs)


@ROM
@skip_unless_pin(COMBAT_PIN)
def test_a_search_leaves_the_live_emulator_byte_identical() -> None:
    """The whole contract. A walk can search on any frame and keep its tape."""
    env = _live(COMBAT_PIN)
    try:
        snap = _ready(env)
        before = np.array(env.get_ram(), copy=True)
        solver = _solver(env)
        plan = solver.solve(snap)
        assert np.array_equal(np.array(env.get_ram(), copy=True), before)
        assert plan.expanded > 0
        assert solver.report()["frames_rolled"] > 0
    finally:
        env.close()


@ROM
@skip_unless_pin(COMBAT_PIN)
def test_a_search_stays_inside_every_budget_it_declares() -> None:
    """Every rung needs a budget, and a budget that is not checked is a claim.

    The plan cap, the expansion cap and the room ledger, on the real machine:
    the same three numbers ``tests/test_solver.py`` proves offline, asserted
    once against a search that really did run the ROM.
    """
    env = _live(COMBAT_PIN)
    try:
        snap = _ready(env)
        config = SearchConfig(beam_width=2, max_depth=2, frame_budget=40)
        solver = _solver(env, config=config)
        plan = solver.solve(snap)
        assert plan.frame_count <= config.frame_budget
        assert plan.expanded <= SOLVER_EXPANSION_BUDGET
        report = solver.report()
        assert report["room_frames"] == report["frames_rolled"] > 0
        assert report["searches"] == 1
    finally:
        env.close()


@ROM
@skip_unless_pin(COMBAT_PIN)
def test_the_walk_continues_identically_after_a_search() -> None:
    """RAM equality is necessary but not sufficient: the core's own hidden
    state (PPU phase, PRNG, APU) has to survive the whole search too, or every
    tape recorded after a solver frame is a different run."""
    env = _live(COMBAT_PIN)
    try:
        _ready(env)
        pin = env.em.get_state()
        left = np.asarray(press("LEFT"), dtype=np.uint8)

        clean = []
        for _ in range(150):
            env.em.set_button_mask(left, 0)
            env.em.step()
            clean.append(np.array(env.get_ram(), copy=True))

        env.em.set_state(pin)
        _solver(env).solve(read_snapshot(env.get_ram()))
        after = []
        for _ in range(150):
            env.em.set_button_mask(left, 0)
            env.em.step()
            after.append(np.array(env.get_ram(), copy=True))

        divergence = next(
            (i for i, (a, b) in enumerate(zip(clean, after)) if not np.array_equal(a, b)),
            None,
        )
        assert divergence is None, f"the walk diverged at frame {divergence}"
    finally:
        env.close()


@ROM
@skip_unless_pin(COMBAT_PIN)
def test_the_room_budget_stops_the_solver_dead() -> None:
    """The drop-out, on the real machine: a solver with no budget left never
    touches the core, so a controller under it pays nothing for the rung."""
    env = _live(COMBAT_PIN)
    try:
        snap = _ready(env)
        solver = _solver(env, room_budget=0)
        for _ in range(10):
            assert solver.act(snap) is None
        assert solver.searches == 0
        assert solver.report()["frames_rolled"] == 0
        assert solver.declines["room_budget"] == 10
    finally:
        env.close()
