from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from punch_out.policy import BoutMode, GlassJoePolicy
from punch_out.ram import (
    ADDR_CLOCK_ON,
    ADDR_FIGHT_FLAG,
    ADDR_HEALTH,
    ADDR_OPP_HEALTH,
    FIGHT_BETWEEN,
    FIGHT_IN_RING,
)
from retro_harness.nes import nes_idle_action

# Import via module path used by scripts (package-local helper).
_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_glass_joe.py"
_spec = importlib.util.spec_from_file_location("run_glass_joe", _RUNNER)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

classify_bout_outcome = _mod.classify_bout_outcome
ensure_match_live = _mod._ensure_match_live
normalize_goal = _mod.normalize_goal


class _PreBellEnv:
    """Health live from the start; clock+fight after ``live_after`` idle steps."""

    def __init__(self, live_after: int) -> None:
        self.steps = 0
        self.live_after = live_after
        self.actions: list = []

    def get_ram(self):
        ram = np.zeros(0x800, dtype=np.uint8)
        ram[ADDR_HEALTH] = 96
        ram[ADDR_OPP_HEALTH] = 96
        if self.steps >= self.live_after:
            ram[ADDR_CLOCK_ON] = 1
            ram[ADDR_FIGHT_FLAG] = FIGHT_IN_RING
        return ram

    def step(self, action):
        self.actions.append(action)
        self.steps += 1
        return (f"obs{self.steps}",)

    def load_state(self, *args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("no mid-run load")


def test_normalize_goal_aliases_bout() -> None:
    assert normalize_goal("bout") == "win"
    assert normalize_goal("win") == "win"
    assert normalize_goal("knockdown") == "knockdown"


def test_classify_knockdown_goal() -> None:
    pol = GlassJoePolicy()
    pol.opp_kd = 1
    assert (
        classify_bout_outcome(
            goal="knockdown",
            policy=pol,
            opp_hp=0,
            fight=FIGHT_IN_RING,
            rnd=1,
        )
        == "knockdown"
    )


def test_classify_ko_win_long_count() -> None:
    pol = GlassJoePolicy()
    pol.opp_kd = 2
    pol.mode = BoutMode.WATCH_KD
    pol.mode_t = 700
    assert (
        classify_bout_outcome(
            goal="win",
            policy=pol,
            opp_hp=0,
            fight=FIGHT_IN_RING,
            rnd=2,
        )
        == "ko_win"
    )


def test_classify_loss_tko() -> None:
    pol = GlassJoePolicy()
    pol.mac_kd = 3
    assert (
        classify_bout_outcome(
            goal="win",
            policy=pol,
            opp_hp=50,
            fight=FIGHT_IN_RING,
            rnd=2,
        )
        == "loss_tko"
    )


def test_classify_continues_when_open() -> None:
    pol = GlassJoePolicy()
    assert (
        classify_bout_outcome(
            goal="win",
            policy=pol,
            opp_hp=96,
            fight=FIGHT_IN_RING,
            rnd=1,
        )
        is None
    )


def test_ensure_match_live_already_live_waits_zero() -> None:
    env = _PreBellEnv(live_after=0)
    obs, waited = ensure_match_live(env, "start", max_wait=50)
    assert waited == 0
    assert obs == "start"
    assert env.actions == []


def test_ensure_match_live_idles_until_clock_no_load() -> None:
    env = _PreBellEnv(live_after=7)
    obs, waited = ensure_match_live(env, "start", max_wait=50)
    assert waited == 7
    assert obs == "obs7"
    idle = nes_idle_action()
    assert len(env.actions) == 7
    assert all(np.array_equal(a, idle) for a in env.actions)


def test_ensure_match_live_caps_at_max_wait() -> None:
    env = _PreBellEnv(live_after=10_000)
    obs, waited = ensure_match_live(env, "start", max_wait=4)
    assert waited == 4
    assert obs == "obs4"
    assert len(env.actions) == 4


def test_classify_decision_win() -> None:
    pol = GlassJoePolicy()
    pol.opp_kd = 1
    pol.mac_kd = 0
    pol.mode = BoutMode.BETWEEN
    pol.mode_t = 150
    assert (
        classify_bout_outcome(
            goal="win",
            policy=pol,
            opp_hp=40,
            fight=FIGHT_BETWEEN,
            rnd=3,
        )
        == "decision_win"
    )
