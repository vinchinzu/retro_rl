from __future__ import annotations

from punch_out.policy import BoutMode, GlassJoePolicy
from punch_out.ram import FIGHT_BETWEEN, FIGHT_IN_RING

# Import via module path used by scripts (package-local helper).
import importlib.util
from pathlib import Path

_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_glass_joe.py"
_spec = importlib.util.spec_from_file_location("run_glass_joe", _RUNNER)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

classify_bout_outcome = _mod.classify_bout_outcome
normalize_goal = _mod.normalize_goal


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
