"""Unit tests for Ceres Ridley tail-tank / wait policies (no emulator)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from super_metroid.combat.ceres_ridley import (
    DOOR_X,
    ENERGY_LEAVE,
    NUDGE_X,
    ROOM_CERES_RIDLEY,
    WALL_X_MIN,
    CeresRidleyEvidence,
    CeresRidleyStrategy,
    countdown_started,
    fight_ceres_ridley_action,
    fight_terminal,
    is_knockback,
    require_ceres_ridley_countdown,
)
from super_metroid.combat.protocol import wrap_ceres_ridley_as_boss_strategy
from super_metroid.ram import GameplayPhase, parse_state
from super_metroid.room_timer import NTSC_FPS, format_segment_time


def _state(**overrides):
    ram = np.zeros(0x2000, dtype=np.uint8)
    base = parse_state(ram, frame=0)
    values = {
        "phase": GameplayPhase.ORDINARY_GAMEPLAY,
        "room_id": ROOM_CERES_RIDLEY,
        "samus_x": WALL_X_MIN,
        "samus_y": 180,
        "pose": 1,
        "health": 99,
        "max_health": 99,
        "timer_type": 0,
        "enemy0_x": 200,
        "enemy0_y": 120,
        "enemy0_hp": 300,
        "num_enemies": 1,
    }
    values.update(overrides)
    return replace(base, **values)


def test_wait_policy_never_moves() -> None:
    state = _state(samus_x=40)
    assert fight_ceres_ridley_action(
        state, hits_taken=0, strategy=CeresRidleyStrategy(policy="wait")
    ) == ()


def test_approach_runs_to_right_wall() -> None:
    state = _state(samus_x=40)
    assert fight_ceres_ridley_action(state, hits_taken=0) == ("RIGHT", "B")


def test_at_wall_idles_before_hover() -> None:
    assert fight_ceres_ridley_action(_state(), hits_taken=0) == ()
    assert fight_ceres_ridley_action(_state(), hits_taken=2) == ()


def test_after_third_hit_jumps() -> None:
    action = fight_ceres_ridley_action(_state(), hits_taken=3, frames_since_hit=0)
    assert "A" in action


def test_after_fourth_hit_nudges_and_jumps() -> None:
    state = _state(samus_x=WALL_X_MIN)
    action = fight_ceres_ridley_action(state, hits_taken=4, frames_since_hit=0)
    assert "LEFT" in action
    assert "A" in action
    parked = _state(samus_x=NUDGE_X)
    assert "A" in fight_ceres_ridley_action(parked, hits_taken=4, frames_since_hit=0)


def test_low_energy_runs_to_left_door() -> None:
    state = _state(samus_x=200, health=24)
    assert fight_ceres_ridley_action(state, hits_taken=5) == ("LEFT", "B")
    at_door = _state(samus_x=DOOR_X, health=24)
    assert fight_ceres_ridley_action(at_door, hits_taken=5) == ()


def test_five_hits_still_tanks_if_energy_high() -> None:
    state = _state(samus_x=NUDGE_X, health=35)
    action = fight_ceres_ridley_action(state, hits_taken=5, frames_since_hit=0)
    assert "LEFT" not in action or "B" not in action
    assert "A" in action


def test_countdown_stops_input() -> None:
    state = _state(health=19, timer_type=3)
    assert countdown_started(state)
    assert fight_ceres_ridley_action(state, hits_taken=5) == ()
    assert state.health < ENERGY_LEAVE


def test_knockback_idles() -> None:
    state = _state(pose=137, samus_x=40)
    assert is_knockback(state)
    assert fight_ceres_ridley_action(state, hits_taken=0) == ()


def test_segment_timing_reports_frames_and_seconds() -> None:
    timing = format_segment_time(3296)
    assert timing["frames"] == 3296
    assert timing["ntsc_fps"] == NTSC_FPS
    assert abs(float(timing["seconds"]) - 3296 / NTSC_FPS) < 0.001
    assert isinstance(timing["clock"], str)
    assert ":" in str(timing["clock"])


def test_evidence_dict_includes_clock() -> None:
    ev = CeresRidleyEvidence(
        start_frame=10,
        end_frame=110,
        policy="tail_tank",
        start_health=99,
        end_health=19,
        hits=5,
        outcome="ceres_ridley_countdown",
    )
    payload = ev.to_dict()
    assert payload["action_frames"] == 100
    assert "seconds" in payload
    assert "clock" in payload


def test_wrapper_matches_catalog() -> None:
    strategy = wrap_ceres_ridley_as_boss_strategy()
    assert strategy.boss_id == "ceres_ridley"
    assert strategy.catalog.room_id == ROOM_CERES_RIDLEY
    assert strategy.entry.room_id == ROOM_CERES_RIDLEY


def test_product_outbound_is_tail_tank() -> None:
    from super_metroid.combat.ceres_ridley import CeresRidleyStrategy

    assert CeresRidleyStrategy().policy == "tail_tank"


def test_fight_terminal_and_product_raise() -> None:
    assert fight_terminal(_state(health=19, timer_type=3)) == "ceres_ridley_countdown"
    assert fight_terminal(_state(game_state=26)) == "death"
    assert fight_terminal(_state(room_id=0xDF45)) == "left_room"
    assert fight_terminal(_state()) is None

    ok = CeresRidleyEvidence(
        start_frame=0,
        end_frame=10,
        policy="tail_tank",
        start_health=99,
        end_health=19,
        hits=5,
        outcome="ceres_ridley_countdown",
    )
    require_ceres_ridley_countdown(ok)
    timed = CeresRidleyEvidence(
        start_frame=0,
        end_frame=10,
        policy="tail_tank",
        start_health=99,
        end_health=99,
        hits=0,
        outcome="timeout",
    )
    with pytest.raises(TimeoutError, match="did not start escape"):
        require_ceres_ridley_countdown(timed)
