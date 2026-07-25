"""Unit tests for snes_oneshot shared helpers."""

from __future__ import annotations

import numpy as np

from snes_oneshot.actions import ActionBuilder, buttons, idle_action
from snes_oneshot.behavior import ActionNode, Condition, NodeStatus, Selector, Sequence
from snes_oneshot.game_state import EnemyState, GameMode, GameState
from snes_oneshot.primitives import FrameAction, mash_start, walk_right
from snes_oneshot.ram_diff import candidates_increasing, diff_changed, snapshot
from snes_oneshot.watchdog import StuckDetector, WatchdogEvent


def test_idle_and_buttons() -> None:
    assert idle_action() == [0] * 12
    right = buttons("RIGHT")
    assert right[7] == 1
    assert sum(right) == 1


def test_action_builder() -> None:
    action = ActionBuilder().press("Y", "RIGHT").build()
    assert action[1] == 1
    assert action[7] == 1


def test_walk_right_frames() -> None:
    frames = list(walk_right(3).frames())
    assert len(frames) == 3
    assert frames[0].action[7] == 1


def test_mash_start_has_pulses() -> None:
    frames = mash_start(pulses=2, hold=2, gap=1)
    assert any(f.action[3] == 1 for f in frames)


def test_nearest_enemy() -> None:
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        player_x=10,
        player_y=10,
        enemies=(
            EnemyState(0, 100, 10, 5, True),
            EnemyState(1, 20, 10, 5, True),
        ),
    )
    nearest = state.nearest_enemy()
    assert nearest is not None
    assert nearest.slot == 1


def test_threat_enemies_include_hp0() -> None:
    living = EnemyState(0, 40, 10, 12, True)
    ghost = EnemyState(1, 25, 10, 0, True)
    inactive = EnemyState(2, 80, 10, 0, False)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        player_x=10,
        player_y=10,
        enemies=(living, ghost, inactive),
    )
    assert len(state.living_enemies) == 1
    assert state.living_enemies[0].slot == 0
    assert len(state.threat_enemies) == 2
    threat = state.nearest_threat()
    assert threat is not None
    assert threat.slot == 1


def test_selector_prefers_first_success() -> None:
    state = GameState(frame=0)
    tree = Selector(
        [
            Condition(lambda s: False, name="no"),
            ActionNode(
                lambda s: FrameAction(action=buttons("A"), reason="go"),
                name="go",
            ),
        ]
    )
    result = tree.tick(state)
    assert result.status is NodeStatus.RUNNING
    assert result.action is not None
    assert result.action.action[8] == 1


def test_sequence_advances() -> None:
    state = GameState(frame=0)
    seq = Sequence(
        [
            Condition(lambda s: True, name="ok"),
            Condition(lambda s: True, name="ok2"),
        ]
    )
    assert seq.tick(state).status is NodeStatus.SUCCESS


def test_ram_diff_and_increasing() -> None:
    before = snapshot(np.zeros(16, dtype=np.uint8))
    after = before.copy()
    after[4] = 9
    deltas = diff_changed(before, after)
    assert len(deltas) == 1
    assert deltas[0].address == 4
    assert candidates_increasing(deltas)[0].after == 9


def test_stuck_detector_position() -> None:
    detector = StuckDetector(position_window=3)
    state = GameState(frame=0, player_x=5, player_y=5)
    assert detector.update(state) is WatchdogEvent.NONE
    assert detector.update(state) is WatchdogEvent.NONE
    assert detector.update(state) is WatchdogEvent.NONE
    assert detector.update(state) is WatchdogEvent.POSITION_STALLED
