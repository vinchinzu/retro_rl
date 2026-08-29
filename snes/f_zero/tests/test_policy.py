from __future__ import annotations

from f_zero.policy import CenterlinePolicy
from f_zero.ram import SCREEN_LAPS_LEFT, SCREEN_REVERSE
from retro_harness.bot_runner import NodeStatus
from retro_harness.controls import SNES_B, SNES_L, SNES_LEFT, SNES_R, SNES_RIGHT
from retro_harness.ram_state import GameMode, GameState


def _state(**extras: object) -> GameState:
    payload = {
        "racing": True,
        "heading_error": 0,
        "damaged": False,
        "screen_text": 0,
    }
    payload.update(extras)
    return GameState(
        frame=int(payload.get("frame", 0)),
        mode=GameMode.PLAYING,
        extras=payload,
    )


def test_countdown_holds_throttle() -> None:
    result = CenterlinePolicy().tick(_state(racing=False))
    assert result.status is NodeStatus.RUNNING
    assert result.reason == "countdown"
    assert result.action is not None
    assert result.action.action[SNES_B] == 1
    assert result.action.action[SNES_LEFT] == 0


def test_center_holds_throttle_in_deadzone() -> None:
    result = CenterlinePolicy().tick(_state(heading_error=2))
    assert result.reason == "center"
    assert result.action is not None
    assert result.action.action[SNES_B] == 1
    assert result.action.action[SNES_LEFT] == 0
    assert result.action.action[SNES_RIGHT] == 0


def test_positive_error_steers_left_then_sharpens() -> None:
    policy = CenterlinePolicy()
    mild = policy.tick(_state(heading_error=8))
    sharp = policy.tick(_state(heading_error=20))
    assert mild.reason == "left"
    assert mild.action is not None
    assert mild.action.action[SNES_LEFT] == 1
    assert mild.action.action[SNES_L] == 0
    assert sharp.reason == "sharp_left"
    assert sharp.action is not None
    assert sharp.action.action[SNES_LEFT] == 1
    assert sharp.action.action[SNES_L] == 1


def test_negative_error_steers_right() -> None:
    result = CenterlinePolicy().tick(_state(heading_error=-9))
    assert result.reason == "right"
    assert result.action is not None
    assert result.action.action[SNES_RIGHT] == 1
    assert result.action.action[SNES_R] == 0


def test_damage_recovers_toward_checkpoint() -> None:
    result = CenterlinePolicy().tick(_state(heading_error=5, damaged=True))
    assert result.reason == "recover_left"
    assert result.action is not None
    assert result.action.action[SNES_LEFT] == 1
    assert result.action.action[SNES_L] == 0


def test_reverse_hud_uses_recovery() -> None:
    result = CenterlinePolicy().tick(
        _state(heading_error=-4, screen_text=SCREEN_REVERSE)
    )
    assert result.reason == "recover_right"


def test_lap_text_stops_with_success() -> None:
    state = GameState(
        frame=12,
        mode=GameMode.PLAYING,
        level_complete=True,
        extras={"screen_text": SCREEN_LAPS_LEFT},
    )
    result = CenterlinePolicy().tick(state)
    assert result.status is NodeStatus.SUCCESS
    assert result.reason == "lap"


def test_explosion_stops_with_failure() -> None:
    state = GameState(frame=9, mode=GameMode.MENU, player_dead=True)
    result = CenterlinePolicy().tick(state)
    assert result.status is NodeStatus.FAILURE
    assert result.reason == "crash"
