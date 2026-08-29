from __future__ import annotations

from magical_quest.policy import DOOR_HOLD_FRAMES, Stage1Policy
from magical_quest.ram import FIRST_DOOR_X, FIRST_DOOR_Y
from retro_harness.bot_runner import NodeStatus
from retro_harness.controls import SNES_RIGHT
from retro_harness.ram_state import GameMode, GameState


def _state(
    *,
    player_x: int = 360,
    player_y: int = 34,
    health: int = 3,
    player_dead: bool = False,
    mode: GameMode = GameMode.PLAYING,
) -> GameState:
    return GameState(
        frame=0,
        mode=mode,
        player_x=player_x,
        player_y=player_y,
        health=health,
        player_dead=player_dead,
    )


def test_walks_right_from_stage1_spawn() -> None:
    result = Stage1Policy().tick(_state())
    assert result.status is NodeStatus.RUNNING
    assert result.reason == "walk_right"
    assert result.action is not None
    assert result.action.action[SNES_RIGHT] == 1


def test_door_hold_then_success() -> None:
    policy = Stage1Policy()
    door = _state(player_x=FIRST_DOOR_X, player_y=FIRST_DOOR_Y, health=1)
    running = [policy.tick(door) for _ in range(DOOR_HOLD_FRAMES - 1)]
    done = policy.tick(door)
    assert all(tick.status is NodeStatus.RUNNING for tick in running)
    assert running[-1].reason == "door_hold"
    assert done.status is NodeStatus.SUCCESS
    assert done.reason == "first_door"


def test_door_hold_resets_if_x_drops() -> None:
    policy = Stage1Policy()
    door = _state(player_x=FIRST_DOOR_X, player_y=FIRST_DOOR_Y, health=1)
    spawn = _state()
    for _ in range(5):
        policy.tick(door)
    assert policy.tick(spawn).reason == "walk_right"
    running = [policy.tick(door) for _ in range(DOOR_HOLD_FRAMES - 1)]
    assert all(tick.status is NodeStatus.RUNNING for tick in running)
    assert policy.tick(door).status is NodeStatus.SUCCESS


def test_zero_health_is_failure() -> None:
    result = Stage1Policy().tick(_state(health=0, player_dead=True))
    assert result.status is NodeStatus.FAILURE
    assert result.reason == "dead"
