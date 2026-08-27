"""Unit tests for Raphael Starbase jump-kick."""

from __future__ import annotations

from dataclasses import replace

from retro_harness.ram_state import EnemyState, GameMode, GameState
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.tactics.raph_air import raph_starbase_jump_action

_A = 8
_B = 0
_Y = 1
_LEFT = 6
_RIGHT = 7
_UP = 4
_DOWN = 5


def _playing(
    *,
    player_x: int = 80,
    player_y: int = 160,
    enemies: tuple[EnemyState, ...] = (),
    char_id: int = 8,
    stage: int = 8,
    frame: int = 0,
    boss_active: bool = False,
) -> GameState:
    extras: dict[str, object] = {}
    if char_id is not None:
        extras["char_id"] = char_id
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING,
        stage=stage,
        player_x=player_x,
        player_y=player_y,
        health=80,
        lives=2,
        enemies=enemies,
        screen_locked=bool(enemies),
        boss_active=boss_active,
        extras=extras,
    )


def _enemy(
    x: int,
    y: int,
    *,
    health: int = 16,
    kind: int = 0xB0,
    slot: int = 0,
) -> EnemyState:
    return EnemyState(
        slot=slot,
        x=x,
        y=y,
        health=health,
        active=True,
        kind=kind,
    )


def _actions_over(state: GameState, frames: int = 8) -> list:
    out = []
    for frame in range(frames):
        out.append(raph_starbase_jump_action(replace(state, frame=frame)))
    return out


def test_non_raph_returns_none() -> None:
    target = _enemy(40, 160)
    state = _playing(player_x=73, enemies=(target,), char_id=2)
    assert raph_starbase_jump_action(state) is None
    missing = _playing(player_x=73, enemies=(target,), char_id=-1)
    assert raph_starbase_jump_action(missing) is None


def test_wrong_stage_or_boss_returns_none() -> None:
    target = _enemy(40, 160)
    assert raph_starbase_jump_action(
        _playing(player_x=73, enemies=(target,), stage=6)
    ) is None
    assert raph_starbase_jump_action(
        _playing(player_x=73, enemies=(target,), boss_active=True)
    ) is None


def test_ground_bruisers_abort() -> None:
    """Living 0xB2/0xB4 jump-lock Raphael; fall through to grounded poke."""
    hover = _enemy(40, 160, kind=0x6A)
    for kind in (0xB2, 0xB4):
        bruiser = _enemy(50, 160, kind=kind, slot=1)
        state = _playing(player_x=73, enemies=(hover, bruiser))
        assert raph_starbase_jump_action(state) is None, hex(kind)


def test_close_stack_jump_and_gap() -> None:
    """Period 4: frame 0 is B+Y; 1–3 are steer-only."""
    stack = _enemy(40, 190, kind=0xB0)
    state = _playing(player_x=73, player_y=176, enemies=(stack,))
    actions = _actions_over(state, frames=4)
    assert all(a is not None for a in actions)
    assert [a.reason for a in actions] == [
        "raph_starbase_jump",
        "raph_starbase_close_gap",
        "raph_starbase_close_gap",
        "raph_starbase_close_gap",
    ]
    jump = actions[0]
    assert jump.action[_B] == 1
    assert jump.action[_Y] == 1
    assert jump.action[_LEFT] == 1
    assert jump.action[_A] == 0
    gap = actions[1]
    assert gap.action[_B] == 0
    assert gap.action[_Y] == 0
    assert gap.action[_LEFT] == 1
    assert gap.action[_A] == 0


def test_out_of_range_returns_none() -> None:
    far = _enemy(200, 160, kind=0x6A)
    assert raph_starbase_jump_action(
        _playing(player_x=80, enemies=(far,))
    ) is None
    high = _enemy(90, 80, kind=0xBA)
    assert raph_starbase_jump_action(
        _playing(player_x=80, player_y=160, enemies=(high,))
    ) is None


def test_vertical_steer_when_elevated() -> None:
    stack = _enemy(90, 140, kind=0xB0)
    state = _playing(player_x=80, player_y=176, enemies=(stack,), frame=0)
    action = raph_starbase_jump_action(state)
    assert action is not None
    assert action.reason == "raph_starbase_jump"
    assert action.action[_UP] == 1
    assert action.action[_DOWN] == 0
    assert action.action[_RIGHT] == 1


def test_no_action_ever_presses_a() -> None:
    samples = [
        _playing(player_x=73, player_y=176, enemies=(_enemy(40, 190, kind=0xB0),)),
        _playing(player_x=80, enemies=(_enemy(110, 160, kind=0x6A),)),
        _playing(player_x=80, enemies=(_enemy(90, 160, kind=0xBA),)),
        _playing(player_x=80, enemies=(_enemy(140, 160, kind=0xB2),)),
        _playing(player_x=80, enemies=(_enemy(140, 160, kind=0x60),)),
        _playing(player_x=80, enemies=(_enemy(140, 160, kind=0x50),), boss_active=True),
    ]
    for state in samples:
        for action in _actions_over(state, frames=8):
            if action is None:
                continue
            assert action.action[_A] == 0, action.reason


def test_jump_is_not_grounded_power_attack() -> None:
    """B+Y only on jump frames; gap frames never combine them."""
    stack = _enemy(40, 190, kind=0xB0)
    state = _playing(player_x=73, player_y=176, enemies=(stack,))
    for action in _actions_over(state, frames=8):
        assert action is not None
        if action.reason == "raph_starbase_jump":
            assert action.action[_B] and action.action[_Y]
        else:
            assert action.reason == "raph_starbase_close_gap"
            assert not (action.action[_B] and action.action[_Y])
            assert action.action[_B] == 0
            assert action.action[_Y] == 0


def test_policy_emits_starbase_jump_and_gap() -> None:
    stack = _enemy(40, 190, kind=0xB0)
    jump = Stage1Policy().tick(
        _playing(player_x=73, player_y=176, enemies=(stack,), frame=0)
    ).action
    assert jump is not None
    assert jump.reason == "raph_starbase_jump"
    assert jump.action[_B] == 1
    assert jump.action[_Y] == 1
    assert jump.action[_A] == 0
    gap = Stage1Policy().tick(
        _playing(player_x=73, player_y=176, enemies=(stack,), frame=1)
    ).action
    assert gap is not None
    assert gap.reason == "raph_starbase_close_gap"
    assert gap.action[_B] == 0
    assert gap.action[_Y] == 0
    assert gap.action[_A] == 0
