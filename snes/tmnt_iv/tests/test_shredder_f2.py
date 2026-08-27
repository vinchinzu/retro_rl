"""Unit tests for Super Shredder form-2 vertical-offset tactics."""

from __future__ import annotations

from dataclasses import replace

from retro_harness.ram_state import EnemyState, GameMode, GameState
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.tactics.shredder_f2 import SuperShredderForm2Tactics


def _form2(
    *,
    player_x: int,
    player_y: int,
    boss_x: int = 160,
    boss_y: int = 180,
    animation: int = 0xBB,
    health: int = 190,
    frame: int = 1,
) -> GameState:
    boss = EnemyState(
        slot=0,
        x=boss_x,
        y=boss_y,
        health=health,
        active=True,
        kind=0xAE,
        animation=animation,
    )
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING,
        stage=9,
        player_x=player_x,
        player_y=player_y,
        health=80,
        lives=2,
        enemies=(boss,),
        boss_active=True,
        screen_locked=True,
        extras={"event": 0x0A, "iframes": 0, "char_id": 8},
    )


def _assert_clean(action) -> None:
    assert action is not None
    assert action.action[8] == 0  # never A
    assert not (action.action[0] and action.action[1])  # never grounded Y+B


def test_in_front_prefers_vertical_or_behind() -> None:
    """Overlapping his lane must not walk into the green-fireball face."""
    state = _form2(player_x=160, player_y=180, boss_x=160, boss_y=180)
    action = SuperShredderForm2Tactics().next(state)
    _assert_clean(action)
    assert action.reason in {"shredder_offset", "shredder_behind"}
    assert action.action[1] == 0  # not Y in his face
    vertical = action.action[4] or action.action[5]
    hop = action.action[0] == 1
    assert vertical or hop
    if action.action[6] or action.action[7]:
        assert hop  # lateral only as a behind hop, never a walk-in


def test_in_front_never_walks_into_left_wall() -> None:
    state = _form2(player_x=24, player_y=166, boss_x=17, boss_y=166)
    action = SuperShredderForm2Tactics().next(state)
    _assert_clean(action)
    assert action.reason in {"shredder_offset", "shredder_behind"}
    assert action.action[6] == 0  # never LEFT into the wall


def test_offset_band_waits_during_aura() -> None:
    """16–28px above/beside: hold, do not mash Y while aura is up."""
    state = _form2(player_x=128, player_y=158, boss_x=160, boss_y=180)
    tactics = SuperShredderForm2Tactics()
    action = tactics.next(state)
    _assert_clean(action)
    assert action.reason == "shredder_wait"
    assert action.action[1] == 0
    assert action.action[0] == 0


def test_drop_window_steps_in_with_y() -> None:
    """Leaving 0xFE is the aura drop — after the shot passes, mash Y."""
    tactics = SuperShredderForm2Tactics()
    aura = _form2(
        player_x=128,
        player_y=158,
        boss_x=160,
        boss_y=180,
        animation=0xFE,
    )
    first = tactics.next(aura)
    _assert_clean(first)
    assert first.reason == "shredder_wait"
    drop = _form2(
        player_x=144,
        player_y=176,
        boss_x=160,
        boss_y=180,
        animation=0x29,
        frame=2,
    )
    reasons = []
    action = first
    for frame in range(16):
        action = tactics.next(replace(drop, frame=frame + 2))
        _assert_clean(action)
        reasons.append(action.reason)
        if action.reason == "shredder_attack" and action.action[1] == 1:
            break
    else:
        raise AssertionError(f"no drop-window Y in {reasons}")
    assert action.action[0] == 0  # no B


def test_stage1_policy_dispatches_form2() -> None:
    state = _form2(player_x=160, player_y=180, boss_x=160, boss_y=180)
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason.startswith("shredder_")
    _assert_clean(result.action)


def test_outside_arena_returns_none() -> None:
    state = replace(_form2(player_x=80, player_y=160), stage=8)
    assert SuperShredderForm2Tactics().next(state) is None
