"""Slash jump-over / behind-combo tactics (Raph Hard wiki recipe)."""

from __future__ import annotations

from dataclasses import replace

from retro_harness.ram_state import EnemyState, GameMode, GameState
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.tactics.slash import SlashTactics


def _playing(
    *,
    player_x: int = 80,
    player_y: int = 160,
    enemies: tuple[EnemyState, ...] = (),
    health: int = 80,
    frame: int = 1,
) -> GameState:
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING,
        camera_x=0,
        player_x=player_x,
        player_y=player_y,
        health=health,
        lives=2,
        enemies=enemies,
        screen_locked=bool(enemies),
    )


def _slash(
    *,
    x: int,
    y: int = 160,
    health: int = 160,
    animation: int = 0x43,
) -> EnemyState:
    return EnemyState(
        slot=0,
        x=x,
        y=y,
        health=health,
        active=True,
        kind=0x50,
        animation=animation,
    )


def _state(
    *,
    player_x: int,
    player_y: int = 160,
    slash_x: int,
    slash_y: int = 160,
    animation: int = 0x43,
    char_id: int | None = 8,
    iframes: int = 0,
    health: int = 160,
    frame: int = 1,
) -> GameState:
    extras: dict[str, object] = {
        "event": 0x0A,
        "iframes": iframes,
        "boss_hp": health,
    }
    if char_id is not None:
        extras["char_id"] = char_id
    return replace(
        _playing(
            player_x=player_x,
            player_y=player_y,
            enemies=(_slash(x=slash_x, y=slash_y, health=health, animation=animation),),
            frame=frame,
        ),
        stage=4,
        boss_active=True,
        extras=extras,
    )


def _press(action, *names: str) -> None:
    idx = {"B": 0, "Y": 1, "LEFT": 6, "RIGHT": 7, "A": 8}
    for name in names:
        assert action.action[idx[name]] == 1, f"expected {name} on {action.reason}"


def _release(action, *names: str) -> None:
    idx = {"B": 0, "Y": 1, "LEFT": 6, "RIGHT": 7, "A": 8}
    for name in names:
        assert action.action[idx[name]] == 0, f"did not expect {name} on {action.reason}"


def test_spin_dodge_hops_away_inside_adx_52() -> None:
    tactics = SlashTactics()
    close = tactics.next(
        _state(player_x=80, slash_x=120, animation=0xEE)
    )
    assert close is not None
    assert close.reason == "slash_dodge"
    _press(close, "B", "LEFT")
    _release(close, "A")

    far = SlashTactics().next(
        _state(player_x=60, slash_x=120, animation=0xEE)
    )
    assert far is not None
    assert far.reason != "slash_dodge"


def test_claw_dodge_only_when_already_close() -> None:
    close = SlashTactics().next(
        _state(player_x=90, slash_x=120, animation=0x09)
    )
    assert close is not None
    assert close.reason == "slash_dodge"
    _press(close, "B")
    _release(close, "A")

    windup = SlashTactics().next(
        _state(player_x=90, slash_x=120, animation=0x83)
    )
    assert windup is not None
    assert windup.reason == "slash_dodge"

    far = SlashTactics().next(
        _state(player_x=40, slash_x=120, animation=0x09)
    )
    assert far is not None
    assert far.reason != "slash_dodge"


def test_punish_window_mashes_toward_y() -> None:
    action = SlashTactics().next(
        _state(player_x=82, slash_x=130, animation=0x3E)
    )
    assert action is not None
    assert action.reason == "slash_back_attack"
    _press(action, "Y", "RIGHT")
    _release(action, "A")


def test_raph_jump_over_rises_before_kick() -> None:
    """In front, mid band: B+toward jump-over. No grounded Y+B. Never A."""
    tactics = SlashTactics()
    state = _state(player_x=80, slash_x=128, animation=0x00)
    actions = [tactics.next(state) for _ in range(12)]
    assert all(a is not None for a in actions)
    first = actions[0]
    assert first.reason == "slash_jump_over"
    _press(first, "B", "RIGHT")
    _release(first, "Y", "A")
    # Same-Y over is B-only (lab cross). Grounded Y+B is the HP special.
    over = [a for a in actions if a.reason == "slash_jump_over"]
    assert over
    for action in over:
        _press(action, "B")
        _release(action, "Y", "A")
    assert all(a.action[8] == 0 for a in actions)


def test_raph_behind_grounded_combo_then_hop_away() -> None:
    """Player behind Slash at adx 12–40: grounded Y, then hop. No infinite mash."""
    tactics = SlashTactics()
    # Far in front so we only approach (facing left); then teleport behind.
    approach = tactics.next(_state(player_x=30, slash_x=120, animation=0x00))
    assert approach is not None
    assert approach.reason == "slash_approach"

    behind = _state(player_x=140, slash_x=120, animation=0x00)
    actions = [tactics.next(behind) for _ in range(48)]
    assert all(a is not None for a in actions)
    assert actions[0].reason == "slash_back_attack"
    _press(actions[0], "Y")
    _release(actions[0], "B", "A")
    reasons = [a.reason for a in actions]
    assert "slash_hop_away" in reasons
    hop = next(a for a in actions if a.reason == "slash_hop_away")
    _press(hop, "B")
    _release(hop, "A")
    assert all(a.action[8] == 0 for a in actions)
    # Combo must end; hopping away is the Hard disengage.
    combo = [a for a in actions if a.reason == "slash_back_attack"]
    assert 1 <= len(combo) <= 40


def test_raph_does_not_walk_into_body() -> None:
    tactics = SlashTactics()
    # Establish facing toward player (left), then stand on his chest.
    tactics.next(_state(player_x=30, slash_x=120, animation=0x00))
    action = tactics.next(_state(player_x=110, slash_x=120, animation=0x00))
    assert action is not None
    assert action.reason in {"slash_space", "slash_hop_away", "slash_dodge"}
    _release(action, "A")
    # Must not walk RIGHT into Slash (player is already on his left, close).
    assert action.action[7] == 0


def test_raph_baits_in_front_when_far() -> None:
    action = SlashTactics().next(
        _state(player_x=30, slash_x=120, animation=0x00)
    )
    assert action is not None
    assert action.reason == "slash_approach"
    _press(action, "RIGHT")
    _release(action, "A", "Y")


def test_raph_jump_kick_meets_elevated_slash() -> None:
    tactics = SlashTactics()
    state = _state(
        player_x=100, player_y=170, slash_x=140, slash_y=120, animation=0x00
    )
    actions = [tactics.next(state) for _ in range(12)]
    assert actions[0] is not None
    assert actions[0].reason in {"slash_jump_over", "slash_jump_kick"}
    assert actions[0].reason != "jump_slash"
    _press(actions[0], "B")
    _release(actions[0], "Y", "A")
    kick = next((a for a in actions if a.reason == "slash_jump_kick"), None)
    assert kick is not None
    _press(kick, "B", "Y")
    _release(kick, "A")


def test_missing_or_non_raph_char_still_jump_over() -> None:
    """Missing/non-8 char_id still jump-overs, not slash_cross thrash."""
    for char_id in (None, 0, 1):
        action = SlashTactics().next(
            _state(player_x=80, slash_x=128, animation=0x00, char_id=char_id)
        )
        assert action is not None
        assert action.reason == "slash_jump_over"
        assert action.reason != "slash_cross"
        _press(action, "B", "RIGHT")
        _release(action, "Y", "A")


def test_never_emits_a_across_cycle() -> None:
    tactics = SlashTactics()
    states = [
        _state(player_x=80, slash_x=128, animation=0x00),
        _state(player_x=80, slash_x=120, animation=0xEE),
        _state(player_x=90, slash_x=120, animation=0x09),
        _state(player_x=82, slash_x=130, animation=0xB7),
        _state(player_x=100, player_y=170, slash_x=140, slash_y=120),
    ]
    for state in states:
        action = tactics.next(state)
        assert action is not None
        _release(action, "A")


def test_stage_policy_uses_slash_tactics_not_generic_jump() -> None:
    slash = _slash(x=140, y=120, animation=0x43)
    state = replace(
        _playing(player_x=100, player_y=170, enemies=(slash,)),
        boss_active=True,
        stage=4,
        extras={"event": 0x0A, "boss_hp": 160, "char_id": 8, "iframes": 0},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason != "jump_slash"
    assert result.action.reason != "baxter_jump_slash"
    assert result.action.reason.startswith("slash_")
    assert result.action.action[8] == 0
