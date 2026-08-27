"""ROM-free tests for Slash pattern-lab adapters and grind-knob seams."""

from __future__ import annotations

from dataclasses import replace

from retro_harness.ram_state import EnemyState, GameMode, GameState
from tmnt_iv.grind_knobs import override_knobs
from tmnt_iv.scripts.slash_pattern_lab import PATTERNS
from tmnt_iv.scripts.slash_patterns import HybridWhiplash, ProductionSlash
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


def _cross_then_attack(pattern: HybridWhiplash, state: GameState, n: int = 90) -> tuple[int, int]:
    reasons = [pattern.next(state).reason for _ in range(n)]
    n_cross = 0
    for reason in reasons:
        if reason != "slash_cross":
            break
        n_cross += 1
    n_attack = 0
    for reason in reasons[n_cross:]:
        if reason != "slash_back_attack":
            break
        n_attack += 1
    return n_cross, n_attack


def test_production_pattern_name_registered() -> None:
    assert ProductionSlash.name == "production"
    assert PATTERNS["production"] is ProductionSlash


def test_production_maps_none_to_idle_slash_wait() -> None:
    action = ProductionSlash().next(_playing())
    assert action is not None
    assert action.reason == "slash_wait"
    assert all(int(v) == 0 for v in action.action)


def test_production_delegates_to_slash_tactics_in_fight() -> None:
    state = _state(player_x=80, slash_x=128, animation=0x00)
    got = ProductionSlash().next(state)
    expected = SlashTactics().next(state)
    assert expected is not None
    assert got.reason == expected.reason
    assert list(got.action) == list(expected.action)
    assert got.action[8] == 0


def test_hybrid_whiplash_reads_cross_and_attack_knobs() -> None:
    mid = _state(player_x=80, slash_x=120, animation=0x43, health=160)
    default_cross, default_attack = _cross_then_attack(HybridWhiplash(), mid)
    with override_knobs({"slash_cross_frames": 8, "slash_attack_frames": 16}):
        short_cross, short_attack = _cross_then_attack(HybridWhiplash(), mid)
    with override_knobs({"slash_cross_frames": 30, "slash_attack_frames": 48}):
        long_cross, long_attack = _cross_then_attack(HybridWhiplash(), mid)
    assert short_cross < default_cross < long_cross
    assert short_attack < default_attack < long_attack
    assert default_cross == 23
    assert default_attack == 36


def test_hybrid_whiplash_low_hp_reads_low_knobs() -> None:
    low = _state(player_x=80, slash_x=120, animation=0x43, health=40)
    with override_knobs(
        {
            "slash_low_hp": 16,
            "slash_cross_frames": 30,
            "slash_cross_frames_low": 8,
            "slash_attack_frames": 48,
            "slash_attack_frames_low": 16,
        }
    ):
        high_path = _cross_then_attack(HybridWhiplash(), low)
    with override_knobs(
        {
            "slash_low_hp": 48,
            "slash_cross_frames": 30,
            "slash_cross_frames_low": 8,
            "slash_attack_frames": 48,
            "slash_attack_frames_low": 16,
        }
    ):
        low_path = _cross_then_attack(HybridWhiplash(), low)
    assert high_path[0] > low_path[0]
    assert high_path[1] > low_path[1]
    assert low_path == (9, 16)
    assert high_path == (31, 48)
