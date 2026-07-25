"""Pure-logic tests for segment stop heuristics and tracker."""

from __future__ import annotations

from snes_oneshot.game_state import EnemyState, GameMode, GameState
from snes_oneshot.segment_runner import (
    SegmentOutcome,
    SegmentTracker,
    WaveChainTracker,
    is_screen_clear,
)


def _playing(
    *,
    enemies: tuple[EnemyState, ...] = (),
    camera_x: int = 100,
    health: int = 80,
    lives: int = 3,
    go_flashing: bool = False,
    level_complete: bool = False,
    player_dead: bool = False,
) -> GameState:
    living = tuple(e for e in enemies if e.active and e.health > 0)
    return GameState(
        frame=1,
        mode=GameMode.PLAYING,
        player_x=80,
        player_y=140,
        health=health,
        lives=lives,
        camera_x=camera_x,
        enemies=enemies,
        level_complete=level_complete,
        player_dead=player_dead,
        screen_locked=bool(living) and not go_flashing,
        extras={"go_flashing": go_flashing},
    )


def test_is_screen_clear_requires_prior_enemies() -> None:
    clear = _playing(go_flashing=True)
    assert not is_screen_clear(
        clear,
        start_camera_x=100,
        had_enemies=False,
        clear_hold_frames=5,
        clear_frames_seen=10,
    )


def test_is_screen_clear_on_go_flashing() -> None:
    state = _playing(go_flashing=True)
    assert is_screen_clear(
        state,
        start_camera_x=100,
        had_enemies=True,
        clear_hold_frames=5,
        clear_frames_seen=1,
    )


def test_is_screen_clear_on_camera_unlock() -> None:
    state = _playing(camera_x=120)
    assert is_screen_clear(
        state,
        start_camera_x=100,
        had_enemies=True,
        clear_hold_frames=5,
        clear_frames_seen=1,
        camera_unlock_delta=8,
    )


def test_tracker_success_after_hold() -> None:
    enemy = EnemyState(0, 100, 140, 20, True)
    tracker = SegmentTracker(max_frames=100, clear_hold_frames=3)
    tracker.begin(_playing(enemies=(enemy,)))
    assert tracker.update(_playing(enemies=(enemy,))) is None
    for _ in range(2):
        assert tracker.update(_playing()) is None
    assert tracker.update(_playing()) is SegmentOutcome.SUCCESS
    assert tracker.kills == 1


def test_tracker_death() -> None:
    enemy = EnemyState(0, 100, 140, 20, True)
    tracker = SegmentTracker(max_frames=100)
    tracker.begin(_playing(enemies=(enemy,)))
    outcome = tracker.update(
        _playing(enemies=(enemy,), health=0, player_dead=True)
    )
    assert outcome is SegmentOutcome.DEATH


def test_tracker_corpse_hp_is_death() -> None:
    """HP wrap (>128) must not false-clear after enemy despawn."""
    enemy = EnemyState(0, 100, 140, 20, True)
    tracker = WaveChainTracker(
        max_frames=100,
        clear_hold_frames=2,
        target_waves=1,
        stop_on_boss=False,
    )
    tracker.begin(_playing(enemies=(enemy,), camera_x=100, health=31))
    assert tracker.update(
        _playing(enemies=(enemy,), camera_x=100, health=31)
    ) is None
    # Corpse HP with empty living — death, not wave clear.
    outcome = tracker.update(
        _playing(camera_x=4608, health=215, lives=0)
    )
    assert outcome is SegmentOutcome.DEATH
    assert tracker.waves_cleared == 0


def test_tracker_timeout() -> None:
    enemy = EnemyState(0, 100, 140, 20, True)
    tracker = SegmentTracker(max_frames=2)
    tracker.begin(_playing(enemies=(enemy,)))
    assert tracker.update(_playing(enemies=(enemy,))) is None
    assert tracker.update(_playing(enemies=(enemy,))) is (
        SegmentOutcome.TIMEOUT
    )


def test_wave_chain_clears_multiple_waves() -> None:
    enemy = EnemyState(0, 100, 140, 20, True)
    tracker = WaveChainTracker(
        max_frames=100,
        clear_hold_frames=2,
        target_waves=2,
        stop_on_boss=False,
    )
    tracker.begin(_playing(enemies=(enemy,), camera_x=100))
    assert tracker.update(_playing(enemies=(enemy,), camera_x=100)) is None
    # Wave 1 clear via camera unlock (kills required).
    assert tracker.update(_playing(camera_x=120)) is None
    assert tracker.waves_cleared == 1
    # Walk / idle then wave 2.
    assert tracker.update(_playing(camera_x=120)) is None
    assert tracker.update(
        _playing(enemies=(enemy,), camera_x=120)
    ) is None
    # No camera move: need longer hold (90) + kills.
    for _ in range(89):
        assert tracker.update(_playing(camera_x=120)) is None
    outcome = tracker.update(_playing(camera_x=120))
    assert outcome is SegmentOutcome.SUCCESS
    assert tracker.waves_cleared == 2


def test_wave_chain_stop_on_boss() -> None:
    enemy = EnemyState(0, 100, 140, 20, True)
    tracker = WaveChainTracker(max_frames=50, stop_on_boss=True)
    tracker.begin(_playing(enemies=(enemy,)))
    # Status 01 = present undrawn still counts as Damnd spawned.
    boss_loading = GameState(
        frame=2,
        mode=GameMode.PLAYING,
        player_x=80,
        player_y=140,
        health=70,
        lives=3,
        camera_x=200,
        enemies=(),
        boss_active=True,
        extras={"go_flashing": False, "boss_status": 1, "boss_hp": 100},
    )
    assert tracker.update(boss_loading) is SegmentOutcome.SUCCESS
    assert tracker.boss_reached is True
    tracker2 = WaveChainTracker(max_frames=50, stop_on_boss=True)
    tracker2.begin(_playing(enemies=(enemy,)))
    boss_drawn = GameState(
        frame=2,
        mode=GameMode.PLAYING,
        player_x=80,
        player_y=140,
        health=70,
        lives=3,
        camera_x=200,
        enemies=(),
        boss_active=True,
        extras={"go_flashing": False, "boss_status": 3, "boss_hp": 100},
    )
    assert tracker2.update(boss_drawn) is SegmentOutcome.SUCCESS
    assert tracker2.boss_reached is True

