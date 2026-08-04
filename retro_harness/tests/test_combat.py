"""Tests for shared beat-em-up combat helpers."""

from __future__ import annotations

from retro_harness.bot_runner import NodeStatus
from retro_harness.combat import (
    PreferredFlank,
    align_vertical_action,
    build_segment_tree,
    fight_nearest_action,
    flank_approach_x_action,
    grab_throw_action,
    is_vertically_aligned,
    select_combat_target,
)
from retro_harness.ram_state import EnemyState, GameMode, GameState


def test_align_vertical_directions() -> None:
    state = GameState(frame=0, player_y=50)
    up = align_vertical_action(state, 40)
    assert up.reason == "align_up"
    down = align_vertical_action(state, 60)
    assert down.reason == "align_down"
    assert is_vertically_aligned(50, 55, tolerance=8)
    # Inverted axis (Final Fight): higher target Y → UP.
    inv_up = align_vertical_action(state, 60, invert_vertical=True)
    assert inv_up.reason == "align_up"
    inv_down = align_vertical_action(state, 40, invert_vertical=True)
    assert inv_down.reason == "align_down"


def test_fight_nearest_orders_steps() -> None:
    enemy = EnemyState(0, 200, 50, 10, True)
    far_y = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=200,
        player_y=80,
        enemies=(enemy,),
    )
    assert fight_nearest_action(far_y).reason == "align_up"
    far_x = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=100,
        player_y=50,
        enemies=(enemy,),
    )
    assert fight_nearest_action(far_x).reason == "approach_right"
    close = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=180,
        player_y=50,
        enemies=(enemy,),
    )
    assert fight_nearest_action(close).reason == "attack"
    overlap = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=200,
        player_y=50,
        enemies=(enemy,),
    )
    assert fight_nearest_action(overlap).reason == "space_left"


def test_flank_prefers_right_side() -> None:
    enemy = EnemyState(0, 200, 50, 10, True)
    # Enemy to the right, outside band → close from this face (no circle-past).
    left_far = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=140,
        player_y=50,
        enemies=(enemy,),
    )
    action = flank_approach_x_action(
        left_far,
        enemy.x,
        preferred_flank=PreferredFlank.RIGHT,
        standoff=18,
    )
    assert action.reason == "approach_right"
    # Already on right flank in range → idle.
    right_side = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=218,
        player_y=50,
        enemies=(enemy,),
    )
    idle = flank_approach_x_action(
        right_side,
        enemy.x,
        preferred_flank=PreferredFlank.RIGHT,
        standoff=18,
    )
    assert idle.reason == "in_range"
    # Inside attack band on the wrong side → still punch (no flank starve).
    left_close = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=185,
        player_y=50,
        enemies=(enemy,),
    )
    punch = fight_nearest_action(
        left_close,
        preferred_flank=PreferredFlank.RIGHT,
        standoff=18,
        min_range=10,
    )
    assert punch.reason == "attack"
    # Left-spawn on preferred shoulder outside band → ease left to standoff,
    # not through the body.
    left_spawn = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=240,
        player_y=50,
        camera_x=100,
        enemies=(enemy,),
    )
    ease = flank_approach_x_action(
        left_spawn,
        enemy.x,
        preferred_flank=PreferredFlank.RIGHT,
        standoff=18,
    )
    assert ease.reason == "approach_left"


def test_left_edge_spaces_instead_of_chasing() -> None:
    """Do not walk into the scroll wall while a left-spawn thugs chips."""
    enemy = EnemyState(0, 969, 70, 80, True)
    # Camera 992, player 1016 → 24px from left edge; enemy further left.
    state = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=1016,
        player_y=70,
        camera_x=992,
        enemies=(enemy,),
    )
    action = fight_nearest_action(
        state,
        preferred_flank=PreferredFlank.RIGHT,
        standoff=18,
        min_range=10,
        attack_range=28,
        camera_left_margin=48,
        edge_attack_bonus=28,
    )
    assert action.reason in {"edge_space", "attack", "attack_gap"}
    assert action.reason != "approach_left"


def test_no_throw_right_during_screen_lock() -> None:
    """Locked fights must not RIGHT+Y toward an enemy on the right."""
    enemy = EnemyState(0, 1700, 70, 40, True)
    state = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=1688,
        player_y=70,
        camera_x=1536,
        screen_locked=True,
        enemies=(enemy,),
    )
    action = fight_nearest_action(
        state,
        use_throw=True,
        grab_range=18,
        attack_range=34,
        min_range=10,
        preferred_flank=PreferredFlank.RIGHT,
        camera_right_margin=160,
        invert_vertical=True,
    )
    assert action.reason != "throw_right"


def test_grab_throw_and_left_threat() -> None:
    left = EnemyState(0, 80, 50, 10, True)
    right = EnemyState(1, 220, 50, 10, True)
    state = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=150,
        player_y=50,
        enemies=(left, right),
    )
    target = select_combat_target(state, prefer_left_threat=True)
    assert target is not None
    assert target.slot == 0
    throw = grab_throw_action(state, left.x)
    assert throw.reason == "throw_left"
    close = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=90,
        player_y=50,
        enemies=(left,),
    )
    fight = fight_nearest_action(
        close,
        use_throw=True,
        grab_range=16,
        preferred_flank=PreferredFlank.RIGHT,
    )
    assert fight.reason in {"throw_left", "throw_right", "flank_right"}


def test_segment_tree_walks_when_clear() -> None:
    tree = build_segment_tree()
    clear = GameState(frame=0, mode=GameMode.PLAYING)
    result = tree.tick(clear)
    assert result.status is NodeStatus.RUNNING
    assert result.action is not None
    assert result.action.reason == "walk_right"


def test_segment_tree_walks_after_enemy_dies() -> None:
    """Reactive sequence must re-check enemies_present each tick."""
    tree = build_segment_tree()
    enemy = EnemyState(0, 200, 50, 10, True)
    fighting = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=180,
        player_y=50,
        enemies=(enemy,),
    )
    assert tree.tick(fighting).action is not None
    assert tree.tick(fighting).action.reason in {
        "attack",
        "attack_gap",
        "space_left",
        "space_right",
        "in_range",
    }
    clear = GameState(frame=1, mode=GameMode.PLAYING, camera_x=100)
    result = tree.tick(clear)
    assert result.action is not None
    assert result.action.reason == "walk_right"
