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


def test_unreachable_right_thug_presses_walk_limit() -> None:
    """Far park waits mid; engage band presses the walk-limit hold."""
    far = EnemyState(0, 1780, 70, 28, True)  # sx≈244 > engage(~205)
    mid = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=1648,
        player_y=70,
        camera_x=1536,
        screen_locked=True,
        enemies=(far,),
    )
    wait = fight_nearest_action(
        mid,
        preferred_flank=PreferredFlank.RIGHT,
        attack_range=34,
        camera_right_margin=160,
        invert_vertical=True,
    )
    assert wait.reason in {"edge_wait", "edge_mid", "edge_recenter"}
    mid_far = EnemyState(0, 1746, 70, 28, True)  # sx≈210
    mid_wait = fight_nearest_action(
        GameState(
            frame=0,
            mode=GameMode.PLAYING,
            player_x=1648,
            player_y=70,
            camera_x=1536,
            screen_locked=True,
            enemies=(mid_far,),
        ),
        preferred_flank=PreferredFlank.RIGHT,
        attack_range=34,
        camera_right_margin=160,
        invert_vertical=True,
    )
    assert mid_wait.reason in {"edge_wait", "edge_mid", "edge_recenter"}
    assert mid_wait.reason != "edge_press"
    near = EnemyState(0, 1726, 70, 28, True)  # sx≈190
    press = fight_nearest_action(
        GameState(
            frame=0,
            mode=GameMode.PLAYING,
            player_x=1648,
            player_y=70,
            camera_x=1536,
            screen_locked=True,
            enemies=(near,),
        ),
        preferred_flank=PreferredFlank.RIGHT,
        attack_range=34,
        camera_right_margin=160,
        invert_vertical=True,
    )
    # sx≈190 engage, player near wait: press toward hold (dx≈78 kick may
    # instead retreat toward wait — either is fine vs approach_right).
    assert press.reason in {
        "edge_press",
        "edge_recenter",
        "edge_wait",
    }


def test_kick_band_retreats_instead_of_idle() -> None:
    """Far-park past wait in kick: LEFT — never edge_wait for chips."""
    enemy = EnemyState(0, 1780, 70, 28, True)  # sx≈244, not engage
    state = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=1700,  # hold, dx≈80
        player_y=70,
        camera_x=1536,
        screen_locked=True,
        enemies=(enemy,),
    )
    action = fight_nearest_action(
        state,
        preferred_flank=PreferredFlank.RIGHT,
        attack_range=34,
        camera_right_margin=160,
        invert_vertical=True,
    )
    assert action.reason == "edge_recenter"
    assert action.action[6] == 1  # LEFT
    assert action.reason != "edge_wait"


def test_hold_kick_band_retreats_left() -> None:
    """W3 chip geometry: hold psx≈170 / dx≈81 must not idle."""
    enemy = EnemyState(0, 1781, 70, 28, True)  # sx≈245, dx≈81
    state = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=1700,  # psx≈164 hold
        player_y=70,
        camera_x=1536,
        screen_locked=True,
        enemies=(enemy,),
    )
    action = fight_nearest_action(
        state,
        preferred_flank=PreferredFlank.RIGHT,
        attack_range=34,
        camera_right_margin=160,
        invert_vertical=True,
    )
    assert action.reason == "edge_recenter"
    assert action.reason != "edge_wait"


def test_engage_kick_at_wait_no_hold_idle() -> None:
    """Engaged kick past wait retreats toward wait."""
    enemy = EnemyState(0, 1750, 70, 28, True)  # sx≈214, dx≈70 from 1680
    past = fight_nearest_action(
        GameState(
            frame=0,
            mode=GameMode.PLAYING,
            player_x=1680,  # past wait toward hold
            player_y=70,
            camera_x=1536,
            screen_locked=True,
            enemies=(enemy,),
        ),
        preferred_flank=PreferredFlank.RIGHT,
        attack_range=34,
        camera_right_margin=160,
        invert_vertical=True,
    )
    assert past.reason == "edge_recenter"
    at_wait = fight_nearest_action(
        GameState(
            frame=0,
            mode=GameMode.PLAYING,
            player_x=1636,
            player_y=70,
            camera_x=1536,
            screen_locked=True,
            enemies=(enemy,),
        ),
        preferred_flank=PreferredFlank.RIGHT,
        attack_range=34,
        camera_right_margin=160,
        invert_vertical=True,
    )
    assert at_wait.reason in {"edge_wait", "edge_press", "edge_recenter"}


def test_no_punch_with_large_dy() -> None:
    """In punch X but large Y: wait — do not trade."""
    enemy = EnemyState(0, 1720, 95, 28, True)  # sx≈184, dy=25
    state = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=1687,  # dx≈33
        player_y=70,
        camera_x=1536,
        screen_locked=True,
        enemies=(enemy,),
    )
    action = fight_nearest_action(
        state,
        preferred_flank=PreferredFlank.RIGHT,
        attack_range=34,
        camera_right_margin=160,
        invert_vertical=True,
        y_tolerance=10,
    )
    assert action.reason in {"edge_wait", "align_up", "align_down"}
    assert action.reason not in {"attack", "attack_gap"}


def test_far_park_does_not_chase_enemy_y() -> None:
    """Far right-edge park: hold lane instead of aligning into jump kicks."""
    far = EnemyState(0, 1780, 100, 28, True)  # sx≈244, different Y
    state = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=1636,  # near wait_x = cam+100
        player_y=50,
        camera_x=1536,
        screen_locked=True,
        enemies=(far,),
    )
    action = fight_nearest_action(
        state,
        preferred_flank=PreferredFlank.RIGHT,
        attack_range=34,
        camera_right_margin=160,
        invert_vertical=True,
        y_tolerance=10,
    )
    assert action.reason == "edge_wait"
    assert action.reason not in {"align_up", "align_down"}


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


def test_no_recenter_into_left_thug() -> None:
    """Dual-thug: do not edge_recenter LEFT into a second threat."""
    right = EnemyState(0, 1231, 46, 40, True)
    left = EnemyState(1, 1100, 46, 40, True)
    state = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=1165,
        player_y=48,
        camera_x=992,
        screen_locked=True,
        enemies=(right, left),
    )
    action = fight_nearest_action(
        state,
        preferred_flank=PreferredFlank.RIGHT,
        prefer_left_threat=True,
        attack_range=36,
        camera_right_margin=160,
        use_throw=True,
        grab_range=18,
    )
    assert action.reason != "edge_recenter"


def test_right_edge_waits_instead_of_chasing() -> None:
    """Do not mash RIGHT into the lock edge when a thug sits further right."""
    enemy = EnemyState(0, 1231, 46, 40, True)
    state = GameState(
        frame=0,
        mode=GameMode.PLAYING,
        player_x=1165,
        player_y=48,
        camera_x=992,
        screen_locked=True,
        enemies=(enemy,),
    )
    action = fight_nearest_action(
        state,
        preferred_flank=PreferredFlank.RIGHT,
        standoff=18,
        min_range=10,
        attack_range=30,
        camera_right_margin=160,
        edge_attack_bonus=12,
    )
    # From cam+173 toward enemy at cam+239: press to walk-limit hold.
    assert action.reason in {
        "edge_press",
        "edge_wait",
        "edge_recenter",
        "attack",
        "attack_gap",
    }
    assert action.reason != "approach_right"


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
