"""Final Fight right-edge / patient-approach combat tests."""

from __future__ import annotations

from final_fight.edge_combat import ff_fight_nearest_action
from retro_harness.combat import PreferredFlank
from retro_harness.ram_state import EnemyState, GameMode, GameState


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
    wait = ff_fight_nearest_action(
        mid,
        preferred_flank=PreferredFlank.RIGHT,
        attack_range=34,
        camera_right_margin=160,
        invert_vertical=True,
    )
    assert wait.reason in {"edge_wait", "edge_mid", "edge_recenter"}
    mid_far = EnemyState(0, 1746, 70, 28, True)  # sx≈210
    mid_wait = ff_fight_nearest_action(
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
    press = ff_fight_nearest_action(
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
    action = ff_fight_nearest_action(
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
    action = ff_fight_nearest_action(
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
    past = ff_fight_nearest_action(
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
    at_wait = ff_fight_nearest_action(
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
    action = ff_fight_nearest_action(
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
    action = ff_fight_nearest_action(
        state,
        preferred_flank=PreferredFlank.RIGHT,
        attack_range=34,
        camera_right_margin=160,
        invert_vertical=True,
        y_tolerance=10,
    )
    assert action.reason == "edge_wait"
    assert action.reason not in {"align_up", "align_down"}


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
    action = ff_fight_nearest_action(
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
    action = ff_fight_nearest_action(
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


def test_area1_andore_face_y_behind_then_throw() -> None:
    from final_fight.edge_combat import area1_andore_action

    face, faced = area1_andore_action(
        frame=1, sx=110, dx=-52, dy=8, faced=False
    )
    assert face.reason == "face"
    assert faced is True
    y, _ = area1_andore_action(frame=24, sx=110, dx=-52, dy=8, faced=True)
    assert y.reason in {"y", "close", "throw"}
    throw, _ = area1_andore_action(frame=4, sx=110, dx=-8, dy=0, faced=True)
    assert throw.reason == "throw"
    far, _ = area1_andore_action(frame=10, sx=110, dx=-120, dy=0, faced=True)
    assert far.reason in {"wait_far", "wait_desync"}
    fence, _ = area1_andore_action(frame=10, sx=180, dx=-40, dy=0, faced=True)
    assert fence.reason == "clamp_l"
    crumb, _ = area1_andore_action(
        frame=8, sx=110, dx=-20, dy=0, faced=True, enemy_hp=40
    )
    assert crumb.reason in {"throw", "gap"}
    hop, _ = area1_andore_action(
        frame=8, sx=110, dx=-6, dy=0, faced=True, enemy_hp=40
    )
    assert hop.reason == "space"
    wait, _ = area1_andore_action(
        frame=8, sx=110, dx=-50, dy=8, faced=True, enemy_hp=40
    )
    assert wait.reason in {"wait_far", "desync"}
