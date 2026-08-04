"""Pure-logic tests for Final Fight's Stage 1 policy."""

from __future__ import annotations

import numpy as np

from final_fight.policy import Stage1Policy, build_stage1_tree
from final_fight.ram import GameStatus, parse_game_state
from retro_harness.bot_runner import NodeStatus
from retro_harness.ram_state import EnemyState, GameMode, GameState


def _playing(
    *,
    player_x: int = 80,
    player_y: int = 140,
    enemies: tuple[EnemyState, ...] = (),
    boss_active: bool = False,
    health: int = 80,
    lives: int = 3,
    stage: int = 0,
    camera_x: int = 0,
    frame: int = 1,
) -> GameState:
    """Build a playing state with the fields policy tests vary."""
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING,
        stage=stage,
        camera_x=camera_x,
        player_x=player_x,
        player_y=player_y,
        health=health,
        lives=lives,
        enemies=enemies,
        boss_active=boss_active,
    )


def _enemy(
    x: int,
    y: int,
    health: int,
    *,
    slot: int = 0,
    active: bool = True,
) -> EnemyState:
    """Build a combat-slot enemy."""
    return EnemyState(slot=slot, x=x, y=y, health=health, active=active, animation=3)


def _ram(*, camera_x: int = 0, stage: int = 0) -> np.ndarray:
    """Build initialized Final Fight WRAM for parser tests."""
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[0x0CA0] = GameStatus.ACTIVE_GAMEPLAY
    ram[0x0CB0] = stage
    ram[0x0D00] = 1
    ram[0x0D14] = 80
    ram[0x0D6E] = 3
    _write_u16(ram, 0x0E07, camera_x)
    return ram


def _write_u16(ram: np.ndarray, address: int, value: int) -> None:
    """Write a little-endian 16-bit WRAM value."""
    ram[address] = value & 0xFF
    ram[address + 1] = value >> 8


def _write_entity(
    ram: np.ndarray,
    base: int,
    *,
    status: int,
    x: int,
    y: int = 80,
    health: int,
) -> None:
    """Write the combat-relevant bytes for one entity slot."""
    ram[base] = status
    _write_u16(ram, base + 0x07, x)
    _write_u16(ram, base + 0x0D, y)
    ram[base + 0x14] = health


def test_walk_right_when_clear() -> None:
    result = Stage1Policy().tick(_playing())
    assert result.status is NodeStatus.RUNNING
    assert result.action is not None
    assert result.action.reason == "walk_right"
    assert result.action.action[7] == 1


def test_align_then_attack_nearest() -> None:
    enemy = _enemy(100, 100, 20)
    policy = Stage1Policy()
    aligned = policy.tick(_playing(player_y=140, enemies=(enemy,)))
    attacked = policy.tick(_playing(player_x=70, player_y=100, enemies=(enemy,)))
    assert aligned.action is not None
    assert aligned.action.reason in {"align_up", "align_down"}
    assert attacked.action is not None
    assert attacked.action.reason == "attack"
    assert attacked.action.action[1] == 1


def test_attack_cadence_gaps() -> None:
    enemy = _enemy(110, 140, 20)
    policy = Stage1Policy()
    reasons = [policy.tick(_playing(enemies=(enemy,))).action.reason for _ in range(3)]
    assert reasons[:2] == ["attack", "attack"]
    assert reasons[2] == "attack_gap"


def test_throw_when_overlapping() -> None:
    enemy = _enemy(90, 55, 53)
    result = Stage1Policy().tick(
        _playing(player_x=100, player_y=55, stage=1, camera_x=50, enemies=(enemy,))
    )
    assert result.action is not None
    assert result.action.reason == "throw_behind"
    assert result.action.action[6] == result.action.action[1] == 1


def test_continue_on_life_loss() -> None:
    result = Stage1Policy().tick(_playing(health=0, lives=2))
    assert result.action is not None
    assert result.action.reason == "ko_wait"


def test_boss_branch_stub() -> None:
    result = build_stage1_tree().tick(_playing(boss_active=True))
    assert result.action is not None
    assert result.action.reason.startswith("boss")


def test_boss_door_jump_dash_in_kick_band() -> None:
    thug = _enemy(200, 50, 60)
    result = Stage1Policy().tick(
        _playing(
            player_x=130,
            player_y=50,
            camera_x=100,
            boss_active=True,
            enemies=(thug,),
        )
    )
    assert result.action is not None
    assert result.action.reason == "jump_dash"
    assert result.action.action[0] == result.action.action[7] == 1


def test_boss_door_park_bait_low_hp_kick_band() -> None:
    thug = _enemy(200, 50, 40)
    result = Stage1Policy().tick(
        _playing(
            player_x=130,
            player_y=50,
            camera_x=100,
            boss_active=True,
            enemies=(thug,),
        )
    )
    assert result.action is not None
    assert result.action.reason == "hold_left"
    assert result.action.action[6] == result.action.action[7] == 0


def test_boss_door_punch_in_band() -> None:
    thug = _enemy(160, 50, 60)
    result = Stage1Policy().tick(
        _playing(
            player_x=130,
            player_y=50,
            camera_x=100,
            boss_active=True,
            enemies=(thug,),
        )
    )
    assert result.action is not None
    assert result.action.reason == "attack"
    assert result.action.action[1] == result.action.action[7] == 1


def test_post_unlock_softens_attack_cadence() -> None:
    enemy = _enemy(110, 140, 20)
    policy = Stage1Policy()
    reasons = [
        policy.tick(_playing(camera_x=1600, enemies=(enemy,))).action.reason
        for _ in range(3)
    ]
    assert reasons == ["attack", "attack", "attack_gap"]


def test_parse_game_state_player_and_enemy() -> None:
    ram = _ram(camera_x=256)
    _write_entity(ram, 0x0D00, status=1, x=0x140, y=0x90, health=80)
    _write_entity(ram, 0x1000, status=3, x=0x180, y=0x90, health=12)
    state = parse_game_state(ram, frame=9)
    assert state.mode is GameMode.PLAYING
    assert (state.frame, state.player_x, state.player_y) == (9, 0x140, 0x90)
    assert state.health == 80
    assert state.living_enemies[0].health == 12
    assert state.screen_locked is True


def test_parse_hp0_corpse_is_threat_not_living() -> None:
    ram = _ram(camera_x=256)
    _write_entity(ram, 0x1000, status=3, x=384, health=0)
    _write_entity(ram, 0x10B0, status=2, x=384, health=50)
    _write_entity(ram, 0x1140, status=3, x=4095, health=40)
    state = parse_game_state(ram)
    assert state.living_enemies == ()
    assert len(state.threat_enemies) == 1
    assert state.threat_enemies[0].health == 0


def test_parse_wave5_uf254_is_threat_not_living() -> None:
    ram = _ram(camera_x=640, stage=2)
    _write_entity(ram, 0x0D00, status=1, x=740, health=48)
    _write_entity(ram, 0x1000, status=3, x=640, health=254)
    state = parse_game_state(ram)
    assert state.living_enemies == ()
    assert len(state.threat_enemies) == 1
    assert state.threat_enemies[0].health == 0


def test_parse_west_area1_hp250_is_living() -> None:
    """West Area1 thug peaks ~250 — must not be UF-normalized away."""
    ram = _ram(camera_x=2560, stage=2)
    _write_entity(ram, 0x0D00, status=1, x=2676, health=50)
    _write_entity(ram, 0x1000, status=3, x=2624, health=250)
    state = parse_game_state(ram)
    assert len(state.living_enemies) == 1
    assert state.living_enemies[0].health == 250


def test_parse_cam840_distant_ghost_does_not_lock() -> None:
    ram = _ram(camera_x=848, stage=1)
    _write_entity(ram, 0x0D00, status=1, x=980, health=67)
    _write_entity(ram, 0x1000, status=3, x=1094, health=0)
    state = parse_game_state(ram)
    assert len(state.threat_enemies) == 1
    assert state.screen_locked is False


def test_parse_subway_hp148_is_living_not_ghost() -> None:
    ram = _ram(camera_x=844, stage=1)
    _write_entity(ram, 0x1000, status=3, x=920, health=148)
    state = parse_game_state(ram)
    assert state.living_enemies[0].health == 148
    assert state.threat_enemies == state.living_enemies


def test_parse_hp0_chaser_is_threat_not_living() -> None:
    ram = _ram()
    _write_entity(ram, 0x1000, status=3, x=100, health=0)
    state = parse_game_state(ram)
    assert state.living_enemies == ()
    assert len(state.threat_enemies) == 1


def test_boss_stub_plant_punches_hp0_ghost() -> None:
    ghost = _enemy(100, 55, 0)
    result = Stage1Policy().tick(
        _playing(player_x=50, player_y=55, boss_active=True, enemies=(ghost,))
    )
    assert result.action is not None
    assert result.action.reason == "z_punch"
    assert result.action.action[1] == 1


def test_parse_clear_round_not_area() -> None:
    ram = _ram()
    ram[0x0CA0] = GameStatus.CLEAR_AREA
    area = parse_game_state(ram)
    ram[0x0CA0] = GameStatus.CLEAR_ROUND
    round_clear = parse_game_state(ram)
    assert area.level_complete is False
    assert area.extras["area_clear"] is True
    assert round_clear.level_complete is True


def test_parse_includes_combat_boss() -> None:
    ram = _ram(camera_x=512)
    _write_entity(ram, 0x11E0, status=3, x=544, health=100)
    state = parse_game_state(ram)
    assert state.boss_active is True
    assert state.living_enemies[0].slot == 3
    assert state.nearest_enemy() is not None


def test_subway_left_edge_jump_dashes_kick_band() -> None:
    enemy = _enemy(700, 40, 34)
    result = Stage1Policy().tick(
        _playing(player_x=640, player_y=40, camera_x=585, stage=1, enemies=(enemy,))
    )
    assert result.action is not None
    assert result.action.reason == "jump_dash"
    assert result.action.action[0] == result.action.action[7] == 1


def test_parse_spawn_status_near_camera_is_living() -> None:
    ram = _ram(camera_x=600, stage=1)
    _write_entity(ram, 0x1000, status=1, x=550, health=148)
    state = parse_game_state(ram)
    assert state.living_enemies[0].health == 148


def test_parse_spawn_hp0_near_camera_is_threat() -> None:
    """Cam994 status-01 HP0 corpses still hurt; count as ghosts."""
    ram = _ram(camera_x=994, stage=1)
    _write_entity(ram, 0x0D00, status=3, x=1033, health=60)
    _write_entity(ram, 0x1000, status=1, x=941, health=0)
    state = parse_game_state(ram)
    assert state.living_enemies == ()
    assert len(state.threat_enemies) == 1
    assert state.threat_enemies[0].health == 0


def test_subway_plant_punches_underflow_ghost() -> None:
    ghost = _enemy(713, 52, 0)
    result = Stage1Policy().tick(
        _playing(player_x=675, player_y=48, camera_x=634, stage=1, enemies=(ghost,))
    )
    assert result.action is not None
    assert result.action.reason in {"z_punch", "z_gap", "z_flee"}


def test_ghost_jump_flees_close_corpse() -> None:
    ghost = _enemy(1014, 55, 0)
    result = Stage1Policy().tick(
        _playing(player_x=980, player_y=55, camera_x=844, stage=1, enemies=(ghost,))
    )
    assert result.action is not None
    assert result.action.reason == "z_flee"
    assert result.action.action[0] == result.action.action[6] == 1


def test_subway_fights_hp148_after_spacing_corpse() -> None:
    ghost = _enemy(1000, 55, 0)
    tough = _enemy(943, 55, 148, slot=2)
    result = Stage1Policy().tick(
        _playing(
            player_x=980,
            player_y=55,
            camera_x=844,
            stage=1,
            enemies=(ghost, tough),
        )
    )
    assert result.action is not None
    assert result.action.reason == "z_flee"


def test_subway_hp148_living_jump_dashes() -> None:
    tough = _enemy(1060, 55, 148)
    result = Stage1Policy().tick(
        _playing(player_x=980, player_y=55, camera_x=844, stage=1, enemies=(tough,))
    )
    assert result.action is not None
    assert result.action.reason == "jump_dash"


def test_subway_dual_living_focuses_tough() -> None:
    weak = _enemy(956, 58, 20)
    tough = _enemy(943, 55, 148, slot=2)
    result = Stage1Policy().tick(
        _playing(
            player_x=980,
            player_y=55,
            camera_x=844,
            stage=1,
            enemies=(weak, tough),
        )
    )
    assert result.action is not None
    assert result.action.reason == "space"
    assert result.action.action[0] == result.action.action[7] == 1


def test_subway_tough_behind_walks_past() -> None:
    tough = _enemy(943, 55, 53, slot=2)
    result = Stage1Policy().tick(
        _playing(player_x=980, player_y=55, camera_x=844, stage=1, enemies=(tough,))
    )
    assert result.action is not None
    assert result.action.reason == "walk_past"
    assert result.action.action[0] == 0


def test_subway_tough_behind_walks_past_at_hp72() -> None:
    tough = _enemy(943, 55, 72, slot=2)
    result = Stage1Policy().tick(
        _playing(player_x=980, player_y=55, camera_x=844, stage=1, enemies=(tough,))
    )
    assert result.action is not None
    assert result.action.reason == "walk_past"


def test_subway_tough_behind_critical_player_faces_y() -> None:
    """Critical player vs high-HP behind: face-Y, do not walk_past into kicks."""
    tough = _enemy(943, 55, 148, slot=2)
    result = Stage1Policy().tick(
        _playing(
            player_x=980,
            player_y=55,
            health=40,
            camera_x=844,
            stage=1,
            enemies=(tough,),
        )
    )
    assert result.action is not None
    assert result.action.reason == "attack"
    assert result.action.action[0] == 0  # no B


def test_subway_tough_behind_critical_stall_walks_past() -> None:
    """Critical player vs stall HP≤72 behind: walk_past."""
    tough = _enemy(943, 55, 60, slot=2)
    result = Stage1Policy().tick(
        _playing(
            player_x=980,
            player_y=55,
            health=40,
            camera_x=844,
            stage=1,
            enemies=(tough,),
        )
    )
    assert result.action is not None
    assert result.action.reason == "walk_past"


def test_subway_behind_hp48_walks_past_not_jd() -> None:
    leftover = _enemy(855, 55, 48, slot=2)
    weak = _enemy(956, 58, 20)
    result = Stage1Policy().tick(
        _playing(
            player_x=902,
            player_y=55,
            camera_x=848,
            stage=1,
            health=46,
            lives=2,
            enemies=(weak, leftover),
        )
    )
    assert result.action is not None
    assert result.action.reason == "walk_past"
    assert result.action.action[0] == 0


def test_subway_cam840_ghost_mashes_right() -> None:
    ghost = _enemy(1014, 55, 0)
    result = Stage1Policy().tick(
        _playing(
            player_x=900,
            player_y=70,
            camera_x=847,
            stage=1,
            health=47,
            lives=2,
            enemies=(ghost,),
            frame=10,
        )
    )
    assert result.action is not None
    assert result.action.reason == "scroll_mash"
    assert result.action.action[7] == result.action.action[1] == 1


def test_subway_cam840_nearby_ghost_plant_punches() -> None:
    """Kick-band UF at cam≥840 must be plant-punched, not scroll-mashed."""
    ghost = _enemy(970, 55, 0)
    result = Stage1Policy().tick(
        _playing(
            player_x=900,
            player_y=70,
            camera_x=847,
            stage=1,
            health=60,
            lives=2,
            enemies=(ghost,),
            frame=1,
        )
    )
    assert result.action is not None
    assert result.action.reason == "z_punch"
    assert result.action.action[1] == 1


def test_area2_unlocked_far_ghost_plant_punches() -> None:
    """Area2 unlocked: plant UF at dx≈120 (open leftovers), not mash."""
    ghost = _enemy(4117 + 120, 58, 0)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3968,
        player_x=4117,
        player_y=58,
        health=54,
        lives=1,
        enemies=(ghost,),
        screen_locked=False,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason in {"z_punch", "z_gap"}
    assert result.action.reason != "scroll_mash"


def test_subway_cam840_low_y_rises_while_scrolling() -> None:
    ghost = _enemy(1014, 55, 0)
    result = Stage1Policy().tick(
        _playing(
            player_x=900,
            player_y=43,
            camera_x=848,
            stage=1,
            health=34,
            lives=2,
            enemies=(ghost,),
            frame=10,
        )
    )
    assert result.action is not None
    assert result.action.reason == "scroll_rise"
    assert result.action.action[4] == result.action.action[7] == 1


def test_subway_tough_behind_high_hp_face_y() -> None:
    tough = _enemy(943, 52, 148, slot=2)
    result = Stage1Policy().tick(
        _playing(
            player_x=980,
            player_y=55,
            camera_x=844,
            stage=1,
            health=80,
            lives=2,
            enemies=(tough,),
        )
    )
    assert result.action is not None
    assert result.action.reason == "attack"
    assert result.action.action[6] == result.action.action[1] == 1


def test_subway_tough_behind_throws_when_close() -> None:
    tough = _enemy(968, 55, 53, slot=2)
    result = Stage1Policy().tick(
        _playing(
            player_x=980,
            player_y=55,
            camera_x=844,
            stage=1,
            health=54,
            lives=2,
            enemies=(tough,),
        )
    )
    assert result.action is not None
    assert result.action.reason == "throw_behind"
    assert result.action.action[6] == result.action.action[1] == 1


def test_subway_tough_behind_walks_past_at_left_gutter() -> None:
    """Gutter sx<40 with stall behind: face-Y, not walk_past into wall."""
    tough = _enemy(850, 55, 53, slot=2)
    result = Stage1Policy().tick(
        _playing(
            player_x=875,
            player_y=55,
            camera_x=844,
            stage=1,
            health=54,
            lives=2,
            enemies=(tough,),
        )
    )
    assert result.action is not None
    assert result.action.reason == "attack"
    assert result.action.action[0] == 0


def test_subway_cam994_front_weak_before_far_behind() -> None:
    """Cam994: clear front weak before JD-left to far-behind tough."""
    weak = _enemy(1210, 78, 34)
    far_tough = _enemy(962, 48, 100, slot=2)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        camera_x=994,
        player_x=1181,
        player_y=63,
        health=67,
        lives=1,
        enemies=(weak, far_tough),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason in {"attack", "attack_gap", "nudge", "align"}
    assert result.action.reason != "jump_dash"


def test_subway_cam994_unlock_mashes_right() -> None:
    """Unlocked cam≥990: mash right even with a living behind spawn."""
    tough = _enemy(950, 55, 62, slot=2)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        camera_x=994,
        player_x=1080,
        player_y=70,
        health=38,
        lives=1,
        enemies=(tough,),
        screen_locked=False,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "scroll_mash"
    assert result.action.action[7] == 1


def test_parse_round_id_subway() -> None:
    ram = _ram(stage=1)
    state = parse_game_state(ram)
    assert state.stage == 1
    assert state.room == 0
    assert state.extras["boss_dead_flag"] == 0


def test_west_side_beyond_kick_closes() -> None:
    """West Side engage often starts dx≈109; do not park-bait forever."""
    enemy = _enemy(850, 40, 78)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=2,
        room=0,
        camera_x=619,
        player_x=738,
        player_y=40,
        health=80,
        lives=1,
        enemies=(enemy,),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "jump_dash"
    assert result.action.reason != "park"
    assert result.action.reason != "far_bait"


def test_area1_dual_pack_park_baits_kick_band() -> None:
    """Area1 dual kick: pulse JD / retreat, never hold_left."""
    weak = _enemy(1930, 55, 34)
    tough = _enemy(1950, 58, 91, slot=2)
    state = GameState(
        frame=20,
        mode=GameMode.PLAYING,
        stage=1,
        room=1,
        camera_x=1792,
        player_x=1860,
        player_y=55,
        health=60,
        lives=1,
        enemies=(weak, tough),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason in {"retreat", "jump_dash"}
    assert result.action.reason != "hold_left"


def test_area1_fights_near_weak_not_far_tough() -> None:
    """Area1: punch the in-band thug, not a far tough."""
    near = _enemy(1840, 55, 42)
    far_tough = _enemy(1990, 58, 91, slot=2)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=1,
        camera_x=1792,
        player_x=1810,
        player_y=55,
        health=60,
        lives=1,
        enemies=(near, far_tough),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason in {"attack", "attack_gap", "nudge"}
    assert result.action.reason != "far_bait"
    assert result.action.reason != "jump_dash"


def test_area1_far_dual_park_baits_beyond_kick() -> None:
    """Area1 cam≥1792 must not force JD just because cam≥900."""
    weak = _enemy(2000, 55, 34)
    tough = _enemy(2020, 58, 91, slot=2)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=1,
        camera_x=1792,
        player_x=1850,
        player_y=55,
        health=60,
        lives=1,
        enemies=(weak, tough),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason in {"park", "far_bait", "align"}
    assert result.action.reason != "jump_dash"


def test_area1_finishes_low_hp_beyond_kick() -> None:
    """Area1: chase HP≤80 leftovers instead of far_bait."""
    low = _enemy(2000, 55, 2)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=1,
        camera_x=1792,
        player_x=1850,
        player_y=55,
        health=60,
        lives=1,
        enemies=(low,),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "jump_dash"


def test_area2_ultra_dual_spaces_tight_overlap() -> None:
    """Area2 HP112/134: hop out of dx<26 before kick one-shot."""
    near = _enemy(4000, 55, 112)
    far = _enemy(4100, 58, 134, slot=2)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3920,
        player_x=3990,
        player_y=55,
        health=54,
        lives=1,
        enemies=(near, far),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "space"


def test_area2_1v1_leftover_jds_kick_band() -> None:
    """Area2 1v1 in front kick band (cam<3915): JD-pass to behind."""
    low = _enemy(4040, 70, 79)
    state = GameState(
        frame=20,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3892,
        player_x=3970,
        player_y=70,
        health=54,
        lives=1,
        enemies=(low,),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "jump_dash"


def test_area2_1v1_crumb_face_y_not_jd() -> None:
    """Area2 HP≤8 at cam3968 in band: face-Y (kill geometry)."""
    crumb = _enemy(4117 + 56, 58, 3)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3968,
        player_x=4117,
        player_y=58,
        health=15,
        lives=1,
        enemies=(crumb,),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason in {"attack", "attack_gap"}
    assert result.action.reason != "park"


def test_area2_1v1_crumb_jds_to_scroll_kill() -> None:
    """Area2 HP≤8 before cam3960: JD-scroll into kill geometry."""
    crumb = _enemy(4084, 55, 3)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3914,
        player_x=4047,
        player_y=58,
        health=15,
        lives=1,
        enemies=(crumb,),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "jump_dash"
    assert result.action.reason != "park"


def test_area2_dual_focuses_crumb() -> None:
    """Area2 dual with HP≤8 crumb: focus crumb, allow toward+Y."""
    crumb = _enemy(4117 + 56, 58, 3)
    tough = _enemy(4117 + 88, 58, 134, slot=2)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3968,
        player_x=4117,
        player_y=58,
        health=15,
        lives=1,
        enemies=(crumb, tough),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason in {"attack", "attack_gap"}
    assert result.action.reason != "space"


def test_area2_1v1_crumb_face_y_at_scroll_lock() -> None:
    """Cam 3916 HP≤8 crumb: JD-scroll (kill needs cam≈3968)."""
    crumb = _enemy(3916 + 133 + 37, 55, 3)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3916,
        player_x=3916 + 133,
        player_y=58,
        health=15,
        lives=1,
        enemies=(crumb,),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "jump_dash"
    assert result.action.reason != "park"


def test_area2_1v1_parks_before_dual_scroll() -> None:
    """Area2 1v1 near cam3915 outside punch: ground-close, not JD."""
    low = _enemy(3916 + 120 + 55, 70, 79)
    state = GameState(
        frame=20,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3916,
        player_x=3916 + 120,
        player_y=70,
        health=54,
        lives=1,
        enemies=(low,),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "nudge"
    assert result.action.reason != "jump_dash"


def test_area2_1v1_punches_at_scroll_lock() -> None:
    """Cam≥3915 in punch band: face-Y, do not park-stall."""
    low = _enemy(3916 + 120 + 32, 70, 69)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3916,
        player_x=3916 + 120,
        player_y=70,
        health=54,
        lives=1,
        enemies=(low,),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason in {"attack", "attack_gap", "throw"}
    assert result.action.reason != "park"
    assert result.action.reason != "jump_dash"


def test_area2_1v1_parks_before_scroll_spawn() -> None:
    """Area2 far ahead (adx>kick, sx>125): park left."""
    low = _enemy(4200, 70, 79)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3891,
        player_x=4027,
        player_y=70,
        health=54,
        lives=1,
        enemies=(low,),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "park"


def test_area2_1v1_punches_when_already_in_range() -> None:
    """Mid save dx≈23/sx≈136: JD-pass to behind (front Y whiffs)."""
    low = _enemy(4050, 51, 79)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3891,
        player_x=4027,
        player_y=51,
        health=54,
        lives=1,
        enemies=(low,),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "jump_dash"
    assert result.action.reason != "park"


def test_area2_behind_closes_before_face_y() -> None:
    """Stall-band behind outside punch: close, do not whiff Y / scroll."""
    low = _enemy(4105, 56, 28)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3969,
        player_x=4157,
        player_y=56,
        health=54,
        lives=1,
        enemies=(low,),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason in {"nudge", "jump_dash"}
    assert result.action.reason != "scroll_edge"
    assert result.action.reason != "attack"


def test_area2_ultra_dual_parks_high_sx() -> None:
    """Ultra dual: climb to sx>125 before chipping the weak thug."""
    weak = _enemy(4000, 55, 79)
    tough = _enemy(4080, 58, 134, slot=2)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3920,
        player_x=4020,
        player_y=55,
        health=54,
        lives=1,
        enemies=(weak, tough),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "space"
    sx = state.player_x - state.camera_x
    assert sx <= 125


def test_area2_behind_face_y_not_walk_past() -> None:
    """Area2 behind: face-Y chips; never gutter walk_past."""
    low = _enemy(4000, 70, 79)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3891,
        player_x=4030,
        player_y=70,
        health=54,
        lives=1,
        enemies=(low,),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason in {"attack", "attack_gap", "throw_behind"}
    assert result.action.reason != "walk_past"


def test_area2_high_sx_behind_closes_gap() -> None:
    """Area2 sx>150 behind adx=52: JD-left close (ground LEFT sticks)."""
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3969,
        player_x=3969 + 188,
        player_y=56,
        health=54,
        lives=1,
        enemies=(_enemy(3969 + 188 - 52, 56, 28),),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "jump_dash"
    assert result.action.reason != "scroll_edge"
    assert result.action.reason != "walk_past"


def test_area2_high_sx_behind_face_y_in_punch() -> None:
    """Area2 sx>150 behind in punch band: face-Y in place."""
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=2,
        camera_x=3969,
        player_x=3969 + 188,
        player_y=56,
        health=54,
        lives=1,
        enemies=(_enemy(3969 + 188 - 32, 56, 28),),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason in {"attack", "attack_gap"}
    assert result.action.reason != "walk_past"


def test_area1_scroll_past_far_behind_leftover() -> None:
    """Unlocked area1: mash past far-behind spawn (avoid 54→38 chip)."""
    behind = _enemy(2300, 67, 34)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=1,
        camera_x=2500,
        player_x=2636,
        player_y=70,
        health=54,
        lives=1,
        enemies=(behind,),
        screen_locked=False,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "scroll_mash"


def test_area1_scroll_past_locked_far_behind() -> None:
    """Cam2488 lock with HP134 behind: keep RIGHT to softlock 2561."""
    behind = _enemy(2322, 67, 134)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=1,
        camera_x=2488,
        player_x=2622,
        player_y=70,
        health=54,
        lives=1,
        enemies=(behind,),
        screen_locked=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "scroll_mash"
    assert result.action.reason != "park"


def test_area1_unlocked_plants_ghost_before_mash() -> None:
    """Area1 unlocked: plant kick-band ghost; do not area0-style mash."""
    ghost = _enemy(2050, 55, 0)
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=1,
        camera_x=1998,
        player_x=2080,
        player_y=70,
        health=54,
        lives=1,
        enemies=(ghost,),
        screen_locked=False,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason in {"z_punch", "z_gap", "z_flee"}
    assert result.action.reason != "scroll_mash"


def test_subway_scroll_edge_when_overshot() -> None:
    """Cam≥840 clear: pull back when sx>170 (off-edge HP drain)."""
    state = GameState(
        frame=1,
        mode=GameMode.PLAYING,
        stage=1,
        room=1,
        camera_x=2048,
        player_x=2280,
        player_y=70,
        health=54,
        lives=1,
        enemies=(),
        screen_locked=False,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "scroll_edge"
