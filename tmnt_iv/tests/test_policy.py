"""Pure-logic tests for TMNT IV Stage 1 policy and RAM adapter."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from snes_oneshot.behavior import NodeStatus
from snes_oneshot.game_state import EnemyState, GameMode, GameState
from tmnt_iv.policy import Stage1Policy, TechnodromeTactics, build_stage1_tree
from tmnt_iv.ram import (
    ADDR_EVENT,
    ADDR_LIVES,
    ADDR_MENU,
    ADDR_STAGE,
    ENEMY_BASES,
    MenuId,
    OFF_CHAR,
    OFF_HP,
    OFF_X,
    OFF_Y,
    PLAYER_BASE,
    parse_game_state,
    write_u16le,
)


def _playing(
    *,
    player_x: int = 80,
    player_y: int = 160,
    enemies: tuple[EnemyState, ...] = (),
    health: int = 80,
    lives: int = 2,
    camera_x: int = 0,
    frame: int = 1,
) -> GameState:
    """Build a playing state with the fields policy tests vary."""
    return GameState(
        frame=frame,
        mode=GameMode.PLAYING,
        camera_x=camera_x,
        player_x=player_x,
        player_y=player_y,
        health=health,
        lives=lives,
        enemies=enemies,
        screen_locked=bool(enemies),
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
    return EnemyState(
        slot=slot,
        x=x,
        y=y,
        health=health,
        active=active,
        animation=0x43,
    )


def _ram() -> np.ndarray:
    """Build initialized TMNT IV WRAM for parser tests."""
    ram = np.zeros(0x20000, dtype=np.uint8)
    ram[ADDR_MENU] = MenuId.PLAYING
    ram[ADDR_LIVES] = 2
    ram[PLAYER_BASE + OFF_HP] = 80
    write_u16le(ram, PLAYER_BASE + OFF_X, 64)
    write_u16le(ram, PLAYER_BASE + OFF_Y, 160)
    return ram


def _write_enemy(
    ram: np.ndarray,
    base: int,
    *,
    x: int,
    y: int,
    health: int,
    char_id: int = 0x60,
) -> None:
    """Write combat-relevant bytes for one enemy slot."""
    write_u16le(ram, base + OFF_X, x)
    write_u16le(ram, base + OFF_Y, y)
    ram[base + OFF_HP] = health
    ram[base + OFF_CHAR] = char_id


def test_walk_right_when_clear() -> None:
    result = Stage1Policy().tick(_playing())
    assert result.status is NodeStatus.RUNNING
    assert result.action is not None
    assert result.action.reason == "walk_right"
    assert result.action.action[7] == 1


def test_align_then_attack_nearest() -> None:
    enemy = _enemy(120, 120, 16)
    policy = Stage1Policy()
    aligned = policy.tick(_playing(player_y=160, enemies=(enemy,)))
    attacked = policy.tick(
        _playing(player_x=90, player_y=120, enemies=(enemy,))
    )
    assert aligned.action is not None
    assert aligned.action.reason in {"align_up", "align_down"}
    assert attacked.action is not None
    assert attacked.action.reason == "attack"
    assert attacked.action.action[1] == 1


def test_attack_cadence_gaps() -> None:
    enemy = _enemy(110, 160, 16)
    policy = Stage1Policy()
    reasons = [
        policy.tick(_playing(enemies=(enemy,))).action.reason
        for _ in range(3)
    ]
    assert reasons[:2] == ["attack", "attack"]
    assert reasons[2] == "attack_gap"


def test_continue_on_life_loss() -> None:
    result = Stage1Policy().tick(_playing(health=0, lives=2))
    assert result.action is not None
    assert result.action.reason == "ko_wait"


def test_up_decreases_world_y_align() -> None:
    """TMNT uses normal screen Y — align toward lower Y presses UP."""
    enemy = _enemy(100, 140, 16)
    result = Stage1Policy().tick(
        _playing(player_x=100, player_y=170, enemies=(enemy,))
    )
    assert result.action is not None
    assert result.action.reason == "align_up"
    assert result.action.action[4] == 1


def test_parse_game_state_player_and_enemy() -> None:
    ram = _ram()
    _write_enemy(ram, ENEMY_BASES[0], x=200, y=170, health=16)
    state = parse_game_state(ram)
    assert state.mode is GameMode.PLAYING
    assert state.player_x == 64
    assert state.player_y == 160
    assert state.health == 80
    assert state.lives == 2
    assert len(state.living_enemies) == 1
    assert state.living_enemies[0].health == 16
    assert state.living_enemies[0].x == 200


def test_despawn_sentinel_not_living() -> None:
    ram = _ram()
    _write_enemy(ram, ENEMY_BASES[0], x=65504, y=170, health=0)
    state = parse_game_state(ram)
    assert state.living_enemies == ()


def test_build_tree_name() -> None:
    tree = build_stage1_tree()
    assert tree.name == "segment_clear"


def test_screen_space_combat_ignores_progress_camera() -> None:
    """Progress word must not force perpetual edge_space LEFT approaches."""
    enemy = _enemy(40, 160, 16)
    result = Stage1Policy().tick(
        _playing(
            player_x=120,
            player_y=160,
            camera_x=1200,
            enemies=(enemy,),
        )
    )
    assert result.action is not None
    assert result.action.reason == "approach_left"
    assert result.action.action[6] == 1


def test_boss_candidate_from_high_hp_slot() -> None:
    ram = _ram()
    _write_enemy(ram, ENEMY_BASES[0], x=180, y=160, health=128)
    state = parse_game_state(ram)
    assert state.boss_active is True
    assert state.extras["boss_hp"] == 128
    assert len(state.living_enemies) == 1


def test_rat_king_stays_boss_below_hp_threshold() -> None:
    """Rat King (char 0x4A) remains boss_active after HP drops < 80."""
    ram = _ram()
    _write_enemy(
        ram, ENEMY_BASES[0], x=200, y=140, health=52, char_id=0x4A
    )
    state = parse_game_state(ram)
    assert state.boss_active is True
    assert state.extras["boss_hp"] == 52
    assert state.living_enemies[0].kind == 0x4A


def test_last_life_still_playing() -> None:
    """lives==0 with HP left is last-life PLAYING, not cutscene."""
    ram = _ram()
    ram[ADDR_LIVES] = 0
    ram[ADDR_EVENT] = 0x0A
    ram[ADDR_STAGE] = 1
    state = parse_game_state(ram)
    assert state.mode is GameMode.PLAYING
    assert state.lives == 0
    assert state.stage == 1


def test_transition_event_is_cutscene() -> None:
    ram = _ram()
    ram[ADDR_EVENT] = 0x19
    write_u16le(ram, PLAYER_BASE + OFF_X, 0)
    write_u16le(ram, PLAYER_BASE + OFF_Y, 0)
    state = parse_game_state(ram)
    assert state.mode is GameMode.CUTSCENE


def test_slash_boss_char_stays_active_at_low_hp() -> None:
    """Slash (0x50) remains boss_active after HP drops below 80."""
    ram = _ram()
    _write_enemy(
        ram, ENEMY_BASES[0], x=180, y=160, health=24, char_id=0x50
    )
    state = parse_game_state(ram)
    assert state.boss_active is True
    assert state.extras["boss_hp"] == 24
    assert state.living_enemies[0].kind == 0x50


def test_april_npc_not_living_enemy() -> None:
    """April O'Neil (char 0xC4) shares enemy slots — must not be fought."""
    ram = _ram()
    _write_enemy(ram, ENEMY_BASES[0], x=208, y=176, health=48)
    ram[ENEMY_BASES[0] + OFF_CHAR] = 0xC4
    state = parse_game_state(ram)
    assert state.living_enemies == ()
    assert state.screen_locked is False


def test_prehistoric_pterodactyl_not_living_dino_is() -> None:
    """Stage 5 pterodactyl 0xEE is non-combat; dino 0x6C is fought."""
    ram = _ram()
    _write_enemy(
        ram, ENEMY_BASES[0], x=100, y=180, health=12, char_id=0x6C
    )
    _write_enemy(
        ram, ENEMY_BASES[1], x=190, y=150, health=16, char_id=0xEE
    )
    _write_enemy(
        ram, ENEMY_BASES[2], x=200, y=170, health=16, char_id=0x60
    )
    state = parse_game_state(ram)
    kinds = {e.kind for e in state.living_enemies}
    assert kinds == {0x6C, 0x60}


def test_prehistoric_jump_slash_on_dino() -> None:
    """Stage byte 4 + living dino → jump-slash instead of grounded Y."""
    dino = EnemyState(
        slot=0, x=110, y=160, health=12, active=True, kind=0x6C
    )
    result = Stage1Policy().tick(
        replace(
            _playing(player_x=90, player_y=160, enemies=(dino,), frame=3),
            stage=4,
        )
    )
    assert result.action is not None
    assert result.action.reason == "jump_slash"
    assert result.action.action[0] == 1  # B
    assert result.action.action[1] == 1  # Y


def test_slash_opener_pokes_from_left_standoff() -> None:
    """0x3E opener at left adx≈48 → toward+Y, never A."""
    slash = EnemyState(
        slot=0, x=130, y=160, health=80, active=True, kind=0x50, animation=0x3E
    )
    state = replace(
        _playing(
            player_x=82,  # adx=48 on left
            player_y=160,
            enemies=(slash,),
            frame=3,
        ),
        stage=4,
        boss_active=True,
        extras={"event": 0x0A, "iframes": 0},
    )
    action = Stage1Policy().tick(state).action
    assert action is not None
    assert action.reason == "slash_back_attack"
    assert action.action[1] == 1  # Y
    assert action.action[8] == 0  # never A


def test_slash_dodges_shell_spin_when_close() -> None:
    """Close + shell-spin status → hop away instead of tanking."""
    slash = EnemyState(
        slot=0, x=100, y=160, health=80, active=True, kind=0x50, animation=0xEE
    )
    state = replace(
        _playing(player_x=80, player_y=160, enemies=(slash,), frame=1),
        stage=4,
        boss_active=True,
        extras={"event": 0x0A, "iframes": 0},
    )
    action = Stage1Policy().tick(state).action
    assert action is not None
    assert action.reason == "slash_dodge"
    assert action.action[0] == 1  # B


def test_slash_dodges_claw_active() -> None:
    """Claw active 0x09 is the real damage — hop out past reach."""
    slash = EnemyState(
        slot=0, x=120, y=160, health=80, active=True, kind=0x50, animation=0x09
    )
    state = replace(
        _playing(player_x=80, player_y=160, enemies=(slash,), frame=1),
        stage=4,
        boss_active=True,
        extras={"event": 0x0A, "iframes": 0},
    )
    action = Stage1Policy().tick(state).action
    assert action is not None
    assert action.reason == "slash_dodge"
    assert action.action[0] == 1  # B


def test_slash_big_punish_b7_attacks() -> None:
    """0xB7 is the 8-dmg window — mash/cross, never flee or A."""
    slash = EnemyState(
        slot=0, x=120, y=160, health=80, active=True, kind=0x50, animation=0xB7
    )
    state = replace(
        _playing(player_x=100, player_y=160, enemies=(slash,), frame=1),
        stage=4,
        boss_active=True,
        extras={"event": 0x0A, "iframes": 0},
    )
    policy = Stage1Policy()
    reasons = []
    for _ in range(12):
        action = policy.tick(state).action
        assert action is not None
        reasons.append(action.reason)
        assert action.reason != "slash_dodge"
        assert action.action[8] == 0
    assert "slash_back_attack" in reasons or "slash_cross" in reasons


def test_duo_boss_forces_left_flank_before_attack() -> None:
    """Tokka/Rahzar: do not mash Y from the right; walk to left standoff."""
    tokka = EnemyState(
        slot=0, x=32, y=176, health=96, active=True, kind=0x48
    )
    rahzar = EnemyState(
        slot=1, x=132, y=176, health=96, active=True, kind=0xA0
    )
    state = replace(
        _playing(
            player_x=191,
            player_y=181,
            enemies=(tokka, rahzar),
            frame=1,
        ),
        stage=3,
        boss_active=True,
        extras={"event": 0x0A},
    )
    action = Stage1Policy().tick(state).action
    assert action is not None
    assert action.reason == "approach_left"
    assert action.action[6] == 1  # LEFT
    assert action.action[1] == 0  # not Y yet


def test_super_shredder_form2_uses_dodge_cycle() -> None:
    """Form 2 keeps a left standoff and hops instead of standing still."""
    boss = EnemyState(
        slot=0, x=160, y=180, health=120, active=True, kind=0xAE
    )
    state = replace(
        _playing(player_x=120, player_y=180, enemies=(boss,), frame=1),
        stage=9,
        boss_active=True,
        extras={"event": 0x0A},
    )
    policy = Stage1Policy()
    reasons = [
        policy.tick(state).action.reason for _ in range(80)
    ]
    assert any(
        r in {
            "shredder_attack",
            "shredder_dodge",
            "shredder_wait",
            "shredder_space",
            "shredder_approach",
            "shredder_align",
        }
        for r in reasons
    )
    assert "shredder_dodge" in reasons


def test_super_shredder_form2_switches_flank_at_left_wall() -> None:
    """A left-wall boss must not produce an unreachable left target."""
    boss = replace(
        _enemy(17, 166, 190),
        kind=0xAE,
    )
    state = replace(
        _playing(
            player_x=24,
            player_y=166,
            enemies=(boss,),
        ),
        stage=9,
        boss_active=True,
        extras={"event": 0x0A},
    )

    action = Stage1Policy().tick(state).action

    assert action is not None
    assert action.reason == "shredder_approach"
    assert action.action[7] == 1  # RIGHT, into the open arena
    assert action.action[6] == 0  # never walk into the left wall


def test_wounded_knee_jump_slash_on_stack() -> None:
    """Stage byte 6 + stacked bazooka Foot (0xb0) → jump-slash."""
    stack = EnemyState(
        slot=0, x=110, y=160, health=35, active=True, kind=0xB0
    )
    result = Stage1Policy().tick(
        replace(
            _playing(player_x=90, player_y=160, enemies=(stack,), frame=3),
            stage=6,
        )
    )
    assert result.action is not None
    assert result.action.reason == "jump_slash"
    assert result.action.action[0] == 1  # B
    assert result.action.action[1] == 1  # Y


def test_starbase_jump_slash_on_hover_foot() -> None:
    """Stage byte 8 + hover Foot (0x6A) → jump-slash."""
    hover = EnemyState(
        slot=0, x=110, y=160, health=20, active=True, kind=0x6A
    )
    result = Stage1Policy().tick(
        replace(
            _playing(player_x=90, player_y=160, enemies=(hover,), frame=3),
            stage=8,
        )
    )
    assert result.action is not None
    assert result.action.reason == "jump_slash"
    assert result.action.action[0] == 1  # B
    assert result.action.action[1] == 1  # Y


def test_super_shredder_form1_is_boss() -> None:
    """Starbase Super Shredder (0x52) stays boss_active below HP 80."""
    ram = _ram()
    ram[ADDR_STAGE] = 8
    _write_enemy(
        ram, ENEMY_BASES[0], x=180, y=160, health=40, char_id=0x52
    )
    state = parse_game_state(ram)
    assert state.boss_active
    assert state.extras["boss_hp"] == 40
    assert state.living_enemies[0].kind == 0x52


def test_super_shredder_form2_is_boss() -> None:
    """Finale Super Shredder (0xAE) is boss_active at spawn HP."""
    ram = _ram()
    ram[ADDR_STAGE] = 9
    _write_enemy(
        ram, ENEMY_BASES[0], x=180, y=200, health=190, char_id=0xAE
    )
    state = parse_game_state(ram)
    assert state.boss_active
    assert state.extras["boss_hp"] == 190
    assert state.living_enemies[0].kind == 0xAE


def test_ending_stage_is_cutscene() -> None:
    """Stage byte ≥10 (ending sequence) is CUTSCENE with HP 0 / px 0."""
    ram = _ram()
    ram[ADDR_STAGE] = 10
    ram[ADDR_EVENT] = 0x0F
    ram[PLAYER_BASE + OFF_HP] = 0
    write_u16le(ram, PLAYER_BASE + OFF_X, 0)
    state = parse_game_state(ram)
    assert state.mode is GameMode.CUTSCENE
    assert state.stage == 10


def test_leatherhead_char_is_boss() -> None:
    """Wounded Knee Leatherhead stays boss_active below HP 80."""
    ram = _ram()
    ram[ADDR_STAGE] = 6
    _write_enemy(
        ram, ENEMY_BASES[0], x=80, y=136, health=48, char_id=0xA2
    )
    state = parse_game_state(ram)
    assert state.boss_active
    assert state.extras["boss_hp"] == 48
    assert state.living_enemies[0].kind == 0xA2


def test_zero_char_ghost_slot_not_living() -> None:
    """Surf-stage ghost slots (char 0 / x 0) must not count as enemies."""
    ram = _ram()
    _write_enemy(ram, ENEMY_BASES[0], x=0, y=0, health=3)
    ram[ENEMY_BASES[0] + OFF_CHAR] = 0x00
    state = parse_game_state(ram)
    assert state.living_enemies == ()
    assert state.boss_active is False


def test_player_x_stall_prefers_down_jump() -> None:
    """Frozen player X drops + JUMP+RIGHT (Stage 2 dumpster)."""
    policy = Stage1Policy()
    reasons: list[str] = []
    for frame in range(1, 100):
        tick = policy.tick(
            _playing(player_x=109, player_y=205, camera_x=frame * 2)
        )
        assert tick.action is not None
        reasons.append(tick.action.reason)
    assert "stall_down" in reasons
    assert "stall_jump_right" in reasons
    first_stall = next(
        i for i, r in enumerate(reasons) if r.startswith("stall_")
    )
    assert reasons[first_stall] == "stall_down"


def test_far_park_enemy_approaches_not_edge_wait() -> None:
    """Foot parked past screen walk-band must be chased, not edge_wait."""
    enemy = _enemy(286, 214, 8)
    result = Stage1Policy().tick(
        _playing(player_x=128, player_y=214, enemies=(enemy,))
    )
    assert result.action is not None
    assert result.action.reason != "edge_wait"
    assert result.action.reason in {
        "approach_right",
        "align_up",
        "align_down",
        "attack",
        "attack_gap",
    }


def test_frozen_combat_position_triggers_jump_escape() -> None:
    """Unchanged player position + enemy HP cannot soft-lock forever."""
    enemy = _enemy(87, 192, 11)
    state = _playing(
        player_x=54,
        player_y=151,
        enemies=(enemy,),
    )
    policy = Stage1Policy()
    actions = [policy.tick(state).action for _ in range(242)]
    assert actions[-1] is not None
    assert actions[-1].reason == "combat_stall_escape"
    assert actions[-1].action[0] == 1  # B
    assert actions[-1].action[5] == 1  # DOWN
    assert actions[-1].action[7] == 1  # RIGHT
    assert actions[-1].action[8] == 0  # never A


def test_duo_boss_suppresses_combat_stall_escape() -> None:
    """Tokka/Rahzar left-flank poke must not be overridden by stall jumps."""
    tokka = EnemyState(
        slot=0, x=120, y=176, health=96, active=True, kind=0x48
    )
    rahzar = EnemyState(
        slot=1, x=160, y=176, health=96, active=True, kind=0xA0
    )
    state = replace(
        _playing(
            player_x=84,
            player_y=176,
            enemies=(tokka, rahzar),
            frame=1,
        ),
        stage=3,
        boss_active=True,
        extras={"event": 0x0A},
    )
    policy = Stage1Policy()
    reasons = [policy.tick(state).action.reason for _ in range(260)]
    assert "combat_stall_escape" not in reasons
    assert any(
        r in {"attack", "attack_gap", "approach_left", "approach_right", "space_left", "align_up", "align_down"}
        for r in reasons
    )


def test_duo_boss_jump_left_escapes_right_wall_pin() -> None:
    """Continuous Technodrome can enter the duo fight behind the right door."""
    tokka = EnemyState(
        slot=0, x=32, y=176, health=96, active=True, kind=0x48
    )
    rahzar = EnemyState(
        slot=1, x=181, y=177, health=96, active=True, kind=0xA0
    )
    state = replace(
        _playing(
            player_x=224,
            player_y=192,
            enemies=(tokka, rahzar),
        ),
        stage=3,
        boss_active=True,
        extras={"event": 0x0A},
    )

    action = Stage1Policy().tick(state).action

    assert action is not None
    assert action.reason == "duo_wall_escape"
    assert action.action[0] == 1  # B
    assert action.action[6] == 1  # LEFT
    assert action.action[8] == 0  # never A


def test_sewer_stage_avoids_spike_band_align_up() -> None:
    """Stage byte ==2 clamps fight Y so UP does not chase into spikes."""
    enemy = _enemy(140, 100, 16)
    state = replace(
        _playing(player_x=100, player_y=180, enemies=(enemy,)),
        stage=2,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason != "align_up"


def test_technodrome_not_forced_right() -> None:
    """Stage byte 3 must not inherit Sewer Surfin' forced-RIGHT pace."""
    enemy = _enemy(140, 160, 16)
    state = replace(
        _playing(player_x=100, player_y=160, enemies=(enemy,)),
        stage=3,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    # Spacing / approach left is allowed on Technodrome corridor.
    assert result.action.reason != "boss_jump_right"


def _tank_state(*, player_y: int, enemy_y: int) -> GameState:
    """Build a hard-mode Technodrome tank state with one blocking Foot."""
    foot = replace(
        _enemy(100, enemy_y, 12, slot=0),
        kind=0x6C,
    )
    return replace(
        _playing(
            player_x=80,
            player_y=player_y,
            enemies=(foot,),
        ),
        stage=3,
        extras={"event": 0x18},
    )


def test_tank_slot_reuse_drops_stale_grab_phase() -> None:
    """A fresh full-HP Foot in the same slot starts a new stun cycle."""
    tactics = TechnodromeTactics()
    tactics._in_tank = True
    tactics._target_slot = 0
    tactics._target_kind = 0x6C
    tactics._target_health = 4
    tactics._phase = "grab_gap"
    tactics._timer = 8

    action = tactics.next(_tank_state(player_y=160, enemy_y=160))

    assert action is not None
    assert action.reason == "blocker_retreat"
    assert tactics._phase == "retreat"


def test_tokka_rahzar_chars_are_boss() -> None:
    """Technodrome duo chars stay boss_active below HP 80."""
    ram = _ram()
    ram[ADDR_STAGE] = 3
    _write_enemy(ram, ENEMY_BASES[0], x=100, y=160, health=40)
    ram[ENEMY_BASES[0] + OFF_CHAR] = 0x48
    _write_enemy(ram, ENEMY_BASES[1], x=180, y=160, health=40)
    ram[ENEMY_BASES[1] + OFF_CHAR] = 0xA0
    state = parse_game_state(ram)
    assert state.boss_active
    assert state.extras["boss_hp"] == 40
    assert {e.kind for e in state.living_enemies} == {0x48, 0xA0}


def test_bebop_rocksteady_chars_are_boss() -> None:
    """Skull and Crossbones duo stays boss_active below HP 80."""
    ram = _ram()
    ram[ADDR_STAGE] = 5
    _write_enemy(
        ram, ENEMY_BASES[0], x=120, y=160, health=48, char_id=0xA8
    )
    _write_enemy(
        ram, ENEMY_BASES[1], x=200, y=160, health=48, char_id=0xAC
    )
    state = parse_game_state(ram)
    assert state.boss_active
    assert state.extras["boss_hp"] == 48
    assert {e.kind for e in state.living_enemies} == {0xA8, 0xAC}


def test_rat_king_left_chip_jump_right() -> None:
    """Pinned near auto-scroll left wall → JUMP+RIGHT escape."""
    boss = replace(_enemy(230, 144, 60, slot=0), kind=0x4A)
    state = replace(
        _playing(player_x=111, player_y=156, enemies=(boss,)),
        stage=2,
        boss_active=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "boss_jump_right"
    assert result.action.action[0] == 1  # B
    assert result.action.action[7] == 1  # RIGHT


def test_rat_king_long_poke_attacks() -> None:
    """Rat King at dx≈120 is still in the extended poke band."""
    boss = replace(_enemy(231, 144, 60, slot=0), kind=0x4A)
    state = replace(
        _playing(player_x=111, player_y=144, enemies=(boss,)),
        stage=2,
        boss_active=True,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason in {
        "attack",
        "attack_gap",
        "boss_jump_right",
        "space_left",
    }


def test_deep_y_ghost_not_living() -> None:
    """Post-boss Foot ghosts at y≥256 must not pull align_down."""
    ram = _ram()
    _write_enemy(ram, ENEMY_BASES[0], x=100, y=296, health=2)
    ram[ENEMY_BASES[0] + OFF_CHAR] = 0x66
    state = parse_game_state(ram)
    assert state.living_enemies == ()


def test_neon_waits_on_far_depth_enemy() -> None:
    """Stage byte 7: far Mode-7 Foot (low Y) → wait, not align_up."""
    enemy = _enemy(120, 100, 2)
    enemy = replace(enemy, kind=0x86)
    result = Stage1Policy().tick(
        replace(
            _playing(player_x=140, player_y=176, enemies=(enemy,)),
            stage=7,
        )
    )
    assert result.action is not None
    assert result.action.reason == "neon_wait"
    assert result.action.reason != "align_up"


def test_neon_attacks_near_band_enemy() -> None:
    """Stage byte 7: near-band Foot → grounded Y attack."""
    enemy = replace(_enemy(140, 165, 2), kind=0x86)
    result = Stage1Policy().tick(
        replace(
            _playing(player_x=120, player_y=176, enemies=(enemy,)),
            stage=7,
        )
    )
    assert result.action is not None
    assert result.action.reason in {"attack", "attack_gap", "approach_right"}


def test_krang_char_is_boss() -> None:
    """Neon Krang (0x4E) stays boss_active below HP 96."""
    ram = _ram()
    ram[ADDR_STAGE] = 7
    _write_enemy(
        ram, ENEMY_BASES[0], x=180, y=160, health=48, char_id=0x4E
    )
    state = parse_game_state(ram)
    assert state.boss_active is True
    assert state.extras["boss_hp"] == 48
    assert state.living_enemies[0].kind == 0x4E


def test_neon_prop_board_not_living() -> None:
    """Mode-7 board/debris chars must not count as combat targets."""
    ram = _ram()
    ram[ADDR_STAGE] = 7
    _write_enemy(
        ram, ENEMY_BASES[0], x=120, y=160, health=2, char_id=0x36
    )
    _write_enemy(
        ram, ENEMY_BASES[1], x=160, y=160, health=2, char_id=0xAC
    )
    _write_enemy(
        ram, ENEMY_BASES[2], x=200, y=165, health=2, char_id=0x86
    )
    state = parse_game_state(ram)
    assert {e.kind for e in state.living_enemies} == {0x86}
    assert state.boss_active is False


def test_jetpack_foot_hp80_not_boss() -> None:
    """Neon jetpack Foot at HP80 must not trip HP-only boss detect."""
    ram = _ram()
    ram[ADDR_STAGE] = 7
    _write_enemy(
        ram, ENEMY_BASES[0], x=180, y=160, health=80, char_id=0x1E
    )
    state = parse_game_state(ram)
    assert state.boss_active is False
    assert len(state.living_enemies) == 1
