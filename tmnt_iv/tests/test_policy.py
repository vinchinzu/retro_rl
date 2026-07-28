"""Pure-logic tests for TMNT IV Stage 1 policy and RAM adapter."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from snes_oneshot.behavior import NodeStatus
from snes_oneshot.game_state import EnemyState, GameMode, GameState
from tmnt_iv.policy import (
    HazardAvoid,
    Stage1Policy,
    TechnodromeTactics,
    build_stage1_tree,
)
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


def test_alleycat_uses_short_poke_cadence() -> None:
    """Stage byte 1 uses one attack frame followed by two release frames."""
    enemy = _enemy(110, 160, 16)
    policy = Stage1Policy()
    state = replace(_playing(enemies=(enemy,)), stage=1)
    reasons = [policy.tick(state).action.reason for _ in range(4)]
    assert reasons == ["attack", "attack_gap", "attack_gap", "attack"]


def test_alleycat_keeps_tight_vertical_lane() -> None:
    """A seven-pixel Alleycat offset is outside the tuned six-pixel band."""
    enemy = _enemy(110, 153, 16)
    result = Stage1Policy().tick(
        replace(_playing(player_y=160, enemies=(enemy,)), stage=1)
    )
    assert result.action is not None
    assert result.action.reason == "align_up"



def test_sewer_accepts_broad_vertical_lane() -> None:
    """Sewer waves attack across a 36-pixel lane instead of chasing spikes."""
    enemy = _enemy(110, 196, 16)
    result = Stage1Policy().tick(
        replace(_playing(player_y=160, enemies=(enemy,)), stage=2)
    )
    assert result.action is not None
    assert result.action.reason == "attack"


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


def test_rat_king_stays_boss_at_hp_1_to_3() -> None:
    """Finishers must keep Rat King as boss_active (old floor was HP ≥ 4)."""
    for hp in (3, 2, 1):
        ram = _ram()
        _write_enemy(
            ram, ENEMY_BASES[0], x=200, y=140, health=hp, char_id=0x4A
        )
        state = parse_game_state(ram)
        assert state.boss_active is True, f"hp={hp}"
        assert state.extras["boss_hp"] == hp


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


def test_wounded_knee_y_thrash_triggers_stall_escape() -> None:
    """Frozen X + bobbing Y + full 0xb0 HP eventually forces jump escape."""
    stack = EnemyState(
        slot=0, x=80, y=148, health=40, active=True, kind=0xB0
    )
    policy = Stage1Policy()
    first_escape: FrameAction | None = None
    # 240 stall frames of same signature, then escape phase.
    for frame in range(280):
        y = 148 + (frame % 8)  # bob within one Stage-7 y_bin of 16
        result = policy.tick(
            replace(
                _playing(player_x=128, player_y=y, enemies=(stack,), frame=frame),
                stage=6,
            )
        )
        assert result.action is not None
        if result.action.reason == "combat_stall_escape" and first_escape is None:
            first_escape = result.action
    assert first_escape is not None
    # Escape uses B+Y laterally against elevated stacks (first phases).
    assert first_escape.action[0] == 1  # B
    assert first_escape.action[1] == 1  # Y
    assert first_escape.action[6] + first_escape.action[7] == 1  # LEFT or RIGHT


def test_raphael_uses_tighter_wounded_knee_cadences() -> None:
    """Raph releases for three wave frames and one Leatherhead frame."""
    foot = replace(_enemy(110, 160, 16), kind=0x60)
    wave = replace(
        _playing(player_x=90, player_y=160, enemies=(foot,)),
        stage=6,
        extras={"char_id": 8},
    )
    wave_policy = Stage1Policy()
    wave_reasons = [
        wave_policy.tick(replace(wave, frame=frame)).action.reason
        for frame in range(6)
    ]
    assert wave_reasons == [
        "attack",
        "attack",
        "attack_gap",
        "attack_gap",
        "attack_gap",
        "attack",
    ]

    leatherhead = replace(foot, kind=0xA2)
    boss = replace(wave, enemies=(leatherhead,), boss_active=True)
    boss_policy = Stage1Policy()
    boss_reasons = [
        boss_policy.tick(replace(boss, frame=frame)).action.reason
        for frame in range(4)
    ]
    assert boss_reasons == ["attack", "attack", "attack_gap", "attack"]


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


def test_starbase_spawn_delay_does_not_trigger_dumpster_escape() -> None:
    """Frozen Starbase launch frames must keep the opening lane."""
    policy = Stage1Policy()
    for frame in range(60):
        action = policy.tick(
            replace(_playing(player_x=64, player_y=192, frame=frame), stage=8)
        ).action
        assert action is not None
        assert action.reason == "starbase_launch_right"
        assert action.action[7] == 1  # RIGHT
        assert action.action[0] == 0  # no dumpster jump


def test_raphael_closes_starbase_stack_with_jump_slash() -> None:
    """Raph must not Y-align forever beside the 0xB0/0xBA stack."""
    stack = replace(_enemy(40, 190, 8), kind=0xB0)
    state = replace(
        _playing(player_x=73, player_y=176, enemies=(stack,), frame=0),
        stage=8,
        extras={"char_id": 8},
    )
    action = Stage1Policy().tick(state).action
    assert action is not None
    assert action.reason == "raph_starbase_jump"
    assert action.action[0] == 1  # B
    assert action.action[1] == 1  # Y
    assert action.action[6] == 1  # LEFT toward the target
    assert action.action[8] == 0  # never A


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
            replace(
                _playing(player_x=109, player_y=205, camera_x=frame * 2),
                stage=1,
            )
        )
        assert tick.action is not None
        reasons.append(tick.action.reason)
    assert "stall_down" in reasons
    assert "stall_jump_right" in reasons
    first_stall = next(
        i for i, r in enumerate(reasons) if r.startswith("stall_")
    )
    assert reasons[first_stall] == "stall_down"


def test_stage0_skips_dumpster_stall() -> None:
    """Big Apple must not DOWN-thrash on frozen X (wave locks)."""
    policy = Stage1Policy()
    reasons: list[str] = []
    for frame in range(1, 100):
        tick = policy.tick(
            _playing(player_x=109, player_y=205, camera_x=frame * 2)
        )
        assert tick.action is not None
        reasons.append(tick.action.reason)
    assert not any(r.startswith("stall_") for r in reasons)


def test_raphael_uses_stage0_dumpster_escape() -> None:
    """Raphael's shorter collision route catches the late Big Apple dumpster."""
    policy = Stage1Policy()
    reasons: list[str] = []
    for frame in range(1, 100):
        state = replace(
            _playing(player_x=128, player_y=156, camera_x=frame * 2),
            extras={"char_id": 8},
        )
        tick = policy.tick(state)
        assert tick.action is not None
        reasons.append(tick.action.reason)
    assert "stall_down" in reasons
    assert "stall_jump_right" in reasons


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
    """Near left wall (56–80): hop-right; deeper wall uses grounded run."""
    boss = replace(_enemy(230, 144, 60, slot=0), kind=0x4A)
    hop = replace(
        _playing(player_x=70, player_y=156, enemies=(boss,)),
        stage=2,
        boss_active=True,
    )
    hop_result = Stage1Policy().tick(hop)
    assert hop_result.action is not None
    assert hop_result.action.reason == "boss_jump_right"
    assert hop_result.action.action[0] == 1  # B
    assert hop_result.action.action[7] == 1  # RIGHT
    # Continuous B+RIGHT at x≈24 soft-locks; grounded RIGHT recovers.
    deep = replace(
        _playing(player_x=40, player_y=156, enemies=(boss,)),
        stage=2,
        boss_active=True,
    )
    deep_result = Stage1Policy().tick(deep)
    assert deep_result.action is not None
    assert deep_result.action.reason == "boss_run_right"
    assert deep_result.action.action[0] == 0  # no B
    assert deep_result.action.action[7] == 1  # RIGHT


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
        "space_left",
    }


def test_sewer_skips_dumpster_stall_thrash() -> None:
    """Sewer auto-scroll freezes X; dumpster UP/DOWN thrash hits spikes."""
    policy = Stage1Policy()
    reasons: list[str] = []
    for frame in range(1, 120):
        state = replace(
            _playing(player_x=207, player_y=193, frame=frame),
            stage=2,
        )
        result = policy.tick(state)
        assert result.action is not None
        reasons.append(result.action.reason)
    assert not any(r.startswith("stall_") for r in reasons)
    # Prefer holding the bottom lane over idle thrash.
    assert any(r in {"walk_right", "sewer_drop_lane", "walk"} for r in reasons)


def test_sewer_spike_jump_when_near() -> None:
    """Hanging spike prop 0x1C within adx 56 → jump-right (A/B best)."""
    state = replace(
        _playing(player_x=200, player_y=192),
        stage=2,
        extras={"hazards": ((230, 202, 0x1C),)},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "sewer_spike_jump"
    assert result.action.action[0] == 1  # B
    assert result.action.action[7] == 1  # RIGHT


def test_sewer_spike_skips_when_foot_closer() -> None:
    """Do not abandon a closer Foot to chase a far spike column."""
    enemy = _enemy(190, 190, 16)
    state = replace(
        _playing(player_x=200, player_y=192, enemies=(enemy,)),
        stage=2,
        extras={"hazards": ((280, 202, 0x1C),)},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason != "sewer_spike_jump"


def test_sewer_drops_lane_when_high_between_waves() -> None:
    """Between-wave walk from mid height drops toward the water lane."""
    state = replace(
        _playing(player_x=200, player_y=140),
        stage=2,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "sewer_drop_lane"
    assert result.action.action[5] == 1  # DOWN
    assert result.action.action[7] == 1  # RIGHT


def test_rat_king_suppresses_combat_stall_escape() -> None:
    """Long Rat King poke must not be overridden by jump-escape thrash."""
    boss = replace(_enemy(231, 160, 40, slot=0), kind=0x4A)
    policy = Stage1Policy()
    reasons: list[str] = []
    for frame in range(1, 280):
        state = replace(
            _playing(player_x=140, player_y=160, enemies=(boss,), frame=frame),
            stage=2,
            boss_active=True,
        )
        result = policy.tick(state)
        if result.action is not None:
            reasons.append(result.action.reason)
    assert "combat_stall_escape" not in reasons


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


def test_pizza_pickup_in_extras_not_living() -> None:
    """Pizza box (0x30, HP0) is a pickup, not a combat target."""
    ram = _ram()
    _write_enemy(
        ram, ENEMY_BASES[0], x=140, y=180, health=0, char_id=0x30
    )
    state = parse_game_state(ram)
    assert state.living_enemies == ()
    assert state.extras["pickups"] == ((140, 180, 0x30),)


def test_pizza_seek_when_hurt() -> None:
    """After a real HP chunk is missing, walk toward on-screen pizza."""
    state = replace(
        _playing(player_x=80, player_y=180, health=48),
        extras={"pickups": ((160, 180, 0x30),)},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "pizza_seek"
    assert result.action.action[7] == 1  # RIGHT


def test_pizza_pickup_underfoot() -> None:
    """Tap Y when standing on the pizza box."""
    state = replace(
        _playing(player_x=150, player_y=180, health=48),
        extras={"pickups": ((152, 182, 0x30),)},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "pizza_pickup"


def test_pizza_ignored_at_full_hp() -> None:
    """Do not divert for pizza while already at Leo max HP."""
    enemy = _enemy(120, 180, 16)
    state = replace(
        _playing(player_x=80, player_y=180, health=80, enemies=(enemy,)),
        extras={"pickups": ((160, 180, 0x30),)},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason != "pizza_seek"
    assert result.action.reason != "pizza_pickup"


def test_pizza_seek_is_limited_to_big_apple() -> None:
    """Far pizza chase mid-wave / Skull is blocked; Alleycat/Sewer between-wave OK."""
    enemy = _enemy(120, 180, 16)
    # Skull: never chase.
    skull = replace(
        _playing(player_x=80, player_y=180, health=24, enemies=(enemy,)),
        stage=5,
        extras={"pickups": ((240, 180, 0x30),)},
    )
    skull_tick = Stage1Policy().tick(skull)
    assert skull_tick.action is not None
    assert skull_tick.action.reason not in {
        "pizza_seek",
        "pizza_pickup",
        "pizza_disengage",
    }
    # Alleycat mid-wave far pizza: no seek (desyncs packs).
    alley_mid = replace(
        _playing(player_x=80, player_y=180, health=24, enemies=(enemy,)),
        stage=1,
        extras={"pickups": ((240, 180, 0x30),)},
    )
    mid_tick = Stage1Policy().tick(alley_mid)
    assert mid_tick.action is not None
    assert mid_tick.action.reason not in {
        "pizza_seek",
        "pizza_pickup",
        "pizza_disengage",
    }
    # Alleycat between waves: far seek allowed.
    alley_clear = replace(
        _playing(player_x=80, player_y=180, health=24),
        stage=1,
        extras={"pickups": ((240, 180, 0x30),)},
    )
    clear_tick = Stage1Policy().tick(alley_clear)
    assert clear_tick.action is not None
    assert clear_tick.action.reason == "pizza_seek"
    # Sewer between waves: same underfoot + between-wave rule as Alleycat.
    sewer_clear = replace(
        _playing(player_x=80, player_y=190, health=24),
        stage=2,
        extras={"pickups": ((240, 190, 0x30),)},
    )
    sewer_tick = Stage1Policy().tick(sewer_clear)
    assert sewer_tick.action is not None
    assert sewer_tick.action.reason == "pizza_seek"


def test_alleycat_underfoot_pizza_pickup() -> None:
    """Alleycat may tap Y on pizza underfoot even mid-wave."""
    enemy = _enemy(200, 180, 16)
    state = replace(
        _playing(player_x=150, player_y=180, health=24, enemies=(enemy,)),
        stage=1,
        extras={"pickups": ((152, 182, 0x30),)},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "pizza_pickup"


def test_alleycat_uses_left_flank() -> None:
    """Alleycat left shoulder: wrong-side approach closes then spaces left."""
    enemy = _enemy(100, 160, 16)
    # Player right of enemy → not on preferred left flank → approach.
    result = Stage1Policy().tick(
        replace(
            _playing(player_x=140, player_y=160, enemies=(enemy,)),
            stage=1,
        )
    )
    assert result.action is not None
    # Left flank ideal is left of enemy; from the right we approach or space.
    assert result.action.reason in {
        "approach_left",
        "space_left",
        "attack",
        "attack_gap",
        "in_range",
    }


def test_pizza_ignored_for_scratch_damage_when_far() -> None:
    """Scratch damage must not chase a distant pizza across the screen."""
    enemy = _enemy(120, 180, 16)
    state = replace(
        _playing(player_x=80, player_y=180, health=72, enemies=(enemy,)),
        extras={"pickups": ((200, 180, 0x30),)},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason not in {"pizza_seek", "pizza_pickup"}


def test_pizza_grabbed_on_scratch_when_near() -> None:
    """Walking past a pizza at 76 HP should still pick it up."""
    state = replace(
        _playing(player_x=100, player_y=180, health=76),
        extras={"pickups": ((120, 190, 0x30),)},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason in {"pizza_seek", "pizza_pickup"}


def test_pizza_sought_across_screen_when_critical() -> None:
    """Critical HP must cross the screen for a visible pizza box.

    Stage1 heal=none died with pizza ~200px right while Leo walked left.
    """
    enemy = _enemy(60, 180, 16)
    state = replace(
        _playing(
            player_x=87,
            player_y=183,
            health=28,
            enemies=(enemy,),
        ),
        extras={"pickups": ((284, 147, 0x30),)},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "pizza_seek"
    assert result.action.action[7] == 1  # RIGHT toward pizza


def test_pizza_not_sought_during_boss_when_healthy() -> None:
    """Baxter + full-ish HP: do not abandon poke for distant pizza."""
    boss = replace(_enemy(285, 192, 96, slot=0), kind=0x44)
    state = replace(
        _playing(player_x=212, player_y=180, health=56, enemies=(boss,)),
        boss_active=True,
        stage=0,
        extras={"pickups": ((65, 147, 0x30),), "boss_hp": 45},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason not in {
        "pizza_seek",
        "pizza_pickup",
        "pizza_disengage",
    }


def test_pizza_sought_during_boss_when_critical() -> None:
    """Clean survival: HP ≤ 32 during Baxter may grab arena pizza."""
    boss = replace(_enemy(285, 192, 96, slot=0), kind=0x44)
    state = replace(
        _playing(player_x=212, player_y=180, health=12, enemies=(boss,)),
        boss_active=True,
        stage=0,
        extras={"pickups": ((65, 147, 0x30),), "boss_hp": 45},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason in {"pizza_seek", "pizza_pickup"}


def test_production_policy_does_not_hijack_for_hazards() -> None:
    """Production tick does not run active hazard dodge (Clean A/B winner)."""
    for health in (28, 72):
        state = replace(
            _playing(player_x=120, player_y=180, health=health),
            stage=0,
            extras={"hazards": ((130, 90, 0x36),), "pickups": ()},
        )
        result = Stage1Policy().tick(state)
        assert result.action is not None
        assert result.action.reason not in {"hazard_jump", "hazard_dodge"}


def test_baxter_lane_recenters_from_right() -> None:
    """Without arena pizza, leave the right-edge thrash band (stay left of Baxter)."""
    # Overlapping / right of Baxter body → forced releft for Clean standoff.
    boss = replace(_enemy(160, 192, 96, slot=0), kind=0x44)
    state = replace(
        _playing(player_x=170, player_y=176, health=52, enemies=(boss,)),
        boss_active=True,
        stage=0,
        extras={"boss_hp": 96, "pickups": ()},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "baxter_releft"
    assert result.action.action[6] == 1  # LEFT


def test_baxter_skips_lane_when_pizza_and_healthy() -> None:
    """Boss.state left pizza: do not thrash lane while HP is not critical."""
    boss = replace(_enemy(285, 192, 96, slot=0), kind=0x44)
    state = replace(
        _playing(player_x=212, player_y=180, health=56, enemies=(boss,)),
        boss_active=True,
        stage=0,
        extras={"boss_hp": 96, "pickups": ((65, 147, 0x30),)},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason not in {
        "baxter_lane",
        "baxter_releft",
        "baxter_ground_poke",
        "pizza_seek",
        "pizza_pickup",
    }


def test_pizza_seek_mid_damage_crosses_far_box() -> None:
    """Clean Stage1: HP 50 must pull pizza at dist≈186 (old MID=96 missed it)."""
    state = replace(
        _playing(player_x=100, player_y=180, health=50),
        stage=0,
        extras={"pickups": ((260, 180, 0x30),), "hazards": ()},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "pizza_seek"
    assert result.action.action[7] == 1  # RIGHT


def test_empty_screen_walk_does_not_stutter_attack() -> None:
    """No enemies and no pizza → pure walk; never spam Y every N frames."""
    policy = Stage1Policy()
    reasons: list[str] = []
    y_pressed = 0
    for frame in range(1, 60):
        result = policy.tick(_playing(frame=frame))
        assert result.action is not None
        reasons.append(result.action.reason)
        # Y is button index 1 on the SNES action vector.
        if result.action.action[1] == 1:
            y_pressed += 1
    assert y_pressed == 0
    assert all(r == "walk_right" for r in reasons)


def test_hazard_avoid_helper_jumps_past_ground_ball() -> None:
    """HazardAvoid helper: empty screen + ground ball → jump-right."""
    state = replace(
        _playing(player_x=120, player_y=180),
        stage=0,
        extras={"hazards": ((150, 170, 0x32),), "pickups": ()},
    )
    action = HazardAvoid().next(state)
    assert action is not None
    assert action.reason == "hazard_jump"
    assert action.action[0] == 1
    assert action.action[7] == 1


def test_hazard_avoid_helper_ceiling_ball() -> None:
    """HazardAvoid helper reacts to ceiling ball in the clear band."""
    state = replace(
        _playing(player_x=120, player_y=180),
        stage=0,
        extras={"hazards": ((128, 80, 0x36),), "pickups": ()},
    )
    action = HazardAvoid().next(state)
    assert action is not None
    assert action.reason in {"hazard_jump", "hazard_dodge", "hazard_wait"}


def test_hazard_avoid_skips_other_stages() -> None:
    """HazardAvoid is Big Apple only."""
    state = replace(
        _playing(player_x=120, player_y=180),
        stage=1,
        extras={"hazards": ((130, 80, 0x36),), "pickups": ()},
    )
    assert HazardAvoid().next(state) is None


def test_baxter_jump_slash_when_elevated() -> None:
    """Elevated Baxter (elev≥10) → pulsed jump-slash (frame%4==0 hits B+Y)."""
    # elev = 171 - 158 = 13 (Stage1 probe max band). Stand left of body.
    boss = replace(_enemy(160, 158, 96, slot=0), kind=0x44)
    policy = Stage1Policy()
    reasons: list[str] = []
    saw_jump = False
    for frame in (0, 4, 8, 1, 2, 3):
        state = replace(
            _playing(
                player_x=110,
                player_y=171,
                health=72,
                enemies=(boss,),
                frame=frame,
            ),
            boss_active=True,
            stage=0,
            extras={"boss_hp": 96, "pickups": ()},
        )
        result = policy.tick(state)
        assert result.action is not None
        reasons.append(result.action.reason)
        if result.action.reason == "baxter_jump_slash":
            saw_jump = True
            assert result.action.action[0] == 1  # B jump
            assert result.action.action[1] == 1  # Y attack
    assert saw_jump
    assert any(
        r in {"baxter_jump_slash", "baxter_reclose", "baxter_ground_poke"}
        for r in reasons
    )


def test_elevated_foot_jump_slash() -> None:
    """True-air Foot (well above lane offsets) get a jump-slash."""
    # elev = 170 - 120 = 50 ≥ 44; ordinary 20–40px lane offsets still align.
    foot = replace(_enemy(140, 120, 16, slot=0), kind=0x60)
    state = replace(
        _playing(player_x=120, player_y=170, enemies=(foot,)),
        stage=0,
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason == "jump_slash"
    assert result.action.action[0] == 1
    assert result.action.action[1] == 1


def test_lane_offset_still_aligns_not_jump() -> None:
    """A normal upper-lane Foot uses walk-align, not jump-slash."""
    foot = replace(_enemy(120, 130, 16, slot=0), kind=0x60)
    # elev = 160 - 130 = 30 < 44 elevated threshold.
    result = Stage1Policy().tick(
        _playing(player_x=90, player_y=160, enemies=(foot,))
    )
    assert result.action is not None
    assert result.action.reason == "align_up"


def test_slash_boss_does_not_elevated_jump() -> None:
    """Slash shell still uses SlashTactics / grounded poke, not jump-slash."""
    slash = replace(_enemy(140, 120, 160, slot=0), kind=0x50)
    state = replace(
        _playing(player_x=100, player_y=170, enemies=(slash,)),
        boss_active=True,
        stage=4,
        extras={"event": 0x0A, "boss_hp": 160, "char_id": 8},
    )
    result = Stage1Policy().tick(state)
    assert result.action is not None
    assert result.action.reason != "jump_slash"
    assert result.action.reason != "baxter_jump_slash"
