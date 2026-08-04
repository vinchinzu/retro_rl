"""Final Fight WRAM layout and GameState adapter.

Addresses are WRAM offsets (stable-retro ``get_ram()`` / data.json).
Source notes: TCRF Notes:Final Fight (SNES), USA.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from retro_harness.ram_state import EnemyState, GameMode, GameState

# Player / enemy entity layout (direct-page relative offsets).
OFF_STATUS = 0x00
OFF_X = 0x07
OFF_Y = 0x0D
OFF_HP = 0x14
OFF_LIVES = 0x6E

PLAYER_BASE = 0x0D00
ENEMY_BASES: tuple[int, ...] = (0x1000, 0x10B0, 0x1140)
BOSS_BASE = 0x11E0
ENEMY_STRIDE = 0xB0

ADDR_GAME_STATUS = 0x0CA0
ADDR_ROUND = 0x0CB0
ADDR_AREA = 0x0CB1
ADDR_ROUNDS_CLEARED = 0x0CB2
ADDR_LEVEL_END = 0x0CD0
ADDR_BOSS_DEAD_FLAG = 0x0CD2
ADDR_CAMERA_X = 0x0E07
ADDR_GO_FLASHING = 0x0CD7
ADDR_CHAR_SELECT = 0x008F

# Living peaks: subway HP148, West Andore ≈216, West Area1 ≈250.
# Kill-frame UF on regular slots often lands ~254 (Clear_w5 corpse);
# Damnd/Sodom boss UF ~237–254 — boss kill uses boss HP / CLEAR_AREA.
ENTITY_HP_MAX = 252  # Area1 ≤250 living; ≥253 = UF ghost
# Drawn/fighting entities use status 0x03. Status 0x01 is spawn/intro —
# usually ignored, but subway left-edge spawners (sx≈-50, living HP) still
# chip while the policy walks right. Junk slots use 0x02. Door/subway
# post-kill: status 0x03 with HP==0 or UF≥253 can still damage.
ENTITY_STATUS_COMBAT: frozenset[int] = frozenset({0x03})
ENTITY_STATUS_SPAWN: frozenset[int] = frozenset({0x01})
# Slum waves often spawn just left of the camera; keep a wide left band.
ENTITY_ONSCREEN_MARGIN_LEFT = 128
ENTITY_ONSCREEN_MARGIN_RIGHT = 48
ENTITY_SCREEN_WIDTH = 256
# Spawn-status fighters only matter when near the playable band.
# West Side cam640 intros at sx≈310–400 are handled by segment
# ``west_spawn_plant`` (raw st=01 scan) — keep this margin tight so
# far intros do not replace living combat targets mid-wave.
ENTITY_SPAWN_MARGIN_LEFT = 80
ENTITY_SPAWN_MARGIN_RIGHT = 48


class GameStatus(IntEnum):
    """Values observed at ``ADDR_GAME_STATUS`` (TCRF)."""

    CHARACTER_SELECT = 0x00
    OPEN_STAGE_A = 0x02
    OPEN_STAGE_B = 0x04
    ACTIVE_GAMEPLAY = 0x06
    CLEAR_AREA = 0x08
    CLEAR_ROUND = 0x0A
    # Break Car / glass bonus between stages (observed post-Sodom).
    BONUS_GAMEPLAY = 0x0E


class RoundId(IntEnum):
    """Values at ``ADDR_ROUND`` / ``0x0CB0`` (TCRF)."""

    SLUM = 0x00
    SUBWAY = 0x01
    WEST_SIDE = 0x02
    INDUSTRIAL = 0x03
    BAY_AREA = 0x04
    UP_TOWN = 0x05
    BREAK_CAR = 0x06
    BREAK_GLASS = 0x07


def read_u8(ram: np.ndarray, address: int) -> int:
    """Read one unsigned byte from WRAM."""
    return int(ram[address])


def read_u16le(ram: np.ndarray, address: int) -> int:
    """Read a little-endian unsigned 16-bit value from WRAM."""
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def entity_is_combat_active(
    status: int,
    hp: int,
    *,
    x: int | None = None,
    camera_x: int | None = None,
) -> bool:
    """True for living fighters or damaging on-screen corpse chasers.

    Living: ``0 < hp <= ENTITY_HP_MAX`` (subway ≈148, West Andore ≈216,
    Area1 ≈250). Damaging ghost: ``hp == 0`` or UF
    ``hp > ENTITY_HP_MAX`` (≥253 after wave5 kill-frame).
    Status ``0x01`` spawners with living HP near the camera also count —
    subway left-edge intros chip through ``walk_right`` if ignored.
    """
    living = 0 < hp <= ENTITY_HP_MAX
    corpse_threat = hp == 0 or hp > ENTITY_HP_MAX

    if status in ENTITY_STATUS_COMBAT:
        if not living and not corpse_threat:
            return False
        if x is not None and camera_x is not None:
            left = camera_x - ENTITY_ONSCREEN_MARGIN_LEFT
            right = (
                camera_x
                + ENTITY_SCREEN_WIDTH
                + ENTITY_ONSCREEN_MARGIN_RIGHT
            )
            if x < left or x > right:
                return False
        return True

    if status in ENTITY_STATUS_SPAWN:
        # Living intros near the camera chip if ignored. Cam994 also
        # leaves status-01 underflow corpses that still hurtbox until
        # plant-punched — same on-screen band as combat ghosts.
        if not living and not corpse_threat:
            return False
        if x is None or camera_x is None:
            return False
        left = camera_x - ENTITY_SPAWN_MARGIN_LEFT
        right = (
            camera_x + ENTITY_SCREEN_WIDTH + ENTITY_SPAWN_MARGIN_RIGHT
        )
        return left <= x <= right

    return False


def _mode_from_status(status: int) -> GameMode:
    if status == GameStatus.CHARACTER_SELECT:
        return GameMode.MENU
    if status in (GameStatus.OPEN_STAGE_A, GameStatus.OPEN_STAGE_B):
        return GameMode.CUTSCENE
    if status == GameStatus.ACTIVE_GAMEPLAY:
        return GameMode.PLAYING
    if status in (GameStatus.CLEAR_AREA, GameStatus.CLEAR_ROUND):
        return GameMode.LEVEL_COMPLETE
    return GameMode.UNKNOWN


def _read_entity(
    ram: np.ndarray,
    base: int,
    *,
    slot: int,
    camera_x: int,
) -> EnemyState:
    status = read_u8(ram, base + OFF_STATUS)
    hp_raw = read_u8(ram, base + OFF_HP)
    x = read_u16le(ram, base + OFF_X)
    y = read_u16le(ram, base + OFF_Y)
    # Normalize underflow corpses to HP0 so living_enemies excludes them
    # while threat_enemies still sees active=True (same as door HP0).
    hp = 0 if hp_raw > ENTITY_HP_MAX else hp_raw
    return EnemyState(
        slot=slot,
        x=x,
        y=y,
        health=hp,
        active=entity_is_combat_active(
            status, hp_raw, x=x, camera_x=camera_x
        ),
        animation=status,
    )


def parse_game_state(ram: np.ndarray, frame: int = 0) -> GameState:
    """Project Final Fight WRAM into a normalized ``GameState``."""
    status = read_u8(ram, ADDR_GAME_STATUS)
    mode = _mode_from_status(status)
    player_active = read_u8(ram, PLAYER_BASE + OFF_STATUS) != 0
    health = read_u8(ram, PLAYER_BASE + OFF_HP)
    lives = read_u8(ram, PLAYER_BASE + OFF_LIVES)
    camera_x = read_u16le(ram, ADDR_CAMERA_X)
    regular = tuple(
        _read_entity(ram, base, slot=i, camera_x=camera_x)
        for i, base in enumerate(ENEMY_BASES)
    )
    boss = _read_entity(ram, BOSS_BASE, slot=3, camera_x=camera_x)
    boss_status = read_u8(ram, BOSS_BASE + OFF_STATUS)
    # Include boss in the enemy tuple when combat-active so nearest-enemy
    # targeting reaches Damnd / Haggar without a separate path.
    enemies = regular + ((boss,) if boss.active else ())
    boss_active = boss_status != 0
    # CLEAR_AREA (0x08) is a sub-section flash; only CLEAR_ROUND means
    # the stage (Damnd) is done. Expose area_clear for wave heuristics.
    level_complete = status == GameStatus.CLEAR_ROUND
    area_clear = status == GameStatus.CLEAR_AREA
    player_dead = lives == 0 and (
        health == 0
        or not player_active
        or health > ENTITY_HP_MAX
    )
    living = tuple(e for e in enemies if e.active and e.health > 0)
    threats = tuple(e for e in enemies if e.active)
    go_flashing = read_u8(ram, ADDR_GO_FLASHING) == 1
    player_x = read_u16le(ram, PLAYER_BASE + OFF_X)
    # Cam≥840: distant HP0 softlocks scroll if they hold screen_locked.
    # Only living + overlapping corpses (dx<50) keep the lock there.
    if camera_x >= 840:
        lock_threats = tuple(
            e
            for e in threats
            if e.health > 0 or abs(e.x - player_x) < 50
        )
    else:
        lock_threats = threats
    return GameState(
        frame=frame,
        mode=mode,
        stage=read_u8(ram, ADDR_ROUND),
        room=read_u8(ram, ADDR_AREA),
        player_x=player_x,
        player_y=read_u16le(ram, PLAYER_BASE + OFF_Y),
        health=health,
        lives=lives,
        camera_x=camera_x,
        enemies=enemies,
        boss_active=boss_active,
        level_complete=level_complete,
        player_dead=player_dead,
        screen_locked=bool(lock_threats) and not go_flashing,
        extras={
            "game_status": status,
            "player_active": player_active,
            "go_flashing": go_flashing,
            "boss_hp": boss.health,
            "boss_status": boss_status,
            "boss_x": boss.x,
            "boss_y": boss.y,
            "char_select": read_u8(ram, ADDR_CHAR_SELECT),
            "area_clear": area_clear,
            "rounds_cleared": read_u8(ram, ADDR_ROUNDS_CLEARED),
            "boss_dead_flag": read_u8(ram, ADDR_BOSS_DEAD_FLAG),
            "level_end_flag": read_u8(ram, ADDR_LEVEL_END),
        },
    )
