"""TMNT IV WRAM layout and GameState adapter.

Addresses are WRAM offsets (stable-retro ``get_ram()`` / data.json).
Seeded from GameHacking PAR codes (USA); coords confirmed by walk/attack
probes from ``Stage1.state``.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from retro_harness.ram_state import EnemyState, GameMode, GameState

# Player 1 entity block (P2 at +0x70). Same HP offset as enemy slots.
PLAYER_BASE = 0x0400
PLAYER_STRIDE = 0x70
OFF_X = 0x08
OFF_Y = 0x0C
OFF_CHAR = 0x14
OFF_ANIM = 0x28
OFF_HP = 0x4A
OFF_IFRAMES = 0x6E

# Enemy entity bases: HP PAR codes at base+OFF_HP (stride 0x70).
ENEMY_BASES: tuple[int, ...] = tuple(
    0x08D0 + i * 0x70 for i in range(7)
)
ENEMY_HP_BASES: tuple[int, ...] = tuple(b + OFF_HP for b in ENEMY_BASES)
ENEMY_STRIDE = 0x70

# Globals
ADDR_MENU = 0x0032
ADDR_EVENT = 0x0070
ADDR_STAGE = 0x0082
ADDR_TIMER = 0x0096
ADDR_LIVES = 0x1AA0

# Progress / scroll heuristic (increases while advancing). Player and
# enemy X/Y stay screen-space during Stage 1 locks — combat zeros
# camera_x so edge clamps use raw screen X.
ADDR_CAMERA_X = 0x003A

ENTITY_HP_MAX = 0xC0  # Foot ~16; Rocksteady / bosses can exceed 0x60
ENTITY_X_MAX = 512  # reject despawn sentinels (e.g. 65504)
# Jetpack Foot on Neon can sit at HP 80 — require ≥96 for HP-only
# boss detection; known boss chars cover mid-fight low HP.
BOSS_HP_MIN = 96  # Baxter / Metalhead / Rat King spawn ≥96
# Boss chars stay "boss" after HP drops below BOSS_HP_MIN.
BOSS_CHAR_IDS: frozenset[int] = frozenset(
    {
        0x44,  # Baxter Stockman
        0x46,  # Metalhead
        0x48,  # Tokka (Technodrome duo)
        0x4A,  # Rat King (Sewer Surfin')
        0x4E,  # Krang (Neon Night Riders)
        0x50,  # Slash (Prehistoric)
        0x52,  # Super Shredder form 1 (Starbase)
        0xAE,  # Super Shredder form 2 (finale)
        0xA0,  # Rahzar (Technodrome duo)
        0xA2,  # Leatherhead (Wounded Knee)
        0xA8,  # Bebop (Skull and Crossbones)
        0xAC,  # Rocksteady (Skull and Crossbones)
    }
)
# Friendly / non-combat slots that share the enemy entity table.
NPC_CHAR_IDS: frozenset[int] = frozenset(
    {
        0xC4,  # April O'Neil
        0xEE,  # Pterodactyl carrier (drops Foot, then leaves)
    }
)
# Ground pizza box (full HP restore). HP byte stays 0; do not fight.
PIZZA_CHAR_IDS: frozenset[int] = frozenset({0x30})
# Stage 1 ceiling/wrecking hazards (HP 0; walking under 0x36 is a −24).
# Stage 3 (Sewer) hanging spike props: char 0x1C (and occasional 0x2C)
# deal −16 with HP 0; not living enemies. Dodge with jump-right when near.
HAZARD_CHAR_IDS: frozenset[int] = frozenset({0x32, 0x36, 0x1C, 0x2C})
# Leo's full health bar (used for pizza-seek gating).
LEO_MAX_HP = 80
# Mode-7 Neon Night Riders (stage byte 7): boards / shots / debris that
# reuse combat-looking chars (incl. 0xAC Rocksteady id at HP 2).
NEON_PROP_CHAR_IDS: frozenset[int] = frozenset(
    {0x36, 0x3C, 0xAC, 0xAE, 0xB0, 0xB2}
)
# Prehistoric dinos (0x6C) are combat targets but need jump-slash (B+Y);
# ground Y alone does not reduce their HP.


class MenuId(IntEnum):
    """Observed values at ``ADDR_MENU`` during boot."""

    TITLE = 0x00
    CHAR_SELECT = 0x02
    TRANSITION = 0x04
    PLAYING = 0x06


def read_u8(ram: np.ndarray, address: int) -> int:
    """Read one unsigned byte from WRAM."""
    return int(ram[address])


def read_u16le(ram: np.ndarray, address: int) -> int:
    """Read a little-endian unsigned 16-bit value from WRAM."""
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def write_u16le(ram: np.ndarray, address: int, value: int) -> None:
    """Write a little-endian unsigned 16-bit value (tests)."""
    ram[address] = value & 0xFF
    ram[address + 1] = (value >> 8) & 0xFF


# Intermission / stage-load events (Baxter clear → Alleycat Blues).
# 0x0D/0x0E/0x0F appear during the Super Shredder ending sequence.
_TRANSITION_EVENTS: frozenset[int] = frozenset(
    {0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0D, 0x0E, 0x0F, 0x19}
)


def _mode_from_ram(
    menu: int,
    health: int,
    lives: int,
    *,
    player_x: int,
    event: int,
    stage: int = 0,
) -> GameMode:
    """Infer mode. ``lives==0`` with HP left is last-life PLAYING."""
    if menu == MenuId.TITLE:
        return GameMode.TITLE
    if menu in (MenuId.CHAR_SELECT, 0x01, 0x03, 0x05):
        return GameMode.MENU
    if menu != MenuId.PLAYING:
        return GameMode.UNKNOWN
    # The ending sequence (stage byte ≥10) zeros the player entity.
    if stage >= 10:
        return GameMode.CUTSCENE
    if health == 0 or health > ENTITY_HP_MAX:
        if lives > 0:
            return GameMode.CONTINUE
        return GameMode.GAME_OVER
    # Stage load zeros the player entity; event walks 0x19 → 0x04..0x09.
    if player_x == 0 or event in _TRANSITION_EVENTS:
        return GameMode.CUTSCENE
    return GameMode.PLAYING


def _read_enemy(
    ram: np.ndarray,
    base: int,
    *,
    slot: int,
    stage: int = 0,
) -> EnemyState:
    hp_raw = read_u8(ram, base + OFF_HP)
    x = read_u16le(ram, base + OFF_X)
    y = read_u16le(ram, base + OFF_Y)
    status = read_u8(ram, base)
    char_id = read_u8(ram, base + OFF_CHAR)
    hp = 0 if hp_raw > ENTITY_HP_MAX else hp_raw
    # Stage 3 surf sometimes leaves char=0 / x=0 ghost slots with
    # residual HP — prefer_left_threat then soft-locks at screen-left.
    neon_prop = stage == 7 and char_id in NEON_PROP_CHAR_IDS
    living = (
        0 < hp <= ENTITY_HP_MAX
        and 0 < x < ENTITY_X_MAX
        and 0 < y < 256
        and char_id != 0
        and char_id not in NPC_CHAR_IDS
        and not neon_prop
    )
    return EnemyState(
        slot=slot,
        x=x,
        y=y,
        health=hp,
        active=living,
        animation=status,
        kind=char_id,
    )


def read_pizza_pickups(ram: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    """Return ``(x, y, char)`` for on-screen pizza boxes (HP byte is 0)."""
    found: list[tuple[int, int, int]] = []
    for base in ENEMY_BASES:
        char_id = read_u8(ram, base + OFF_CHAR)
        if char_id not in PIZZA_CHAR_IDS:
            continue
        x = read_u16le(ram, base + OFF_X)
        y = read_u16le(ram, base + OFF_Y)
        if 0 < x < ENTITY_X_MAX and 0 < y < 256:
            found.append((x, y, char_id))
    return tuple(found)


def read_hazards(ram: np.ndarray) -> tuple[tuple[int, int, int], ...]:
    """Return ``(x, y, char)`` for on-screen wrecking-ball hazards."""
    found: list[tuple[int, int, int]] = []
    for base in ENEMY_BASES:
        char_id = read_u8(ram, base + OFF_CHAR)
        if char_id not in HAZARD_CHAR_IDS:
            continue
        x = read_u16le(ram, base + OFF_X)
        y = read_u16le(ram, base + OFF_Y)
        if 0 < x < ENTITY_X_MAX and 0 < y < 256:
            found.append((x, y, char_id))
    return tuple(found)


def parse_game_state(ram: np.ndarray, frame: int = 0) -> GameState:
    """Project TMNT IV WRAM into a normalized ``GameState``."""
    menu = read_u8(ram, ADDR_MENU)
    health = read_u8(ram, PLAYER_BASE + OFF_HP)
    lives = read_u8(ram, ADDR_LIVES)
    event = read_u8(ram, ADDR_EVENT)
    stage = read_u8(ram, ADDR_STAGE)
    player_x = read_u16le(ram, PLAYER_BASE + OFF_X)
    player_y = read_u16le(ram, PLAYER_BASE + OFF_Y)
    mode = _mode_from_ram(
        menu,
        health,
        lives,
        player_x=player_x,
        event=event,
        stage=stage,
    )
    progress_x = read_u16le(ram, ADDR_CAMERA_X)
    enemies = tuple(
        _read_enemy(ram, base, slot=i, stage=stage)
        for i, base in enumerate(ENEMY_BASES)
    )
    living = tuple(e for e in enemies if e.active and e.health > 0)
    # Known boss chars stay boss_active down to HP 1 so finishers land
    # (Rat King at HP 3 used to drop off at the old floor of 4, and Clean
    # walks away from a near-dead Footski). Neon board props that reuse
    # Rocksteady's id (0xAC @ HP2) are filtered in ``_read_enemy`` on
    # stage 7, so they never reach this candidate list.
    boss_candidates = tuple(
        e
        for e in living
        if e.health >= BOSS_HP_MIN
        or (e.kind in BOSS_CHAR_IDS and e.health >= 1)
    )
    boss = boss_candidates[0] if boss_candidates else None
    player_dead = lives == 0 and (health == 0 or health > ENTITY_HP_MAX)
    pickups = read_pizza_pickups(ram)
    hazards = read_hazards(ram)
    return GameState(
        frame=frame,
        mode=mode,
        # 0 = Big Apple (Stage 1), 1 = Alleycat Blues (Stage 2), …
        stage=stage,
        room=0,
        player_x=player_x,
        player_y=player_y,
        health=health,
        lives=lives,
        # Progress word doubles as camera for wave-unlock / walk stall.
        camera_x=progress_x,
        enemies=enemies,
        boss_active=boss is not None,
        level_complete=False,
        player_dead=player_dead,
        screen_locked=bool(living),
        go_flashing=False,
        area_clear=False,
        extras={
            "menu": menu,
            "event": event,
            "timer": read_u8(ram, ADDR_TIMER),
            "char_id": read_u8(ram, PLAYER_BASE + OFF_CHAR),
            "anim": read_u8(ram, PLAYER_BASE + OFF_ANIM),
            "iframes": read_u8(ram, PLAYER_BASE + OFF_IFRAMES),
            # Dual-written for one-release back-compat with extras readers.
            "go_flashing": False,
            "area_clear": False,
            "progress_x": progress_x,
            "coords_are_screen": True,
            "boss_hp": boss.health if boss is not None else 0,
            "boss_status": 1 if boss is not None else 0,
            "boss_slot": boss.slot if boss is not None else -1,
            "pickups": pickups,
            "hazards": hazards,
        },
    )
