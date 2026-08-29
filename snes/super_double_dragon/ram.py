"""Super Double Dragon WRAM layout and normalized state adapter.

The game uses page-sized actor records.  Actor allocation is mostly stable in
Mission 1, but the enemy page changes as waves recycle.  Coordinates used for
combat are the rendered screen X at ``+0x74`` and logical lane Y at ``+0x10``.
The latter increases when UP is pressed, so it is inverted here to match the
screen-coordinate convention used by ``retro_harness``.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from retro_harness.ram_state import EnemyState, GameMode, GameState

ADDR_STAGE = 0x001C
ADDR_SCENE_LOCK = 0x0018
ADDR_SCENE_SUB = 0x0019
ADDR_CREDITS = 0x00D9
ADDR_LIVES = 0x00DC
ADDR_FLOOR = 0x00DE
# High byte of the current P1 actor page (for example 0x12 -> 0x1200).
# It is 0xFF while no player actor exists during an area transition.
ADDR_PLAYER_PAGE = 0x1CF9

# Pages 0x06-0x17 are fighters.  0x17 holds the leftover gym fighter on
# Area19_Clear and a third 0x1A enemy the 0x06-0x16 window missed.
ACTOR_BASES: tuple[int, ...] = tuple(page << 8 for page in range(0x06, 0x18))
# Mission 1 area 0x10 starts here, but the actor allocator moves P1 between
# pages at area transitions. ``PLAYER_KIND`` is only a fallback for tests and
# transition frames; ``ADDR_PLAYER_PAGE`` is authoritative during play.
PLAYER_BASE = 0x1200
PLAYER_KIND = 0x09

OFF_STATUS = 0x00
OFF_KIND = 0x02
OFF_WORLD_X = 0x0C
OFF_Y = 0x10
OFF_HP = 0x27
OFF_SCREEN_Y = 0x72
OFF_SCREEN_X = 0x74

ACTOR_STATUS_DOWN = 0x02
ACTOR_STATUS_DRAWN = 0x03
UNUSED_KIND = 0xFF
ENTITY_HP_MAX = 200


class MissionId(IntEnum):
    """Observed values at ``ADDR_STAGE`` for the seven missions."""

    MISSION_1 = 0x10
    MISSION_2 = 0x14
    MISSION_3 = 0x17
    MISSION_4 = 0x1C
    MISSION_5 = 0x1D
    MISSION_6 = 0x1F
    MISSION_7 = 0x20


MIN_GAMEPLAY_AREA = 0x10
MAX_GAMEPLAY_AREA = 0x20


def read_u8(ram: np.ndarray, address: int) -> int:
    """Read one unsigned WRAM byte."""
    return int(ram[address])


def read_u16le(ram: np.ndarray, address: int) -> int:
    """Read a little-endian WRAM word."""
    return read_u8(ram, address) | (read_u8(ram, address + 1) << 8)


def write_u16le(ram: np.ndarray, address: int, value: int) -> None:
    """Write a little-endian word (used by pure parser tests)."""
    ram[address] = value & 0xFF
    ram[address + 1] = (value >> 8) & 0xFF


def _player_mode(stage: int, status: int, health: int) -> GameMode:
    if not MIN_GAMEPLAY_AREA <= stage <= MAX_GAMEPLAY_AREA:
        return GameMode.MENU
    if status == ACTOR_STATUS_DRAWN and 0 < health <= ENTITY_HP_MAX:
        return GameMode.PLAYING
    if health == 0 and status in {ACTOR_STATUS_DOWN, ACTOR_STATUS_DRAWN}:
        return GameMode.CONTINUE
    return GameMode.CUTSCENE


def mission_number(area: int) -> int:
    """Map the internal area byte onto the manual's Mission 1–7 labels."""
    if area < MissionId.MISSION_2:
        return 1
    if area < MissionId.MISSION_3:
        return 2
    if area < MissionId.MISSION_4:
        return 3
    if area < MissionId.MISSION_5:
        return 4
    if area < MissionId.MISSION_6:
        return 5
    if area < MissionId.MISSION_7:
        return 6
    return 7


def find_player_base(ram: np.ndarray) -> int:
    """Return the current P1 actor page.

    Actor kind is not a stable discriminator: Billy is kind ``0x09`` in the
    street areas but kind ``0x05`` on the Mission 1 elevator.  The engine
    exposes the assigned page at ``ADDR_PLAYER_PAGE``; kind scanning remains
    only as a fallback for synthetic tests and transition frames.
    """
    pointed_base = read_u8(ram, ADDR_PLAYER_PAGE) << 8
    if pointed_base in ACTOR_BASES:
        return pointed_base
    candidates = [
        base
        for base in ACTOR_BASES
        if read_u8(ram, base + OFF_KIND) == PLAYER_KIND
    ]
    for wanted_status in (ACTOR_STATUS_DRAWN, ACTOR_STATUS_DOWN):
        for base in candidates:
            if read_u8(ram, base + OFF_STATUS) == wanted_status:
                return base
    return candidates[0] if candidates else PLAYER_BASE


def _read_enemy(ram: np.ndarray, base: int, *, slot: int) -> EnemyState:
    status = read_u8(ram, base + OFF_STATUS)
    kind = read_u8(ram, base + OFF_KIND)
    raw_hp = read_u8(ram, base + OFF_HP)
    # HP0 fighters remain visible, dangerous, and hittable until their final
    # knockdown.  Give them normalized HP1 so shared living-enemy logic keeps
    # targeting them; preserve raw_hp in animation-independent diagnostics.
    active = status == ACTOR_STATUS_DRAWN and kind != UNUSED_KIND
    health = raw_hp if 0 < raw_hp <= ENTITY_HP_MAX else (1 if active else 0)
    return EnemyState(
        slot=slot,
        x=read_u8(ram, base + OFF_SCREEN_X),
        y=255 - read_u8(ram, base + OFF_Y),
        health=health,
        active=active,
        animation=status,
        kind=kind,
    )


def parse_game_state(ram: np.ndarray, frame: int = 0) -> GameState:
    """Project the actor pool and global bytes into ``GameState``."""
    stage = read_u8(ram, ADDR_STAGE)
    player_base = find_player_base(ram)
    enemy_bases = tuple(base for base in ACTOR_BASES if base != player_base)
    status = read_u8(ram, player_base + OFF_STATUS)
    health = read_u8(ram, player_base + OFF_HP)
    lives = read_u8(ram, ADDR_LIVES)
    enemies = tuple(
        _read_enemy(ram, base, slot=index)
        for index, base in enumerate(enemy_bases)
    )
    living = tuple(enemy for enemy in enemies if enemy.active)
    boss = max(living, key=lambda enemy: enemy.health, default=None)
    drawn = [
        base
        for base in enemy_bases
        if read_u8(ram, base + OFF_STATUS) == ACTOR_STATUS_DRAWN
        and read_u8(ram, base + OFF_KIND) != UNUSED_KIND
    ]
    return GameState(
        frame=frame,
        mode=_player_mode(stage, status, health),
        stage=stage,
        player_x=read_u8(ram, player_base + OFF_SCREEN_X),
        player_y=255 - read_u8(ram, player_base + OFF_Y),
        health=health,
        lives=lives,
        camera_x=0,
        enemies=enemies,
        boss_active=bool(boss is not None and boss.health >= 80),
        player_dead=health == 0,
        screen_locked=bool(living),
        extras={
            "credits": read_u8(ram, ADDR_CREDITS),
            "mission": mission_number(stage),
            "player_base": player_base,
            "player_status": status,
            "player_kind": read_u8(ram, player_base + OFF_KIND),
            "player_world_x": read_u16le(ram, player_base + OFF_WORLD_X),
            "player_screen_y": read_u8(ram, player_base + OFF_SCREEN_Y),
            "scene_lock": read_u8(ram, ADDR_SCENE_LOCK),
            "scene_sub": read_u8(ram, ADDR_SCENE_SUB),
            "floor": read_u8(ram, ADDR_FLOOR),
            "active_actor_bases": drawn,
            "drawn_actors": [
                {
                    "base": base,
                    "kind": read_u8(ram, base + OFF_KIND),
                    "world_x": read_u16le(ram, base + OFF_WORLD_X),
                    "y": read_u8(ram, base + OFF_Y),
                    "hp": read_u8(ram, base + OFF_HP),
                    "screen_x": read_u8(ram, base + OFF_SCREEN_X),
                }
                for base in drawn
            ],
            "raw_enemy_hp": {
                f"0x{base:04X}": read_u8(ram, base + OFF_HP) for base in drawn
            },
        },
    )
