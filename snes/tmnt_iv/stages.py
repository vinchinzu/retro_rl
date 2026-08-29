"""Stage-byte predicates shared by TMNT IV tactics (no policy import)."""

from __future__ import annotations

from retro_harness.ram_state import GameState

SEWER = 2
PREHISTORIC = 4
WOUNDED_KNEE = 6
NEON_HIGHWAY = 7
STARBASE_WAVES = 8
STARBASE_SHREDDER = 9
STARBASE = frozenset({STARBASE_WAVES, STARBASE_SHREDDER})

RAPH_CHAR = 8
# Bottom-ish water lane preference between Sewer waves.
SEWER_SAFE_WALK_Y = 190
# Stage 7 stacked / elevated Foot (bazooka stack top is char 0xb0).
WOUNDED_KNEE_JUMP_CHARS: frozenset[int] = frozenset({0xB0})
# Duo bosses: Tokka/Rahzar and Bebop/Rocksteady. Adds must not steal targeting.
DUO_BOSS_CHARS: frozenset[int] = frozenset({0x48, 0xA0, 0xA8, 0xAC})
# Raphael jump-locks beside these Starbase bruisers — skip B+Y overlay.
RAPH_STARBASE_GROUND_CHARS: frozenset[int] = frozenset({0xB2, 0xB4})
# Hover / stack tops the Starbase jump-kick must close.
RAPH_STARBASE_CLOSE_CHARS: frozenset[int] = frozenset({0x6A, 0xB0, 0xBA})
# Mode-7: enemies approach in depth (rising Y). Player Y clamps ~160–213;
# fight the near band (y >= this) or Krang. Shared by NeonLane + fight().
NEON_MIN_FIGHT_Y = 140
KRANG_CHAR = 0x4E


def is_sewer(state: GameState) -> bool:
    """True on Stage 3 Sewer Surfin' (stage byte 2)."""
    return state.stage == SEWER


def is_prehistoric(state: GameState) -> bool:
    """True on Stage 5 Prehistoric (stage byte 4)."""
    return state.stage == PREHISTORIC


def is_wounded_knee(state: GameState) -> bool:
    """True on Stage 7 Bury My Shell at Wounded Knee (byte 6)."""
    return state.stage == WOUNDED_KNEE


def is_neon_highway(state: GameState) -> bool:
    """True on Stage 8 Neon Night Riders Mode-7 highway (byte 7)."""
    return state.stage == NEON_HIGHWAY


def is_starbase(state: GameState) -> bool:
    """True on Stage 9 Starbase waves / Super Shredder (bytes 8–9)."""
    return state.stage in STARBASE
