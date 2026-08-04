"""Declarative RAM reading for SNES games via stable-retro.

Provides typed value readers for numpy RAM arrays, a declarative RAMSchema
for mapping field names to addresses, and a RAMWatcher for detecting
value changes between frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

# -- Typed value readers -----------------------------------------------------

def read_u8(ram: np.ndarray, addr: int) -> int:
    return int(ram[addr])

def read_u16(ram: np.ndarray, addr: int) -> int:
    """Little-endian unsigned 16-bit read."""
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)

def read_u16_be(ram: np.ndarray, addr: int) -> int:
    """Big-endian unsigned 16-bit read."""
    return (int(ram[addr]) << 8) | int(ram[addr + 1])

def read_s8(ram: np.ndarray, addr: int) -> int:
    v = int(ram[addr])
    return v - 256 if v > 127 else v

def read_s16(ram: np.ndarray, addr: int) -> int:
    """Little-endian signed 16-bit read."""
    v = read_u16(ram, addr)
    return v - 65536 if v > 32767 else v

_READERS = {
    "u8": read_u8,
    "u16": read_u16,
    "u16_be": read_u16_be,
    "s8": read_s8,
    "s16": read_s16,
}

# -- RAMSchema ---------------------------------------------------------------

class RAMSchema:
    """Declarative RAM address map.

    Usage::

        schema = RAMSchema({
            "player_x": (0x00D6, "u8"),
            "player_y": (0x00D8, "u8"),
            "level_id": (0x0076, "u16"),
            "health":   (0x04B9, "u16_be"),
        })
        values = schema.read(ram_array)
        # {"player_x": 42, "player_y": 100, "level_id": 233, "health": 161}

    Supported types: ``"u8"``, ``"u16"``, ``"u16_be"``, ``"s8"``, ``"s16"``
    """

    def __init__(self, addresses: dict[str, tuple[int, str]]) -> None:
        for name, (addr, type_str) in addresses.items():
            if type_str not in _READERS:
                raise ValueError(f"Unknown type {type_str!r} for field {name!r}")
        self._addresses = dict(addresses)

    @classmethod
    def from_dict(cls, d: dict[str, tuple[int, str]]) -> RAMSchema:
        """Create a RAMSchema from a plain dict."""
        return cls(d)

    @property
    def fields(self) -> list[str]:
        """Return the list of field names."""
        return list(self._addresses)

    def read(self, ram: np.ndarray) -> dict[str, int]:
        """Read all fields from *ram* and return a name -> value dict."""
        return {
            name: _READERS[type_str](ram, addr)
            for name, (addr, type_str) in self._addresses.items()
        }

    def read_one(self, ram: np.ndarray, field_name: str) -> int:
        """Read a single field by name."""
        addr, type_str = self._addresses[field_name]
        return _READERS[type_str](ram, addr)

# -- RAMWatcher --------------------------------------------------------------

class RAMWatcher:
    """Track RAM value changes between frames.

    Usage::

        watcher = RAMWatcher(schema)
        changes = watcher.update(ram)
        # {"health": (161, 150), "level_id": (1, 2)}  -- (old, new)
    """

    def __init__(self, schema: RAMSchema) -> None:
        self._schema = schema
        self._prev: dict[str, int] | None = None

    def update(self, ram: np.ndarray) -> dict[str, tuple[int, int]]:
        """Return fields that changed since the last call.

        Returns ``{field_name: (old_value, new_value)}`` for every field whose
        value differs.  First call returns an empty dict (no previous state).
        """
        current = self._schema.read(ram)
        if self._prev is None:
            self._prev = current
            return {}
        changes = {
            name: (self._prev[name], current[name])
            for name in current
            if current[name] != self._prev[name]
        }
        self._prev = current
        return changes


# -- Normalized scripted-agent state (from retro_harness.game_state) ------------


class GameMode(Enum):
    """High-level mode inferred from RAM / scene adapters."""

    UNKNOWN = auto()
    BOOT = auto()
    TITLE = auto()
    MENU = auto()
    PLAYING = auto()
    PAUSED = auto()
    CUTSCENE = auto()
    GAME_OVER = auto()
    CONTINUE = auto()
    LEVEL_COMPLETE = auto()
    ENDING = auto()


@dataclass(frozen=True)
class EnemyState:
    """One enemy slot projected into normalized coordinates."""

    slot: int
    x: int
    y: int
    health: int
    active: bool
    animation: int = 0
    kind: int = 0


@dataclass(frozen=True)
class ProjectileState:
    """Projectile / hazard slot."""

    slot: int
    x: int
    y: int
    active: bool
    vx: int = 0
    vy: int = 0


@dataclass(frozen=True)
class GameState:
    """Common state object produced by per-game RAM adapters."""

    frame: int
    mode: GameMode = GameMode.UNKNOWN
    stage: int = 0
    room: int = 0
    player_x: int = 0
    player_y: int = 0
    velocity_x: int = 0
    velocity_y: int = 0
    health: int = 0
    lives: int = 0
    camera_x: int = 0
    camera_y: int = 0
    enemies: tuple[EnemyState, ...] = ()
    projectiles: tuple[ProjectileState, ...] = ()
    boss_active: bool = False
    level_complete: bool = False
    player_dead: bool = False
    screen_locked: bool = False
    grounded: bool = False
    go_flashing: bool = False
    area_clear: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def living_enemies(self) -> tuple[EnemyState, ...]:
        """Active enemies with remaining (non-corpse) health.

        HP ``0`` is a corpse/ghost threat (adapters may normalize HP
        underflow to 0). Living HP may exceed 128 depending on the game.
        """
        return tuple(
            e for e in self.enemies if e.active and e.health > 0
        )

    @property
    def threat_enemies(self) -> tuple[EnemyState, ...]:
        """Active combat threats, including damaging HP0 corpses/ghosts.

        Adapters mark zero-HP chasers ``active=True`` when they still hurt;
        ``living_enemies`` excludes them so clear/wave logic stays HP-based.
        """
        return tuple(e for e in self.enemies if e.active)

    def nearest_enemy(self) -> EnemyState | None:
        """Return the closest living enemy by Manhattan distance, if any."""
        living = self.living_enemies
        if not living:
            return None
        return min(
            living,
            key=lambda e: abs(e.x - self.player_x) + abs(e.y - self.player_y),
        )

    def nearest_threat(self) -> EnemyState | None:
        """Closest active threat (living or damaging HP0), Manhattan distance."""
        threats = self.threat_enemies
        if not threats:
            return None
        return min(
            threats,
            key=lambda e: abs(e.x - self.player_x) + abs(e.y - self.player_y),
        )


# -- Differential RAM discovery (from retro_harness.ram_diff) --------------------


@dataclass(frozen=True)
class RamDelta:
    """One address that changed between two RAM snapshots."""

    address: int
    before: int
    after: int

    @property
    def delta(self) -> int:
        """Signed change after - before."""
        return self.after - self.before


def snapshot(ram: np.ndarray) -> np.ndarray:
    """Copy a RAM buffer for later comparison."""
    return np.array(ram, dtype=np.uint8, copy=True)


def diff_changed(
    before: np.ndarray,
    after: np.ndarray,
    *,
    limit: int | None = 256,
) -> list[RamDelta]:
    """Return addresses whose byte values changed.

    Args:
        before: Earlier RAM snapshot.
        after: Later RAM snapshot.
        limit: Optional max number of deltas to return (sorted by address).
    """
    if before.shape != after.shape:
        raise ValueError("RAM snapshots must have the same shape")
    changed = np.flatnonzero(before != after)
    deltas = [
        RamDelta(address=int(addr), before=int(before[addr]), after=int(after[addr]))
        for addr in changed
    ]
    if limit is not None:
        return deltas[:limit]
    return deltas


def candidates_increasing(deltas: list[RamDelta]) -> list[RamDelta]:
    """Filter deltas that increased (useful for X/score probes)."""
    return [d for d in deltas if d.delta > 0]


def candidates_decreasing(deltas: list[RamDelta]) -> list[RamDelta]:
    """Filter deltas that decreased (useful for health probes)."""
    return [d for d in deltas if d.delta < 0]

