"""Normalized game state shared across oneshot SNES agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


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
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def living_enemies(self) -> tuple[EnemyState, ...]:
        """Active enemies with remaining (non-corpse) health.

        HP ``0`` is a corpse/ghost threat (adapters also normalize true
        underflow bytes to 0). Living HP may exceed 128 — Final Fight
        subway tough thugs peak around 148.
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
