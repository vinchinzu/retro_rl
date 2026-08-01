"""Per-frame / per-phase observation cache for day-plan and task builders.

Heavy tasks re-read the same RAM fields many times per step. ``WorldContext``
batches named reads for one WorldState frame so builders and skills can share
a small hot cache without coupling to WorldSnapshot's full export surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import numpy as np

from harvest.core.ram_catalog import read_ram_value
from retro_harness import WorldState

CacheLoader = Callable[[np.ndarray], Any]


@dataclass
class WorldContext:
    """Observation cache keyed by (frame, ram object id) + named fields."""

    _frame: int = -1
    _ram_token: int = 0
    _cache: dict[str, Any] = field(default_factory=dict)

    def bind(self, world: WorldState) -> "WorldContext":
        """Reset cache when the frame or RAM buffer identity changes."""
        ram_token = id(world.ram)
        frame = int(getattr(world, "frame", -1))
        if frame != self._frame or ram_token != self._ram_token:
            self._frame = frame
            self._ram_token = ram_token
            self._cache.clear()
        return self

    def get(self, key: str, loader: CacheLoader, ram: np.ndarray) -> Any:
        if key not in self._cache:
            self._cache[key] = loader(ram)
        return self._cache[key]

    def invalidate(self, *keys: str) -> None:
        if not keys:
            self._cache.clear()
            return
        for key in keys:
            self._cache.pop(key, None)

    # ── Common harvest reads ──────────────────────────────────────────

    def tilemap(self, ram: np.ndarray) -> int:
        return int(self.get("tilemap", lambda r: read_ram_value(r, "tilemap", raw=True), ram))

    def player_px(self, ram: np.ndarray) -> Tuple[int, int]:
        def _load(r: np.ndarray) -> Tuple[int, int]:
            return (
                int(read_ram_value(r, "player_x", raw=True)),
                int(read_ram_value(r, "player_y", raw=True)),
            )

        return self.get("player_px", _load, ram)

    def day_time(self, ram: np.ndarray) -> Tuple[int, int, int]:
        def _load(r: np.ndarray) -> Tuple[int, int, int]:
            return (
                int(read_ram_value(r, "day", raw=True)),
                int(read_ram_value(r, "hour", raw=True)),
                int(read_ram_value(r, "minute", raw=True)),
            )

        return self.get("day_time", _load, ram)

    def stamina(self, ram: np.ndarray) -> int:
        return int(self.get("stamina", lambda r: read_ram_value(r, "stamina", raw=True), ram))

    def money(self, ram: np.ndarray) -> int:
        return int(self.get("money", lambda r: read_ram_value(r, "money", raw=False), ram))

    def weekday(self, ram: np.ndarray) -> int:
        return int(self.get("weekday", lambda r: read_ram_value(r, "weekday", raw=True), ram))

    def season(self, ram: np.ndarray) -> int:
        return int(self.get("season", lambda r: read_ram_value(r, "season", raw=True), ram))

    def snapshot_dict(self, ram: np.ndarray) -> dict[str, Any]:
        """Small JSON-friendly policy snapshot for builders / advisors."""
        day, hour, minute = self.day_time(ram)
        px, py = self.player_px(ram)
        return {
            "tilemap": self.tilemap(ram),
            "player_x": px,
            "player_y": py,
            "day": day,
            "hour": hour,
            "minute": minute,
            "weekday": self.weekday(ram),
            "season": self.season(ram),
            "stamina": self.stamina(ram),
            "money": self.money(ram),
        }


def world_context_from_world(
    world: WorldState,
    existing: Optional[WorldContext] = None,
) -> WorldContext:
    ctx = existing if existing is not None else WorldContext()
    return ctx.bind(world)


__all__ = ["WorldContext", "world_context_from_world"]
