"""SMZ3 — Super Metroid + A Link to the Past combined randomizer bots.

Long-term goal: roll a seed and race two bots on it. Near-term: solid seed
tooling, room-timeout game-over, and reuse of vanilla ``alttp`` /
``super_metroid`` primitives once each world is controllable.
"""

from __future__ import annotations

__all__ = [
    "boot",
    "early_route",
    "paths",
    "portals",
    "ram",
    "room_timeout",
    "rom_builder",
    "seed",
    "world",
]
