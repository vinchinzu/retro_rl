"""Dungeon boss phase machines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FightResult:
    """One ``fight(env)``. ``stage_replay`` reads ``ok``, ``frames``, and ``blocker``."""

    ok: bool
    frames: int
    blocker: str = ""
    strikes: int = 0
    floor_hits: int = 0
    fire_hits: int = 0
    sword_hits: int = 0
