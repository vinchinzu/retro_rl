"""Shared HP-delta and policy-tick unwrap used by every TMNT IV probe loop.

Runners import these instead of copying ``0 < health <= 0x60`` and
``tick.action.action if tick.action else idle``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retro_harness.actions import idle_action
from retro_harness.ram_state import GameState
from tmnt_iv.policy import Stage1Policy

# Foot/Leo bar is 0x00–0x60; 0x61+ is a despawn/sentinel during loads.
_LIVE_HP_MAX = 0x60


def living_hp(health: int) -> bool:
    """True when HP is a real in-combat bar (not 0, not a load sentinel)."""
    return 0 < health <= _LIVE_HP_MAX


@dataclass
class HpDelta:
    """Accumulate natural HP drops across frames.

    ``count_zero=True`` also counts the frame that hits 0 (full-run / capture).
    Probe loops that ignore the KO frame leave it false (Clean / grind).
    """

    prev: int | None = None
    damage: int = 0
    max_hit: int = 0
    min_hp: int | None = None
    count_zero: bool = False

    @classmethod
    def start(cls, health: int, *, count_zero: bool = False) -> HpDelta:
        """Seed from the first parsed state."""
        prev = health if living_hp(health) else None
        return cls(prev=prev, min_hp=prev, count_zero=count_zero)

    def note(self, health: int) -> int:
        """Apply one frame of HP. Return the drop this frame (0 if none)."""
        hit = 0
        countable = living_hp(health) or (self.count_zero and health == 0)
        if countable and self.prev is not None and health < self.prev:
            hit = self.prev - max(0, health)
            self.damage += hit
            self.max_hit = max(self.max_hit, hit)
        if living_hp(health):
            self.prev = health
            if self.min_hp is None or health < self.min_hp:
                self.min_hp = health
        elif self.count_zero and health == 0:
            self.prev = 0
        return hit


def policy_input(policy: Stage1Policy, state: GameState) -> tuple[Any, str]:
    """Unwrap ``policy.tick`` to an emulator action + reason (idle if none)."""
    tick = policy.tick(state)
    if tick.action is not None:
        return tick.action.action, tick.action.reason
    return idle_action(), tick.reason or "idle"
