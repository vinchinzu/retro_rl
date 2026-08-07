"""Contract-guarded survival assist for Zelda I first-pass routing.

See ``docs/ASSIST_CONTRACT.md``. Default runners stay Clean; enable only via
``--infinite-life`` / explicit ``UnlimitedHealthAssist(enabled=True)``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from zelda_i.ram import (
    ADDR_HEALTH,
    PLAY_MODE,
    ZeldaSnapshot,
    full_health_byte,
    read_snapshot,
)


class _RetroData(Protocol):
    def set_value(self, key: str, value: int) -> None: ...


@dataclass
class ResourceCounter:
    restored: int = 0
    writes: int = 0
    first_active_frame: int | None = None


@dataclass
class AssistTelemetry:
    health: ResourceCounter = field(default_factory=ResourceCounter)
    suspended_phase_frames: Counter[str] = field(default_factory=Counter)
    maximum_single_frame_damage: int = 0
    deaths: int = 0
    progression_writes: int = 0
    capacity_writes: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "health": asdict(self.health),
            "suspended_phase_frames": dict(self.suspended_phase_frames),
            "maximum_single_frame_damage": self.maximum_single_frame_damage,
            "deaths": self.deaths,
            "progression_writes": self.progression_writes,
            "capacity_writes": self.capacity_writes,
        }


def assist_phase_name(snap: ZeldaSnapshot) -> str:
    """Coarse phase for assist guards (mirrors SM GameplayPhase idea)."""
    if snap.mode == 17:
        return "death"
    if snap.mode == 18:
        return "triforce_fanfare"
    if snap.mode in (0, 1, 2, 3, 4):
        return "menu_or_boot"
    if snap.transitioning:
        return "transition"
    if snap.mode == PLAY_MODE or snap.in_cave:
        return "ordinary_gameplay"
    return f"mode_{snap.mode}"


class UnlimitedHealthAssist:
    """Refill filled hearts to the natural container max; never grant containers.

    Writes only ``health`` (``ADDR_HEALTH`` / data.json key) under the contract.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.telemetry = AssistTelemetry()
        self._prev_filled: int | None = None
        self._prev_phase: str | None = None

    def report(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "class": "survival",
            "kind": "unlimited_health",
            **self.telemetry.to_dict(),
        }

    def apply_snapshot(
        self,
        data: _RetroData,
        snap: ZeldaSnapshot,
        *,
        frame: int = 0,
    ) -> ZeldaSnapshot | None:
        """Apply assist from a snapshot. Returns None if no write happened."""
        if not self.enabled:
            return None

        phase = assist_phase_name(snap)
        if phase == "death" and self._prev_phase != "death":
            self.telemetry.deaths += 1
        self._prev_phase = phase

        if phase != "ordinary_gameplay":
            self.telemetry.suspended_phase_frames[phase] += 1
            self._prev_filled = None
            return None

        filled = snap.filled_hearts
        if self._prev_filled is not None:
            damage = max(0, self._prev_filled - filled)
            self.telemetry.maximum_single_frame_damage = max(
                self.telemetry.maximum_single_frame_damage,
                damage,
            )

        target = full_health_byte(snap.health)
        if snap.health == target or snap.heart_containers <= 0:
            self._prev_filled = filled
            return None

        counter = self.telemetry.health
        if counter.first_active_frame is None:
            counter.first_active_frame = frame
        restored = max(0, (target & 0x0F) - (snap.health & 0x0F))
        data.set_value("health", target)
        counter.restored += restored
        counter.writes += 1
        self._prev_filled = target & 0x0F
        return None

    def apply_env(self, env: Any, *, frame: int = 0) -> None:
        """Read RAM from ``env``, apply, leave env mutated when writing."""
        if not self.enabled:
            return
        snap = read_snapshot(env.get_ram())
        self.apply_snapshot(env.data, snap, frame=frame)


def write_health_u8(env: Any, value: int) -> None:
    """Low-level health write (tests / diagnostics). Prefer the assist class."""
    env.data.set_value("health", int(value) & 0xFF)


__all__ = [
    "AssistTelemetry",
    "ResourceCounter",
    "UnlimitedHealthAssist",
    "assist_phase_name",
    "write_health_u8",
    "ADDR_HEALTH",
]
