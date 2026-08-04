"""Contract-guarded unlimited-ammo controller with auditable telemetry."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Protocol

from super_metroid.ram import GameplayPhase, SuperMetroidState


class _RetroData(Protocol):
    def set_value(self, key: str, value: int) -> None: ...


@dataclass
class ResourceCounter:
    restored: int = 0
    writes: int = 0
    first_unlocked_frame: int | None = None


@dataclass
class AssistTelemetry:
    energy: ResourceCounter = field(default_factory=ResourceCounter)
    ammo: dict[str, ResourceCounter] = field(
        default_factory=lambda: {
            "missiles": ResourceCounter(),
            "super_missiles": ResourceCounter(),
            "power_bombs": ResourceCounter(),
        }
    )
    suspended_phase_frames: Counter[str] = field(default_factory=Counter)
    progression_writes: int = 0
    capacity_writes: int = 0
    maximum_single_frame_damage: int = 0
    deaths: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "energy": asdict(self.energy),
            "ammo": {name: asdict(counter) for name, counter in self.ammo.items()},
            "suspended_phase_frames": dict(self.suspended_phase_frames),
            "progression_writes": self.progression_writes,
            "capacity_writes": self.capacity_writes,
            "maximum_single_frame_damage": self.maximum_single_frame_damage,
            "deaths": self.deaths,
        }


class UnlimitedAmmoAssist:
    """Refill naturally unlocked ammo, never capacity or progression."""

    _FIELDS = (
        ("missiles", "max_missiles"),
        ("super_missiles", "max_super_missiles"),
        ("power_bombs", "max_power_bombs"),
    )

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.telemetry = AssistTelemetry()

    def report(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            **self.telemetry.to_dict(),
        }

    def apply(self, data: _RetroData, state: SuperMetroidState) -> None:
        if not self.enabled:
            return

        if state.phase is not GameplayPhase.ORDINARY_GAMEPLAY:
            self.telemetry.suspended_phase_frames[state.phase.value] += 1
            return

        for current_name, capacity_name in self._FIELDS:
            current = int(getattr(state, current_name))
            capacity = int(getattr(state, capacity_name))
            counter = self.telemetry.ammo[current_name]
            if capacity > 0 and counter.first_unlocked_frame is None:
                counter.first_unlocked_frame = state.frame
            if capacity <= 0 or current >= capacity:
                continue
            data.set_value(current_name, capacity)
            counter.restored += capacity - current
            counter.writes += 1


class UnlimitedResourcesAssist(UnlimitedAmmoAssist):
    """Restore current energy and naturally unlocked ammo under one guard."""

    def __init__(
        self,
        *,
        unlimited_energy: bool = True,
        unlimited_ammo: bool = True,
    ) -> None:
        super().__init__(enabled=unlimited_ammo)
        self.unlimited_energy = unlimited_energy
        self._effective_health: int | None = None
        self._previous_phase: GameplayPhase | None = None

    def report(self) -> dict[str, object]:
        return {
            "enabled": self.enabled or self.unlimited_energy,
            "unlimited_energy_enabled": self.unlimited_energy,
            "unlimited_ammo_enabled": self.enabled,
            **self.telemetry.to_dict(),
        }

    def apply(self, data: _RetroData, state: SuperMetroidState) -> None:
        if (
            state.phase is GameplayPhase.DEATH_OR_GAME_OVER
            and self._previous_phase is not GameplayPhase.DEATH_OR_GAME_OVER
        ):
            self.telemetry.deaths += 1
        self._previous_phase = state.phase

        energy_allowed = (
            state.phase is GameplayPhase.ORDINARY_GAMEPLAY
            and state.area_name != "Ceres"
        )
        if energy_allowed:
            energy = self.telemetry.energy
            if state.max_health > 0 and energy.first_unlocked_frame is None:
                energy.first_unlocked_frame = state.frame
            if self._effective_health is not None:
                damage = max(0, self._effective_health - state.health)
                self.telemetry.maximum_single_frame_damage = max(
                    self.telemetry.maximum_single_frame_damage,
                    damage,
                )
            if (
                self.unlimited_energy
                and 0 < state.health < state.max_health
            ):
                data.set_value("health", state.max_health)
                energy.restored += state.max_health - state.health
                energy.writes += 1
                self._effective_health = state.max_health
            else:
                self._effective_health = state.health
        else:
            self._effective_health = None
            if (
                self.unlimited_energy
                and state.phase is GameplayPhase.ORDINARY_GAMEPLAY
                and state.area_name == "Ceres"
            ):
                self.telemetry.suspended_phase_frames["energy:ceres"] += 1
            if self.unlimited_energy and not self.enabled:
                self.telemetry.suspended_phase_frames[state.phase.value] += 1

        super().apply(data, state)
