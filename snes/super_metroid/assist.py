"""Contract-guarded unlimited-ammo controller with auditable telemetry."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol

from super_metroid.ram import GameplayPhase, SuperMetroidState

# always: restore whenever current < capacity (product continuous default).
# at_zero: practice handicap — ammo tops up only at 0; energy tops up at 0 **or**
# when current is at/below a low floor (death-save before the death phase latches).
# One-hit Phantoon flame can skip the ordinary health==0 frame (gs→death same
# tick), so a pure-zero energy policy never fires. Death phase still never
# revives a completed transition.
RefillWhen = Literal["always", "at_zero"]

# Energy floor for at_zero practice (inclusive). Ammo still waits for exact 0.
# 40 covers GT / acid 40-damage chips before death phase steals the frame.
AT_ZERO_ENERGY_FLOOR = 40


class _RetroData(Protocol):
    def set_value(self, key: str, value: int) -> None: ...


@dataclass
class ResourceCounter:
    restored: int = 0
    writes: int = 0
    # Discrete empty→full restores (skill metric under refill_when=at_zero).
    top_ups: int = 0
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
            "top_ups_total": self.top_ups_total(),
        }

    def top_ups_total(self) -> int:
        n = int(self.energy.top_ups)
        for counter in self.ammo.values():
            n += int(counter.top_ups)
        return n


class UnlimitedAmmoAssist:
    """Refill naturally unlocked ammo, never capacity or progression."""

    _FIELDS = (
        ("missiles", "max_missiles"),
        ("super_missiles", "max_super_missiles"),
        ("power_bombs", "max_power_bombs"),
    )

    def __init__(
        self,
        *,
        enabled: bool = True,
        refill_when: RefillWhen = "always",
    ) -> None:
        self.enabled = enabled
        self.refill_when: RefillWhen = refill_when
        self.telemetry = AssistTelemetry()

    def report(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "refill_when": self.refill_when,
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
            if self.refill_when == "at_zero" and current > 0:
                continue
            data.set_value(current_name, capacity)
            counter.restored += capacity - current
            counter.writes += 1
            if current == 0:
                counter.top_ups += 1


# Energy-drain scripted sequences (refill softlocks progression).
# Big Boy latch: pose $E8 (232), mov $15/$1B (21/27); early stick via $7FFF HP.
# Mother Brain rainbow: pose $54 (84) + mov $0A (10); stun poses $E9/$EB (233/235).
# Movement $0A is also ordinary knockback outside Tourian — do not match it
# globally or LN/GT hits suspend energy for the 180f hold after every chip.
_ENERGY_DRAIN_POSES = frozenset({84, 232, 233, 235})
_ENERGY_DRAIN_MOVEMENT = frozenset({10, 21, 27})
_BABY_METROID_HP = 0x7FFF
_LATCH_PROXIMITY_PX = 48
_TOURIAN_AREA = 5


def _metroid_latched(state: SuperMetroidState) -> bool:
    """True while energy must fall (Metroid latch or MB rainbow drain)."""
    # Attached baby Metroid: invincible HP band and glued to Samus.
    if int(state.enemy0_hp) == _BABY_METROID_HP:
        dx = abs(int(state.enemy0_x) - int(state.samus_x))
        dy = abs(int(state.enemy0_y) - int(state.samus_y))
        if dx <= _LATCH_PROXIMITY_PX and dy <= _LATCH_PROXIMITY_PX:
            return True
    if int(state.area_index) != _TOURIAN_AREA:
        return False
    return (
        int(state.pose) in _ENERGY_DRAIN_POSES
        or int(state.movement_type) in _ENERGY_DRAIN_MOVEMENT
    )

# Hold energy suspend across 1–2f pose gaps (e.g. MB rainbow → stun pose 42).
_ENERGY_DRAIN_HOLD_FRAMES = 180


class UnlimitedResourcesAssist(UnlimitedAmmoAssist):
    """Restore current energy and naturally unlocked ammo under one guard."""

    def __init__(
        self,
        *,
        unlimited_energy: bool = True,
        unlimited_ammo: bool = True,
        refill_when: RefillWhen = "always",
    ) -> None:
        super().__init__(enabled=unlimited_ammo, refill_when=refill_when)
        self.unlimited_energy = unlimited_energy
        self._effective_health: int | None = None
        self._previous_phase: GameplayPhase | None = None
        self._energy_drain_hold = 0

    def report(self) -> dict[str, object]:
        return {
            "enabled": self.enabled or self.unlimited_energy,
            "unlimited_energy_enabled": self.unlimited_energy,
            "unlimited_ammo_enabled": self.enabled,
            "refill_when": self.refill_when,
            **self.telemetry.to_dict(),
        }

    def apply(self, data: _RetroData, state: SuperMetroidState) -> None:
        if (
            state.phase is GameplayPhase.DEATH_OR_GAME_OVER
            and self._previous_phase is not GameplayPhase.DEATH_OR_GAME_OVER
        ):
            self.telemetry.deaths += 1
        self._previous_phase = state.phase

        latched = _metroid_latched(state)
        if latched:
            self._energy_drain_hold = _ENERGY_DRAIN_HOLD_FRAMES
        elif self._energy_drain_hold > 0:
            self._energy_drain_hold -= 1
        drain_suspend = latched or self._energy_drain_hold > 0

        energy_allowed = (
            state.phase is GameplayPhase.ORDINARY_GAMEPLAY
            and state.area_name != "Ceres"
            and not drain_suspend
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
            if self.unlimited_energy and state.max_health > 0:
                health = int(state.health)
                max_h = int(state.max_health)
                should_refill = False
                if self.refill_when == "at_zero":
                    # Death-save before death phase: pure health==0 often never
                    # appears in ordinary gameplay (boss one-shots latch gs=death
                    # same tick). Floor catches low tanks; 0 still counts.
                    should_refill = health <= AT_ZERO_ENERGY_FLOOR
                else:
                    # Product continuous: keep current full without reviving death.
                    should_refill = 0 < health < max_h
                if should_refill:
                    data.set_value("health", max_h)
                    energy.restored += max_h - health
                    energy.writes += 1
                    if self.refill_when == "at_zero":
                        # Discrete practice top-up (empty or floor trip → full).
                        energy.top_ups += 1
                    self._effective_health = max_h
                else:
                    self._effective_health = health
            else:
                self._effective_health = state.health
        else:
            self._effective_health = None
            if self.unlimited_energy and drain_suspend:
                self.telemetry.suspended_phase_frames["energy:metroid_latch"] += 1
            elif (
                self.unlimited_energy
                and state.phase is GameplayPhase.ORDINARY_GAMEPLAY
                and state.area_name == "Ceres"
            ):
                self.telemetry.suspended_phase_frames["energy:ceres"] += 1
            if self.unlimited_energy and not self.enabled and not drain_suspend:
                self.telemetry.suspended_phase_frames[state.phase.value] += 1

        super().apply(data, state)
