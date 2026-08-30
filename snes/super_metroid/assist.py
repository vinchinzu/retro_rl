"""Contract-guarded unlimited-ammo controller with auditable telemetry."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from super_metroid.combat.enemies.scan import ENEMY_BASE, ENEMY_STRIDE, enemies_from_ram
from super_metroid.combat.enemies.species import ATOMIC_ID
from super_metroid.ram import (
    GameplayPhase,
    SuperMetroidState,
    parse_env_state,
    write_wram_u16,
)

# always: restore whenever current < capacity (product continuous default).
# at_zero: practice handicap — ammo tops up only at 0; energy tops up at 0 **or**
# when current is at/below a low floor (death-save before the death phase latches).
# One-hit Phantoon flame can skip the ordinary health==0 frame (gs→death same
# tick), so a pure-zero energy policy never fires. Death phase still never
# revives a completed transition.
RefillWhen = Literal["always", "at_zero"]
AssistProfile = Literal["clean", "survival", "scaffold"]

# Energy floor for at_zero practice (inclusive). Ammo still waits for exact 0.
# 40 covers GT / acid 40-damage chips before death phase steals the frame.
AT_ZERO_ENERGY_FLOOR = 40


class _RetroData(Protocol):
    def set_value(self, key: str, value: int) -> None: ...


@dataclass(frozen=True)
class ScaffoldAllowlistEntry:
    """One eligible (room, species) clamp target. Unknown pairs are never written."""

    room_id: int
    enemy_id: int
    spawn_state: int | None = None
    phase: int | None = None


@dataclass(frozen=True)
class HpClampWrite:
    frame: int
    room_id: int
    slot: int
    enemy_id: int
    old: int
    new: int
    reason: str


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
    hp_clamp_writes: list[HpClampWrite] = field(default_factory=list)
    hp_clamp_counts_by_room: Counter[str] = field(default_factory=Counter)
    hp_clamp_counts_by_entity: Counter[str] = field(default_factory=Counter)

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
            "hp_clamp_writes": [asdict(row) for row in self.hp_clamp_writes],
            "hp_clamp_counts_by_room": dict(self.hp_clamp_counts_by_room),
            "hp_clamp_counts_by_entity": dict(self.hp_clamp_counts_by_entity),
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


# Attic (Wrecked Ship) gray-door kill-all is the first ordinary-enemy pilot.
# Live species is not ROM-verified here; unknown ids in this room fail closed.
ATTIC_ROOM_ID = 0xCA52
ATTIC_PILOT_ENEMY_ID = ATOMIC_ID
_CLAMP_HP = 1
_OFF_ENEMY_HP = 0x14
_OFF_ENEMY_SPAWN = 0x1A  # instruction list
_OFF_ENEMY_PHASE = 0x30  # AI var 0


def attic_ordinary_enemy_allowlist() -> tuple[ScaffoldAllowlistEntry, ...]:
    """Development-only Attic pilot. Placeholder species; fail closed if unseen."""
    return (
        ScaffoldAllowlistEntry(
            room_id=ATTIC_ROOM_ID,
            enemy_id=ATTIC_PILOT_ENEMY_ID,
        ),
    )


def _resolve_ram(source: object) -> Any | None:
    """env.get_ram(), a ``.ram`` buffer, or an indexable WRAM snapshot."""
    get_ram = getattr(source, "get_ram", None)
    if callable(get_ram):
        try:
            ram = get_ram()
        except Exception:  # noqa: BLE001
            ram = None
        if ram is not None:
            return ram
    ram = getattr(source, "ram", None)
    if ram is not None:
        return ram
    try:
        _ = source[0]  # type: ignore[index]
    except Exception:  # noqa: BLE001
        return None
    return source


def _slot_u16(ram: Any, slot: int, offset: int) -> int | None:
    addr = ENEMY_BASE + slot * ENEMY_STRIDE + offset
    try:
        return int(ram[addr]) | (int(ram[addr + 1]) << 8)
    except Exception:  # noqa: BLE001
        return None


def _write_slot_hp(source: object, ram: Any, slot: int, hp: int) -> bool:
    addr = ENEMY_BASE + slot * ENEMY_STRIDE + _OFF_ENEMY_HP
    data = getattr(source, "data", source)
    assign = getattr(getattr(data, "memory", None), "assign", None)
    if callable(assign):
        try:
            write_wram_u16(source, addr, hp)
            return True
        except Exception:  # noqa: BLE001
            pass
    try:
        ram[addr] = hp & 0xFF
        ram[addr + 1] = (hp >> 8) & 0xFF
        return True
    except Exception:  # noqa: BLE001
        return False


def _index_allowlist(
    entries: Sequence[ScaffoldAllowlistEntry],
) -> dict[tuple[int, int], tuple[ScaffoldAllowlistEntry, ...]]:
    grouped: dict[tuple[int, int], list[ScaffoldAllowlistEntry]] = {}
    for entry in entries:
        grouped.setdefault((int(entry.room_id), int(entry.enemy_id)), []).append(entry)
    return {key: tuple(rows) for key, rows in grouped.items()}


class ScaffoldHpClamp:
    """Allowlisted live-enemy HP→1 poke. Development-only; never STATUS/Finish."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        allowlist: Sequence[ScaffoldAllowlistEntry] = (),
        telemetry: AssistTelemetry | None = None,
    ) -> None:
        self.enabled = enabled
        self.allowlist = tuple(allowlist)
        self.telemetry = telemetry if telemetry is not None else AssistTelemetry()
        self._index = _index_allowlist(self.allowlist)
        self._clamped: set[tuple[int, int, int, int]] = set()

    def report(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "allowlist": [asdict(entry) for entry in self.allowlist],
            "writes": [asdict(row) for row in self.telemetry.hp_clamp_writes],
            "counts_by_room": dict(self.telemetry.hp_clamp_counts_by_room),
            "counts_by_entity": dict(self.telemetry.hp_clamp_counts_by_entity),
        }

    def _match(
        self,
        room_id: int,
        enemy_id: int,
        spawn_state: int | None,
        phase: int | None,
    ) -> ScaffoldAllowlistEntry | None:
        rows = self._index.get((int(room_id), int(enemy_id)))
        if not rows:
            return None
        for row in rows:
            if row.spawn_state is not None and (
                spawn_state is None or int(row.spawn_state) != int(spawn_state)
            ):
                continue
            if row.phase is not None and (
                phase is None or int(row.phase) != int(phase)
            ):
                continue
            return row
        return None

    def _forget_absent(self, room_id: int, live: set[tuple[int, int]]) -> None:
        stale = [
            key
            for key in self._clamped
            if key[0] == int(room_id) and (key[1], key[2]) not in live
        ]
        for key in stale:
            self._clamped.discard(key)

    def apply(self, source: object, state: SuperMetroidState) -> None:
        if not self.enabled:
            return
        if state.phase is not GameplayPhase.ORDINARY_GAMEPLAY:
            self.telemetry.suspended_phase_frames[f"hp_clamp:{state.phase.value}"] += 1
            return
        ram = _resolve_ram(source)
        if ram is None:
            return
        enemies = enemies_from_ram(ram)
        live = {(int(enemy.slot), int(enemy.enemy_id)) for enemy in enemies}
        self._forget_absent(int(state.room_id), live)
        if not self._index:
            return
        for enemy in enemies:
            spawn_state = _slot_u16(ram, enemy.slot, _OFF_ENEMY_SPAWN)
            phase = _slot_u16(ram, enemy.slot, _OFF_ENEMY_PHASE)
            row = self._match(int(state.room_id), int(enemy.enemy_id), spawn_state, phase)
            if row is None:
                continue
            phase_key = int(row.phase) if row.phase is not None else 0
            key = (int(state.room_id), int(enemy.slot), int(enemy.enemy_id), phase_key)
            hp = int(enemy.hp)
            if hp <= _CLAMP_HP:
                if hp == _CLAMP_HP:
                    self._clamped.add(key)
                continue
            if key in self._clamped:
                continue
            if not _write_slot_hp(source, ram, enemy.slot, _CLAMP_HP):
                continue
            self._clamped.add(key)
            write = HpClampWrite(
                frame=int(state.frame),
                room_id=int(state.room_id),
                slot=int(enemy.slot),
                enemy_id=int(enemy.enemy_id),
                old=hp,
                new=_CLAMP_HP,
                reason="scaffold_hp_clamp",
            )
            self.telemetry.hp_clamp_writes.append(write)
            self.telemetry.hp_clamp_counts_by_room[f"0x{int(state.room_id):04X}"] += 1
            self.telemetry.hp_clamp_counts_by_entity[f"0x{int(enemy.enemy_id):04X}"] += 1


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
        profile: AssistProfile = "survival",
        hp_clamp: ScaffoldHpClamp | None = None,
        scaffold_allowlist: Sequence[ScaffoldAllowlistEntry] | None = None,
    ) -> None:
        super().__init__(enabled=unlimited_ammo, refill_when=refill_when)
        self.unlimited_energy = unlimited_energy
        self.profile: AssistProfile = profile
        self._effective_health: int | None = None
        self._previous_phase: GameplayPhase | None = None
        self._energy_drain_hold = 0
        if hp_clamp is not None:
            self.hp_clamp = hp_clamp
            self.hp_clamp.telemetry = self.telemetry
        else:
            enable_clamp = profile == "scaffold" or scaffold_allowlist is not None
            if scaffold_allowlist is not None:
                entries = tuple(scaffold_allowlist)
            elif enable_clamp:
                entries = attic_ordinary_enemy_allowlist()
            else:
                entries = ()
            self.hp_clamp = ScaffoldHpClamp(
                enabled=enable_clamp,
                allowlist=entries,
                telemetry=self.telemetry,
            )

    def report(self) -> dict[str, object]:
        scaffold = bool(self.hp_clamp.enabled)
        return {
            "enabled": self.enabled or self.unlimited_energy or scaffold,
            "profile": "scaffold" if scaffold else self.profile,
            "development_only": scaffold,
            "unlimited_energy_enabled": self.unlimited_energy,
            "unlimited_ammo_enabled": self.enabled,
            "refill_when": self.refill_when,
            "hp_clamp": self.hp_clamp.report(),
            **self.telemetry.to_dict(),
        }

    def apply(
        self,
        data: _RetroData,
        state: SuperMetroidState,
        *,
        ram: object | None = None,
    ) -> None:
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
        self.hp_clamp.apply(ram if ram is not None else data, state)

    def attach_env(self, env) -> None:
        """Refill inside ``env.step`` so a headed HUD sees topped-up health.

        Wrap this *before* ``retro_harness.headed.attach_headed``. Idle after a
        hop uses ``env.step`` directly and would otherwise skip ``_Sess``.
        """
        orig = env.step

        def step(action):
            out = orig(action)
            st = parse_env_state(env, mode="nav")
            data = getattr(env, "data", env)
            try:
                self.apply(data, st, ram=env)
            except Exception:  # noqa: BLE001
                self.apply(env, st, ram=env)
            return out

        env.step = step  # type: ignore[method-assign]
