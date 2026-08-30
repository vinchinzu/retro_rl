from __future__ import annotations

from dataclasses import replace
import numpy as np

from super_metroid.assist import (
    ATTIC_PILOT_ENEMY_ID,
    ATTIC_ROOM_ID,
    ScaffoldAllowlistEntry,
    ScaffoldHpClamp,
    UnlimitedAmmoAssist,
    UnlimitedResourcesAssist,
    attic_ordinary_enemy_allowlist,
)
from super_metroid.combat.enemies.scan import ENEMY_BASE, ENEMY_STRIDE
from super_metroid.ram import SNES_WRAM_BANK, GameplayPhase, parse_state

# Synthetic ordinary-enemy id for RAM-buffer tests. Not a live Attic header.
_SYNTHETIC_ENEMY_ID = 0xBEEF
_OFF_HP = 0x14
_OFF_SPAWN = 0x1A
_OFF_PHASE = 0x30


class FakeData:
    def __init__(self, ram: np.ndarray | None = None) -> None:
        self.writes: list[tuple[str, int]] = []
        self.ram = ram

    def set_value(self, key: str, value: int) -> None:
        self.writes.append((key, value))

    def get_ram(self):
        return self.ram


class _RetroMemory:
    """env.data.memory stand-in: assign + blocks, no get_ram."""

    def __init__(self, ram: np.ndarray) -> None:
        self._ram = ram
        self.blocks = {SNES_WRAM_BANK: ram}
        self.assigns: list[tuple[int, str, int]] = []

    def assign(self, addr: int, dtype: str, value: int) -> None:
        self.assigns.append((int(addr), str(dtype), int(value)))
        addr = int(addr)
        if dtype == "<u2":
            self._ram[addr] = int(value) & 0xFF
            self._ram[addr + 1] = (int(value) >> 8) & 0xFF
        else:
            self._ram[addr] = int(value) & 0xFF


class RetroData:
    """Looks like live env.data: set_value + memory.assign, no get_ram / .ram."""

    def __init__(self, ram: np.ndarray) -> None:
        self.writes: list[tuple[str, int]] = []
        self.memory = _RetroMemory(ram)

    def set_value(self, key: str, value: int) -> None:
        self.writes.append((key, value))


def _u16(ram: np.ndarray, addr: int, value: int) -> None:
    ram[addr] = value & 0xFF
    ram[addr + 1] = (value >> 8) & 0xFF


def _poke_enemy(
    ram: np.ndarray,
    slot: int,
    *,
    enemy_id: int,
    hp: int,
    x: int = 100,
    y: int = 120,
    spawn_state: int = 0,
    phase: int = 0,
) -> None:
    base = ENEMY_BASE + slot * ENEMY_STRIDE
    _u16(ram, base, enemy_id)
    _u16(ram, base + 0x02, x)
    _u16(ram, base + 0x06, y)
    _u16(ram, base + _OFF_HP, hp)
    _u16(ram, base + _OFF_SPAWN, spawn_state)
    _u16(ram, base + _OFF_PHASE, phase)


def _slot_hp(ram: np.ndarray, slot: int) -> int:
    addr = ENEMY_BASE + slot * ENEMY_STRIDE + _OFF_HP
    return int(ram[addr]) | (int(ram[addr + 1]) << 8)


def _attic(enemy_id: int = _SYNTHETIC_ENEMY_ID, **kwargs) -> ScaffoldAllowlistEntry:
    return ScaffoldAllowlistEntry(room_id=ATTIC_ROOM_ID, enemy_id=enemy_id, **kwargs)


def _scaffold(*, allowlist, ram: np.ndarray | None = None) -> tuple[UnlimitedResourcesAssist, FakeData]:
    assist = UnlimitedResourcesAssist(scaffold_allowlist=allowlist)
    return assist, FakeData(ram)


def _ordinary_attic(**kwargs):
    return replace(
        _state(),
        frame=kwargs.pop("frame", 10),
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        room_id=ATTIC_ROOM_ID,
        **kwargs,
    )


def _state():
    ram = np.zeros(0x10000, dtype=np.uint8)
    return parse_state(ram)


def test_locked_ammo_is_never_granted() -> None:
    assist = UnlimitedAmmoAssist()
    data = FakeData()
    state = replace(
        _state(),
        frame=10,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        missiles=0,
        max_missiles=0,
    )

    assist.apply(data, state)

    assert data.writes == []


def test_only_current_unlocked_ammo_is_refilled() -> None:
    assist = UnlimitedAmmoAssist()
    data = FakeData()
    state = replace(
        _state(),
        frame=100,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        missiles=2,
        max_missiles=5,
        super_missiles=0,
        max_super_missiles=0,
    )

    assist.apply(data, state)

    assert data.writes == [("missiles", 5)]
    assert assist.telemetry.ammo["missiles"].restored == 3
    assert assist.telemetry.ammo["missiles"].first_unlocked_frame == 100
    assert assist.telemetry.ammo["missiles"].top_ups == 0  # partial, not empty
    assert assist.telemetry.capacity_writes == 0
    assert assist.telemetry.progression_writes == 0


def test_at_zero_ammo_waits_until_empty() -> None:
    assist = UnlimitedAmmoAssist(refill_when="at_zero")
    data = FakeData()
    partial = replace(
        _state(),
        frame=10,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        missiles=2,
        max_missiles=5,
    )
    empty = replace(partial, frame=11, missiles=0)

    assist.apply(data, partial)
    assert data.writes == []
    assert assist.telemetry.ammo["missiles"].top_ups == 0

    assist.apply(data, empty)
    assert data.writes == [("missiles", 5)]
    assert assist.telemetry.ammo["missiles"].restored == 5
    assert assist.telemetry.ammo["missiles"].top_ups == 1
    assert assist.telemetry.top_ups_total() == 1


def test_transition_phase_suspends_writes() -> None:
    assist = UnlimitedAmmoAssist()
    data = FakeData()
    state = replace(
        _state(),
        phase=GameplayPhase.ROOM_TRANSITION,
        missiles=1,
        max_missiles=5,
    )

    assist.apply(data, state)

    assert data.writes == []
    assert assist.telemetry.suspended_phase_frames["room_transition"] == 1


def test_reset_garbage_does_not_claim_ammo_was_unlocked() -> None:
    assist = UnlimitedAmmoAssist()
    data = FakeData()
    state = replace(
        _state(),
        phase=GameplayPhase.UNKNOWN,
        missiles=0x5555,
        max_missiles=0x5555,
    )

    assist.apply(data, state)

    assert assist.telemetry.ammo["missiles"].first_unlocked_frame is None
    assert data.writes == []


def test_unlimited_energy_restores_current_health_only() -> None:
    assist = UnlimitedResourcesAssist()
    data = FakeData()
    state = replace(
        _state(),
        frame=20,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        health=37,
        max_health=199,
    )

    assist.apply(data, state)

    assert data.writes == [("health", 199)]
    assert assist.telemetry.energy.restored == 162
    assert assist.telemetry.energy.writes == 1
    assert assist.telemetry.energy.top_ups == 0  # always mode, not empty
    assert assist.telemetry.capacity_writes == 0
    assert assist.telemetry.progression_writes == 0


def test_at_zero_energy_waits_until_empty() -> None:
    assist = UnlimitedResourcesAssist(unlimited_ammo=False, refill_when="at_zero")
    data = FakeData()
    damaged = replace(
        _state(),
        frame=1,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        health=80,
        max_health=199,
    )
    empty = replace(damaged, frame=2, health=0)

    assist.apply(data, damaged)
    assert data.writes == []
    assert assist.telemetry.energy.top_ups == 0

    assist.apply(data, empty)
    assert data.writes == [("health", 199)]
    assert assist.telemetry.energy.restored == 199
    assert assist.telemetry.energy.top_ups == 1
    assert assist.telemetry.top_ups_total() == 1


def test_at_zero_energy_tops_up_at_low_floor() -> None:
    """Boss one-shots skip ordinary health==0; floor death-saves before latch."""
    from super_metroid.assist import AT_ZERO_ENERGY_FLOOR

    assist = UnlimitedResourcesAssist(unlimited_ammo=False, refill_when="at_zero")
    data = FakeData()
    above = replace(
        _state(),
        frame=1,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        health=AT_ZERO_ENERGY_FLOOR + 1,
        max_health=299,
    )
    low = replace(above, frame=2, health=AT_ZERO_ENERGY_FLOOR)

    assist.apply(data, above)
    assert data.writes == []

    assist.apply(data, low)
    assert data.writes == [("health", 299)]
    assert assist.telemetry.energy.top_ups == 1
    assert assist.telemetry.energy.restored == 299 - AT_ZERO_ENERGY_FLOOR


def test_unlimited_energy_observes_damage_and_suspends_outside_gameplay() -> None:
    assist = UnlimitedResourcesAssist(unlimited_ammo=False)
    data = FakeData()
    ordinary = replace(
        _state(),
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        health=99,
        max_health=99,
    )
    damaged = replace(ordinary, frame=1, health=79)
    transition = replace(
        damaged,
        frame=2,
        phase=GameplayPhase.ROOM_TRANSITION,
        health=1,
    )

    assist.apply(data, ordinary)
    assist.apply(data, damaged)
    assist.apply(data, transition)

    assert data.writes == [("health", 99)]
    assert assist.telemetry.maximum_single_frame_damage == 20
    assert assist.telemetry.suspended_phase_frames["room_transition"] == 1


def test_unlimited_energy_is_suspended_during_ceres_countdown_route() -> None:
    assist = UnlimitedResourcesAssist()
    data = FakeData()
    state = replace(
        _state(),
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        area_index=6,
        health=27,
        max_health=99,
    )

    assist.apply(data, state)

    assert data.writes == []
    assert assist.telemetry.suspended_phase_frames["energy:ceres"] == 1


def test_unlimited_energy_is_suspended_while_metroid_latched() -> None:
    """Baby Metroid / MB rainbow drain must reach low energy — refill softlocks."""
    assist = UnlimitedResourcesAssist()
    data = FakeData()
    latched = replace(
        _state(),
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        area_index=5,  # Tourian — drain movement is not global knockback
        pose=137,
        movement_type=21,
        health=50,
        max_health=499,
    )
    early_stick = replace(
        _state(),
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        pose=9,
        movement_type=1,
        health=400,
        max_health=499,
        samus_x=200,
        samus_y=187,
        enemy0_x=200,
        enemy0_y=166,
        enemy0_hp=0x7FFF,
    )
    mb_rainbow = replace(
        _state(),
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        area_index=5,
        pose=84,
        movement_type=10,
        health=300,
        max_health=499,
        room_id=0xDD58,
    )
    drained = replace(latched, pose=232, movement_type=27, health=1)
    free = replace(
        latched,
        pose=1,
        movement_type=0,
        health=50,
        enemy0_hp=0,
        enemy0_x=0,
        enemy0_y=0,
    )

    assist.apply(data, latched)
    assert data.writes == []
    assert assist.telemetry.suspended_phase_frames["energy:metroid_latch"] == 1

    data.writes.clear()
    assist.apply(data, early_stick)
    assert data.writes == []
    assert assist.telemetry.suspended_phase_frames["energy:metroid_latch"] == 2

    data.writes.clear()
    assist.apply(data, mb_rainbow)
    assert data.writes == []
    assert assist.telemetry.suspended_phase_frames["energy:metroid_latch"] == 3

    data.writes.clear()
    assist.apply(data, drained)
    assert data.writes == []
    assert assist.telemetry.suspended_phase_frames["energy:metroid_latch"] == 4

    data.writes.clear()
    assist.apply(data, free)
    # Sticky hold still active after drain poses — no refill yet.
    assert data.writes == []

    # Expire sticky hold, then free standing should refill.
    assist._energy_drain_hold = 0
    data.writes.clear()
    assist.apply(data, free)
    assert data.writes == [("health", 499)]


def test_mb_rainbow_pose_gap_keeps_energy_suspended() -> None:
    """Pose 42 between rainbow and stun must not get a one-frame refill."""
    assist = UnlimitedResourcesAssist()
    data = FakeData()
    rainbow = replace(
        _state(),
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        area_index=5,
        pose=84,
        movement_type=10,
        health=200,
        max_health=499,
        room_id=0xDD58,
    )
    gap = replace(rainbow, pose=42, movement_type=6, health=199)
    stun = replace(rainbow, pose=233, movement_type=27, health=199)

    assist.apply(data, rainbow)
    assert data.writes == []
    data.writes.clear()
    assist.apply(data, gap)
    assert data.writes == []
    data.writes.clear()
    assist.apply(data, stun)
    assert data.writes == []


def test_norfair_knockback_does_not_suspend_energy() -> None:
    """Movement $0A is ordinary LN/GT knockback, not a Tourian latch."""
    assist = UnlimitedResourcesAssist(unlimited_ammo=False)
    data = FakeData()
    hit = replace(
        _state(),
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        area_index=2,  # Norfair
        pose=137,
        movement_type=10,
        health=50,
        max_health=299,
        room_id=0xB283,
    )

    assist.apply(data, hit)

    assert data.writes == [("health", 299)]
    assert "energy:metroid_latch" not in assist.telemetry.suspended_phase_frames


def test_attach_env_refills_inside_env_step() -> None:
    """Headed HUD reads RAM after env.step — refill must happen in that wrap."""
    ram = np.zeros(0x10000, dtype=np.uint8)
    # Ordinary gameplay + Zebes + health 10/99 so always-refill writes.
    ram[0x0998] = 8  # game_state ordinary-ish; parse_state uses its own addrs

    class Env:
        def __init__(self):
            self.data = FakeData()
            self.ram = ram

        def step(self, action):
            del action
            return None

        def get_ram(self):
            return self.ram

    env = Env()
    assist = UnlimitedResourcesAssist()
    low = replace(
        _state(),
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        area_index=0,
        health=10,
        max_health=99,
    )
    from super_metroid import assist as assist_mod

    orig_parse = assist_mod.parse_env_state
    assist_mod.parse_env_state = lambda *args, **kwargs: low  # type: ignore[assignment]
    try:
        assist.attach_env(env)
        env.step(None)
    finally:
        assist_mod.parse_env_state = orig_parse
    assert env.data.writes == [("health", 99)]


def test_survival_default_does_not_clamp_live_enemies() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=250)
    assist = UnlimitedResourcesAssist()
    data = FakeData(ram)
    assist.apply(data, _ordinary_attic(health=99, max_health=99))

    assert _slot_hp(ram, 0) == 250
    assert assist.hp_clamp.enabled is False
    assert assist.telemetry.hp_clamp_writes == []
    assert assist.report()["profile"] == "survival"
    assert assist.report()["development_only"] is False


def test_empty_allowlist_writes_no_hp() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=250)
    assist, data = _scaffold(allowlist=(), ram=ram)
    assist.apply(data, _ordinary_attic())

    assert _slot_hp(ram, 0) == 250
    assert assist.hp_clamp.enabled is True
    assert assist.telemetry.hp_clamp_writes == []


def test_unknown_species_in_attic_fail_closed() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=250)
    assist = UnlimitedResourcesAssist(profile="scaffold")
    data = FakeData(ram)
    assist.apply(data, _ordinary_attic())

    assert _slot_hp(ram, 0) == 250
    assert assist.telemetry.hp_clamp_writes == []
    factory = attic_ordinary_enemy_allowlist()
    assert factory[0].room_id == ATTIC_ROOM_ID
    assert factory[0].enemy_id == ATTIC_PILOT_ENEMY_ID


def test_clamp_live_allowlisted_enemy_hp_to_one_not_zero() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=250)
    assist, data = _scaffold(allowlist=(_attic(),), ram=ram)
    assist.apply(data, _ordinary_attic(frame=7, health=40, max_health=99))

    assert _slot_hp(ram, 0) == 1
    assert ("enemy0_hp", 0) not in data.writes
    assert ("enemy0_hp", 1) not in data.writes
    writes = assist.telemetry.hp_clamp_writes
    assert len(writes) == 1
    assert writes[0].old == 250
    assert writes[0].new == 1
    assert writes[0].slot == 0
    assert writes[0].enemy_id == _SYNTHETIC_ENEMY_ID
    assert writes[0].room_id == ATTIC_ROOM_ID
    assert writes[0].reason == "scaffold_hp_clamp"
    assert assist.telemetry.hp_clamp_counts_by_room == {"0xCA52": 1}
    assert assist.telemetry.hp_clamp_counts_by_entity == {"0xBEEF": 1}
    assert assist.telemetry.progression_writes == 0
    assert assist.telemetry.capacity_writes == 0
    report = assist.report()
    assert report["profile"] == "scaffold"
    assert report["development_only"] is True
    assert report["hp_clamp"]["writes"][0]["new"] == 1


def test_clamp_once_per_phase_and_skips_hp_already_one() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=250)
    assist, data = _scaffold(allowlist=(_attic(),), ram=ram)
    state = _ordinary_attic(frame=1)
    assist.apply(data, state)
    _u16(ram, ENEMY_BASE + _OFF_HP, 80)
    assist.apply(data, replace(state, frame=2))

    assert _slot_hp(ram, 0) == 80
    assert len(assist.telemetry.hp_clamp_writes) == 1

    ram2 = np.zeros(0x2000, dtype=np.uint8)
    _poke_enemy(ram2, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=1)
    assist2, data2 = _scaffold(allowlist=(_attic(),), ram=ram2)
    assist2.apply(data2, _ordinary_attic())
    assert _slot_hp(ram2, 0) == 1
    assert assist2.telemetry.hp_clamp_writes == []


def test_clamp_skips_dead_and_off_map() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=0)
    _poke_enemy(ram, 1, enemy_id=_SYNTHETIC_ENEMY_ID, hp=250, x=0xFE00)
    assist, data = _scaffold(allowlist=(_attic(),), ram=ram)
    assist.apply(data, _ordinary_attic())

    assert _slot_hp(ram, 0) == 0
    assert _slot_hp(ram, 1) == 250
    assert assist.telemetry.hp_clamp_writes == []


def test_clamp_scans_all_slots() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=90)
    _poke_enemy(ram, 31, enemy_id=_SYNTHETIC_ENEMY_ID, hp=40)
    _poke_enemy(ram, 2, enemy_id=0x1111, hp=99)
    assist, data = _scaffold(allowlist=(_attic(),), ram=ram)
    assist.apply(data, _ordinary_attic())

    assert _slot_hp(ram, 0) == 1
    assert _slot_hp(ram, 31) == 1
    assert _slot_hp(ram, 2) == 99
    assert len(assist.telemetry.hp_clamp_writes) == 2
    slots = {row.slot for row in assist.telemetry.hp_clamp_writes}
    assert slots == {0, 31}


def test_clamp_suspends_outside_ordinary_gameplay() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=250)
    assist, data = _scaffold(allowlist=(_attic(),), ram=ram)
    state = replace(
        _state(),
        phase=GameplayPhase.ROOM_TRANSITION,
        room_id=ATTIC_ROOM_ID,
        missiles=1,
        max_missiles=5,
    )
    assist.apply(data, state)

    assert _slot_hp(ram, 0) == 250
    assert assist.telemetry.hp_clamp_writes == []
    assert assist.telemetry.suspended_phase_frames["hp_clamp:room_transition"] == 1


def test_clamp_spawn_state_mismatch_fail_closed() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=250, spawn_state=0)
    allow = (_attic(spawn_state=0x1234),)
    assist, data = _scaffold(allowlist=allow, ram=ram)
    assist.apply(data, _ordinary_attic())
    assert _slot_hp(ram, 0) == 250

    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=250, spawn_state=0x1234)
    assist.apply(data, _ordinary_attic(frame=2))
    assert _slot_hp(ram, 0) == 1
    assert len(assist.telemetry.hp_clamp_writes) == 1


def test_clamp_new_phase_reclamps_once() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=500, phase=1)
    allow = (
        _attic(phase=1),
        _attic(phase=2),
    )
    assist, data = _scaffold(allowlist=allow, ram=ram)
    assist.apply(data, _ordinary_attic(frame=1))
    assert _slot_hp(ram, 0) == 1

    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=500, phase=2)
    assist.apply(data, _ordinary_attic(frame=2))
    assert _slot_hp(ram, 0) == 1
    assert [row.old for row in assist.telemetry.hp_clamp_writes] == [500, 500]
    assert assist.telemetry.hp_clamp_counts_by_entity["0xBEEF"] == 2


def test_disabled_clamp_controller_writes_nothing() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=250)
    clamp = ScaffoldHpClamp(enabled=False, allowlist=(_attic(),))
    clamp.apply(FakeData(ram), _ordinary_attic())
    assert _slot_hp(ram, 0) == 250
    assert clamp.telemetry.hp_clamp_writes == []


def test_attach_env_runs_survival_refill_and_clamp() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=180)

    class Env:
        def __init__(self):
            self.data = FakeData(ram)
            self.ram = ram

        def step(self, action):
            del action
            return None

        def get_ram(self):
            return self.ram

    env = Env()
    assist = UnlimitedResourcesAssist(scaffold_allowlist=(_attic(),))
    low = _ordinary_attic(health=10, max_health=99)
    from super_metroid import assist as assist_mod

    orig_parse = assist_mod.parse_env_state
    assist_mod.parse_env_state = lambda *args, **kwargs: low  # type: ignore[assignment]
    try:
        assist.attach_env(env)
        env.step(None)
    finally:
        assist_mod.parse_env_state = orig_parse
    assert env.data.writes == [("health", 99)]
    assert _slot_hp(ram, 0) == 1
    assert len(assist.telemetry.hp_clamp_writes) == 1
    assert assist.telemetry.progression_writes == 0


def test_apply_env_data_without_get_ram_clamps_via_memory_assign() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=250)
    data = RetroData(ram)
    assist = UnlimitedResourcesAssist(scaffold_allowlist=(_attic(),))
    assist.apply(data, _ordinary_attic(health=40, max_health=99))

    assert _slot_hp(ram, 0) == 1
    assert data.memory.assigns == [
        (ENEMY_BASE + _OFF_HP, "<u2", 1),
    ]
    assert ("enemy0_hp", 1) not in data.writes
    assert len(assist.telemetry.hp_clamp_writes) == 1
    assert assist.telemetry.energy.writes == 1


def test_clamp_reclamps_after_leaving_and_reentering_room() -> None:
    ram = np.zeros(0x2000, dtype=np.uint8)
    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=250)
    assist, data = _scaffold(allowlist=(_attic(),), ram=ram)
    assist.apply(data, _ordinary_attic(frame=1))
    assert _slot_hp(ram, 0) == 1
    assert len(assist.telemetry.hp_clamp_writes) == 1

    other = replace(
        _state(),
        frame=2,
        phase=GameplayPhase.ORDINARY_GAMEPLAY,
        room_id=0x93FE,
    )
    assist.apply(data, other)

    _poke_enemy(ram, 0, enemy_id=_SYNTHETIC_ENEMY_ID, hp=250)
    assist.apply(data, _ordinary_attic(frame=3))
    assert _slot_hp(ram, 0) == 1
    assert len(assist.telemetry.hp_clamp_writes) == 2
    assert [row.old for row in assist.telemetry.hp_clamp_writes] == [250, 250]


def test_report_profile_follows_actual_writes_not_constructor() -> None:
    mixed = UnlimitedResourcesAssist(profile="clean")
    assert mixed.unlimited_energy is True
    assert mixed.enabled is True
    assert mixed.hp_clamp.enabled is False
    assert mixed.report()["profile"] == "survival"

    clean = UnlimitedResourcesAssist(unlimited_energy=False, unlimited_ammo=False)
    assert clean.report()["profile"] == "clean"
    assert clean.report()["development_only"] is False

    scaffold = UnlimitedResourcesAssist(scaffold_allowlist=(_attic(),))
    assert scaffold.report()["profile"] == "scaffold"
    assert scaffold.report()["development_only"] is True
