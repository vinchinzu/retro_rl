from __future__ import annotations

from dataclasses import replace
import numpy as np

from super_metroid.assist import UnlimitedAmmoAssist, UnlimitedResourcesAssist
from super_metroid.ram import GameplayPhase, parse_state


class FakeData:
    def __init__(self) -> None:
        self.writes: list[tuple[str, int]] = []

    def set_value(self, key: str, value: int) -> None:
        self.writes.append((key, value))


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
