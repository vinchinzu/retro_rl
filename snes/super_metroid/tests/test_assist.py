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
    assert assist.telemetry.capacity_writes == 0
    assert assist.telemetry.progression_writes == 0


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
    assert assist.telemetry.capacity_writes == 0
    assert assist.telemetry.progression_writes == 0


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
