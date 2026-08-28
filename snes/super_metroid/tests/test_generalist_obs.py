"""Generalist observation contract (no emulator)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from super_metroid.combat.enemies.scan import ENEMY_BASE, ENEMY_STRIDE, Enemy
from super_metroid.combat.enemies.species import ATOMIC_ID
from super_metroid.generalist.goals import Goal
from super_metroid.generalist.obs import (
    N_ACTIONS,
    N_GRID,
    OBS_DIM,
    GeneralistObs,
    observe,
    observe_parts,
    occupancy_grid,
    samus_vector,
    schema_digests,
)
from super_metroid.paths import VANILLA_ROM_SHA1


def _state(**kwargs: object) -> SimpleNamespace:
    base = dict(
        samus_x=640,
        samus_y=480,
        pose=1,
        facing=8,
        velocity_x=2,
        velocity_y=0,
        momentum_x=0,
        health=99,
        max_health=99,
        collected_items=4,
        game_state=8,
        door_transition=0,
        movement_type=1,
        speed_counter=0,
        room_id=0x91F8,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_obs_dim_is_226() -> None:
    assert OBS_DIM == 226
    assert N_ACTIONS == 27


def test_occupancy_solid_and_enemy() -> None:
    enemy = Enemy(0, ATOMIC_ID, 640 + 16, 480, 250, 0)
    grid = occupancy_grid(
        640,
        480,
        (enemy,),
        solid={(640 // 16, 480 // 16): True},
    )
    assert grid.shape == (N_GRID,)
    center = (13 // 2) * 13 + (13 // 2)
    assert grid[center] == 1.0
    right = (13 // 2) * 13 + (13 // 2 + 1)
    assert grid[right] == -1.0


def test_observe_shape_and_goal_tail() -> None:
    goal = Goal("kpdr25/crateria/parlor", 0x91F8, 121, 1179, start_room_id=0x91F8)
    vec = observe(_state(), goal)
    assert vec.shape == (OBS_DIM,)
    assert vec.dtype == np.float32
    assert vec[-1] == 1.0
    assert 0.0 <= samus_vector(_state())[8] <= 1.0
    steered = observe(_state(), goal, steer_x=640 + 512, steer_y=480)
    assert GeneralistObs.from_array(steered).goal_dx > GeneralistObs.from_array(vec).goal_dx
    parts = observe_parts(_state(), goal, steer_x=640 + 512, steer_y=480)
    assert parts.to_array().shape == (OBS_DIM,)
    assert np.allclose(parts.to_array(), steered)
    assert np.allclose(GeneralistObs.from_array(steered).goal, parts.goal)


def test_observe_with_ram_enemy_slot() -> None:
    ram = np.zeros(ENEMY_BASE + ENEMY_STRIDE, dtype=np.uint8)

    def put(addr: int, value: int) -> None:
        ram[addr] = value & 0xFF
        ram[addr + 1] = (value >> 8) & 0xFF

    put(ENEMY_BASE, ATOMIC_ID)
    put(ENEMY_BASE + 0x02, 700)
    put(ENEMY_BASE + 0x06, 480)
    put(ENEMY_BASE + 0x14, 250)
    goal = Goal("any", 0x91F8, 700, 480, any_door=True)
    vec = observe(_state(), goal, ram=ram)
    assert vec.shape == (OBS_DIM,)


def test_schema_digests_are_stable() -> None:
    first = schema_digests()
    second = schema_digests()
    assert first == second
    assert first["obs_dim"] == "226"
    assert first["n_actions"] == "27"
    assert VANILLA_ROM_SHA1.startswith("da957f")
