"""226-float generalist observation: 13×13 occupancy + Samus + enemies + Goal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from retro_harness.contracts import (
    ActionContract,
    ObservationContract,
    ObservationField,
    RewardComponent,
    RewardContract,
    SNES_BUTTONS,
    WrapperContract,
    WrapperSpec,
)
from retro_harness.platformer.neuro.net import GRID_SIZE, N_GRID
from super_metroid.combat.enemies.scan import Enemy, list_enemies
from super_metroid.combat.enemies.species import Contact, species_of
from super_metroid.generalist.goals import GOAL_VEC_DIM, Goal, goal_vector
from super_metroid.hop_glance import pose_class
from super_metroid.platformer_levels import SM_ACTIONS
from super_metroid.ram import (
    FACING_LEFT,
    GS_ORDINARY,
    HI_JUMP_MASK,
    MORPH_BALL_MASK,
    VARIA_MASK,
)

SPEED_BOOSTER_MASK = 0x2000
SAMUS_DIM = 20
ENEMY_SLOTS = 5
ENEMY_FEAT = 5
ENEMY_DIM = ENEMY_SLOTS * ENEMY_FEAT
OBS_DIM = N_GRID + SAMUS_DIM + ENEMY_DIM + GOAL_VEC_DIM  # 226
TILE_PX = 16
N_ACTIONS = len(SM_ACTIONS)

# Offsets inside the Samus / Goal parts — not concatenated OBS_DIM indices.
SAMUS_ORDINARY = 13
SAMUS_DOOR_TRANSITION = 14
GOAL_DX = 0
GOAL_DY = 1
GOAL_PREVIOUS_ACTION = 5
GOAL_SAME_ROOM = 8

CollisionFn = Callable[[int, int], bool]


@dataclass(frozen=True)
class GeneralistObs:
    """Named 226-float observation. ``to_array`` is the locked SB3 layout."""

    occupancy: np.ndarray
    samus: np.ndarray
    enemies: np.ndarray
    goal: np.ndarray
    SAMUS_ORDINARY = SAMUS_ORDINARY
    SAMUS_DOOR_TRANSITION = SAMUS_DOOR_TRANSITION
    GOAL_DX = GOAL_DX
    GOAL_DY = GOAL_DY
    GOAL_PREVIOUS_ACTION = GOAL_PREVIOUS_ACTION
    GOAL_SAME_ROOM = GOAL_SAME_ROOM

    @classmethod
    def blank(cls) -> GeneralistObs:
        return cls(
            occupancy=np.zeros(N_GRID, dtype=np.float32),
            samus=np.zeros(SAMUS_DIM, dtype=np.float32),
            enemies=np.zeros(ENEMY_DIM, dtype=np.float32),
            goal=np.zeros(GOAL_VEC_DIM, dtype=np.float32),
        )

    @classmethod
    def from_array(cls, vec: np.ndarray | Sequence[float]) -> GeneralistObs:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.shape[0] != OBS_DIM:
            raise ValueError(f"obs shape {arr.shape} != ({OBS_DIM},)")
        samus_end = N_GRID + SAMUS_DIM
        enemy_end = samus_end + ENEMY_DIM
        return cls(
            occupancy=arr[:N_GRID],
            samus=arr[N_GRID:samus_end],
            enemies=arr[samus_end:enemy_end],
            goal=arr[enemy_end:],
        )

    def to_array(self) -> np.ndarray:
        vec = np.concatenate(
            [self.occupancy, self.samus, self.enemies, self.goal]
        ).astype(np.float32, copy=False)
        if vec.shape != (OBS_DIM,):
            raise ValueError(f"obs shape {vec.shape} != ({OBS_DIM},)")
        return vec

    @property
    def grid(self) -> np.ndarray:
        return np.asarray(self.occupancy, dtype=np.float32).reshape(GRID_SIZE, GRID_SIZE)

    @property
    def ordinary(self) -> float:
        return float(self.samus[SAMUS_ORDINARY])

    @property
    def door_transition(self) -> float:
        return float(self.samus[SAMUS_DOOR_TRANSITION])

    @property
    def goal_dx(self) -> float:
        return float(self.goal[GOAL_DX])

    @property
    def goal_dy(self) -> float:
        return float(self.goal[GOAL_DY])

    @property
    def previous_action(self) -> int:
        return int(round(float(self.goal[GOAL_PREVIOUS_ACTION]) * 26.0))

    @property
    def same_room(self) -> bool:
        return float(self.goal[GOAL_SAME_ROOM]) > 0.5


def occupancy_grid(
    samus_x: int,
    samus_y: int,
    enemies: Sequence[Enemy] = (),
    *,
    solid: CollisionFn | Mapping[tuple[int, int], bool] | None = None,
) -> np.ndarray:
    """13×13 local occupancy: +1 solid, −1 enemy, 0 air/unknown."""

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    origin = GRID_SIZE // 2
    if solid is not None:
        lookup: CollisionFn
        if callable(solid):
            lookup = solid
        else:
            blocks = solid

            def lookup(wx: int, wy: int, _blocks=blocks) -> bool:
                return bool(_blocks.get((wx // TILE_PX, wy // TILE_PX), False))

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                wx = int(samus_x) + (col - origin) * TILE_PX
                wy = int(samus_y) + (row - origin) * TILE_PX
                if lookup(wx, wy):
                    grid[row, col] = 1.0
    for enemy in enemies:
        col = int(round((int(enemy.x) - int(samus_x)) / TILE_PX)) + origin
        row = int(round((int(enemy.y) - int(samus_y)) / TILE_PX)) + origin
        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            grid[row, col] = -1.0
    return grid.reshape(-1)


def _clip(value: float, limit: float = 4.0) -> float:
    return float(max(-limit, min(limit, value)))


def samus_vector(state: Any) -> list[float]:
    """20-float Samus body. Locked by the observation contract."""

    pose = int(getattr(state, "pose", 0) or 0)
    klass = pose_class(pose)
    facing = int(getattr(state, "facing", 0) or 0)
    health = float(getattr(state, "health", 0) or 0)
    max_health = float(getattr(state, "max_health", 0) or 0)
    items = int(getattr(state, "collected_items", 0) or 0)
    gs = int(getattr(state, "game_state", GS_ORDINARY) or 0)
    dt = int(getattr(state, "door_transition", 0) or 0)
    return [
        1.0 if klass == "stand" else 0.0,
        1.0 if klass == "morph" else 0.0,
        1.0 if klass == "air" else 0.0,
        1.0 if klass == "other" else 0.0,
        -1.0 if facing == FACING_LEFT else 1.0,
        _clip(float(getattr(state, "velocity_x", 0) or 0) / 8.0),
        _clip(float(getattr(state, "velocity_y", 0) or 0) / 8.0),
        _clip(float(getattr(state, "momentum_x", 0) or 0) / 8.0),
        health / max_health if max_health > 0 else 0.0,
        1.0 if items & MORPH_BALL_MASK else 0.0,
        1.0 if items & HI_JUMP_MASK else 0.0,
        1.0 if items & SPEED_BOOSTER_MASK else 0.0,
        1.0 if items & VARIA_MASK else 0.0,
        1.0 if gs == GS_ORDINARY else 0.0,
        1.0 if dt else 0.0,
        float(pose) / 255.0,
        float(getattr(state, "movement_type", 0) or 0) / 32.0,
        float(getattr(state, "speed_counter", 0) or 0) / 4.0,
        float(getattr(state, "samus_x", 0) or 0) / 4096.0,
        float(getattr(state, "samus_y", 0) or 0) / 4096.0,
    ]


def _contact_value(enemy: Enemy) -> float:
    species = species_of(enemy.enemy_id)
    contact = species.frozen_contact if int(enemy.freeze_timer) > 0 else species.live_contact
    return {
        Contact.NONE: 0.0,
        Contact.KNOCKBACK: 0.33,
        Contact.SOLID: 0.66,
        Contact.PLATFORM: 1.0,
    }.get(contact, 0.0)


def enemy_vector(state: Any, enemies: Sequence[Enemy]) -> list[float]:
    """5 slots × (rel_x, rel_y, species-id, contact, health-frac)."""

    sx = int(getattr(state, "samus_x", 0) or 0)
    sy = int(getattr(state, "samus_y", 0) or 0)
    ranked = sorted(
        enemies,
        key=lambda enemy: abs(int(enemy.x) - sx) + abs(int(enemy.y) - sy),
    )[:ENEMY_SLOTS]
    out: list[float] = []
    for enemy in ranked:
        species = species_of(enemy.enemy_id)
        max_hp = float(species.max_hp) if species.max_hp else float(enemy.hp)
        out.extend(
            [
                _clip((int(enemy.x) - sx) / 256.0),
                _clip((int(enemy.y) - sy) / 256.0),
                float(int(enemy.enemy_id) & 0xFFFF) / 65535.0,
                _contact_value(enemy),
                float(enemy.hp) / max_hp if max_hp > 0 else 0.0,
            ]
        )
    missing = ENEMY_SLOTS - len(ranked)
    out.extend([0.0] * (missing * ENEMY_FEAT))
    return out


def observe_parts(
    state: Any,
    goal: Goal,
    *,
    ram: Any | None = None,
    prev_action: int = 0,
    solid: CollisionFn | Mapping[tuple[int, int], bool] | None = None,
    steer_x: int | None = None,
    steer_y: int | None = None,
) -> GeneralistObs:
    """Build the named observation (same layout as ``observe``)."""

    enemies = list_enemies(ram) if ram is not None else ()
    return GeneralistObs(
        occupancy=occupancy_grid(
            int(getattr(state, "samus_x", 0) or 0),
            int(getattr(state, "samus_y", 0) or 0),
            enemies,
            solid=solid,
        ),
        samus=np.asarray(samus_vector(state), dtype=np.float32),
        enemies=np.asarray(enemy_vector(state, enemies), dtype=np.float32),
        goal=np.asarray(
            goal_vector(
                state,
                goal,
                prev_action=prev_action,
                steer_x=steer_x,
                steer_y=steer_y,
            ),
            dtype=np.float32,
        ),
    )


def observe(
    state: Any,
    goal: Goal,
    *,
    ram: Any | None = None,
    prev_action: int = 0,
    solid: CollisionFn | Mapping[tuple[int, int], bool] | None = None,
    steer_x: int | None = None,
    steer_y: int | None = None,
) -> np.ndarray:
    """Build the locked 226-float vector."""

    return observe_parts(
        state,
        goal,
        ram=ram,
        prev_action=prev_action,
        solid=solid,
        steer_x=steer_x,
        steer_y=steer_y,
    ).to_array()


def observation_contract() -> ObservationContract:
    return ObservationContract(
        fields=(
            ObservationField("occupancy", "float32", (N_GRID,), "13x13 local occupancy"),
            ObservationField("samus", "float32", (SAMUS_DIM,), "pose facing vel items"),
            ObservationField("enemies", "float32", (ENEMY_DIM,), "five nearest room enemies"),
            ObservationField("goal", "float32", (GOAL_VEC_DIM,), "contractor Goal"),
        ),
        preprocessing={"concat": True, "pixels": False, "dim": OBS_DIM},
        version="1",
    )


def action_contract() -> ActionContract:
    return ActionContract.from_button_rows(
        SM_ACTIONS,
        controller_buttons=SNES_BUTTONS,
        version="1",
    )


def reward_contract() -> RewardContract:
    return RewardContract(
        components=(
            RewardComponent(
                "delta_distance",
                1.0,
                "pixels closer to Join xy or first bounded Goal-route door, with "
                "monotone remaining-room distance and nearest-door fallback",
            ),
            RewardComponent("join", 1.0, "hop_glance LeaveSpec pass"),
            RewardComponent("death", -1.0, "energy reached zero"),
            RewardComponent("stall", -0.05, "no Oπ progress"),
        ),
        aggregation="sum",
        version="2",
    )


def wrapper_contract(*, frame_skip: int = 4) -> WrapperContract:
    return WrapperContract(
        stack=(
            WrapperSpec("frame_skip", {"n": int(frame_skip)}, version="1"),
            WrapperSpec("join_eval", {"skip": 1}, version="1"),
        )
    )


def schema_digests(*, frame_skip: int = 4) -> dict[str, str]:
    """ROM-free observation/action/reward/wrapper identity digests."""

    return {
        "observation": observation_contract().identity_digest,
        "action": action_contract().identity_digest,
        "reward": reward_contract().identity_digest,
        "wrapper": wrapper_contract(frame_skip=frame_skip).identity_digest,
        "obs_dim": str(OBS_DIM),
        "n_actions": str(N_ACTIONS),
    }


__all__ = [
    "ENEMY_DIM",
    "GOAL_DX",
    "GOAL_DY",
    "GOAL_PREVIOUS_ACTION",
    "GOAL_SAME_ROOM",
    "GOAL_VEC_DIM",
    "GRID_SIZE",
    "GeneralistObs",
    "N_ACTIONS",
    "N_GRID",
    "OBS_DIM",
    "SAMUS_DIM",
    "SAMUS_DOOR_TRANSITION",
    "SAMUS_ORDINARY",
    "SPEED_BOOSTER_MASK",
    "action_contract",
    "enemy_vector",
    "observe",
    "observe_parts",
    "observation_contract",
    "occupancy_grid",
    "reward_contract",
    "samus_vector",
    "schema_digests",
    "wrapper_contract",
]
