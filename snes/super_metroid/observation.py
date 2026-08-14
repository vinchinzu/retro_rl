"""Shared observation tuple API for physics sim and emulator validation.

Provides a unified observation interface readable from both SimState/TrajectoryFrame
(MiniStep physics predictor) and SuperMetroidState (emulator RAM). Designed for
residual profiling — measuring where MiniStep diverges from ground truth.

Observation levels (Bob's locked lattice):
- Oπ (coarsest): ($0AF6, $0AFA, $0A1C, $079B) pixels x/y, pose, room
- Oσ: Oπ plus ($0AF8, $0AFC) subpixels
- Oσ+: Oσ plus optional ($0F8C, $18A8) enemy energy / i-frames
- O† (separate): ($09C2, dead) energy/death - NOT a coarsening of Oπ

Speeds ($0B2C/$0B2E/$0B42/$0B44) live in first-differing-field, not a second σ+.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "Observation",
    "observation_from_kinematics",
    "observation_from_sim_state",
    "observation_from_trajectory_frame",
]


@dataclass(frozen=True)
class Observation:
    """Shared observation tuple for physics sim and emulator validation.

    Bob's locked observation lattice:
    - Oπ: pixels x/y ($0AF6, $0AFA), pose ($0A1C), room ($079B)
    - Oσ: Oπ plus subpixels ($0AF8, $0AFC)
    - Oσ+: Oσ plus optional enemy energy ($0F8C) / i-frames ($18A8)
    - O†: energy ($09C2) / death (separate from Oπ)

    Speeds ($0B2C/$0B2E/$0B42/$0B44) live in first-differing-field, not σ+.

    RAM addresses (WRAM $7E:xxxx):
    - x/y: $0AF6 / $0AFA (Oπ)
    - pose: $0A1C (Oπ)
    - room: $079B (Oπ, room header pointer, not door ID)
    - subX/subY: $0AF8 / $0AFC (Oσ)
    - enemy_energy: $0F8C (Oσ+, optional)
    - invulnerability_timer: $18A8 (Oσ+, optional)
    - energy: $09C2 (O†)
    - frame_counter_1: $1842 (lag detection)
    - frame_counter_2: $09DA (lag detection)
    - velocity_x/y: $0B2C/$0B2E (first-differing-field)
    - momentum_x: $0B42/$0B44 (first-differing-field)
    """

    frame: int

    # Oπ (coarsest): pixels x/y, pose, room
    x: int
    y: int
    pose: int
    room: int

    # Oσ: Oπ plus subpixels
    sub_x: int
    sub_y: int

    # Speeds (first-differing-field, not σ+)
    velocity_x: int
    velocity_y: int
    velocity_x_sub: int
    velocity_y_sub: int
    momentum_x: int
    momentum_x_sub: int
    speed_counter: int
    speed_flag: int

    # O† (separate): energy/death — None when Mini does not track
    energy: int | None

    # Lag detection — None when Mini does not track
    frame_counter_1: int | None
    frame_counter_2: int | None

    # Oσ+ (optional): enemy energy / i-frames
    enemy_energy: int = 0
    invulnerability_timer: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Export to dict, omitting zero optional fields."""
        result = asdict(self)
        # Omit Oσ+ fields when zero
        if result["enemy_energy"] == 0:
            del result["enemy_energy"]
        if result["invulnerability_timer"] == 0:
            del result["invulnerability_timer"]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        """Restore from dict with optional field defaults."""
        data_copy = dict(data)
        data_copy.setdefault("enemy_energy", 0)
        data_copy.setdefault("invulnerability_timer", 0)
        return cls(**data_copy)


def observation_from_kinematics(state: Any) -> Observation:
    """Map any HasKinematics object (SimState, SuperMetroidState, door snap).

    O† uses ``health`` when present (emulator). Mini/stub have no health → None.
    Oσ+ uses ``enemy0_hp`` when present; otherwise 0 (unobserved on Mini).
    Lag counters stay None until SuperMetroidState grows $1842/$09DA.
    """
    enemy = getattr(state, "enemy0_hp", None)
    return Observation(
        frame=state.frame,
        x=state.samus_x,
        y=state.samus_y,
        pose=state.pose,
        room=state.room_id,
        sub_x=state.samus_x_sub,
        sub_y=state.samus_y_sub,
        velocity_x=state.velocity_x,
        velocity_y=state.velocity_y,
        velocity_x_sub=state.velocity_x_sub,
        velocity_y_sub=state.velocity_y_sub,
        momentum_x=state.momentum_x,
        momentum_x_sub=state.momentum_x_sub,
        speed_counter=state.speed_counter,
        speed_flag=state.speed_flag,
        energy=getattr(state, "health", None),
        frame_counter_1=None,
        frame_counter_2=None,
        enemy_energy=0 if enemy is None else enemy,
        invulnerability_timer=0,
    )


def observation_from_sim_state(state: Any) -> Observation:
    return observation_from_kinematics(state)


def observation_from_trajectory_frame(frame: Any) -> Observation:
    return observation_from_kinematics(frame)
