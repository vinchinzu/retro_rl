"""Shared observation tuple API for physics sim and emulator validation.

Provides a unified observation interface readable from both SimState/TrajectoryFrame
(MiniStep physics predictor) and SuperMetroidState (emulator RAM). Designed for
residual profiling — measuring where MiniStep diverges from ground truth.

Observation levels (from PHYSICS_PREDICTOR.md):
- Oσ+ (extended): enemies + i-frames + knockback (when available)
- Oσ (core): position + velocity + pose + room + energy + frame counters
- Oπ (input): button tape only
- O† (liveness): death / health=0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = [
    "Observation",
    "observation_from_sim_state",
    "observation_from_trajectory_frame",
]


@dataclass(frozen=True)
class Observation:
    """Shared observation tuple for physics sim and emulator validation.

    Core fields (Oσ) always present. Optional enemy/i-frame fields (Oσ+)
    included when RAM has them; omit when empty (MiniStep has no knockback).

    RAM addresses (WRAM $7E:xxxx):
    - x/y: $0AF6 / $0AFA
    - subX/subY: $0AF8 / $0AFC
    - pose: $0A1C
    - room: $079B (room header pointer, not door ID)
    - energy: $09C2
    - frame_counter_1: $1842 (lag-sensitive tape index)
    - frame_counter_2: $09DA (secondary frame counter)
    """

    # Core kinematics (Oσ)
    frame: int
    room: int
    x: int
    y: int
    sub_x: int
    sub_y: int
    velocity_x: int
    velocity_y: int
    velocity_x_sub: int
    velocity_y_sub: int
    momentum_x: int
    momentum_x_sub: int
    pose: int

    # Speed booster state
    speed_counter: int
    speed_flag: int

    # Liveness (O†)
    energy: int

    # Frame counters (lag detection)
    frame_counter_1: int
    frame_counter_2: int

    # Extended state (Oσ+) — optional, omit when empty
    enemies: tuple[dict[str, Any], ...] = ()
    invulnerability_timer: int = 0
    knockback_timer: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Export to dict, omitting empty optional fields."""
        result = asdict(self)
        # Omit enemies when empty
        if not result["enemies"]:
            del result["enemies"]
        # Omit i-frame/knockback when zero
        if result["invulnerability_timer"] == 0:
            del result["invulnerability_timer"]
        if result["knockback_timer"] == 0:
            del result["knockback_timer"]
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        """Restore from dict with optional field defaults."""
        data_copy = dict(data)
        data_copy.setdefault("enemies", [])
        data_copy["enemies"] = tuple(data_copy["enemies"])
        data_copy.setdefault("invulnerability_timer", 0)
        data_copy.setdefault("knockback_timer", 0)
        return cls(**data_copy)


def observation_from_sim_state(state: Any) -> Observation:
    """Extract Observation from SimState (physics_sim.py).

    Args:
        state: SimState or compatible object with required fields

    Returns:
        Observation with core fields. Extended fields (enemies/i-frame) omitted
        since MiniStep does not track them.
    """
    return Observation(
        frame=state.frame,
        room=state.room_id,
        x=state.samus_x,
        y=state.samus_y,
        sub_x=state.samus_x_sub,
        sub_y=state.samus_y_sub,
        velocity_x=state.velocity_x,
        velocity_y=state.velocity_y,
        velocity_x_sub=state.velocity_x_sub,
        velocity_y_sub=state.velocity_y_sub,
        momentum_x=state.momentum_x,
        momentum_x_sub=state.momentum_x_sub,
        pose=state.pose,
        speed_counter=state.speed_counter,
        speed_flag=state.speed_flag,
        energy=0,  # Not tracked in SimState
        frame_counter_1=0,  # Not tracked in SimState
        frame_counter_2=0,  # Not tracked in SimState
    )


def observation_from_trajectory_frame(frame: Any) -> Observation:
    """Extract Observation from TrajectoryFrame (physics_sim.py).

    Args:
        frame: TrajectoryFrame or compatible object with required fields

    Returns:
        Observation with core fields. Extended fields (enemies/i-frame) from
        TrajectoryFrame.enemies when available.
    """
    enemies = tuple(frame.enemies) if hasattr(frame, "enemies") else ()
    return Observation(
        frame=frame.frame,
        room=frame.room_id,
        x=frame.samus_x,
        y=frame.samus_y,
        sub_x=frame.samus_x_sub,
        sub_y=frame.samus_y_sub,
        velocity_x=frame.velocity_x,
        velocity_y=frame.velocity_y,
        velocity_x_sub=frame.velocity_x_sub,
        velocity_y_sub=frame.velocity_y_sub,
        momentum_x=frame.momentum_x,
        momentum_x_sub=frame.momentum_x_sub,
        pose=frame.pose,
        speed_counter=frame.speed_counter,
        speed_flag=frame.speed_flag,
        energy=0,  # Not tracked in TrajectoryFrame
        frame_counter_1=0,  # Not tracked in TrajectoryFrame
        frame_counter_2=0,  # Not tracked in TrajectoryFrame
        enemies=enemies,
    )
