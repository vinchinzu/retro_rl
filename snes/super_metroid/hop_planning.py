"""Hop and takeoff planning with physics trajectory prediction.

Provides trajectory-based evaluation for hop and takeoff candidates using
PhysicsPredictor. Planners can query predicted futures without full emulation.

Integration points:
- door_kinematics: evaluate door entry/exit trajectories
- takeoff: evaluate platform hop trajectories  
- map_planning: route planning with predicted kinematics

Default to StubPredictor in tests (no ROM required). Production can use
SmRevClient or other physics backends via load_predictor().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from super_metroid.door_kinematics import DoorKinematics
from super_metroid.physics_sim import (
    FrameInput,
    PhysicsPredictor,
    SimState,
    StubPredictor,
    Trajectory,
    position_out_of_range,
)
from super_metroid.ram import SuperMetroidState
from super_metroid.takeoff import TakeoffWindow

__all__ = [
    "HopCandidate",
    "TrajectoryEvaluator",
    "evaluate_hop_trajectory",
    "evaluate_takeoff_trajectory",
]


@dataclass(frozen=True)
class HopCandidate:
    """A candidate hop with predicted trajectory.

    Combines planning intent (takeoff window, target) with physics
    prediction outcome for route evaluation.
    """

    takeoff: TakeoffWindow
    start_state: SimState
    inputs: tuple[FrameInput, ...]
    trajectory: Trajectory
    feasible: bool
    reason: str = ""

    @property
    def frames(self) -> int:
        """Total frames for this hop (input sequence length)."""
        return len(self.inputs)

    @property
    def final_x(self) -> int:
        """Final X position after trajectory."""
        if not self.trajectory.frames:
            return self.start_state.samus_x
        return self.trajectory.frames[-1].samus_x

    @property
    def final_y(self) -> int:
        """Final Y position after trajectory."""
        if not self.trajectory.frames:
            return self.start_state.samus_y
        return self.trajectory.frames[-1].samus_y


class TrajectoryEvaluator:
    """Evaluate hop/takeoff candidates using physics prediction.

    Wraps a PhysicsPredictor and provides planning-level APIs for
    trajectory feasibility checks.
    """

    def __init__(self, predictor: PhysicsPredictor | None = None) -> None:
        """Initialize evaluator with predictor backend.

        Args:
            predictor: Physics predictor (defaults to StubPredictor)
        """
        self._predictor = predictor or StubPredictor(name="hop-planning-stub")

    @property
    def predictor(self) -> PhysicsPredictor:
        """Access underlying predictor."""
        return self._predictor

    def evaluate_hop(
        self,
        takeoff: TakeoffWindow,
        start_state: SimState,
        inputs: Sequence[FrameInput],
        *,
        target_x_range: tuple[int, int] | None = None,
        target_y_range: tuple[int, int] | None = None,
    ) -> HopCandidate:
        """Evaluate a hop candidate with predicted trajectory.

        Args:
            takeoff: Takeoff window specification
            start_state: Initial Samus state
            inputs: Button input sequence
            target_x_range: Optional target X band (lo, hi)
            target_y_range: Optional target Y band (lo, hi)

        Returns:
            HopCandidate with trajectory and feasibility assessment
        """
        trajectory = self._predictor.predict(start_state, inputs)
        reason = ""
        if trajectory.frames:
            final = trajectory.frames[-1]
            reason = position_out_of_range(
                final.samus_x,
                final.samus_y,
                x_range=target_x_range,
                y_range=target_y_range,
            )

        return HopCandidate(
            takeoff=takeoff,
            start_state=start_state,
            inputs=tuple(inputs),
            trajectory=trajectory,
            feasible=not reason,
            reason=reason,
        )

    def evaluate_door_transition(
        self,
        door_kin: DoorKinematics,
        inputs: Sequence[FrameInput],
    ) -> Trajectory:
        """Predict trajectory for a door entry/exit sequence.

        Args:
            door_kin: Door kinematics (leave or entry snapshot)
            inputs: Button sequence after door transition

        Returns:
            Predicted trajectory from door kinematics state
        """
        return self._predictor.predict(SimState.from_kinematics(door_kin), inputs)


def evaluate_hop_trajectory(
    takeoff: TakeoffWindow,
    start: SuperMetroidState,
    inputs: Sequence[FrameInput],
    predictor: PhysicsPredictor | None = None,
) -> HopCandidate:
    """Convenience function: evaluate single hop with trajectory.

    Args:
        takeoff: Takeoff window for this hop
        start: Current game state
        inputs: Button input sequence
        predictor: Optional predictor (defaults to StubPredictor)

    Returns:
        HopCandidate with predicted trajectory
    """
    evaluator = TrajectoryEvaluator(predictor)
    return evaluator.evaluate_hop(takeoff, SimState.from_kinematics(start), inputs)


def evaluate_takeoff_trajectory(
    takeoff: TakeoffWindow,
    start_x: int,
    start_y: int,
    inputs: Sequence[FrameInput],
    predictor: PhysicsPredictor | None = None,
) -> HopCandidate:
    """Convenience function: evaluate takeoff from position.

    Args:
        takeoff: Takeoff window specification
        start_x: Starting X pixel
        start_y: Starting Y pixel
        inputs: Button input sequence
        predictor: Optional predictor (defaults to StubPredictor)

    Returns:
        HopCandidate with predicted trajectory
    """
    evaluator = TrajectoryEvaluator(predictor)
    return evaluator.evaluate_hop(
        takeoff,
        SimState.grounded(samus_x=start_x, samus_y=start_y),
        inputs,
    )
