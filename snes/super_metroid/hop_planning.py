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
)
from super_metroid.ram import SuperMetroidState
from super_metroid.takeoff import TakeoffWindow

__all__ = [
    "HopCandidate",
    "TrajectorEvaluator",
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

        # Check feasibility against target ranges if provided
        feasible = True
        reason = ""

        if trajectory.frames:
            final = trajectory.frames[-1]

            if target_x_range is not None:
                x_lo, x_hi = target_x_range
                if not (x_lo <= final.samus_x <= x_hi):
                    feasible = False
                    reason = (
                        f"final x={final.samus_x} outside target "
                        f"[{x_lo}, {x_hi}]"
                    )

            if feasible and target_y_range is not None:
                y_lo, y_hi = target_y_range
                if not (y_lo <= final.samus_y <= y_hi):
                    feasible = False
                    reason = (
                        f"final y={final.samus_y} outside target "
                        f"[{y_lo}, {y_hi}]"
                    )

        return HopCandidate(
            takeoff=takeoff,
            start_state=start_state,
            inputs=tuple(inputs),
            trajectory=trajectory,
            feasible=feasible,
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
        # Convert DoorKinematics to SimState
        start = SimState(
            frame=door_kin.frame,
            room_id=door_kin.room_id,
            samus_x=door_kin.samus_x,
            samus_y=door_kin.samus_y,
            samus_x_sub=door_kin.samus_x_sub,
            samus_y_sub=door_kin.samus_y_sub,
            velocity_x=door_kin.velocity_x,
            velocity_y=door_kin.velocity_y,
            velocity_x_sub=door_kin.velocity_x_sub,
            velocity_y_sub=door_kin.velocity_y_sub,
            momentum_x=door_kin.momentum_x,
            momentum_x_sub=door_kin.momentum_x_sub,
            pose=door_kin.pose,
            facing=door_kin.facing,
            movement_type=door_kin.movement_type,
            speed_counter=door_kin.speed_counter,
            speed_flag=door_kin.speed_flag,
            shinespark_timer=door_kin.shinespark_timer,
        )
        return self._predictor.predict(start, inputs)


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
    sim_state = SimState.from_sm_state(start)
    return evaluator.evaluate_hop(takeoff, sim_state, inputs)


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
    # Create minimal SimState for position-based planning
    sim_state = SimState(
        frame=0,
        room_id=0,
        samus_x=start_x,
        samus_y=start_y,
        samus_x_sub=0,
        samus_y_sub=0,
        velocity_x=0,
        velocity_y=0,
        velocity_x_sub=0,
        velocity_y_sub=0,
        momentum_x=0,
        momentum_x_sub=0,
        pose=0,
        facing=8,  # FACING_RIGHT
        movement_type=0,
        speed_counter=0,
        speed_flag=0,
        shinespark_timer=0,
    )
    return evaluator.evaluate_hop(takeoff, sim_state, inputs)
