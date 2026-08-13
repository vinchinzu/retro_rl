"""Tests for hop_planning trajectory evaluation (offline, no ROM).

Proves that hop/takeoff planning calls PhysicsPredictor and uses
trajectory results for feasibility assessment.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from retro_harness.controls import SNES_DPAD_RIGHT
from super_metroid.door_kinematics import DoorKinematics
from super_metroid.hop_planning import (
    HopCandidate,
    TrajectoryEvaluator,
    evaluate_hop_trajectory,
    evaluate_takeoff_trajectory,
)
from super_metroid.physics_sim import (
    FrameInput,
    PhysicsPredictor,
    SimState,
    StubPredictor,
    Trajectory,
    TrajectoryFrame,
)
from super_metroid.ram import parse_state
from super_metroid.takeoff import TakeoffWindow


class TestTrajectoryEvaluator:
    """Test TrajectoryEvaluator with StubPredictor."""

    def test_creates_with_default_stub_predictor(self) -> None:
        evaluator = TrajectoryEvaluator()
        assert isinstance(evaluator.predictor, StubPredictor)
        assert "stub" in evaluator.predictor.name()

    def test_accepts_custom_predictor(self) -> None:
        custom = StubPredictor(name="custom-test")
        evaluator = TrajectoryEvaluator(custom)
        assert evaluator.predictor is custom
        assert evaluator.predictor.name() == "custom-test"

    def test_evaluate_hop_calls_predictor(self) -> None:
        """Verify predictor.predict() is called with start + inputs."""
        predictor = StubPredictor(name="test-call-check")
        evaluator = TrajectoryEvaluator(predictor)

        start = SimState(
            frame=0,
            room_id=0x91F8,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        takeoff = TakeoffWindow((100, 120), SNES_DPAD_RIGHT)
        inputs = [FrameInput(buttons=0x80) for _ in range(10)]  # RIGHT

        candidate = evaluator.evaluate_hop(takeoff, start, inputs)

        # Verify predictor was called (trajectory exists)
        assert candidate.trajectory is not None
        assert candidate.trajectory.predictor == "test-call-check"
        assert len(candidate.trajectory.frames) == 10
        # StubPredictor moves right with RIGHT button
        assert candidate.final_x > start.samus_x

    def test_evaluate_hop_checks_target_x_range(self) -> None:
        """Feasibility fails when final X is outside target range."""
        evaluator = TrajectoryEvaluator()
        start = SimState(
            frame=0,
            room_id=0,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        takeoff = TakeoffWindow((100, 120), SNES_DPAD_RIGHT)
        # Move right 10 frames -> final x = 100 + 10*2 = 120
        inputs = [FrameInput(buttons=0x80) for _ in range(10)]

        # Target range that final x will miss
        candidate = evaluator.evaluate_hop(
            takeoff, start, inputs, target_x_range=(200, 300)
        )

        assert not candidate.feasible
        assert "outside target" in candidate.reason.lower()
        assert "x=" in candidate.reason

    def test_evaluate_hop_checks_target_y_range(self) -> None:
        """Feasibility fails when final Y is outside target range."""
        evaluator = TrajectoryEvaluator()
        start = SimState(
            frame=0,
            room_id=0,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        takeoff = TakeoffWindow((100, 120), SNES_DPAD_RIGHT)
        inputs = [FrameInput(buttons=0)]

        # Target range that final y will miss
        candidate = evaluator.evaluate_hop(
            takeoff, start, inputs, target_y_range=(100, 150)
        )

        assert not candidate.feasible
        assert "outside target" in candidate.reason.lower()
        assert "y=" in candidate.reason

    def test_evaluate_hop_feasible_when_in_range(self) -> None:
        """Feasibility passes when final position is in target ranges."""
        evaluator = TrajectoryEvaluator()
        start = SimState(
            frame=0,
            room_id=0,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        takeoff = TakeoffWindow((100, 120), SNES_DPAD_RIGHT)
        # Move right 10 frames -> x=120, StubPredictor keeps y=200
        inputs = [FrameInput(buttons=0x80) for _ in range(10)]

        candidate = evaluator.evaluate_hop(
            takeoff,
            start,
            inputs,
            target_x_range=(110, 130),
            target_y_range=(190, 210),
        )

        assert candidate.feasible
        assert candidate.reason == ""
        assert 110 <= candidate.final_x <= 130
        assert 190 <= candidate.final_y <= 210

    def test_evaluate_door_transition(self) -> None:
        """Verify door transition trajectory prediction."""
        evaluator = TrajectoryEvaluator()
        door_kin = DoorKinematics(
            frame=100,
            room_id=0x91F8,
            samus_x=400,
            samus_y=180,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=2,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=2,
            momentum_x_sub=0,
            speed_counter=0,
            speed_flag=0,
            vertical_direction=0,
            facing=8,
            movement_type=0,
            shinespark_timer=0,
            pose=0,
            door_transition=0,
            transition_direction=0,
            door_def_ptr=0,
            game_state=8,
            phase="ordinary",
        )
        inputs = [FrameInput(buttons=0x80) for _ in range(5)]

        trajectory = evaluator.evaluate_door_transition(door_kin, inputs)

        assert len(trajectory.frames) == 5
        assert trajectory.start.room_id == 0x91F8
        assert trajectory.start.samus_x == 400
        # Should predict movement from door kinematics
        assert trajectory.frames[-1].samus_x > 400


class TestHopCandidate:
    """Test HopCandidate data structure."""

    def test_frames_property(self) -> None:
        start = SimState(
            frame=0,
            room_id=0,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        traj = Trajectory(start=start, frames=(), predictor="test")
        takeoff = TakeoffWindow((100, 120), SNES_DPAD_RIGHT)
        inputs = tuple(FrameInput(buttons=0) for _ in range(15))

        candidate = HopCandidate(
            takeoff=takeoff,
            start_state=start,
            inputs=inputs,
            trajectory=traj,
            feasible=True,
        )

        assert candidate.frames == 15

    def test_final_x_from_trajectory(self) -> None:
        start = SimState(
            frame=0,
            room_id=0,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        frames = (
            TrajectoryFrame(
                frame=1,
                room_id=0,
                samus_x=105,
                samus_y=200,
                samus_x_sub=0,
                samus_y_sub=0,
                velocity_x=0,
                velocity_y=0,
                velocity_x_sub=0,
                velocity_y_sub=0,
                momentum_x=0,
                momentum_x_sub=0,
                pose=0,
                facing=8,
                movement_type=0,
                speed_counter=0,
                speed_flag=0,
                shinespark_timer=0,
                enemies=[],
            ),
            TrajectoryFrame(
                frame=2,
                room_id=0,
                samus_x=110,
                samus_y=195,
                samus_x_sub=0,
                samus_y_sub=0,
                velocity_x=0,
                velocity_y=0,
                velocity_x_sub=0,
                velocity_y_sub=0,
                momentum_x=0,
                momentum_x_sub=0,
                pose=0,
                facing=8,
                movement_type=0,
                speed_counter=0,
                speed_flag=0,
                shinespark_timer=0,
                enemies=[],
            ),
        )
        traj = Trajectory(start=start, frames=frames, predictor="test")
        takeoff = TakeoffWindow((100, 120), SNES_DPAD_RIGHT)

        candidate = HopCandidate(
            takeoff=takeoff,
            start_state=start,
            inputs=(FrameInput(0), FrameInput(0)),
            trajectory=traj,
            feasible=True,
        )

        assert candidate.final_x == 110
        assert candidate.final_y == 195

    def test_final_position_from_start_when_empty_trajectory(self) -> None:
        start = SimState(
            frame=0,
            room_id=0,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        traj = Trajectory(start=start, frames=(), predictor="test")
        takeoff = TakeoffWindow((100, 120), SNES_DPAD_RIGHT)

        candidate = HopCandidate(
            takeoff=takeoff,
            start_state=start,
            inputs=(),
            trajectory=traj,
            feasible=True,
        )

        assert candidate.final_x == 100
        assert candidate.final_y == 200


class TestConvenienceFunctions:
    """Test top-level convenience functions."""

    def test_evaluate_hop_trajectory_from_sm_state(self) -> None:
        """Verify evaluate_hop_trajectory converts SuperMetroidState."""
        ram = np.zeros(0x2000, dtype=np.uint8)
        ram[0x0AF6] = 100  # samus_x lo
        ram[0x0AF7] = 0  # samus_x hi
        ram[0x0AFA] = 200  # samus_y lo
        ram[0x0AFB] = 0  # samus_y hi
        state = parse_state(ram, frame=0)

        takeoff = TakeoffWindow((100, 120), SNES_DPAD_RIGHT)
        inputs = [FrameInput(buttons=0x80) for _ in range(5)]

        candidate = evaluate_hop_trajectory(takeoff, state, inputs)

        assert candidate.start_state.samus_x == 100
        assert candidate.start_state.samus_y == 200
        assert len(candidate.trajectory.frames) == 5

    def test_evaluate_takeoff_trajectory_from_position(self) -> None:
        """Verify evaluate_takeoff_trajectory creates minimal SimState."""
        takeoff = TakeoffWindow((100, 120), SNES_DPAD_RIGHT)
        inputs = [FrameInput(buttons=0x80) for _ in range(8)]

        candidate = evaluate_takeoff_trajectory(takeoff, 100, 200, inputs)

        assert candidate.start_state.samus_x == 100
        assert candidate.start_state.samus_y == 200
        assert len(candidate.trajectory.frames) == 8
        # StubPredictor should move right
        assert candidate.final_x > 100


class TestPredictorIntegration:
    """Integration tests proving predictor is used by planning."""

    def test_custom_predictor_is_called(self) -> None:
        """Verify custom predictor backend is invoked."""

        class MockPredictor(PhysicsPredictor):
            """Mock predictor that tracks calls."""

            def __init__(self) -> None:
                self.call_count = 0
                self.last_start: SimState | None = None
                self.last_inputs: list[FrameInput] = []

            def predict(
                self, start: SimState, inputs: list[FrameInput]
            ) -> Trajectory:
                self.call_count += 1
                self.last_start = start
                self.last_inputs = list(inputs)
                # Return empty trajectory
                return Trajectory(
                    start=start, frames=(), predictor="mock"
                )

            def name(self) -> str:
                return "mock-predictor"

        mock = MockPredictor()
        evaluator = TrajectoryEvaluator(mock)

        start = SimState(
            frame=0,
            room_id=0,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        takeoff = TakeoffWindow((100, 120), SNES_DPAD_RIGHT)
        inputs = [FrameInput(buttons=0x80) for _ in range(3)]

        candidate = evaluator.evaluate_hop(takeoff, start, inputs)

        # Verify predictor was called
        assert mock.call_count == 1
        assert mock.last_start == start
        assert len(mock.last_inputs) == 3
        assert mock.last_inputs[0].buttons == 0x80
        assert candidate.trajectory.predictor == "mock"

    def test_stub_predictor_physics_is_used(self) -> None:
        """Verify StubPredictor physics affects planning outcome."""
        evaluator = TrajectoryEvaluator(StubPredictor(name="integration"))
        start = SimState(
            frame=0,
            room_id=0,
            samus_x=100,
            samus_y=200,
            samus_x_sub=0,
            samus_y_sub=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            pose=0,
            facing=8,
            movement_type=0,
            speed_counter=0,
            speed_flag=0,
            shinespark_timer=0,
        )
        takeoff = TakeoffWindow((100, 120), SNES_DPAD_RIGHT)

        # StubPredictor moves +2 px/frame with RIGHT (0x80)
        inputs_right = [FrameInput(buttons=0x80) for _ in range(20)]
        right_candidate = evaluator.evaluate_hop(takeoff, start, inputs_right)

        # Should land around x=140 (100 + 20*2)
        assert 130 <= right_candidate.final_x <= 150

        # StubPredictor moves -2 px/frame with LEFT (0x40)
        inputs_left = [FrameInput(buttons=0x40) for _ in range(20)]
        left_candidate = evaluator.evaluate_hop(takeoff, start, inputs_left)

        # Should land around x=60 (100 - 20*2)
        assert 50 <= left_candidate.final_x <= 70
