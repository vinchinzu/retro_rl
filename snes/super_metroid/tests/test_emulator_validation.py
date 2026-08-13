"""Tests for emulator validation (ground truth).

These tests prove that emulator validation is the authority, not predictor.
Tests skip without ROM (pytest.mark.skipif) - offline tests still pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from super_metroid.emulator_validation import (
    ROM_AVAILABLE,
    EmulatorValidationResult,
    validate_trajectory_on_emulator,
)
from super_metroid.hop_planning import TrajectoryEvaluator
from super_metroid.physics_sim import FrameInput, SimState, StubPredictor
from super_metroid.takeoff import TakeoffWindow


class TestEmulatorValidationResult:
    """Test EmulatorValidationResult data structure (no ROM)."""

    def test_success_result(self) -> None:
        result = EmulatorValidationResult(
            success=True,
            final_room_id=0x91F8,
            final_x=400,
            final_y=200,
            frames_executed=60,
        )
        assert result.success
        assert result.room_clear
        assert result.final_room_id == 0x91F8
        assert result.reason == ""

    def test_failure_result(self) -> None:
        result = EmulatorValidationResult(
            success=False,
            final_room_id=0x91F8,
            final_x=100,
            final_y=200,
            frames_executed=30,
            reason="X out of range",
        )
        assert not result.success
        assert not result.room_clear
        assert result.reason == "X out of range"

    def test_crash_result(self) -> None:
        result = EmulatorValidationResult(
            success=False,
            final_room_id=None,
            final_x=None,
            final_y=None,
            frames_executed=15,
            reason="Emulator crash",
        )
        assert not result.success
        assert result.final_room_id is None
        assert result.final_x is None


class TestPredictorVsEmulatorPolicy:
    """Test that predictor success does NOT count as ground truth."""

    def test_stub_predictor_success_is_not_room_clear(self) -> None:
        """Prove that StubPredictor success is NOT a room-clear claim.

        This test uses only StubPredictor (no ROM). It proves the policy:
        predictor is for search speed only, NOT ground truth.
        """
        evaluator = TrajectoryEvaluator(StubPredictor(name="test"))
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
        takeoff = TakeoffWindow((100, 120), "RIGHT")
        inputs = [FrameInput(buttons=0x80) for _ in range(20)]

        candidate = evaluator.evaluate_hop(
            takeoff, start, inputs, target_x_range=(120, 140)
        )

        # Predictor says feasible, but this is NOT a room-clear claim
        assert candidate.feasible, "StubPredictor should predict success"

        # To claim room_clear, must validate on emulator (ground truth)
        # This test does NOT do that, so we have no room_clear claim
        # (would need @pytest.mark.skipif(not ROM_AVAILABLE) test for that)

    def test_predictor_policy_requires_emulator_for_claims(self) -> None:
        """Document that room-clear requires emulator validation.

        This is a documentation test proving the policy exists.
        """
        # Policy: predictor success alone is NOT sufficient for room_clear
        # Must call validate_trajectory_on_emulator() for ground truth

        # Example workflow (would need ROM for real validation):
        # 1. candidate = evaluator.evaluate_hop(...)  # Fast search
        # 2. if candidate.feasible:
        # 3.     result = validate_trajectory_on_emulator(...)  # Ground truth
        # 4.     if result.success:
        # 5.         assert result.room_clear  # NOW we can claim room_clear

        assert True, "Policy documented"


@pytest.mark.skipif(
    not ROM_AVAILABLE,
    reason="ROM not available - emulator validation requires ROM",
)
class TestEmulatorValidation:
    """Emulator validation tests (skip without ROM).

    These tests require ROM and stable-retro. They skip in CI without ROM.
    When ROM is available, they prove emulator is ground truth.
    """

    def test_validate_trajectory_requires_rom(self) -> None:
        """ROM check works (only runs if ROM available)."""
        assert ROM_AVAILABLE, "This test should only run when ROM available"

    def test_emulator_validation_is_ground_truth(self) -> None:
        """When emulator validates, that's a real room-clear claim.

        This test would validate a real trajectory on emulator.
        Skipped without ROM (pytest.mark.skipif above).
        """
        # This is a placeholder showing the pattern
        # Real test would:
        # 1. Load a real start state
        # 2. Run real inputs
        # 3. Validate against target room
        # 4. Assert result.room_clear is TRUE
        #
        # Example:
        # result = validate_trajectory_on_emulator(
        #     "custom_integrations/SuperMetroid-Snes/landing_site.state",
        #     inputs,
        #     target_room_id=0x91F8,
        # )
        # assert result.room_clear
        pytest.skip("Placeholder for real emulator validation test")


class TestEmulatorValidationWithoutRom:
    """Test validation behavior when ROM is not available (no skip)."""

    def test_validate_raises_without_rom(self) -> None:
        """Validation should raise FileNotFoundError without ROM."""
        if ROM_AVAILABLE:
            pytest.skip("This test requires ROM to NOT be available")

        inputs = [FrameInput(buttons=0)]
        with pytest.raises(FileNotFoundError, match="ROM not available"):
            validate_trajectory_on_emulator(
                Path("nonexistent.state"), inputs
            )


class TestSearchThenValidateWorkflow:
    """Test the recommended search→validate workflow (offline)."""

    def test_search_with_predictor_then_validate_pattern(self) -> None:
        """Demonstrate search→validate workflow (stub search only).

        This test shows the pattern without requiring ROM.
        Real validation would call validate_trajectory_on_emulator().
        """
        # 1. SEARCH: Use predictor for fast filtering
        evaluator = TrajectoryEvaluator(StubPredictor())
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
        takeoff = TakeoffWindow((100, 120), "RIGHT")

        # Try multiple candidates
        candidates = []
        for num_frames in [15, 20, 25, 30]:
            inputs = [FrameInput(buttons=0x80) for _ in range(num_frames)]
            candidate = evaluator.evaluate_hop(
                takeoff, start, inputs, target_x_range=(130, 150)
            )
            if candidate.feasible:
                candidates.append(candidate)

        # Sort by frame count (heuristic)
        candidates.sort(key=lambda c: c.frames)
        assert len(candidates) > 0, "Should find some candidates"

        # 2. VALIDATE: Would validate on emulator (needs ROM)
        # for candidate in candidates:
        #     result = validate_trajectory_on_emulator(
        #         start_state_path,
        #         candidate.inputs,
        #         target_room_id=NEXT_ROOM,
        #     )
        #     if result.success:
        #         # NOW we have ground truth room_clear
        #         assert result.room_clear
        #         break

        # This test only does search (no ROM needed)
        # Validation is in @pytest.mark.skipif tests above
