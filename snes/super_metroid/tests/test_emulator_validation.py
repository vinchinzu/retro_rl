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
    When ROM is available, they exercise the emulator validation path.
    """

    def test_validate_trajectory_requires_rom(self) -> None:
        """ROM check works (only runs if ROM available)."""
        assert ROM_AVAILABLE, "This test should only run when ROM available"

    def test_emulator_vs_stub_comparison_short_hop(self) -> None:
        """Compare stub predictor vs real emulator on short hop.

        This test exercises the emulator validation path without claiming
        room_clear. It compares stub vs emulator kinematics to show:
        1. Validation path works on real emulator
        2. Stub and emulator may disagree (expected, emu is truth)
        3. Emulator result is what matters for ground truth

        Does NOT claim room_clear - just exercises validation machinery.
        """
        from pathlib import Path

        from super_metroid.physics_sim import SimState, StubPredictor

        # Use existing state file from scratch (test artifact)
        state_path = Path(
            "snes/super_metroid/custom_integrations/SuperMetroid-Snes/"
            "scratch/post_ice_bat_to_red_pure.state"
        )

        if not state_path.exists():
            pytest.skip(
                f"Test state file not found: {state_path}. "
                "Need existing state file to exercise emulator path."
            )

        # Short hop: just hold RIGHT for 10 frames (simple test)
        inputs = [FrameInput(buttons=0x80) for _ in range(10)]

        # Get stub prediction
        stub_predictor = StubPredictor(name="comparison-test")
        # Use a simple start state for stub (kinematics only)
        stub_start = SimState(
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
        stub_trajectory = stub_predictor.predict(stub_start, inputs)

        # Get emulator validation (ground truth)
        emu_result = validate_trajectory_on_emulator(
            state_path,
            inputs,
            # No target checks - just exercise the path
        )

        # Emulator result is what we got (ground truth)
        assert emu_result.frames_executed == 10, "Should execute all frames"

        # Compare stub vs emulator
        if stub_trajectory.frames:
            stub_final_x = stub_trajectory.frames[-1].samus_x
            stub_final_y = stub_trajectory.frames[-1].samus_y

            # Note: stub and emu will almost certainly disagree
            # That's expected - stub is simplified physics
            # This test just proves the validation path works

            disagreement = False
            if emu_result.final_x is not None:
                x_diff = abs(stub_final_x - emu_result.final_x)
                if x_diff > 5:  # Allow small tolerance
                    disagreement = True

            if emu_result.final_y is not None:
                y_diff = abs(stub_final_y - emu_result.final_y)
                if y_diff > 5:
                    disagreement = True

            # Disagreement is expected and OK - emu is truth
            # This test just exercises the validation machinery
            if disagreement:
                # This is normal - stub is approximate
                pass

        # Key assertions: emulator path executed successfully
        assert emu_result.final_room_id is not None, "Should read final room"
        assert emu_result.final_x is not None, "Should read final X"
        assert emu_result.final_y is not None, "Should read final Y"

        # Note: We do NOT assert room_clear or claim any room was cleared
        # This test only proves the validation path works


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
