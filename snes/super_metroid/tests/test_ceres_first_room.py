"""Tests for first Ceres room fixture (Elevator → Falling Tile).

Tests the hop/tape search and emulator validation for the first room of
Ceres → Morph → Bomb sequence. Offline tests use StubPredictor (no ROM).
"""

from __future__ import annotations

import pytest

from super_metroid.emulator_validation import ROM_AVAILABLE
from super_metroid.physics_sim import StubPredictor
from super_metroid.routes.kpdr.ceres.first_room_fixture import (
    CERES_ELEVATOR_START_X,
    CERES_ELEVATOR_START_Y,
    CERES_FALLING_TARGET_X,
    CERES_FALLING_TARGET_Y,
    CeresFirstRoomFixture,
    search_ceres_first_room,
    validate_ceres_first_room,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
)


class TestCeresFirstRoomSearch:
    """Test search for first Ceres room (offline, no ROM needed)."""

    def test_search_returns_fixture(self) -> None:
        """Search returns a CeresFirstRoomFixture."""
        predictor = StubPredictor(name="test-ceres-search")
        fixture = search_ceres_first_room(predictor)

        assert isinstance(fixture, CeresFirstRoomFixture)
        assert fixture.from_room_id == ROOM_CERES_ELEVATOR
        assert fixture.to_room_id == ROOM_CERES_FALLING
        assert fixture.predictor_name == "test-ceres-search"

    def test_search_uses_start_position(self) -> None:
        """Search uses Ceres Elevator start position."""
        fixture = search_ceres_first_room()

        assert fixture.start_x == CERES_ELEVATOR_START_X
        assert fixture.start_y == CERES_ELEVATOR_START_Y

    def test_search_targets_falling_tile_room(self) -> None:
        """Search targets Falling Tile room position."""
        fixture = search_ceres_first_room()

        assert fixture.target_x == CERES_FALLING_TARGET_X
        assert fixture.target_y == CERES_FALLING_TARGET_Y

    def test_search_produces_input_sequence(self) -> None:
        """Search produces non-empty input sequence."""
        fixture = search_ceres_first_room()

        assert len(fixture.inputs) > 0
        assert all(hasattr(inp, "buttons") for inp in fixture.inputs)

    def test_search_sets_predictor_results(self) -> None:
        """Search sets predictor feasibility and final position."""
        fixture = search_ceres_first_room()

        # StubPredictor should produce some movement
        assert fixture.predictor_frames == len(fixture.inputs)
        assert fixture.predictor_final_x != fixture.start_x  # Should have moved

    def test_search_not_emulator_validated(self) -> None:
        """Search does not claim emulator validation."""
        fixture = search_ceres_first_room()

        assert not fixture.emulator_validated
        assert not fixture.emulator_success
        assert fixture.emulator_final_room is None

    def test_search_not_room_clear_without_emu(self) -> None:
        """Search never claims room_clear without emulator validation."""
        fixture = search_ceres_first_room()

        # Even if predictor says feasible, room_clear requires emulator
        assert not fixture.room_clear

    def test_search_respects_max_frames(self) -> None:
        """Search respects max_search_frames limit."""
        fixture = search_ceres_first_room(max_search_frames=50)

        # Should find something within 50 frames
        assert fixture.predictor_frames <= 50

    def test_fixture_to_dict_serializable(self) -> None:
        """Fixture can be serialized to dict."""
        fixture = search_ceres_first_room()
        data = fixture.to_dict()

        assert isinstance(data, dict)
        assert "from_room_id" in data
        assert "inputs" in data
        assert isinstance(data["inputs"], list)


class TestCeresFirstRoomValidation:
    """Test emulator validation (requires ROM)."""

    @pytest.mark.skipif(not ROM_AVAILABLE, reason="ROM not available")
    def test_validate_calls_emulator(self) -> None:
        """Validate calls emulator validation path.

        This test requires ROM and will be skipped in CI without ROM.
        """
        # Search with StubPredictor
        fixture = search_ceres_first_room()

        # Note: This will fail without a real Ceres Elevator start state
        # In production, we'd use custom_integrations/.../ceres_elevator.state
        # For now, this test documents the validation interface
        with pytest.raises((FileNotFoundError, RuntimeError)):
            validate_ceres_first_room(fixture, "nonexistent.state")

    @pytest.mark.skipif(not ROM_AVAILABLE, reason="ROM not available")
    def test_validate_with_stub_state(self) -> None:
        """Validate with a stub start state (if available).

        This test will pass only if ROM and a Ceres Elevator start state exist.
        """
        pytest.skip("Requires Ceres Elevator start state (not yet in repo)")


class TestCeresFirstRoomFixture:
    """Test CeresFirstRoomFixture data structure."""

    def test_room_clear_requires_emulator_success(self) -> None:
        """room_clear is False unless emulator_validated and emulator_success."""
        from super_metroid.physics_sim import FrameInput

        # Predictor says feasible but no emulator validation
        fixture = CeresFirstRoomFixture(
            from_room_id=ROOM_CERES_ELEVATOR,
            to_room_id=ROOM_CERES_FALLING,
            start_x=200,
            start_y=180,
            target_x=50,
            target_y=200,
            inputs=(FrameInput(0),),
            predictor_name="test",
            predictor_feasible=True,
            predictor_final_x=50,
            predictor_final_y=200,
            predictor_frames=1,
            emulator_validated=False,
        )
        assert not fixture.room_clear

        # Emulator validated but failed
        fixture_failed = CeresFirstRoomFixture(
            from_room_id=ROOM_CERES_ELEVATOR,
            to_room_id=ROOM_CERES_FALLING,
            start_x=200,
            start_y=180,
            target_x=50,
            target_y=200,
            inputs=(FrameInput(0),),
            predictor_name="test",
            predictor_feasible=True,
            predictor_final_x=50,
            predictor_final_y=200,
            predictor_frames=1,
            emulator_validated=True,
            emulator_success=False,
        )
        assert not fixture_failed.room_clear

        # Emulator validated and succeeded
        fixture_success = CeresFirstRoomFixture(
            from_room_id=ROOM_CERES_ELEVATOR,
            to_room_id=ROOM_CERES_FALLING,
            start_x=200,
            start_y=180,
            target_x=50,
            target_y=200,
            inputs=(FrameInput(0),),
            predictor_name="test",
            predictor_feasible=True,
            predictor_final_x=50,
            predictor_final_y=200,
            predictor_frames=1,
            emulator_validated=True,
            emulator_success=True,
            emulator_final_room=ROOM_CERES_FALLING,
        )
        assert fixture_success.room_clear
