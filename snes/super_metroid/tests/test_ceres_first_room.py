"""Tests for first Ceres room fixture (Elevator → Falling Tile).

Tests the real tape (not search) from routes.kpdr.ceres.outbound for the first
room of Ceres → Morph → Bomb sequence. Offline tests pass without ROM.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from super_metroid.emulator_validation import ROM_AVAILABLE
from super_metroid.routes.kpdr.ceres.first_room_fixture import (
    CeresFirstRoomFixture,
    button_names_to_mask,
    get_ceres_first_room_tape,
    validate_ceres_first_room,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
)


class TestButtonConversion:
    """Test button name to mask conversion."""

    def test_right_button(self) -> None:
        """RIGHT is bit 7 (0x80)."""
        mask = button_names_to_mask(("RIGHT",))
        assert mask == 0x80

    def test_a_button(self) -> None:
        """A is bit 8 (0x100)."""
        mask = button_names_to_mask(("A",))
        assert mask == 0x100

    def test_right_plus_a(self) -> None:
        """RIGHT+A is bits 7+8 (0x180)."""
        mask = button_names_to_mask(("RIGHT", "A"))
        assert mask == 0x180

    def test_empty_tuple(self) -> None:
        """Empty tuple gives idle (0)."""
        mask = button_names_to_mask(())
        assert mask == 0

    def test_case_insensitive(self) -> None:
        """Button names are case insensitive."""
        assert button_names_to_mask(("right",)) == button_names_to_mask(("RIGHT",))
        assert button_names_to_mask(("RiGhT",)) == 0x80


class TestCeresFirstRoomTape:
    """Test real tape extraction (offline, no ROM needed)."""

    def test_get_tape_returns_fixture(self) -> None:
        """get_ceres_first_room_tape returns a CeresFirstRoomFixture."""
        fixture = get_ceres_first_room_tape()

        assert isinstance(fixture, CeresFirstRoomFixture)
        assert fixture.from_room_id == ROOM_CERES_ELEVATOR
        assert fixture.to_room_id == ROOM_CERES_FALLING

    def test_tape_has_real_source(self) -> None:
        """Tape source is documented (not greedy search)."""
        fixture = get_ceres_first_room_tape()

        assert "outbound" in fixture.tape_source.lower()
        assert "greedy" not in fixture.tape_source.lower()
        assert "search" not in fixture.tape_source.lower()

    def test_tape_produces_input_sequence(self) -> None:
        """Tape has non-empty input sequence."""
        fixture = get_ceres_first_room_tape()

        assert len(fixture.inputs) > 0
        assert all(hasattr(inp, "buttons") for inp in fixture.inputs)

    def test_tape_starts_with_right_a(self) -> None:
        """Tape starts with RIGHT+A (jump start) for 24 frames."""
        fixture = get_ceres_first_room_tape()

        # First 24 frames should be RIGHT+A = 0x180
        assert fixture.inputs[0].buttons == 0x180
        assert all(inp.buttons == 0x180 for inp in fixture.inputs[:24])

    def test_tape_exact_frame_count(self) -> None:
        """Tape has EXACT frame count matching product spans.

        Raw spans: RIGHT+A 24, RIGHT 120, LEFT 120, RIGHT+B 240 (arm-pump), idle 60
        Total: 564 frames
        """
        fixture = get_ceres_first_room_tape()

        assert fixture.frames == 564
        assert len(fixture.inputs) == 564

    def test_tape_arm_pump_expansion(self) -> None:
        """RIGHT+B 240 frames expands with L↔R arm-pump (period=2).

        Frames 264-503 (240 frames after RIGHT+A 24, RIGHT 120, LEFT 120):
        - Frames 264-265: RIGHT+B+L (0x481)
        - Frames 266-267: RIGHT+B+R (0x881)
        - Frames 268-269: RIGHT+B+L (0x481)
        - ... (120 spans of 2 frames each)
        """
        fixture = get_ceres_first_room_tape()

        # RIGHT+B section starts at frame 264 (after 24+120+120)
        arm_pump_start = 24 + 120 + 120
        arm_pump_end = arm_pump_start + 240

        # Check first few arm-pump spans
        # RIGHT+B+L: RIGHT (0x80) + B (0x01) + L (0x400) = 0x481
        # RIGHT+B+R: RIGHT (0x80) + B (0x01) + R (0x800) = 0x881
        assert fixture.inputs[arm_pump_start].buttons == 0x481  # L
        assert fixture.inputs[arm_pump_start + 1].buttons == 0x481  # L
        assert fixture.inputs[arm_pump_start + 2].buttons == 0x881  # R
        assert fixture.inputs[arm_pump_start + 3].buttons == 0x881  # R
        assert fixture.inputs[arm_pump_start + 4].buttons == 0x481  # L
        assert fixture.inputs[arm_pump_start + 5].buttons == 0x481  # L

        # Check arm-pump section length
        arm_pump_frames = fixture.inputs[arm_pump_start:arm_pump_end]
        assert len(arm_pump_frames) == 240

    def test_tape_ends_with_idle(self) -> None:
        """Tape ends with 60 frames of idle (all buttons released)."""
        fixture = get_ceres_first_room_tape()

        # Last 60 frames should be idle (0x000)
        idle_start = 24 + 120 + 120 + 240
        assert all(inp.buttons == 0 for inp in fixture.inputs[idle_start:])

    def test_tape_not_emulator_validated(self) -> None:
        """Tape does not claim emulator validation by default."""
        fixture = get_ceres_first_room_tape()

        assert not fixture.emulator_validated
        assert not fixture.emulator_success
        assert fixture.emulator_final_room is None

    def test_tape_not_room_clear_without_emu(self) -> None:
        """Tape never claims room_clear without emulator validation."""
        fixture = get_ceres_first_room_tape()

        assert not fixture.room_clear

    def test_fixture_frames_property(self) -> None:
        """Fixture has frames property (tape length)."""
        fixture = get_ceres_first_room_tape()

        assert fixture.frames == len(fixture.inputs)
        assert fixture.frames > 0

    def test_fixture_to_dict_serializable(self) -> None:
        """Fixture can be serialized to dict (smedit-tas-1 compatible)."""
        fixture = get_ceres_first_room_tape()
        data = fixture.to_dict()

        assert isinstance(data, dict)
        assert "from_room_id" in data
        assert "inputs" in data
        assert isinstance(data["inputs"], list)
        assert all("buttons" in inp for inp in data["inputs"])


class TestCeresFirstRoomValidation:
    """Test emulator validation (requires ROM + start state)."""

    def _has_start_state(self) -> bool:
        """Check if SM_CERES_ELEV_STATE env var is set and file exists."""
        state_path = os.environ.get("SM_CERES_ELEV_STATE")
        return state_path is not None and Path(state_path).exists()

    def test_validate_requires_rom_or_start_state(self) -> None:
        """Validate raises if ROM or start state unavailable."""
        fixture = get_ceres_first_room_tape()

        if not ROM_AVAILABLE or not self._has_start_state():
            # Should raise when prerequisites missing
            with pytest.raises((FileNotFoundError, ValueError)):
                validate_ceres_first_room(fixture)

    @pytest.mark.skipif(
        not ROM_AVAILABLE,
        reason="ROM not available (emulator validation skipped)",
    )
    def test_validate_with_env_var(self) -> None:
        """Validate uses SM_CERES_ELEV_STATE env var if set.

        This test will pass only if ROM and start state are available.
        """
        if not self._has_start_state():
            pytest.skip("SM_CERES_ELEV_STATE not set or file missing")

        fixture = get_ceres_first_room_tape()
        validated = validate_ceres_first_room(fixture)

        assert validated.emulator_validated
        # Don't assert success/room_clear here - that depends on tape quality


class TestCeresFirstRoomFixture:
    """Test CeresFirstRoomFixture data structure."""

    def test_room_clear_requires_emulator_success(self) -> None:
        """room_clear is False unless emulator_validated and emulator_success."""
        from super_metroid.physics_sim import FrameInput

        # Not validated
        fixture = CeresFirstRoomFixture(
            from_room_id=ROOM_CERES_ELEVATOR,
            to_room_id=ROOM_CERES_FALLING,
            inputs=(FrameInput(0),),
            tape_source="test",
            emulator_validated=False,
        )
        assert not fixture.room_clear

        # Validated but failed
        fixture_failed = CeresFirstRoomFixture(
            from_room_id=ROOM_CERES_ELEVATOR,
            to_room_id=ROOM_CERES_FALLING,
            inputs=(FrameInput(0),),
            tape_source="test",
            emulator_validated=True,
            emulator_success=False,
        )
        assert not fixture_failed.room_clear

        # Validated and succeeded
        fixture_success = CeresFirstRoomFixture(
            from_room_id=ROOM_CERES_ELEVATOR,
            to_room_id=ROOM_CERES_FALLING,
            inputs=(FrameInput(0),),
            tape_source="test",
            emulator_validated=True,
            emulator_success=True,
            emulator_final_room=ROOM_CERES_FALLING,
        )
        assert fixture_success.room_clear
