"""Tests for fourth Ceres room fixture (Scientist → Flat).

Tests the room-gated arm-pump pattern from routes.kpdr.ceres.outbound for the
fourth room of Ceres → Morph → Bomb sequence. Offline tests pass without ROM.

Note:
    Product code treats Scientist→Flat→Ridley as one continuous arm-pump.
    This fixture creates an artificial stop at Flat for modular testing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from super_metroid.emulator_validation import ROM_AVAILABLE
from super_metroid.routes.kpdr.ceres.arm_pump import _arm_pump_dash_spans
from super_metroid.routes.kpdr.ceres.fourth_room_fixture import (
    CeresFourthRoomFixture,
    button_names_to_mask,
    get_ceres_fourth_room_tape,
    validate_ceres_fourth_room,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_FLAT,
    ROOM_CERES_SCIENTIST,
)


class TestButtonConversion:
    """Test button name to mask conversion."""

    def test_right_button(self) -> None:
        """RIGHT is bit 7 (0x80)."""
        mask = button_names_to_mask(("RIGHT",))
        assert mask == 0x80

    def test_b_button(self) -> None:
        """B is bit 0 (0x01)."""
        mask = button_names_to_mask(("B",))
        assert mask == 0x01

    def test_l_button(self) -> None:
        """L is bit 10 (0x400)."""
        mask = button_names_to_mask(("L",))
        assert mask == 0x400

    def test_r_button(self) -> None:
        """R is bit 11 (0x800)."""
        mask = button_names_to_mask(("R",))
        assert mask == 0x800

    def test_right_plus_b(self) -> None:
        """RIGHT+B is bits 7+0 (0x81)."""
        mask = button_names_to_mask(("RIGHT", "B"))
        assert mask == 0x81

    def test_right_plus_b_plus_l(self) -> None:
        """RIGHT+B+L is bits 7+0+10 (0x481)."""
        mask = button_names_to_mask(("RIGHT", "B", "L"))
        assert mask == 0x481

    def test_right_plus_b_plus_r(self) -> None:
        """RIGHT+B+R is bits 7+0+11 (0x881)."""
        mask = button_names_to_mask(("RIGHT", "B", "R"))
        assert mask == 0x881

    def test_empty_tuple(self) -> None:
        """Empty tuple gives idle (0)."""
        mask = button_names_to_mask(())
        assert mask == 0

    def test_case_insensitive(self) -> None:
        """Button names are case insensitive."""
        assert button_names_to_mask(("right",)) == button_names_to_mask(
            ("RIGHT",)
        )
        assert button_names_to_mask(("RiGhT",)) == 0x80


class TestCeresFourthRoomTape:
    """Test tape generation (offline, no ROM needed)."""

    def test_get_tape_returns_fixture(self) -> None:
        """get_ceres_fourth_room_tape returns a CeresFourthRoomFixture."""
        fixture = get_ceres_fourth_room_tape()

        assert isinstance(fixture, CeresFourthRoomFixture)
        assert fixture.from_room_id == ROOM_CERES_SCIENTIST
        assert fixture.to_room_id == ROOM_CERES_FLAT

    def test_tape_source_documents_room_gated(self) -> None:
        """Tape source documents room-gated behavior (not fixed tape)."""
        fixture = get_ceres_fourth_room_tape()

        source_lower = fixture.tape_source.lower()
        assert "room-gated" in source_lower or "room gated" in source_lower
        assert "continuous" in source_lower
        # Should NOT claim to be extracted from fixed tape
        assert "exact" not in source_lower

    def test_tape_produces_input_sequence(self) -> None:
        """Tape has non-empty input sequence."""
        fixture = get_ceres_fourth_room_tape()

        assert len(fixture.inputs) > 0
        assert all(hasattr(inp, "buttons") for inp in fixture.inputs)

    def test_tape_estimated_frame_count(self) -> None:
        """Tape has estimated frame count (~300 frames).

        Product comment: Scientist→Flat→Ridley drops ~600f total.
        This fixture: ~300 frames for Scientist→Flat (roughly half).
        Actual frame count determined by emulator validation.
        """
        fixture = get_ceres_fourth_room_tape()

        # Estimated 300 frames (not exact; room-gated behavior)
        assert fixture.frames == 300
        assert len(fixture.inputs) == 300

    def test_tape_uses_arm_pump_pattern(self) -> None:
        """Tape uses classic L↔R arm-pump pattern (RIGHT+B with shoulders).

        All frames should match _arm_pump_dash_spans("RIGHT", 300, ...) output.
        """
        fixture = get_ceres_fourth_room_tape()

        # Get expected arm-pump expansion from the actual helper
        arm_pump_spans = _arm_pump_dash_spans("RIGHT", 300, "test")

        # Verify tape matches helper output frame-by-frame
        tape_idx = 0
        for span in arm_pump_spans:
            expected_mask = button_names_to_mask(span.names)
            for _ in range(span.frames):
                assert (
                    fixture.inputs[tape_idx].buttons == expected_mask
                ), f"Frame {tape_idx} mismatch: got 0x{fixture.inputs[tape_idx].buttons:03X}, expected 0x{expected_mask:03X} for {span.names}"
                tape_idx += 1

        # Verify we consumed all frames
        assert tape_idx == 300

    def test_tape_starts_with_right_b(self) -> None:
        """Tape starts with RIGHT+B arm-pump (should include L or R shoulder)."""
        fixture = get_ceres_fourth_room_tape()

        # First frame should be RIGHT+B + shoulder (L=0x400 or R=0x800)
        first_button = fixture.inputs[0].buttons
        # RIGHT (0x80) + B (0x01) = 0x81 base
        # Plus L (0x400) → 0x481 or R (0x800) → 0x881
        assert first_button in (0x481, 0x881), (
            f"First frame should be RIGHT+B+L (0x481) or RIGHT+B+R (0x881), "
            f"got 0x{first_button:03X}"
        )

    def test_tape_not_emulator_validated(self) -> None:
        """Tape does not claim emulator validation by default."""
        fixture = get_ceres_fourth_room_tape()

        assert not fixture.emulator_validated
        assert not fixture.emulator_success
        assert fixture.emulator_final_room is None

    def test_tape_not_room_clear_without_emu(self) -> None:
        """Tape never claims room_clear without emulator validation."""
        fixture = get_ceres_fourth_room_tape()

        assert not fixture.room_clear

    def test_fixture_frames_property(self) -> None:
        """Fixture has frames property (tape length)."""
        fixture = get_ceres_fourth_room_tape()

        assert fixture.frames == len(fixture.inputs)
        assert fixture.frames > 0

    def test_fixture_to_dict_serializable(self) -> None:
        """Fixture can be serialized to dict (smedit-tas-1 compatible)."""
        fixture = get_ceres_fourth_room_tape()
        data = fixture.to_dict()

        assert isinstance(data, dict)
        assert "from_room_id" in data
        assert "inputs" in data
        assert isinstance(data["inputs"], list)
        assert all("buttons" in inp for inp in data["inputs"])


class TestCeresFourthRoomValidation:
    """Test emulator validation (requires ROM + start state)."""

    def _has_start_state(self) -> bool:
        """Check if SM_CERES_SCIENTIST_STATE env var is set and file exists."""
        state_path = os.environ.get("SM_CERES_SCIENTIST_STATE")
        return state_path is not None and Path(state_path).exists()

    def test_validate_requires_rom_or_start_state(self) -> None:
        """Validate raises if ROM or start state unavailable."""
        fixture = get_ceres_fourth_room_tape()

        if not ROM_AVAILABLE or not self._has_start_state():
            # Should raise when prerequisites missing
            with pytest.raises((FileNotFoundError, ValueError)):
                validate_ceres_fourth_room(fixture)

    @pytest.mark.skipif(
        not ROM_AVAILABLE,
        reason="ROM not available (emulator validation skipped)",
    )
    def test_validate_with_env_var(self) -> None:
        """Validate uses SM_CERES_SCIENTIST_STATE env var if set.

        This test will pass only if ROM and start state are available.
        """
        if not self._has_start_state():
            pytest.skip("SM_CERES_SCIENTIST_STATE not set or file missing")

        fixture = get_ceres_fourth_room_tape()
        validated = validate_ceres_fourth_room(fixture)

        assert validated.emulator_validated


class TestCeresFourthRoomFixture:
    """Test CeresFourthRoomFixture data structure."""

    def test_room_clear_requires_emulator_success(self) -> None:
        """room_clear is False unless emulator_validated and emulator_success."""
        from super_metroid.physics_sim import FrameInput

        # Not validated
        fixture = CeresFourthRoomFixture(
            from_room_id=ROOM_CERES_SCIENTIST,
            to_room_id=ROOM_CERES_FLAT,
            inputs=(FrameInput(0),),
            tape_source="test",
            emulator_validated=False,
        )
        assert not fixture.room_clear

        # Validated but failed
        fixture_failed = CeresFourthRoomFixture(
            from_room_id=ROOM_CERES_SCIENTIST,
            to_room_id=ROOM_CERES_FLAT,
            inputs=(FrameInput(0),),
            tape_source="test",
            emulator_validated=True,
            emulator_success=False,
        )
        assert not fixture_failed.room_clear

        # Validated and succeeded
        fixture_success = CeresFourthRoomFixture(
            from_room_id=ROOM_CERES_SCIENTIST,
            to_room_id=ROOM_CERES_FLAT,
            inputs=(FrameInput(0),),
            tape_source="test",
            emulator_validated=True,
            emulator_success=True,
            emulator_final_room=ROOM_CERES_FLAT,
        )
        assert fixture_success.room_clear
