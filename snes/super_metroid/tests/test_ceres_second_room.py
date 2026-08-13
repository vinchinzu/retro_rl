"""Tests for second Ceres room fixture (Falling Tile → Magnet Stairs).

Tests the real tape (not search) from routes.kpdr.ceres.outbound for the second
room of Ceres → Morph → Bomb sequence. Offline tests pass without ROM.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from super_metroid.emulator_validation import ROM_AVAILABLE
from super_metroid.routes.kpdr.ceres.arm_pump import _arm_pump_dash_spans
from super_metroid.routes.kpdr.ceres.second_room_fixture import (
    CeresSecondRoomFixture,
    button_names_to_mask,
    get_ceres_second_room_tape,
    validate_ceres_second_room,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_FALLING,
    ROOM_CERES_MAGNET,
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

    def test_right_plus_b(self) -> None:
        """RIGHT+B is bits 7+0 (0x81)."""
        mask = button_names_to_mask(("RIGHT", "B"))
        assert mask == 0x81

    def test_right_plus_b_plus_a(self) -> None:
        """RIGHT+B+A is bits 7+0+8 (0x181)."""
        mask = button_names_to_mask(("RIGHT", "B", "A"))
        assert mask == 0x181

    def test_left_button(self) -> None:
        """LEFT is bit 6 (0x40)."""
        mask = button_names_to_mask(("LEFT",))
        assert mask == 0x40

    def test_empty_tuple(self) -> None:
        """Empty tuple gives idle (0)."""
        mask = button_names_to_mask(())
        assert mask == 0

    def test_case_insensitive(self) -> None:
        """Button names are case insensitive."""
        assert button_names_to_mask(("right",)) == button_names_to_mask(("RIGHT",))
        assert button_names_to_mask(("RiGhT",)) == 0x80


class TestCeresSecondRoomTape:
    """Test real tape extraction (offline, no ROM needed)."""

    def test_get_tape_returns_fixture(self) -> None:
        """get_ceres_second_room_tape returns a CeresSecondRoomFixture."""
        fixture = get_ceres_second_room_tape()

        assert isinstance(fixture, CeresSecondRoomFixture)
        assert fixture.from_room_id == ROOM_CERES_FALLING
        assert fixture.to_room_id == ROOM_CERES_MAGNET

    def test_tape_has_real_source(self) -> None:
        """Tape source is documented (not greedy search)."""
        fixture = get_ceres_second_room_tape()

        assert "outbound" in fixture.tape_source.lower()
        assert "greedy" not in fixture.tape_source.lower()
        assert "search" not in fixture.tape_source.lower()

    def test_tape_produces_input_sequence(self) -> None:
        """Tape has non-empty input sequence."""
        fixture = get_ceres_second_room_tape()

        assert len(fixture.inputs) > 0
        assert all(hasattr(inp, "buttons") for inp in fixture.inputs)

    def test_tape_starts_with_right(self) -> None:
        """Tape starts with RIGHT for 24 frames."""
        fixture = get_ceres_second_room_tape()

        # First 24 frames should be RIGHT = 0x80
        assert fixture.inputs[0].buttons == 0x80
        assert all(inp.buttons == 0x80 for inp in fixture.inputs[:24])

    def test_tape_exact_frame_count(self) -> None:
        """Tape has EXACT frame count matching product spans.

        Raw spans:
        RIGHT 24, RIGHT+B 24 (arm-pump), RIGHT+B+A 24, RIGHT+A 24,
        RIGHT 24×4=96, RIGHT+B 24 (arm-pump), idle 12, RIGHT 24,
        idle 140, RIGHT 160, LEFT 120, RIGHT+B 96 (arm-pump), idle 24

        Total: 24 + 24 + 24 + 24 + 96 + 24 + 12 + 24 + 140 + 160 + 120 + 96 + 24 = 792 frames
        """
        fixture = get_ceres_second_room_tape()

        assert fixture.frames == 792
        assert len(fixture.inputs) == 792

    def test_tape_arm_pump_first_span_matches_helper(self) -> None:
        """First RIGHT+B 24 frames (span 2) matches _arm_pump_dash_spans output.

        Frames 24-47 (24 frames after initial RIGHT 24) must match the
        ActionSpans returned by _arm_pump_dash_spans("RIGHT", 24, ...).
        """
        fixture = get_ceres_second_room_tape()

        # Get expected arm-pump expansion from the actual helper
        arm_pump_spans = _arm_pump_dash_spans("RIGHT", 24, "test")

        # First RIGHT+B section starts at frame 24 (after initial RIGHT 24)
        arm_pump_start = 24
        arm_pump_end = arm_pump_start + 24

        # Verify tape matches helper output
        tape_idx = arm_pump_start
        for span in arm_pump_spans:
            expected_mask = button_names_to_mask(span.names)
            for _ in range(span.frames):
                assert (
                    fixture.inputs[tape_idx].buttons == expected_mask
                ), f"Frame {tape_idx} mismatch: got 0x{fixture.inputs[tape_idx].buttons:03X}, expected 0x{expected_mask:03X} for {span.names}"
                tape_idx += 1

        # Verify we consumed exactly 24 frames
        assert tape_idx == arm_pump_end

    def test_tape_arm_pump_second_span_matches_helper(self) -> None:
        """Second RIGHT+B 24 frames matches _arm_pump_dash_spans output.

        This is span 9 at offset 24+24+24+24+96=192 frames.
        """
        fixture = get_ceres_second_room_tape()

        # Get expected arm-pump expansion
        arm_pump_spans = _arm_pump_dash_spans("RIGHT", 24, "test")

        # Second RIGHT+B section starts at frame 192
        arm_pump_start = 192
        arm_pump_end = arm_pump_start + 24

        # Verify tape matches helper output
        tape_idx = arm_pump_start
        for span in arm_pump_spans:
            expected_mask = button_names_to_mask(span.names)
            for _ in range(span.frames):
                assert (
                    fixture.inputs[tape_idx].buttons == expected_mask
                ), f"Frame {tape_idx} mismatch: got 0x{fixture.inputs[tape_idx].buttons:03X}, expected 0x{expected_mask:03X} for {span.names}"
                tape_idx += 1

        assert tape_idx == arm_pump_end

    def test_tape_arm_pump_third_span_matches_helper(self) -> None:
        """Third RIGHT+B 96 frames matches _arm_pump_dash_spans output.

        This is the final arm-pump span (span 15) at offset
        24+24+24+24+96+24+12+24+140+160+120=672 frames.
        """
        fixture = get_ceres_second_room_tape()

        # Get expected arm-pump expansion for 96 frames
        arm_pump_spans = _arm_pump_dash_spans("RIGHT", 96, "test")

        # Third RIGHT+B section starts at frame 672
        arm_pump_start = 672
        arm_pump_end = arm_pump_start + 96

        # Verify tape matches helper output
        tape_idx = arm_pump_start
        for span in arm_pump_spans:
            expected_mask = button_names_to_mask(span.names)
            for _ in range(span.frames):
                assert (
                    fixture.inputs[tape_idx].buttons == expected_mask
                ), f"Frame {tape_idx} mismatch: got 0x{fixture.inputs[tape_idx].buttons:03X}, expected 0x{expected_mask:03X} for {span.names}"
                tape_idx += 1

        # Verify we consumed exactly 96 frames
        assert tape_idx == arm_pump_end

        # Check arm-pump section length
        arm_pump_frames = fixture.inputs[arm_pump_start:arm_pump_end]
        assert len(arm_pump_frames) == 96

    def test_tape_right_b_a_span(self) -> None:
        """Tape contains RIGHT+B+A for 24 frames (span 3)."""
        fixture = get_ceres_second_room_tape()

        # RIGHT+B+A span starts at frame 48 (after RIGHT 24, RIGHT+B 24)
        # RIGHT+B+A = 0x181
        span_start = 48
        span_end = span_start + 24

        assert all(
            inp.buttons == 0x181 for inp in fixture.inputs[span_start:span_end]
        )

    def test_tape_left_span(self) -> None:
        """Tape contains LEFT for 120 frames (span 14)."""
        fixture = get_ceres_second_room_tape()

        # LEFT span starts at frame 552 (24+24+24+24+96+24+12+24+140+160)
        # LEFT = 0x40
        span_start = 552
        span_end = span_start + 120

        assert all(inp.buttons == 0x40 for inp in fixture.inputs[span_start:span_end])

    def test_tape_ends_with_idle(self) -> None:
        """Tape ends with 24 frames of idle (all buttons released)."""
        fixture = get_ceres_second_room_tape()

        # Last 24 frames should be idle (0x000)
        idle_start = 768  # 792 - 24
        assert all(inp.buttons == 0 for inp in fixture.inputs[idle_start:])

    def test_tape_not_emulator_validated(self) -> None:
        """Tape does not claim emulator validation by default."""
        fixture = get_ceres_second_room_tape()

        assert not fixture.emulator_validated
        assert not fixture.emulator_success
        assert fixture.emulator_final_room is None

    def test_tape_not_room_clear_without_emu(self) -> None:
        """Tape never claims room_clear without emulator validation."""
        fixture = get_ceres_second_room_tape()

        assert not fixture.room_clear

    def test_fixture_frames_property(self) -> None:
        """Fixture has frames property (tape length)."""
        fixture = get_ceres_second_room_tape()

        assert fixture.frames == len(fixture.inputs)
        assert fixture.frames > 0

    def test_fixture_to_dict_serializable(self) -> None:
        """Fixture can be serialized to dict (smedit-tas-1 compatible)."""
        fixture = get_ceres_second_room_tape()
        data = fixture.to_dict()

        assert isinstance(data, dict)
        assert "from_room_id" in data
        assert "inputs" in data
        assert isinstance(data["inputs"], list)
        assert all("buttons" in inp for inp in data["inputs"])


class TestCeresSecondRoomValidation:
    """Test emulator validation (requires ROM + start state)."""

    def _has_start_state(self) -> bool:
        """Check if SM_CERES_FALLING_STATE env var is set and file exists."""
        state_path = os.environ.get("SM_CERES_FALLING_STATE")
        return state_path is not None and Path(state_path).exists()

    def test_validate_requires_rom_or_start_state(self) -> None:
        """Validate raises if ROM or start state unavailable."""
        fixture = get_ceres_second_room_tape()

        if not ROM_AVAILABLE or not self._has_start_state():
            # Should raise when prerequisites missing
            with pytest.raises((FileNotFoundError, ValueError)):
                validate_ceres_second_room(fixture)

    @pytest.mark.skipif(
        not ROM_AVAILABLE,
        reason="ROM not available (emulator validation skipped)",
    )
    def test_validate_with_env_var(self) -> None:
        """Validate uses SM_CERES_FALLING_STATE env var if set.

        This test will pass only if ROM and start state are available.
        """
        if not self._has_start_state():
            pytest.skip("SM_CERES_FALLING_STATE not set or file missing")

        fixture = get_ceres_second_room_tape()
        validated = validate_ceres_second_room(fixture)

        assert validated.emulator_validated


class TestCeresSecondRoomFixture:
    """Test CeresSecondRoomFixture data structure."""

    def test_room_clear_requires_emulator_success(self) -> None:
        """room_clear is False unless emulator_validated and emulator_success."""
        from super_metroid.physics_sim import FrameInput

        # Not validated
        fixture = CeresSecondRoomFixture(
            from_room_id=ROOM_CERES_FALLING,
            to_room_id=ROOM_CERES_MAGNET,
            inputs=(FrameInput(0),),
            tape_source="test",
            emulator_validated=False,
        )
        assert not fixture.room_clear

        # Validated but failed
        fixture_failed = CeresSecondRoomFixture(
            from_room_id=ROOM_CERES_FALLING,
            to_room_id=ROOM_CERES_MAGNET,
            inputs=(FrameInput(0),),
            tape_source="test",
            emulator_validated=True,
            emulator_success=False,
        )
        assert not fixture_failed.room_clear

        # Validated and succeeded
        fixture_success = CeresSecondRoomFixture(
            from_room_id=ROOM_CERES_FALLING,
            to_room_id=ROOM_CERES_MAGNET,
            inputs=(FrameInput(0),),
            tape_source="test",
            emulator_validated=True,
            emulator_success=True,
            emulator_final_room=ROOM_CERES_MAGNET,
        )
        assert fixture_success.room_clear
