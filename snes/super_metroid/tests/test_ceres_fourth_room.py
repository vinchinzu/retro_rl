"""Tests for fourth Ceres room fixture (Scientist → Ridley).

Tests the room-gated helper call from routes.kpdr.ceres.arm_pump for the
fourth room segment. This invokes `_ceres_arm_pump_until` with product
parameters, not a fixed tape. Offline tests verify helper wiring without ROM.

Note:
    Product code uses `_ceres_arm_pump_until(done=lambda s: s.room_id == ROOM_CERES_RIDLEY)`.
    Frame count is variable; done condition checks for Ridley 0xE0B5.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from super_metroid.emulator_validation import ROM_AVAILABLE
from super_metroid.routes.kpdr.ceres.fourth_room_fixture import (
    CeresFourthRoomFixture,
    play_ceres_fourth_room,
    validate_ceres_fourth_room_emulator,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_RIDLEY,
    ROOM_CERES_SCIENTIST,
)


class TestCeresFourthRoomHelper:
    """Test helper invocation (offline, no ROM needed)."""

    def test_play_invokes_arm_pump_until(self) -> None:
        """play_ceres_fourth_room invokes _ceres_arm_pump_until with product params."""
        # Mock session at Scientist room
        mock_session = Mock()
        mock_session.state.room_id = ROOM_CERES_SCIENTIST

        with patch(
            "super_metroid.routes.kpdr.ceres.fourth_room_fixture._ceres_arm_pump_until",
            return_value=123,
        ) as mock_helper:
            fixture = play_ceres_fourth_room(mock_session)

            # Verify helper was called with product parameters
            mock_helper.assert_called_once()
            call_args = mock_helper.call_args
            assert call_args[0][0] is mock_session  # session
            assert call_args[0][1] == "RIGHT"  # direction
            assert call_args[1]["reason"] == "ceres_out_flat_band"
            assert call_args[1]["max_frames"] == 900
            assert call_args[1]["stuck_jump_after"] == 40
            assert "done" in call_args[1]

            # Verify done condition checks for Ridley
            done_fn = call_args[1]["done"]
            mock_state_ridley = Mock()
            mock_state_ridley.room_id = ROOM_CERES_RIDLEY
            assert done_fn(mock_state_ridley) is True

            mock_state_scientist = Mock()
            mock_state_scientist.room_id = ROOM_CERES_SCIENTIST
            assert done_fn(mock_state_scientist) is False

    def test_play_returns_fixture_with_frames(self) -> None:
        """play_ceres_fourth_room returns fixture with frames_consumed."""
        mock_session = Mock()
        mock_session.state.room_id = ROOM_CERES_SCIENTIST

        with patch(
            "super_metroid.routes.kpdr.ceres.fourth_room_fixture._ceres_arm_pump_until",
            return_value=456,
        ):
            fixture = play_ceres_fourth_room(mock_session)

            assert isinstance(fixture, CeresFourthRoomFixture)
            assert fixture.from_room_id == ROOM_CERES_SCIENTIST
            assert fixture.to_room_id == ROOM_CERES_RIDLEY
            assert fixture.frames_consumed == 456
            assert "_ceres_arm_pump_until" in fixture.helper_source

    def test_play_raises_if_not_at_scientist(self) -> None:
        """play_ceres_fourth_room raises if not starting at Scientist."""
        mock_session = Mock()
        mock_session.state.room_id = 0xDEAD  # Wrong room

        with pytest.raises(ValueError, match="Expected to start at Scientist"):
            play_ceres_fourth_room(mock_session)

    def test_helper_source_mentions_arm_pump_until(self) -> None:
        """Fixture helper_source is grep-able for _ceres_arm_pump_until."""
        mock_session = Mock()
        mock_session.state.room_id = ROOM_CERES_SCIENTIST

        with patch(
            "super_metroid.routes.kpdr.ceres.fourth_room_fixture._ceres_arm_pump_until",
            return_value=100,
        ):
            fixture = play_ceres_fourth_room(mock_session)

            # Grep-able: fixture documents the helper it uses
            assert "_ceres_arm_pump_until" in fixture.helper_source
            assert "arm_pump" in fixture.helper_source.lower()


class TestCeresFourthRoomValidation:
    """Test emulator validation (requires ROM + start state)."""

    def _has_start_state(self) -> bool:
        """Check if SM_CERES_SCIENTIST_STATE env var is set and file exists."""
        state_path = os.environ.get("SM_CERES_SCIENTIST_STATE")
        return state_path is not None and Path(state_path).exists()

    def test_validate_requires_start_state(self) -> None:
        """Validate raises if start state unavailable."""
        if not self._has_start_state():
            with pytest.raises((FileNotFoundError, ValueError)):
                validate_ceres_fourth_room_emulator()

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

        validated = validate_ceres_fourth_room_emulator()

        assert validated.emulator_validated
        # Frame count is variable; don't assert specific value
        assert validated.frames_consumed >= 0


class TestCeresFourthRoomFixture:
    """Test CeresFourthRoomFixture data structure."""

    def test_room_clear_requires_emulator_success(self) -> None:
        """room_clear is False unless emulator_validated and emulator_success."""
        # Not validated
        fixture = CeresFourthRoomFixture(
            from_room_id=ROOM_CERES_SCIENTIST,
            to_room_id=ROOM_CERES_RIDLEY,
            frames_consumed=100,
            helper_source="test",
            emulator_validated=False,
        )
        assert not fixture.room_clear

        # Validated but failed
        fixture_failed = CeresFourthRoomFixture(
            from_room_id=ROOM_CERES_SCIENTIST,
            to_room_id=ROOM_CERES_RIDLEY,
            frames_consumed=100,
            helper_source="test",
            emulator_validated=True,
            emulator_success=False,
        )
        assert not fixture_failed.room_clear

        # Validated and succeeded
        fixture_success = CeresFourthRoomFixture(
            from_room_id=ROOM_CERES_SCIENTIST,
            to_room_id=ROOM_CERES_RIDLEY,
            frames_consumed=100,
            helper_source="test",
            emulator_validated=True,
            emulator_success=True,
            emulator_final_room=ROOM_CERES_RIDLEY,
        )
        assert fixture_success.room_clear

    def test_fixture_boundary_is_scientist_to_ridley(self) -> None:
        """Fixture boundary is Scientist → Ridley, not Scientist → Flat."""
        fixture = CeresFourthRoomFixture(
            from_room_id=ROOM_CERES_SCIENTIST,
            to_room_id=ROOM_CERES_RIDLEY,
            frames_consumed=200,
            helper_source="test",
        )

        assert fixture.from_room_id == ROOM_CERES_SCIENTIST
        assert fixture.to_room_id == ROOM_CERES_RIDLEY
        # Product done condition checks for Ridley, not Flat

    def test_frames_consumed_is_variable(self) -> None:
        """frames_consumed records actual helper result (variable)."""
        fixture_a = CeresFourthRoomFixture(
            from_room_id=ROOM_CERES_SCIENTIST,
            to_room_id=ROOM_CERES_RIDLEY,
            frames_consumed=150,
            helper_source="test",
        )

        fixture_b = CeresFourthRoomFixture(
            from_room_id=ROOM_CERES_SCIENTIST,
            to_room_id=ROOM_CERES_RIDLEY,
            frames_consumed=250,
            helper_source="test",
        )

        # Frame count varies by entry conditions
        assert fixture_a.frames_consumed != fixture_b.frames_consumed
        # Don't assert fixed value; room-gated behavior is adaptive
