"""Tests for emulator validation adapter."""

from __future__ import annotations

import numpy as np
import pytest

from super_metroid.emulator_validation import (
    _buttons_mask_to_action,
    observation_from_env,
)
from super_metroid.observation import Observation
from super_metroid.physics_sim import FrameInput


class TestButtonConversion:
    """Test packed button mask to 12-length action vector conversion."""

    def test_idle_action(self) -> None:
        """Test all buttons released."""
        action = _buttons_mask_to_action(0)
        assert action == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def test_single_button_b(self) -> None:
        """Test B button (bit 0)."""
        action = _buttons_mask_to_action(1)
        assert action == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def test_single_button_right(self) -> None:
        """Test RIGHT button (bit 7, mask 0x80)."""
        action = _buttons_mask_to_action(0x80)
        assert action == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    def test_multiple_buttons_b_right(self) -> None:
        """Test B + RIGHT (bits 0 and 7, mask 0x81)."""
        action = _buttons_mask_to_action(0x81)
        assert action == [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    def test_multiple_buttons_left_b(self) -> None:
        """Test LEFT + B (bits 6 and 0, mask 0x41)."""
        action = _buttons_mask_to_action(0x41)
        assert action == [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

    def test_all_dpad_buttons(self) -> None:
        """Test all D-pad buttons (UP, DOWN, LEFT, RIGHT)."""
        # UP=bit4, DOWN=bit5, LEFT=bit6, RIGHT=bit7
        action = _buttons_mask_to_action(0xF0)
        assert action == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

    def test_frame_input_conversion(self) -> None:
        """Test conversion from FrameInput.buttons."""
        frame_input = FrameInput(buttons=0x80)  # RIGHT
        action = _buttons_mask_to_action(frame_input.buttons)
        assert action == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]


class TestObservationFromEnv:
    """Test observation extraction from emulator."""

    def test_observation_structure(self) -> None:
        """Test Observation structure and field access (offline test)."""
        # Create an Observation directly to test the structure
        obs = Observation(
            frame=100,
            x=336,
            y=160,
            pose=1,
            room=0x9191,
            sub_x=0x80,
            sub_y=0x40,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            speed_counter=0,
            speed_flag=0,
            energy=99,
            frame_counter_1=0x42,
            frame_counter_2=0x10,
            enemy_energy=0x20,
            invulnerability_timer=5,
        )

        # Test Oπ fields
        assert obs.x == 336
        assert obs.y == 160
        assert obs.pose == 1
        assert obs.room == 0x9191

        # Test Oσ fields
        assert obs.sub_x == 0x80
        assert obs.sub_y == 0x40

        # Test O† fields
        assert obs.energy == 99

        # Test lag detection fields
        assert obs.frame_counter_1 == 0x42
        assert obs.frame_counter_2 == 0x10

        # Test Oσ+ fields
        assert obs.enemy_energy == 0x20
        assert obs.invulnerability_timer == 5


class TestEmulatorValidationIntegration:
    """Integration tests requiring ROM (skipped without ROM)."""

    @pytest.mark.skipif(
        True,  # Always skip for now - requires ROM and state setup
        reason="Requires ROM and state file setup",
    )
    def test_validate_trajectory_basic(self) -> None:
        """Test basic trajectory validation (placeholder)."""
        # This would test the full validate_trajectory_on_emulator flow
        # Skipped because it requires:
        # 1. ROM in roms/ directory
        # 2. Valid .state file
        # 3. Full stable-retro setup
        pass
