"""Tests for emulator validation adapter."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from super_metroid.emulator_validation import (
    ROM_AVAILABLE,
    _buttons_mask_to_action,
    observation_from_env,
    validate_trajectory_on_emulator,
)
from super_metroid.observation import Observation
from super_metroid.physics_sim import FrameInput
from super_metroid.ram import SuperMetroidState, GameplayPhase


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

    def test_observation_from_mocked_parse_env_state(self, monkeypatch) -> None:
        """Test observation_from_env calls parse_env_state and maps fields."""
        # Create a mock SuperMetroidState
        mock_state = SuperMetroidState(
            frame=100,
            game_state=8,
            phase=GameplayPhase.ORDINARY_GAMEPLAY,
            room_id=0x9191,
            area_index=0,
            door_transition=0,
            transition_direction=0,
            samus_x=336,
            samus_y=160,
            velocity_x=0,
            velocity_y=0,
            pose=1,
            health=99,  # This should map to energy
            max_health=99,
            reserve_health=0,
            max_reserve_health=0,
            missiles=0,
            max_missiles=0,
            super_missiles=0,
            max_super_missiles=0,
            power_bombs=0,
            max_power_bombs=0,
            selected_item=0,
            equipped_items=0,
            collected_items=0,
            equipped_beams=0,
            collected_beams=0,
            timer_type=0,
            escape_timer_frames=0,
            escape_timer_seconds=0,
            escape_timer_minutes=0,
            num_enemies=0,
            enemies_killed=0,
            enemy0_x=0,
            enemy0_y=0,
            enemy0_hp=32,  # This should map to enemy_energy
            enemy0_spritemap=0,
            event_flags=(0, 0, 0, 0, 0, 0, 0, 0),
            boss_bits=(0, 0, 0, 0, 0, 0, 0, 0),
            samus_x_sub=0x80,
            samus_y_sub=0x40,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            speed_counter=0,
            speed_flag=0,
        )

        # Mock parse_env_state to return our mock state
        def mock_parse_env_state(env, mode="full"):
            return mock_state

        import super_metroid.emulator_validation as emu_val
        monkeypatch.setattr(emu_val, "parse_env_state", mock_parse_env_state)

        # Create a fake env (doesn't matter what it is since parse_env_state is mocked)
        fake_env = Mock()

        # Call observation_from_env
        obs = observation_from_env(fake_env)

        # Verify Oπ fields
        assert obs.x == 336
        assert obs.y == 160
        assert obs.pose == 1
        assert obs.room == 0x9191

        # Verify Oσ fields
        assert obs.sub_x == 0x80
        assert obs.sub_y == 0x40

        # Verify O† field: energy comes from state.health
        assert obs.energy == 99

        # Verify Oσ+: enemy_energy comes from state.enemy0_hp
        assert obs.enemy_energy == 32

        # Verify unobserved fields are None
        assert obs.frame_counter_1 is None
        assert obs.frame_counter_2 is None
        assert obs.invulnerability_timer is None


class TestEmulatorValidationIntegration:
    """Integration tests requiring ROM (skipped without ROM)."""

    def test_no_rom_raises_file_not_found(self, monkeypatch, tmp_path) -> None:
        """Test that validation raises FileNotFoundError when ROM unavailable."""
        # Mock ROM_AVAILABLE to False
        import super_metroid.emulator_validation as emu_val
        monkeypatch.setattr(emu_val, "ROM_AVAILABLE", False)

        # Create a dummy state file
        state_file = tmp_path / "test.state"
        state_file.write_bytes(b"dummy")

        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="ROM not available"):
            validate_trajectory_on_emulator(
                state_file,
                [FrameInput(buttons=0)],
            )

    @pytest.mark.skipif(
        not ROM_AVAILABLE,
        reason="Requires ROM and state file setup",
    )
    def test_validate_trajectory_basic(self) -> None:
        """Test basic trajectory validation (placeholder for ROM tests)."""
        # This would test the full validate_trajectory_on_emulator flow
        # Skipped because it requires:
        # 1. ROM in roms/ directory
        # 2. Valid .state file
        # 3. Full stable-retro setup
        pass
