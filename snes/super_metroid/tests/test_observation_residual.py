"""Tests for observation tuple and residual profiling.

Pure offline tests — no ROM required. Validates Bob's locked observation lattice:
- Oπ (coarsest): pixels x/y, pose, room
- Oσ: Oπ plus subpixels
- Oσ+: Oσ plus optional enemy energy / i-frames
- O† (separate): energy/death

fd_π is first pixel/pose/room disagreement, NOT "inputs diverge."
"""

from __future__ import annotations

import pytest

from super_metroid.observation import (
    Observation,
    observation_from_sim_state,
    observation_from_trajectory_frame,
)
from super_metroid.physics_sim import SimState, TrajectoryFrame
from super_metroid.residual import (
    DivergenceCause,
    ResidualProfile,
    compute_residual_profile,
)


class TestObservation:
    """Test Observation dataclass per Bob's locked lattice."""

    def test_create_minimal(self) -> None:
        obs = Observation(
            frame=100,
            x=400,
            y=200,
            pose=0,
            room=0x91F8,
            sub_x=0,
            sub_y=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            speed_counter=0,
            speed_flag=0,
            energy=100,
            frame_counter_1=1000,
            frame_counter_2=500,
        )
        # Oπ fields
        assert obs.x == 400
        assert obs.y == 200
        assert obs.pose == 0
        assert obs.room == 0x91F8
        # Oσ fields
        assert obs.sub_x == 0
        assert obs.sub_y == 0
        # O† fields
        assert obs.energy == 100

    def test_to_dict_omits_zero_optional_fields(self) -> None:
        obs = Observation(
            frame=100,
            x=400,
            y=200,
            pose=0,
            room=0x91F8,
            sub_x=0,
            sub_y=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            speed_counter=0,
            speed_flag=0,
            energy=100,
            frame_counter_1=1000,
            frame_counter_2=500,
        )
        data = obs.to_dict()
        # Oσ+ fields should be omitted when zero
        assert "enemy_energy" not in data
        assert "invulnerability_timer" not in data

    def test_to_dict_includes_present_optional_fields(self) -> None:
        obs = Observation(
            frame=100,
            x=400,
            y=200,
            pose=0,
            room=0x91F8,
            sub_x=0,
            sub_y=0,
            velocity_x=0,
            velocity_y=0,
            velocity_x_sub=0,
            velocity_y_sub=0,
            momentum_x=0,
            momentum_x_sub=0,
            speed_counter=0,
            speed_flag=0,
            energy=100,
            frame_counter_1=1000,
            frame_counter_2=500,
            enemy_energy=50,
            invulnerability_timer=60,
        )
        data = obs.to_dict()
        # Oσ+ fields should be present
        assert data["enemy_energy"] == 50
        assert data["invulnerability_timer"] == 60

    def test_from_dict_roundtrip(self) -> None:
        obs = Observation(
            frame=100,
            x=400,
            y=200,
            pose=1,
            room=0x91F8,
            sub_x=32768,
            sub_y=16384,
            velocity_x=2,
            velocity_y=-3,
            velocity_x_sub=1000,
            velocity_y_sub=2000,
            momentum_x=5,
            momentum_x_sub=8192,
            speed_counter=3,
            speed_flag=1,
            energy=99,
            frame_counter_1=5000,
            frame_counter_2=2500,
            enemy_energy=50,
            invulnerability_timer=60,
        )
        data = obs.to_dict()
        restored = Observation.from_dict(data)
        assert restored == obs

    def test_from_dict_with_missing_optional_fields(self) -> None:
        data = {
            "frame": 100,
            "x": 400,
            "y": 200,
            "pose": 0,
            "room": 0x91F8,
            "sub_x": 0,
            "sub_y": 0,
            "velocity_x": 0,
            "velocity_y": 0,
            "velocity_x_sub": 0,
            "velocity_y_sub": 0,
            "momentum_x": 0,
            "momentum_x_sub": 0,
            "speed_counter": 0,
            "speed_flag": 0,
            "energy": 100,
            "frame_counter_1": 1000,
            "frame_counter_2": 500,
        }
        obs = Observation.from_dict(data)
        assert obs.enemy_energy == 0
        assert obs.invulnerability_timer == 0

    def test_from_sim_state(self) -> None:
        state = SimState(
            frame=100,
            room_id=0x91F8,
            samus_x=400,
            samus_y=200,
            samus_x_sub=32768,
            samus_y_sub=16384,
            velocity_x=2,
            velocity_y=-3,
            velocity_x_sub=1000,
            velocity_y_sub=2000,
            momentum_x=5,
            momentum_x_sub=8192,
            pose=1,
            facing=8,
            movement_type=0,
            speed_counter=3,
            speed_flag=1,
            shinespark_timer=0,
        )
        obs = observation_from_sim_state(state)
        # Oπ fields
        assert obs.x == 400
        assert obs.y == 200
        assert obs.pose == 1
        assert obs.room == 0x91F8
        # Oσ fields
        assert obs.sub_x == 32768
        assert obs.sub_y == 16384
        # Speeds
        assert obs.velocity_x == 2
        assert obs.velocity_y == -3
        assert obs.speed_counter == 3
        assert obs.speed_flag == 1
        # SimState does not track O† or Oσ+
        assert obs.energy == 0
        assert obs.enemy_energy == 0
        assert obs.invulnerability_timer == 0

    def test_from_trajectory_frame(self) -> None:
        frame = TrajectoryFrame(
            frame=10,
            room_id=0x91F8,
            samus_x=450,
            samus_y=180,
            samus_x_sub=1000,
            samus_y_sub=2000,
            velocity_x=2,
            velocity_y=-1,
            velocity_x_sub=500,
            velocity_y_sub=1500,
            momentum_x=2,
            momentum_x_sub=100,
            pose=1,
            facing=8,
            movement_type=1,
            speed_counter=2,
            speed_flag=0,
            shinespark_timer=0,
        )
        obs = observation_from_trajectory_frame(frame)
        # Oπ fields
        assert obs.x == 450
        assert obs.y == 180
        assert obs.pose == 1
        assert obs.room == 0x91F8
        # Oσ fields
        assert obs.sub_x == 1000
        assert obs.sub_y == 2000
        assert obs.speed_counter == 2
        # TrajectoryFrame does not track O† or Oσ+
        assert obs.energy == 0
        assert obs.enemy_energy == 0


class TestResidualProfile:
    """Test ResidualProfile per Bob's locked lattice."""

    def test_create_unmeasured(self) -> None:
        profile = ResidualProfile.unmeasured_profile()
        assert profile.unmeasured is True
        assert profile.fd_pi is None
        assert profile.fd_sigma is None
        assert profile.fd_sigma_plus is None
        assert profile.fd_dagger is None

    def test_to_dict_roundtrip(self) -> None:
        profile = ResidualProfile(
            fd_pi=None,
            fd_sigma=10,
            fd_sigma_plus=10,
            fd_dagger=None,
            cause=DivergenceCause.COLLISION,
            first_diff_field="subpixels",
            unmeasured=False,
        )
        data = profile.to_dict()
        restored = ResidualProfile.from_dict(data)
        assert restored == profile

    def test_should_hard_reject_room_divergence(self) -> None:
        """Room ($079B) is part of Oπ — hard-reject on Oπ break."""
        profile = ResidualProfile(
            fd_pi=5,
            fd_sigma=5,
            fd_sigma_plus=5,
            fd_dagger=None,
            cause=DivergenceCause.ROOM,
            first_diff_field="room",
        )
        assert profile.should_hard_reject() is True

    def test_should_hard_reject_death_divergence(self) -> None:
        """O† (death) is separate — hard-reject on O† break."""
        profile = ResidualProfile(
            fd_pi=None,
            fd_sigma=None,
            fd_sigma_plus=None,
            fd_dagger=8,
            cause=None,
            first_diff_field=None,
        )
        assert profile.should_hard_reject() is True

    def test_should_not_hard_reject_collision(self) -> None:
        """Subpixel collision (Oσ break, Oπ holds) is not hard-reject."""
        profile = ResidualProfile(
            fd_pi=None,
            fd_sigma=10,
            fd_sigma_plus=10,
            fd_dagger=None,
            cause=DivergenceCause.COLLISION,
            first_diff_field="subpixels",
        )
        assert profile.should_hard_reject() is False

    def test_needs_emulator_spot_check(self) -> None:
        """Oσ broke, Oπ holds (subpixels diverge, pixels/pose/room agree) → spot-check."""
        profile = ResidualProfile(
            fd_pi=None,
            fd_sigma=10,
            fd_sigma_plus=10,
            fd_dagger=None,
            cause=DivergenceCause.COLLISION,
            first_diff_field="subpixels",
        )
        assert profile.needs_emulator_spot_check() is True

    def test_no_spot_check_when_hard_reject(self) -> None:
        profile = ResidualProfile(
            fd_pi=5,
            fd_sigma=5,
            fd_sigma_plus=5,
            fd_dagger=None,
            cause=DivergenceCause.ROOM,
            first_diff_field="room",
        )
        assert profile.needs_emulator_spot_check() is False

    def test_can_keep_as_search_model_when_pi_holds(self) -> None:
        """Oπ holds for horizon (pixels/pose/room agree) → keep as search model."""
        profile = ResidualProfile(
            fd_pi=None,
            fd_sigma=10,
            fd_sigma_plus=10,
            fd_dagger=None,
            cause=DivergenceCause.COLLISION,
            first_diff_field="subpixels",
        )
        assert profile.can_keep_as_search_model() is True

    def test_cannot_keep_as_search_model_when_hard_reject(self) -> None:
        profile = ResidualProfile(
            fd_pi=5,
            fd_sigma=5,
            fd_sigma_plus=5,
            fd_dagger=None,
            cause=DivergenceCause.ROOM,
            first_diff_field="room",
        )
        assert profile.can_keep_as_search_model() is False

    def test_tag_lag_desync(self) -> None:
        profile = ResidualProfile(
            fd_pi=None,
            fd_sigma=3,
            fd_sigma_plus=3,
            fd_dagger=None,
            cause=DivergenceCause.LAG,
            first_diff_field="frame_counter",
        )
        assert profile.tag_lag_desync() is True

    def test_no_lag_desync_on_collision(self) -> None:
        profile = ResidualProfile(
            fd_pi=None,
            fd_sigma=10,
            fd_sigma_plus=10,
            fd_dagger=None,
            cause=DivergenceCause.COLLISION,
            first_diff_field="subpixels",
        )
        assert profile.tag_lag_desync() is False


class TestComputeResidualProfile:
    """Test compute_residual_profile per Bob's locked lattice."""

    def test_unmeasured_when_emu_obs_is_none(self) -> None:
        """When emu_obs is None, return unmeasured profile."""
        mini_obs = [
            Observation(
                frame=i,
                x=100 + i,
                y=200,
                pose=0,
                room=0x91F8,
                sub_x=0,
                sub_y=0,
                velocity_x=1,
                velocity_y=0,
                velocity_x_sub=0,
                velocity_y_sub=0,
                momentum_x=1,
                momentum_x_sub=0,
                speed_counter=0,
                speed_flag=0,
                energy=99,
                frame_counter_1=i,
                frame_counter_2=i,
            )
            for i in range(10)
        ]
        profile = compute_residual_profile(mini_obs, None)
        assert profile.unmeasured is True
        assert profile.fd_pi is None

    def test_agree_for_horizon(self) -> None:
        """When Mini and Emu agree for horizon, fd fields are None."""
        obs_list = [
            Observation(
                frame=i,
                x=100 + i,
                y=200,
                pose=0,
                room=0x91F8,
                sub_x=0,
                sub_y=0,
                velocity_x=1,
                velocity_y=0,
                velocity_x_sub=0,
                velocity_y_sub=0,
                momentum_x=1,
                momentum_x_sub=0,
                speed_counter=0,
                speed_flag=0,
                energy=99,
                frame_counter_1=i,
                frame_counter_2=i,
            )
            for i in range(10)
        ]
        profile = compute_residual_profile(obs_list, obs_list)
        assert profile.unmeasured is False
        assert profile.fd_pi is None
        assert profile.fd_sigma is None
        assert profile.fd_sigma_plus is None
        assert profile.fd_dagger is None

    def test_room_divergence_hard_reject(self) -> None:
        """Room ($079B) is part of Oπ — fd_pi set on room divergence."""
        mini_obs = [
            Observation(
                frame=i,
                x=100 + i,
                y=200,
                pose=0,
                room=0x91F8,
                sub_x=0,
                sub_y=0,
                velocity_x=1,
                velocity_y=0,
                velocity_x_sub=0,
                velocity_y_sub=0,
                momentum_x=1,
                momentum_x_sub=0,
                speed_counter=0,
                speed_flag=0,
                energy=99,
                frame_counter_1=i,
                frame_counter_2=i,
            )
            for i in range(10)
        ]
        # Emulator transitions to different room at frame 5
        emu_obs = [
            obs if i < 5 else Observation(**{**obs.to_dict(), "room": 0xA000})
            for i, obs in enumerate(mini_obs)
        ]
        profile = compute_residual_profile(mini_obs, emu_obs)
        assert profile.fd_pi == 5  # Oπ broke (room)
        assert profile.fd_sigma == 5  # Oσ also broke (Oσ includes Oπ)
        assert profile.fd_sigma_plus == 5  # Oσ+ also broke (Oσ+ includes Oσ)
        assert profile.cause == DivergenceCause.ROOM
        assert profile.first_diff_field == "room"
        assert profile.should_hard_reject() is True

    def test_death_divergence_hard_reject(self) -> None:
        """O† (death / $09C2=0) is separate — hard-reject on O† break."""
        mini_obs = [
            Observation(
                frame=i,
                x=100 + i,
                y=200,
                pose=0,
                room=0x91F8,
                sub_x=0,
                sub_y=0,
                velocity_x=1,
                velocity_y=0,
                velocity_x_sub=0,
                velocity_y_sub=0,
                momentum_x=1,
                momentum_x_sub=0,
                speed_counter=0,
                speed_flag=0,
                energy=99,
                frame_counter_1=i,
                frame_counter_2=i,
            )
            for i in range(10)
        ]
        # Emulator dies at frame 7
        emu_obs = [
            obs if i < 7 else Observation(**{**obs.to_dict(), "energy": 0})
            for i, obs in enumerate(mini_obs)
        ]
        profile = compute_residual_profile(mini_obs, emu_obs)
        assert profile.fd_dagger == 7
        assert profile.should_hard_reject() is True

    def test_lag_desync_tag(self) -> None:
        """$1842/$09DA diverge → tag `lag`, stop scoring kinematics."""
        mini_obs = [
            Observation(
                frame=i,
                x=100 + i,
                y=200,
                pose=0,
                room=0x91F8,
                sub_x=0,
                sub_y=0,
                velocity_x=1,
                velocity_y=0,
                velocity_x_sub=0,
                velocity_y_sub=0,
                momentum_x=1,
                momentum_x_sub=0,
                speed_counter=0,
                speed_flag=0,
                energy=99,
                frame_counter_1=i,
                frame_counter_2=i,
            )
            for i in range(10)
        ]
        # Emulator frame counter desyncs at frame 3
        emu_obs = [
            obs if i < 3 else Observation(**{**obs.to_dict(), "frame_counter_1": i + 1})
            for i, obs in enumerate(mini_obs)
        ]
        profile = compute_residual_profile(mini_obs, emu_obs)
        assert profile.fd_pi is None  # Oπ holds (pixels/pose/room agree)
        assert profile.fd_sigma == 3  # Oσ broke (frame counter diverged)
        assert profile.cause == DivergenceCause.LAG
        assert profile.first_diff_field == "frame_counter"
        assert profile.tag_lag_desync() is True

    def test_pixel_divergence_pi_break(self) -> None:
        """Pixels x/y are part of Oπ — fd_pi set on pixel divergence."""
        mini_obs = [
            Observation(
                frame=i,
                x=100 + i,
                y=200,
                pose=0,
                room=0x91F8,
                sub_x=0,
                sub_y=0,
                velocity_x=1,
                velocity_y=0,
                velocity_x_sub=0,
                velocity_y_sub=0,
                momentum_x=1,
                momentum_x_sub=0,
                speed_counter=0,
                speed_flag=0,
                energy=99,
                frame_counter_1=i,
                frame_counter_2=i,
            )
            for i in range(10)
        ]
        # Emulator pixels diverge at frame 4
        emu_obs = [
            obs if i < 4 else Observation(**{**obs.to_dict(), "x": 100 + i + 1})
            for i, obs in enumerate(mini_obs)
        ]
        profile = compute_residual_profile(mini_obs, emu_obs)
        assert profile.fd_pi == 4  # Oπ broke (pixels)
        assert profile.fd_sigma == 4  # Oσ also broke
        assert profile.cause == DivergenceCause.COLLISION
        assert profile.first_diff_field == "pixels"

    def test_subpixel_divergence_sigma_break_pi_holds(self) -> None:
        """Subpixels are Oσ (not Oπ) — Oσ broke, Oπ holds → spot_check."""
        mini_obs = [
            Observation(
                frame=i,
                x=100 + i,
                y=200,
                pose=0,
                room=0x91F8,
                sub_x=0,
                sub_y=0,
                velocity_x=1,
                velocity_y=0,
                velocity_x_sub=0,
                velocity_y_sub=0,
                momentum_x=1,
                momentum_x_sub=0,
                speed_counter=0,
                speed_flag=0,
                energy=99,
                frame_counter_1=i,
                frame_counter_2=i,
            )
            for i in range(10)
        ]
        # Emulator subpixels diverge at frame 4
        emu_obs = [
            obs if i < 4 else Observation(**{**obs.to_dict(), "sub_x": 1000})
            for i, obs in enumerate(mini_obs)
        ]
        profile = compute_residual_profile(mini_obs, emu_obs)
        assert profile.fd_pi is None  # Oπ holds (pixels/pose/room agree)
        assert profile.fd_sigma == 4  # Oσ broke (subpixels diverged)
        assert profile.fd_sigma_plus == 4  # Oσ+ also broke
        assert profile.cause == DivergenceCause.COLLISION
        assert profile.first_diff_field == "subpixels"
        assert profile.needs_emulator_spot_check() is True
        assert profile.can_keep_as_search_model() is True  # Oπ holds

    def test_enemy_energy_divergence_sigma_plus(self) -> None:
        """Enemy energy is Oσ+ — fd_σ+ set when enemy energy diverges."""
        mini_obs = [
            Observation(
                frame=i,
                x=100 + i,
                y=200,
                pose=0,
                room=0x91F8,
                sub_x=0,
                sub_y=0,
                velocity_x=1,
                velocity_y=0,
                velocity_x_sub=0,
                velocity_y_sub=0,
                momentum_x=1,
                momentum_x_sub=0,
                speed_counter=0,
                speed_flag=0,
                energy=99,
                frame_counter_1=i,
                frame_counter_2=i,
                enemy_energy=0,
            )
            for i in range(10)
        ]
        # Emulator enemy energy kicks in at frame 2
        emu_obs = [
            obs if i < 2 else Observation(**{**obs.to_dict(), "enemy_energy": 50})
            for i, obs in enumerate(mini_obs)
        ]
        profile = compute_residual_profile(mini_obs, emu_obs)
        assert profile.fd_pi is None  # Oπ holds
        assert profile.fd_sigma is None  # Oσ holds
        assert profile.fd_sigma_plus == 2  # Oσ+ broke (enemy energy)
