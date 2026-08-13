"""Offline residual lattice + thin stepper (no ROM)."""

from __future__ import annotations

import numpy as np

from smb.approx import JUMP_SPEED, idle_action, press, rollout, step
from smb.observation import (
    Observation,
    level1_start_obs,
    observation_from_ram,
    pack_room,
    unpack_room,
)
from smb.ram import (
    ADDR_AREA_POINTER,
    ADDR_LEVEL,
    ADDR_LIVES,
    ADDR_PLAYER_MOTION,
    ADDR_PLAYER_STATE,
    ADDR_PLAYER_X,
    ADDR_PLAYER_X_FRAC,
    ADDR_PLAYER_Y,
    ADDR_WORLD,
    ADDR_X_PAGE,
    ADDR_X_SPEED,
    ADDR_Y_SPEED,
)
from smb.residual import (
    DivergenceCause,
    ResidualProfile,
    compute_residual_profile,
    format_profile,
)
from smb.residual_harness import measure_segment


def _blank_ram() -> np.ndarray:
    return np.zeros(0x800, dtype=np.uint8)


def _obs(**overrides: object) -> Observation:
    base = level1_start_obs()
    data = base.to_dict()
    data.update(overrides)
    return Observation.from_dict(data)


def test_pack_room_roundtrip() -> None:
    packed = pack_room(0, 0, 194)
    assert unpack_room(packed) == (0, 0, 194)
    assert pack_room(7, 3, 16) == (7 << 16) | (3 << 8) | 16


def test_observation_from_ram_level1_shape() -> None:
    ram = _blank_ram()
    ram[ADDR_PLAYER_STATE] = 0x08
    ram[ADDR_X_PAGE] = 0
    ram[ADDR_PLAYER_X] = 40
    ram[ADDR_PLAYER_Y] = 176
    ram[ADDR_PLAYER_X_FRAC] = 16
    ram[ADDR_X_SPEED] = 4
    ram[ADDR_Y_SPEED] = 0
    ram[ADDR_WORLD] = 0
    ram[ADDR_LEVEL] = 0
    ram[ADDR_AREA_POINTER] = 194
    ram[ADDR_LIVES] = 2
    ram[ADDR_PLAYER_MOTION] = 0
    obs = observation_from_ram(ram, frame=3)
    assert obs.x == 40
    assert obs.y == 176
    assert obs.pose == 0x08
    assert obs.room == pack_room(0, 0, 194)
    assert obs.sub_x == 16
    assert obs.velocity_x == 4
    assert obs.on_ground is True
    assert obs.energy == 2
    assert obs.dead is False
    assert obs.frame == 3


def test_observation_from_ram_air_and_death() -> None:
    ram = _blank_ram()
    ram[ADDR_PLAYER_STATE] = 0x0B
    ram[ADDR_PLAYER_Y] = 200
    ram[ADDR_PLAYER_MOTION] = 1
    ram[ADDR_Y_SPEED] = 4
    obs = observation_from_ram(ram)
    assert obs.dead is True
    assert obs.on_ground is False
    assert obs.velocity_y == 4


def test_idle_is_identity_on_ground() -> None:
    start = level1_start_obs()
    nxt = step(start, idle_action())
    assert nxt.x == start.x
    assert nxt.y == start.y
    assert nxt.sub_x == 0
    assert nxt.velocity_x == 0
    assert nxt.on_ground is True
    assert nxt.frame == start.frame + 1
    assert nxt.frame_counter == ((start.frame_counter or 0) + 1) & 0xFF


def test_walk_is_deterministic_and_rightward() -> None:
    start = level1_start_obs()
    a = rollout(start, [press("RIGHT")] * 8)
    b = rollout(start, [press("RIGHT")] * 8)
    assert [obs.to_dict() for obs in a] == [obs.to_dict() for obs in b]
    assert a[-1].x >= start.x
    assert a[-1].velocity_x > 0
    assert a[-1].on_ground is True
    assert all(frame.y == start.y for frame in a)


def test_jump_leaves_ground_then_lands() -> None:
    start = level1_start_obs()
    frames = rollout(start, [press("A")] * 4 + [idle_action()] * 20)
    assert frames[1].y == start.y + JUMP_SPEED
    assert frames[1].on_ground is False
    assert frames[1].velocity_y == JUMP_SPEED
    assert min(frame.y for frame in frames) < start.y
    assert frames[-1].on_ground is True
    assert frames[-1].y == start.ground_y
    assert frames[-1].velocity_y == 0


def test_stepper_does_not_mutate_start() -> None:
    start = level1_start_obs()
    before = start.to_dict()
    step(start, press("RIGHT", "A"))
    assert start.to_dict() == before


def test_residual_unmeasured_and_agree() -> None:
    frames = rollout(level1_start_obs(), [press("RIGHT")] * 5)
    assert compute_residual_profile(frames, None).unmeasured is True
    profile = compute_residual_profile(frames, frames)
    assert profile.fd_pi is None
    assert profile.fd_sigma is None
    assert profile.fd_sigma_plus is None
    assert profile.fd_dagger is None
    assert profile.can_keep_as_search_model() is True


def test_residual_pixel_and_subpixel_and_room() -> None:
    mini = [_obs(frame=i, x=40 + i, sub_x=i) for i in range(8)]
    emu_px = [_obs(frame=i, x=40 + i if i < 3 else 99, sub_x=i) for i in range(8)]
    px = compute_residual_profile(mini, emu_px)
    assert px.fd_pi == 3
    assert px.fd_sigma == 3
    assert px.first_diff_field == "pixels"
    assert px.cause is DivergenceCause.COLLISION

    emu_sub = [_obs(frame=i, x=40 + i, sub_x=i if i < 2 else 9) for i in range(8)]
    sub = compute_residual_profile(mini, emu_sub)
    assert sub.fd_pi is None
    assert sub.fd_sigma == 2
    assert sub.first_diff_field == "subpixels"
    assert sub.needs_emulator_spot_check() is True
    assert sub.can_keep_as_search_model() is True

    # Later pixel break must not overwrite the earlier subpixel first-field.
    emu_both = [
        _obs(
            frame=i,
            x=40 + i if i < 5 else 99,
            sub_x=i if i < 2 else 9,
        )
        for i in range(8)
    ]
    both = compute_residual_profile(mini, emu_both)
    assert both.fd_sigma == 2
    assert both.fd_pi == 5
    assert both.first_diff_field == "subpixels"

    emu_room = [
        _obs(frame=i, x=40 + i, sub_x=i, room=pack_room(0, 0, 194 if i < 4 else 1))
        for i in range(8)
    ]
    room = compute_residual_profile(mini, emu_room)
    assert room.fd_pi == 4
    assert room.cause is DivergenceCause.ROOM
    assert room.should_hard_reject() is True


def test_residual_death_and_lag() -> None:
    mini = [_obs(frame=i, energy=2, dead=False, frame_counter=10 + i) for i in range(6)]
    emu_dead = [
        _obs(frame=i, energy=2, dead=(i >= 4), frame_counter=10 + i) for i in range(6)
    ]
    death = compute_residual_profile(mini, emu_dead)
    assert death.fd_dagger == 4
    assert death.should_hard_reject() is True

    emu_lag = [
        _obs(frame=i, energy=2, dead=False, frame_counter=10 + i + (1 if i >= 2 else 0))
        for i in range(6)
    ]
    lag = compute_residual_profile(mini, emu_lag)
    assert lag.lag is True
    assert lag.cause is DivergenceCause.LAG
    assert lag.tag_lag_desync() is True


def test_profile_roundtrip_and_format() -> None:
    profile = ResidualProfile(
        fd_pi=None,
        fd_sigma=4,
        fd_sigma_plus=4,
        fd_dagger=None,
        cause=DivergenceCause.COLLISION,
        first_diff_field="subpixels",
    )
    assert ResidualProfile.from_dict(profile.to_dict()) == profile
    text = format_profile(profile)
    assert "fdσ=4" in text
    assert "fdπ=—" in text
    assert "first=subpixels" in text


def test_measure_segment_offline_walk() -> None:
    result = measure_segment("walk", run_emulator=False)
    assert result.profile.unmeasured is True
    assert result.approx_obs[0].x == 40
    assert result.approx_obs[-1].x >= 40
    assert result.horizon == 25
