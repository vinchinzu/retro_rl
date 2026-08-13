"""Offline residual lattice + thin stepper (no ROM)."""

from __future__ import annotations

import numpy as np

from smb.approx import (
    AIR_RUN_KEEP,
    JUMP_FORCE_DOWN,
    JUMP_FORCE_UP,
    JUMP_SPEED,
    JUMP_Y_SPEED,
    WALK_MAX,
    idle_action,
    jump_table_index,
    press,
    rollout,
    step,
    takeoff_vertical,
)
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
    frames = rollout(start, [press("A")] * 4 + [idle_action()] * 40)
    assert frames[1].y == start.y + JUMP_SPEED
    assert frames[1].on_ground is False
    assert frames[1].velocity_y == JUMP_SPEED
    assert frames[1].y_move_force == 0x20
    assert frames[1].jump_origin_y == start.y
    assert min(frame.y for frame in frames) < start.y
    assert frames[-1].on_ground is True
    assert frames[-1].y == start.ground_y
    assert frames[-1].velocity_y == 0
    assert frames[-1].sub_y == 128
    assert frames[-1].vertical_force == 0x70


def test_jump_a_release_matches_level1_trace() -> None:
    """Standing 4-A then idle: ImposeGravity + A-release VF copy (live 1-1)."""
    start = level1_start_obs()
    frames = rollout(start, [press("A")] * 4 + [idle_action()] * 4)
    expected = (
        # f, y, sub_y, vy, y_move_force, vertical_force
        (0, 176, 0, 0, 0, 0),
        (1, 172, 0, -4, 32, 32),
        (2, 168, 32, -4, 64, 32),
        (3, 164, 96, -4, 96, 32),
        (4, 160, 192, -4, 128, 32),
        (5, 157, 64, -4, 240, 112),
        (6, 154, 48, -3, 96, 112),
        (7, 151, 144, -3, 208, 112),
        (8, 149, 96, -2, 64, 112),
    )
    for frame, y, sub_y, vy, ymf, vf in expected:
        obs = frames[frame]
        assert (obs.y, obs.sub_y, obs.velocity_y, obs.y_move_force, obs.vertical_force) == (
            y,
            sub_y,
            vy,
            ymf,
            vf,
        ), f"f{frame}"


def test_land_keeps_ymf_dummy() -> None:
    """Standing 4-A then idle: land snaps y, keeps $0416 leftover (live 1-1)."""
    start = level1_start_obs()
    frames = rollout(start, [press("A")] * 4 + [idle_action()] * 28)
    land = frames[25]
    assert land.on_ground is True
    assert land.y == start.ground_y
    assert land.sub_y == 128
    assert land.velocity_y == 0
    assert land.y_move_force == 0
    assert land.vertical_force == 0x70
    assert frames[24].on_ground is False
    assert frames[30].sub_y == 128
    assert frames[30].on_ground is True


def test_air_x_uses_walk_tables_until_run_speed() -> None:
    start = level1_start_obs()
    air = rollout(start, [press("RIGHT", "B", "A")] * 8)
    grounded_run = rollout(start, [press("RIGHT", "B")] * 8)
    assert air[1].velocity_x == 1
    assert air[1].x_force == 48
    assert air[2].velocity_x == 1
    assert air[2].x_force == 200
    assert air[2].sub_x == 32
    assert grounded_run[2].velocity_x == 2
    assert grounded_run[2].x_force == 20
    assert all(frame.velocity_x <= WALK_MAX for frame in air)
    assert AIR_RUN_KEEP == 0x19


def test_takeoff_frame_uses_air_x() -> None:
    """16f run then A: leave-ground frame adds walk $98, not run $E4 (live 1-1)."""
    start = level1_start_obs()
    frames = rollout(
        start,
        [press("RIGHT", "B")] * 16 + [press("RIGHT", "B", "A")] * 4,
    )
    last_ground = frames[16]
    takeoff = frames[17]
    assert last_ground.on_ground is True
    assert last_ground.velocity_x == 14
    assert last_ground.x_force == 140
    assert takeoff.on_ground is False
    assert takeoff.velocity_x == 15
    assert takeoff.x_force == 36
    assert takeoff.y == start.y + JUMP_SPEED


def test_jump_tables_from_abs_vx() -> None:
    """smbdis InitJS land bands — no live tape per threshold."""
    cases = (
        (0, 0),
        (8, 0),
        (9, 1),
        (15, 1),
        (16, 2),
        (24, 2),
        (25, 3),
        (27, 3),
        (28, 4),
        (40, 4),
        (-21, 2),
    )
    for velocity_x, index in cases:
        assert jump_table_index(velocity_x) == index, velocity_x
        vy, vf, vfd, ymf = takeoff_vertical(velocity_x)
        assert (vy, vf, vfd, ymf) == (
            JUMP_Y_SPEED[index],
            JUMP_FORCE_UP[index],
            JUMP_FORCE_DOWN[index],
            0,
        ), velocity_x
        takeoff = step(_obs(velocity_x=velocity_x), press("A"))
        assert takeoff.on_ground is False
        assert takeoff.velocity_y == vy
        assert takeoff.vertical_force == vf
        assert takeoff.vertical_force_down == vfd
        assert takeoff.y == 176 + vy


def test_run24_then_jump_uses_mid_run_table() -> None:
    """24f RIGHT+B → |vx|=21 → InitJS index 2 (vf=$1E, not $20)."""
    start = level1_start_obs()
    frames = rollout(start, [press("RIGHT", "B")] * 24 + [press("RIGHT", "B", "A")] * 2)
    last_ground = frames[24]
    takeoff = frames[25]
    assert last_ground.on_ground is True
    assert last_ground.velocity_x == 21
    assert jump_table_index(last_ground.velocity_x) == 2
    assert takeoff.on_ground is False
    assert takeoff.velocity_y == -4
    assert takeoff.vertical_force == 0x1E
    assert takeoff.vertical_force_down == 0x60
    assert takeoff.sub_y == 0
    assert takeoff.y_move_force == 0x1E
    assert frames[26].sub_y == 0x1E


def test_run32_then_jump_uses_fast_run_table() -> None:
    """32f RIGHT+B → |vx|=28 → InitJS index 4 (vy=-5, vf=$28)."""
    start = level1_start_obs()
    frames = rollout(start, [press("RIGHT", "B")] * 32 + [press("RIGHT", "B", "A")])
    last_ground = frames[32]
    takeoff = frames[33]
    assert last_ground.on_ground is True
    assert last_ground.velocity_x == 28
    assert jump_table_index(last_ground.velocity_x) == 4
    assert takeoff.on_ground is False
    assert takeoff.velocity_y == -5
    assert takeoff.vertical_force == 0x28
    assert takeoff.vertical_force_down == 0x90
    assert takeoff.y == start.y - 5


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


def test_measure_segment_offline_jump_to_land() -> None:
    result = measure_segment("jump_to_land", run_emulator=False)
    assert result.profile.unmeasured is True
    assert result.horizon == 33
    land = result.approx_obs[25]
    assert land.on_ground is True
    assert land.y == 176
    assert land.sub_y == 128
    assert land.vertical_force == 0x70


def test_measure_segment_offline_run_then_jump() -> None:
    result = measure_segment("run_then_jump", run_emulator=False)
    assert result.profile.unmeasured is True
    assert result.horizon == 37
    assert result.approx_obs[16].on_ground is True
    assert result.approx_obs[17].on_ground is False
    assert result.approx_obs[17].x_force == 36


def test_measure_segment_offline_run24_and_run32() -> None:
    mid = measure_segment("run24_then_jump", run_emulator=False)
    fast = measure_segment("run32_then_jump", run_emulator=False)
    assert mid.horizon == 45
    assert fast.horizon == 53
    assert mid.approx_obs[25].vertical_force == 0x1E
    assert fast.approx_obs[33].velocity_y == -5
    assert fast.approx_obs[33].vertical_force == 0x28
