"""Shared R(τ) lattice compare (no game ROM)."""

from __future__ import annotations

from types import SimpleNamespace

from retro_harness.residual import (
    DivergenceCause,
    LatticeSpec,
    ResidualProfile,
    compute_residual_profile,
    format_profile,
)


def _obs(**fields: object) -> SimpleNamespace:
    base = dict(
        x=0,
        y=0,
        pose=0,
        room=1,
        sub_x=0,
        sub_y=0,
        velocity_x=0,
        velocity_y=0,
        energy=2,
        dead=False,
        frame_counter=0,
        enemy0_active=0,
        enemy0_type=0,
    )
    base.update(fields)
    return SimpleNamespace(**base)


SPEC = LatticeSpec(
    sigma_plus=("enemy0_active", "enemy0_type"),
    dagger=("energy", "dead"),
    lag=("frame_counter",),
)


def test_unmeasured_and_agree() -> None:
    frames = [_obs(x=i) for i in range(4)]
    assert compute_residual_profile(frames, None, spec=SPEC).unmeasured is True
    profile = compute_residual_profile(frames, frames, spec=SPEC)
    assert profile.fd_pi is None
    assert profile.can_keep_as_search_model() is True


def test_pixels_then_subpixels() -> None:
    mini = [_obs(x=i, sub_x=i) for i in range(6)]
    emu = [_obs(x=i if i < 3 else 99, sub_x=i if i < 1 else 9) for i in range(6)]
    profile = compute_residual_profile(mini, emu, spec=SPEC)
    assert profile.fd_sigma == 1
    assert profile.fd_pi == 3
    assert profile.first_diff_field == "subpixels"
    assert profile.cause is DivergenceCause.COLLISION


def test_room_hard_reject() -> None:
    mini = [_obs(room=1) for _ in range(4)]
    emu = [_obs(room=1 if i < 2 else 9) for i in range(4)]
    profile = compute_residual_profile(mini, emu, spec=SPEC)
    assert profile.fd_pi == 2
    assert profile.cause is DivergenceCause.ROOM
    assert profile.should_hard_reject() is True


def test_format_and_roundtrip() -> None:
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
