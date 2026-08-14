"""SMB residual wrapper around the shared R(τ) lattice compare."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from retro_harness.residual import (
    DivergenceCause,
    LatticeSpec,
    ResidualProfile,
    compute_residual_profile as _compute,
    format_profile,
)
from smb.observation import Observation

__all__ = [
    "DivergenceCause",
    "ResidualProfile",
    "SMB_LATTICE",
    "compute_residual_profile",
    "format_profile",
]

SMB_LATTICE = LatticeSpec(
    sigma_plus=("enemy0_active", "enemy0_type"),
    dagger=("energy", "dead"),
    lag=("frame_counter",),
    lag_require_both=True,
)


def compute_residual_profile(
    approx_obs: Sequence[Observation | Any],
    emu_obs: Sequence[Observation | Any] | None,
) -> ResidualProfile:
    """Compute R(τ) between the pure stepper and an emulator trajectory."""
    return _compute(approx_obs, emu_obs, spec=SMB_LATTICE)
