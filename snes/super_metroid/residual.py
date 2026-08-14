"""Residual profiling for physics sim vs emulator validation.

Records R(τ) = (fd_σ+, fd_σ, fd_π, fd_†) — first-difference frames where
MiniStep trajectory diverges from SuperMetroidEnv replay.

Bob's locked observation lattice:
- Oπ (coarsest): ($0AF6, $0AFA, $0A1C, $079B) pixels x/y, pose, room
- Oσ: Oπ plus ($0AF8, $0AFC) subpixels
- Oσ+: Oσ plus optional ($0F8C, $18A8) enemy energy / i-frames
- O† (separate): ($09C2, dead) energy/death

fd_π is first pixel/pose/room disagreement, NOT "inputs diverge."

Planner rules:
- residual ≥ Oπ on horizon → keep Mini/Stub as **search** model (NOT room_clear)
- Oσ broke, Oπ holds → emu spot-check (`validate_trajectory_on_emulator`)
- $079B or O† (death / $09C2) → hard-reject (Oπ or O† break)
- $1842/$09DA diverge → tag `lag`, stop scoring later kinematics
- room_clear only from `validate_trajectory_on_emulator` (E = SuperMetroidEnv)

**Until sm_rev --load-state is available, do not fake fd frames** — emit
"unmeasured" / omit fd, never invented numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from retro_harness.residual import (
    DivergenceCause,
    LatticeSpec,
    ResidualProfile,
    compute_residual_profile as _compute,
)
from super_metroid.observation import Observation

__all__ = [
    "DivergenceCause",
    "ResidualProfile",
    "SM_LATTICE",
    "compute_residual_profile",
]

SM_LATTICE = LatticeSpec(
    sigma_plus=("enemy_energy", "invulnerability_timer"),
    dagger=("energy",),
    lag=("frame_counter_1", "frame_counter_2"),
    extra_velocity=("momentum_x", "momentum_x_sub"),
    lag_require_both=False,
)


def compute_residual_profile(
    mini_obs: Sequence[Observation | Any],
    emu_obs: Sequence[Observation | Any] | None,
) -> ResidualProfile:
    """Compute residual profile R(τ) between Mini/Stub and emulator trajectories."""
    return _compute(mini_obs, emu_obs, spec=SM_LATTICE)
