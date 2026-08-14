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

from dataclasses import dataclass
from enum import Enum
from typing import Any

from super_metroid.observation import Observation

__all__ = [
    "DivergenceCause",
    "ResidualProfile",
    "compute_residual_profile",
]


class DivergenceCause(Enum):
    """Probable cause tag for first WRAM field difference.
    
    Speeds ($0B2C/$0B2E/$0B42/$0B44) live in first-differing-field tags.
    """

    COLLISION = "collision"
    LAG = "lag"
    ROOM = "room"
    VELOCITY = "velocity"
    OTHER = "other"


@dataclass(frozen=True)
class ResidualProfile:
    """Residual R(τ) profile for Mini/Stub vs SuperMetroidEnv trajectory.

    Records first-difference frames for Bob's locked observation lattice:
    - fd_π: pixels x/y, pose, room diverge (Oπ break)
    - fd_σ: fd_π or subpixels diverge (Oσ break)
    - fd_σ+: fd_σ or enemy energy / i-frames diverge (Oσ+ break)
    - fd_†: energy/death diverge (O† break, separate from Oπ)

    When fd is None, that level agrees for the entire horizon or is unmeasured.
    fd_π is first pixel/pose/room disagreement, NOT "inputs diverge."
    """

    # First-difference frames (None = agrees for horizon or unmeasured)
    fd_pi: int | None  # Oπ: pixels x/y, pose, room
    fd_sigma: int | None  # Oσ: Oπ plus subpixels
    fd_sigma_plus: int | None  # Oσ+: Oσ plus enemy energy / i-frames
    fd_dagger: int | None  # O†: energy/death (separate)

    # Probable cause + first differing field (when fd_sigma is not None)
    cause: DivergenceCause | None
    first_diff_field: str | None

    # Unmeasured flag (set when start cannot be loaded on both sides)
    unmeasured: bool = False
    # Orthogonal to cause: room desync must still hard-reject if lag also fires.
    lag: bool = False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "fd_pi": self.fd_pi,
            "fd_sigma": self.fd_sigma,
            "fd_sigma_plus": self.fd_sigma_plus,
            "fd_dagger": self.fd_dagger,
            "unmeasured": self.unmeasured,
            "lag": self.lag,
        }
        if self.cause is not None:
            result["cause"] = self.cause.value
        if self.first_diff_field is not None:
            result["first_diff_field"] = self.first_diff_field
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResidualProfile:
        cause = (
            DivergenceCause(data["cause"])
            if "cause" in data and data["cause"] is not None
            else None
        )
        return cls(
            fd_pi=data["fd_pi"],
            fd_sigma=data["fd_sigma"],
            fd_sigma_plus=data["fd_sigma_plus"],
            fd_dagger=data["fd_dagger"],
            cause=cause,
            first_diff_field=data.get("first_diff_field"),
            unmeasured=data.get("unmeasured", False),
            lag=bool(data.get("lag", False)),
        )

    @classmethod
    def unmeasured_profile(cls) -> ResidualProfile:
        """Return unmeasured profile (start cannot be loaded on both sides)."""
        return cls(
            fd_pi=None,
            fd_sigma=None,
            fd_sigma_plus=None,
            fd_dagger=None,
            cause=None,
            first_diff_field=None,
            unmeasured=True,
            lag=False,
        )

    def should_hard_reject(self) -> bool:
        """Hard-reject on death/energy or a $079B room break (not last-write cause)."""
        if self.fd_dagger is not None:
            return True
        if self.fd_pi is not None and (
            self.cause is DivergenceCause.ROOM or self.first_diff_field == "room"
        ):
            return True
        return False

    def needs_emulator_spot_check(self) -> bool:
        """Return True if trajectory needs emulator spot-check.

        Spot-check when:
        - Oσ broke (fd_sigma is not None)
        - Oπ holds (fd_pi is None) — pixels/pose/room agree
        - Not hard-rejected
        """
        if self.unmeasured:
            return False
        if self.should_hard_reject():
            return False
        return self.fd_sigma is not None and self.fd_pi is None

    def can_keep_as_search_model(self) -> bool:
        """Return True if Mini/Stub can be kept as search model.

        Keep as search model when:
        - Oπ holds for horizon (fd_pi is None) — pixels/pose/room agree
        - Not hard-rejected

        Note: This does NOT mean room_clear — that requires emulator validation.
        """
        if self.unmeasured:
            return False
        if self.should_hard_reject():
            return False
        return self.fd_pi is None

    def tag_lag_desync(self) -> bool:
        """True when frame counters diverged (stop scoring later kinematics)."""
        return self.lag or self.cause is DivergenceCause.LAG


def compute_residual_profile(
    mini_obs: list[Observation],
    emu_obs: list[Observation] | None,
) -> ResidualProfile:
    """Compute residual profile R(τ) between Mini/Stub and emulator trajectories.

    Uses Bob's locked observation lattice:
    - fd_π: first pixel/pose/room disagreement (Oπ break)
    - fd_σ: first fd_π or subpixel disagreement (Oσ break)
    - fd_σ+: first fd_σ or enemy energy/i-frame disagreement (Oσ+ break)
    - fd_†: first energy/death disagreement (O† break, separate)

    Args:
        mini_obs: Observations from MiniStep/Stub predictor
        emu_obs: Observations from emulator replay, or None when unmeasured

    Returns:
        ResidualProfile with first-difference frames and cause tag

    Note: Until sm_rev --load-state is available, emu_obs may be None (unmeasured).
    Do not fake fd frames — return unmeasured profile instead.
    """
    if emu_obs is None:
        return ResidualProfile.unmeasured_profile()

    # Align horizons (use shorter length)
    horizon = min(len(mini_obs), len(emu_obs))
    if horizon == 0:
        return ResidualProfile.unmeasured_profile()

    # Find first-difference frames per Bob's locked lattice
    fd_pi: int | None = None  # Oπ: pixels x/y, pose, room
    fd_sigma: int | None = None  # Oσ: Oπ plus subpixels
    fd_sigma_plus: int | None = None  # Oσ+: Oσ plus enemy energy / i-frames
    fd_dagger: int | None = None  # O†: energy/death (separate)
    cause: DivergenceCause | None = None
    first_diff_field: str | None = None
    lag = False

    for i in range(horizon):
        m = mini_obs[i]
        e = emu_obs[i]

        # Check O† (death/energy) — separate from Oπ
        # Skip if Mini energy is unobserved (None)
        if fd_dagger is None and m.energy is not None:
            if m.energy != e.energy or (m.energy == 0) != (e.energy == 0):
                fd_dagger = i
                if first_diff_field is None:
                    first_diff_field = "energy"
                    cause = DivergenceCause.OTHER

        # Check Oπ (pixels x/y, pose, room) — coarsest level
        if fd_pi is None:
            # Check room first (high priority for hard-reject)
            if m.room != e.room:
                fd_pi = i
                first_diff_field = "room"
                cause = DivergenceCause.ROOM
            # Check pixels x/y
            elif m.x != e.x or m.y != e.y:
                fd_pi = i
                first_diff_field = "pixels"
                cause = DivergenceCause.COLLISION
            # Check pose
            elif m.pose != e.pose:
                fd_pi = i
                first_diff_field = "pose"
                cause = DivergenceCause.OTHER

        # Check Oσ (Oπ plus subpixels)
        if fd_sigma is None:
            # If Oπ broke, Oσ also broke
            if fd_pi is not None:
                fd_sigma = fd_pi
            # Check subpixels (only pixels/subpixels set fd_σ)
            elif m.sub_x != e.sub_x or m.sub_y != e.sub_y:
                fd_sigma = i
                first_diff_field = "subpixels"
                cause = DivergenceCause.COLLISION

        # Lag is orthogonal to ROOM/Oπ: tag it, then stop scoring later frames.
        if (
            m.frame_counter_1 is not None
            and (m.frame_counter_1 != e.frame_counter_1 or m.frame_counter_2 != e.frame_counter_2)
        ):
            if first_diff_field is None:
                first_diff_field = "frame_counter"
            if cause is not DivergenceCause.ROOM:
                cause = DivergenceCause.LAG
            # Lattice: Oσ+ includes Oσ — fill before abandoning the horizon.
            if fd_sigma is not None and fd_sigma_plus is None:
                fd_sigma_plus = fd_sigma
            lag = True
            break

        # Tag velocity/momentum divergence in first_diff_field (don't set fd_σ)
        # Speeds live only in first_diff_field, not Oσ break
        if first_diff_field is None and (
            m.velocity_x != e.velocity_x
            or m.velocity_y != e.velocity_y
            or m.velocity_x_sub != e.velocity_x_sub
            or m.velocity_y_sub != e.velocity_y_sub
        ):
            first_diff_field = "velocity"
            if cause is None:
                cause = DivergenceCause.VELOCITY

        if first_diff_field is None and (m.momentum_x != e.momentum_x or m.momentum_x_sub != e.momentum_x_sub):
            first_diff_field = "momentum"
            if cause is None:
                cause = DivergenceCause.VELOCITY

        # Check Oσ+ (Oσ plus enemy energy / i-frames)
        if fd_sigma_plus is None:
            # If Oσ broke, Oσ+ also broke
            if fd_sigma is not None:
                fd_sigma_plus = fd_sigma
            # Check enemy energy / i-frames
            elif (
                m.enemy_energy != e.enemy_energy
                or m.invulnerability_timer != e.invulnerability_timer
            ):
                fd_sigma_plus = i
                if first_diff_field is None:
                    first_diff_field = "enemy_energy_or_invulnerability"
                    cause = DivergenceCause.OTHER

    return ResidualProfile(
        fd_pi=fd_pi,
        fd_sigma=fd_sigma,
        fd_sigma_plus=fd_sigma_plus,
        fd_dagger=fd_dagger,
        cause=cause,
        first_diff_field=first_diff_field,
        unmeasured=False,
        lag=lag,
    )
