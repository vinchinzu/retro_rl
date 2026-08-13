"""Residual profiling for physics sim vs emulator validation.

Records R(τ) = (fd_σ+, fd_σ, fd_π, fd_†) — first-difference frames where
MiniStep trajectory diverges from SuperMetroidEnv replay. Enables:
- Spotting collision / lag / knockback mismatches
- Keeping MiniStep as search model when Oπ holds for horizon
- Hard-rejecting room/death divergence
- Emulator spot-checks when Oσ breaks but Oπ holds

Rules (from task):
- residual ≥ Oπ on horizon → keep Mini/Stub as **search** model (NOT room_clear)
- Oσ broke, Oπ holds → emu spot-check (`validate_trajectory_on_emulator`)
- $079B or O† (death / $09C2) → hard-reject
- $1842/$09DA diverge → tag `lag`, stop scoring later kinematics (desynced tape index)
- room_clear only from `validate_trajectory_on_emulator` (E = SuperMetroidEnv, not SMEDIT snes9x)

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
    """Probable cause tag for first WRAM field difference."""

    COLLISION = "collision"
    LAG = "lag"
    DOOR = "door"
    KNOCKBACK = "knockback"
    OTHER = "other"


@dataclass(frozen=True)
class ResidualProfile:
    """Residual R(τ) profile for Mini/Stub vs SuperMetroidEnv trajectory.

    Records first-difference frames for each observation level:
    - fd_σ+: enemies/i-frames diverge
    - fd_σ: core kinematics diverge (x/y/vel/pose/room/energy)
    - fd_π: inputs diverge (should never happen for same tape)
    - fd_†: death/energy=0 diverge

    When fd is None, that level agrees for the entire horizon or is unmeasured.
    """

    # First-difference frames (None = agrees for horizon or unmeasured)
    fd_sigma_plus: int | None
    fd_sigma: int | None
    fd_pi: int | None
    fd_dagger: int | None

    # Probable cause + first differing field (when fd_sigma is not None)
    cause: DivergenceCause | None
    first_diff_field: str | None

    # Unmeasured flag (set when start cannot be loaded on both sides)
    unmeasured: bool = False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "fd_sigma_plus": self.fd_sigma_plus,
            "fd_sigma": self.fd_sigma,
            "fd_pi": self.fd_pi,
            "fd_dagger": self.fd_dagger,
            "unmeasured": self.unmeasured,
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
            fd_sigma_plus=data["fd_sigma_plus"],
            fd_sigma=data["fd_sigma"],
            fd_pi=data["fd_pi"],
            fd_dagger=data["fd_dagger"],
            cause=cause,
            first_diff_field=data.get("first_diff_field"),
            unmeasured=data.get("unmeasured", False),
        )

    @classmethod
    def unmeasured_profile(cls) -> ResidualProfile:
        """Return unmeasured profile (start cannot be loaded on both sides)."""
        return cls(
            fd_sigma_plus=None,
            fd_sigma=None,
            fd_pi=None,
            fd_dagger=None,
            cause=None,
            first_diff_field=None,
            unmeasured=True,
        )

    def should_hard_reject(self) -> bool:
        """Return True if trajectory should be hard-rejected.

        Hard-reject when:
        - Room ($079B) diverged (fd_sigma is not None and cause is DOOR)
        - Death/energy diverged (fd_dagger is not None)
        """
        if self.fd_dagger is not None:
            return True
        if self.fd_sigma is not None and self.cause is DivergenceCause.DOOR:
            return True
        return False

    def needs_emulator_spot_check(self) -> bool:
        """Return True if trajectory needs emulator spot-check.

        Spot-check when:
        - Oσ broke (fd_sigma is not None)
        - Oπ holds (fd_pi is None)
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
        - Oπ holds for horizon (fd_pi is None)
        - Not hard-rejected

        Note: This does NOT mean room_clear — that requires emulator validation.
        """
        if self.unmeasured:
            return False
        if self.should_hard_reject():
            return False
        return self.fd_pi is None

    def tag_lag_desync(self) -> bool:
        """Return True if lag desync detected (frame counters diverged).

        When true, stop scoring later kinematics (desynced tape index).
        """
        return self.cause is DivergenceCause.LAG


def compute_residual_profile(
    mini_obs: list[Observation],
    emu_obs: list[Observation] | None,
) -> ResidualProfile:
    """Compute residual profile R(τ) between Mini/Stub and emulator trajectories.

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

    # Find first-difference frames
    fd_sigma_plus: int | None = None
    fd_sigma: int | None = None
    fd_pi: int | None = None
    fd_dagger: int | None = None
    cause: DivergenceCause | None = None
    first_diff_field: str | None = None

    for i in range(horizon):
        m = mini_obs[i]
        e = emu_obs[i]

        # Check O† (death/energy)
        if fd_dagger is None:
            if m.energy != e.energy or (m.energy == 0) != (e.energy == 0):
                fd_dagger = i
                if first_diff_field is None:
                    first_diff_field = "energy"
                    cause = DivergenceCause.OTHER

        # Check Oσ (core kinematics)
        if fd_sigma is None:
            # Check room first (high priority for hard-reject)
            if m.room != e.room:
                fd_sigma = i
                first_diff_field = "room"
                cause = DivergenceCause.DOOR
            # Check frame counters (lag detection)
            elif m.frame_counter_1 != e.frame_counter_1 or m.frame_counter_2 != e.frame_counter_2:
                fd_sigma = i
                first_diff_field = "frame_counter"
                cause = DivergenceCause.LAG
            # Check position
            elif m.x != e.x or m.y != e.y:
                fd_sigma = i
                first_diff_field = "position"
                cause = DivergenceCause.COLLISION
            elif m.sub_x != e.sub_x or m.sub_y != e.sub_y:
                fd_sigma = i
                first_diff_field = "subpixel"
                cause = DivergenceCause.COLLISION
            # Check velocity
            elif (
                m.velocity_x != e.velocity_x
                or m.velocity_y != e.velocity_y
                or m.velocity_x_sub != e.velocity_x_sub
                or m.velocity_y_sub != e.velocity_y_sub
            ):
                fd_sigma = i
                first_diff_field = "velocity"
                cause = DivergenceCause.COLLISION
            # Check momentum
            elif m.momentum_x != e.momentum_x or m.momentum_x_sub != e.momentum_x_sub:
                fd_sigma = i
                first_diff_field = "momentum"
                cause = DivergenceCause.COLLISION
            # Check pose
            elif m.pose != e.pose:
                fd_sigma = i
                first_diff_field = "pose"
                cause = DivergenceCause.OTHER

        # Check Oσ+ (enemies/i-frames)
        if fd_sigma_plus is None:
            # Check i-frame timers
            if (
                m.invulnerability_timer != e.invulnerability_timer
                or m.knockback_timer != e.knockback_timer
            ):
                fd_sigma_plus = i
                if first_diff_field is None:
                    first_diff_field = "invulnerability"
                    cause = DivergenceCause.KNOCKBACK
            # Check enemy count
            elif len(m.enemies) != len(e.enemies):
                fd_sigma_plus = i
                if first_diff_field is None:
                    first_diff_field = "enemy_count"
                    cause = DivergenceCause.OTHER

        # Note: fd_pi (input divergence) not checked here — requires separate
        # input tape comparison. Should never happen for same tape.

    return ResidualProfile(
        fd_sigma_plus=fd_sigma_plus,
        fd_sigma=fd_sigma,
        fd_pi=fd_pi,
        fd_dagger=fd_dagger,
        cause=cause,
        first_diff_field=first_diff_field,
        unmeasured=False,
    )
