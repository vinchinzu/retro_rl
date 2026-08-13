"""Residual profiling for the SMB approximate stepper vs emulator.

Records R(τ) = (fd_σ+, fd_σ, fd_π, fd_†) — first-difference frames where the
pure stepper trajectory diverges from a stable-retro replay.

Lattice:
- Oπ: pixel x/y, pose ($000E), room (world/level/area)
- Oσ: Oπ plus subpixels ($0400, $0416)
- Oσ+: Oσ plus enemy slot 0
- O†: lives / death (separate)

fd_π is first pixel/pose/room disagreement, NOT "inputs diverge."

Planner rules:
- residual ≥ Oπ on horizon → keep approx as a **search** model (not route-clear)
- Oσ broke, Oπ holds → emulator spot-check
- room or O† → hard-reject
- $0009 diverges → tag `lag`, stop scoring later kinematics
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from smb.observation import Observation

__all__ = [
    "DivergenceCause",
    "ResidualProfile",
    "compute_residual_profile",
    "format_profile",
]


class DivergenceCause(Enum):
    """Probable cause tag for the first differing field."""

    COLLISION = "collision"
    LAG = "lag"
    ROOM = "room"
    VELOCITY = "velocity"
    OTHER = "other"


@dataclass(frozen=True)
class ResidualProfile:
    """R(τ) first-difference frames. None = agrees for the horizon or unmeasured."""

    fd_pi: int | None
    fd_sigma: int | None
    fd_sigma_plus: int | None
    fd_dagger: int | None
    cause: DivergenceCause | None
    first_diff_field: str | None
    unmeasured: bool = False
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
        if self.fd_dagger is not None:
            return True
        if self.fd_pi is not None and (
            self.cause is DivergenceCause.ROOM or self.first_diff_field == "room"
        ):
            return True
        return False

    def needs_emulator_spot_check(self) -> bool:
        if self.unmeasured or self.should_hard_reject():
            return False
        return self.fd_sigma is not None and self.fd_pi is None

    def can_keep_as_search_model(self) -> bool:
        if self.unmeasured or self.should_hard_reject():
            return False
        return self.fd_pi is None

    def tag_lag_desync(self) -> bool:
        return self.lag or self.cause is DivergenceCause.LAG


def format_profile(profile: ResidualProfile) -> str:
    """Render R(τ) = (fd_σ+, fd_σ, fd_π, fd_†) plus first field / cause."""
    if profile.unmeasured:
        return "R(τ)=unmeasured"

    def _fd(value: int | None) -> str:
        return "—" if value is None else str(value)

    parts = (
        f"fdσ+={_fd(profile.fd_sigma_plus)}",
        f"fdσ={_fd(profile.fd_sigma)}",
        f"fdπ={_fd(profile.fd_pi)}",
        f"fd†={_fd(profile.fd_dagger)}",
    )
    line = "R(τ)=(" + ", ".join(parts) + ")"
    if profile.first_diff_field:
        line += f"  first={profile.first_diff_field}"
    if profile.cause is not None:
        line += f"  cause={profile.cause.value}"
    if profile.lag:
        line += "  lag"
    return line


def compute_residual_profile(
    approx_obs: list[Observation],
    emu_obs: list[Observation] | None,
) -> ResidualProfile:
    """Compute R(τ) between the pure stepper and an emulator trajectory."""
    if emu_obs is None:
        return ResidualProfile.unmeasured_profile()

    horizon = min(len(approx_obs), len(emu_obs))
    if horizon == 0:
        return ResidualProfile.unmeasured_profile()

    fd_pi: int | None = None
    fd_sigma: int | None = None
    fd_sigma_plus: int | None = None
    fd_dagger: int | None = None
    cause: DivergenceCause | None = None
    first_diff_field: str | None = None
    lag = False

    for i in range(horizon):
        m = approx_obs[i]
        e = emu_obs[i]

        if fd_dagger is None and m.energy is not None:
            if m.energy != e.energy or m.dead != e.dead:
                fd_dagger = i
                if first_diff_field is None:
                    first_diff_field = "energy" if m.energy != e.energy else "dead"
                    cause = DivergenceCause.OTHER

        if fd_pi is None:
            if m.room != e.room:
                fd_pi = i
                if first_diff_field is None:
                    first_diff_field = "room"
                cause = DivergenceCause.ROOM
            elif m.x != e.x or m.y != e.y:
                fd_pi = i
                if first_diff_field is None:
                    first_diff_field = "pixels"
                    cause = DivergenceCause.COLLISION
            elif m.pose != e.pose:
                fd_pi = i
                if first_diff_field is None:
                    first_diff_field = "pose"
                    cause = DivergenceCause.OTHER

        if fd_sigma is None:
            if fd_pi is not None:
                fd_sigma = fd_pi
            elif m.sub_x != e.sub_x or m.sub_y != e.sub_y:
                fd_sigma = i
                if first_diff_field is None:
                    first_diff_field = "subpixels"
                    cause = DivergenceCause.COLLISION

        if (
            m.frame_counter is not None
            and e.frame_counter is not None
            and m.frame_counter != e.frame_counter
        ):
            if first_diff_field is None:
                first_diff_field = "frame_counter"
            if cause is not DivergenceCause.ROOM:
                cause = DivergenceCause.LAG
            if fd_sigma is not None and fd_sigma_plus is None:
                fd_sigma_plus = fd_sigma
            lag = True
            break

        if first_diff_field is None and (
            m.velocity_x != e.velocity_x or m.velocity_y != e.velocity_y
        ):
            first_diff_field = "velocity"
            if cause is None:
                cause = DivergenceCause.VELOCITY

        if fd_sigma_plus is None:
            if fd_sigma is not None:
                fd_sigma_plus = fd_sigma
            elif m.enemy0_active != e.enemy0_active or m.enemy0_type != e.enemy0_type:
                fd_sigma_plus = i
                if first_diff_field is None:
                    first_diff_field = "enemy0"
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
