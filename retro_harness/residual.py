"""Shared R(τ) residual profile for approximate steppers vs emulator replay.

Records R(τ) = (fd_σ+, fd_σ, fd_π, fd_†) — first-difference frames where a
pure stepper trajectory diverges from an emulator replay.

Lattice:
- Oπ: pixels, pose, room
- Oσ: Oπ plus subpixels
- Oσ+: Oσ plus game-specific extras (enemy slot, i-frames, …)
- O†: energy / death (separate)

fd_π is first pixel/pose/room disagreement, NOT "inputs diverge."

Planner rules:
- residual ≥ Oπ on horizon → keep approx as a **search** model (not route-clear)
- Oσ broke, Oπ holds → emulator spot-check
- room or O† → hard-reject
- frame-counter diverge → tag `lag`, stop scoring later kinematics
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

__all__ = [
    "DivergenceCause",
    "LatticeSpec",
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


# Frozen pairs so LatticeSpec stays hashable. Games can append more.
_DEFAULT_LABELS: tuple[tuple[str, str], ...] = (
    ("x", "pixels"),
    ("y", "pixels"),
    ("sub_x", "subpixels"),
    ("sub_y", "subpixels"),
    ("velocity_x", "velocity"),
    ("velocity_y", "velocity"),
    ("velocity_x_sub", "velocity"),
    ("velocity_y_sub", "velocity"),
    ("momentum_x", "momentum"),
    ("momentum_x_sub", "momentum"),
    ("frame_counter", "frame_counter"),
    ("frame_counter_1", "frame_counter"),
    ("frame_counter_2", "frame_counter"),
    ("enemy0_active", "enemy0"),
    ("enemy0_type", "enemy0"),
    ("enemy_energy", "enemy_energy_or_invulnerability"),
    ("invulnerability_timer", "enemy_energy_or_invulnerability"),
)


@dataclass(frozen=True)
class LatticeSpec:
    """Field names on observation objects, in check order within each layer."""

    pi: tuple[str, ...] = ("room", "x", "y", "pose")
    sigma: tuple[str, ...] = ("sub_x", "sub_y")
    sigma_plus: tuple[str, ...] = ()
    dagger: tuple[str, ...] = ("energy",)
    lag: tuple[str, ...] = ()
    velocity: tuple[str, ...] = ("velocity_x", "velocity_y")
    extra_velocity: tuple[str, ...] = ()
    labels: tuple[tuple[str, str], ...] = _DEFAULT_LABELS
    # SMB requires both sides present; SM tags if the first counter is live.
    lag_require_both: bool = True

    def label(self, field_name: str) -> str:
        for src, dst in self.labels:
            if src == field_name:
                return dst
        return field_name


def _cause_for(label: str) -> DivergenceCause:
    if label == "room":
        return DivergenceCause.ROOM
    if label in {"pixels", "subpixels"}:
        return DivergenceCause.COLLISION
    if label in {"velocity", "momentum"}:
        return DivergenceCause.VELOCITY
    if label == "frame_counter":
        return DivergenceCause.LAG
    return DivergenceCause.OTHER


def _first_mismatch(left: Any, right: Any, fields: tuple[str, ...]) -> str | None:
    for name in fields:
        if getattr(left, name) != getattr(right, name):
            return name
    return None


def _lag_broke(left: Any, right: Any, spec: LatticeSpec) -> bool:
    if not spec.lag:
        return False
    if spec.lag_require_both:
        if any(getattr(left, name) is None or getattr(right, name) is None for name in spec.lag):
            return False
    elif getattr(left, spec.lag[0]) is None:
        return False
    return any(getattr(left, name) != getattr(right, name) for name in spec.lag)


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
    approx_obs: Sequence[Any],
    emu_obs: Sequence[Any] | None,
    *,
    spec: LatticeSpec,
) -> ResidualProfile:
    """Compute R(τ) between a pure stepper and an emulator trajectory."""
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

    def _note(field_name: str, new_cause: DivergenceCause | None = None) -> None:
        nonlocal cause, first_diff_field
        label = spec.label(field_name)
        if first_diff_field is None:
            first_diff_field = label
        if new_cause is not None:
            cause = new_cause
        elif cause is None:
            cause = _cause_for(label)

    for i in range(horizon):
        model = approx_obs[i]
        emu = emu_obs[i]

        if fd_dagger is None and getattr(model, "energy", None) is not None:
            broke = _first_mismatch(model, emu, spec.dagger)
            if broke is not None:
                fd_dagger = i
                _note(broke)

        if fd_pi is None:
            broke = _first_mismatch(model, emu, spec.pi)
            if broke is not None:
                fd_pi = i
                _note(broke)

        if fd_sigma is None:
            if fd_pi is not None:
                fd_sigma = fd_pi
            else:
                broke = _first_mismatch(model, emu, spec.sigma)
                if broke is not None:
                    fd_sigma = i
                    _note(broke)

        if _lag_broke(model, emu, spec):
            _note("frame_counter")
            if cause is not DivergenceCause.ROOM:
                cause = DivergenceCause.LAG
            if fd_sigma is not None and fd_sigma_plus is None:
                fd_sigma_plus = fd_sigma
            lag = True
            break

        if first_diff_field is None:
            broke = _first_mismatch(model, emu, spec.velocity)
            if broke is not None:
                _note(broke)
            elif spec.extra_velocity:
                broke = _first_mismatch(model, emu, spec.extra_velocity)
                if broke is not None:
                    _note(broke)

        if fd_sigma_plus is None:
            if fd_sigma is not None:
                fd_sigma_plus = fd_sigma
            else:
                broke = _first_mismatch(model, emu, spec.sigma_plus)
                if broke is not None:
                    fd_sigma_plus = i
                    _note(broke)

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
