"""Certified full-environment snapshots for deterministic rollouts.

Implements rr-gbd.32: :class:`SnapshotAdapter` + :class:`SnapshotEnvelope`
with adapter/schema/core/game identity and certification levels
:attr:`SnapshotCertification.EMULATOR_ONLY` versus
:attr:`SnapshotCertification.CERTIFIED_FULL_ENV`.

Raw emulator ``get_state``/``set_state`` snapshots remain supported through
:class:`EmulatorOnlyAdapter` (uncertified).  Full-environment certification
requires an adapter that also captures and restores wrapper counters, caches,
RNGs, and any other Python-side state that affects future behavior.

Identity checks run **before** mutation: a mismatched envelope refuses to call
``set_state`` or restore adapter state.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from retro_harness.identity import digest_record, require_nonempty


SNAPSHOT_SCHEMA_VERSION = 1
EMULATOR_ONLY_ADAPTER_ID = "retro_harness.snapshot.EmulatorOnlyAdapter"


class SnapshotCertification(str, Enum):
    """How much of the environment a snapshot is claimed to restore."""

    EMULATOR_ONLY = "emulator_only"
    CERTIFIED_FULL_ENV = "certified_full_env"

    @classmethod
    def from_value(cls, value: Any) -> "SnapshotCertification":
        if isinstance(value, cls):
            return value
        if isinstance(value, Enum):
            value = value.value
        if not isinstance(value, str):
            raise TypeError("snapshot certification must be a string")
        normalized = value.strip().casefold().replace("-", "_")
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"invalid snapshot certification {value!r}; "
            "expected emulator_only or certified_full_env"
        )


class SnapshotError(ValueError):
    """Base error for snapshot capture/restore failures."""


class SnapshotIdentityMismatch(SnapshotError):
    """Raised when envelope identity does not match the live environment."""


@dataclass(frozen=True, slots=True)
class SnapshotIdentity:
    """Stable identity bound to a snapshot envelope.

    Fields:
        adapter_id: Fully-qualified adapter identity string.
        schema_version: Adapter schema version used when capturing.
        core_identity_digest: Emulator-core / backend identity digest.
        game_identity_digest: Game / ROM / start-state identity digest.
    """

    adapter_id: str
    schema_version: int
    core_identity_digest: str
    game_identity_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "adapter_id", require_nonempty(self.adapter_id, "adapter_id")
        )
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise SnapshotError("schema_version must be a positive int")
        object.__setattr__(
            self,
            "core_identity_digest",
            require_nonempty(self.core_identity_digest, "core_identity_digest"),
        )
        object.__setattr__(
            self,
            "game_identity_digest",
            require_nonempty(self.game_identity_digest, "game_identity_digest"),
        )

    def identity_record(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "schema_version": self.schema_version,
            "core_identity_digest": self.core_identity_digest,
            "game_identity_digest": self.game_identity_digest,
        }

    @property
    def identity_digest(self) -> str:
        return digest_record("snapshot-identity-v1", self.identity_record())

    def to_record(self) -> dict[str, Any]:
        return {**self.identity_record(), "identity_digest": self.identity_digest}

    def matches(self, other: "SnapshotIdentity") -> bool:
        if not isinstance(other, SnapshotIdentity):
            return False
        return self.identity_digest == other.identity_digest


@dataclass(frozen=True, slots=True)
class SnapshotEnvelope:
    """One lane's snapshot with certification and identity metadata.

    ``emulator_state`` is always present.  ``adapter_state`` is required for
    :attr:`SnapshotCertification.CERTIFIED_FULL_ENV` and must be ``None`` for
    :attr:`SnapshotCertification.EMULATOR_ONLY`.
    """

    certification: SnapshotCertification
    identity: SnapshotIdentity
    emulator_state: Any
    adapter_state: Any | None = None

    def __post_init__(self) -> None:
        certification = SnapshotCertification.from_value(self.certification)
        object.__setattr__(self, "certification", certification)
        if not isinstance(self.identity, SnapshotIdentity):
            raise SnapshotError("envelope identity must be a SnapshotIdentity")
        if (
            certification is SnapshotCertification.CERTIFIED_FULL_ENV
            and self.adapter_state is None
        ):
            raise SnapshotError(
                "CERTIFIED_FULL_ENV envelopes require non-None adapter_state"
            )
        if (
            certification is SnapshotCertification.EMULATOR_ONLY
            and self.adapter_state is not None
        ):
            raise SnapshotError(
                "EMULATOR_ONLY envelopes must not carry adapter_state"
            )
        # Independent copy so live env mutation cannot corrupt the envelope.
        object.__setattr__(self, "emulator_state", deepcopy(self.emulator_state))
        if self.adapter_state is not None:
            object.__setattr__(self, "adapter_state", deepcopy(self.adapter_state))

    @property
    def is_certified_full_env(self) -> bool:
        return self.certification is SnapshotCertification.CERTIFIED_FULL_ENV

    @property
    def is_emulator_only(self) -> bool:
        return self.certification is SnapshotCertification.EMULATOR_ONLY


@dataclass(frozen=True, slots=True)
class PoolSnapshot:
    """Pool-wide ordered envelopes (one per lane)."""

    envelopes: tuple[SnapshotEnvelope, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.envelopes, tuple):
            object.__setattr__(self, "envelopes", tuple(self.envelopes))
        if not self.envelopes:
            raise SnapshotError("PoolSnapshot requires at least one envelope")
        if any(not isinstance(item, SnapshotEnvelope) for item in self.envelopes):
            raise SnapshotError("PoolSnapshot envelopes must be SnapshotEnvelope")
        certifications = {item.certification for item in self.envelopes}
        if len(certifications) != 1:
            raise SnapshotError(
                "PoolSnapshot envelopes must share one certification level"
            )

    def __len__(self) -> int:
        return len(self.envelopes)

    @property
    def certification(self) -> SnapshotCertification:
        return self.envelopes[0].certification


@runtime_checkable
class SnapshotAdapter(Protocol):
    """Capture and restore environment state beyond raw emulator bytes.

    Implementations must be deterministic: identical envs produce equal
    adapter state, and restore must put the env on the same trajectory.
    """

    @property
    def adapter_id(self) -> str:
        """Stable fully-qualified adapter identity."""

    @property
    def schema_version(self) -> int:
        """Adapter schema version participating in envelope identity."""

    @property
    def certification(self) -> SnapshotCertification:
        """Certification level this adapter can produce."""

    def core_identity_digest(self, env: Any) -> str:
        """Digest identifying the emulator core / backend for ``env``."""

    def game_identity_digest(self, env: Any) -> str:
        """Digest identifying the game / ROM / start for ``env``."""

    def capture_adapter_state(self, env: Any) -> Any | None:
        """Capture Python-side state (counters, caches, RNG, …).

        Emulator-only adapters return ``None``.
        """

    def restore_adapter_state(self, env: Any, state: Any | None) -> None:
        """Restore Python-side state previously returned by capture."""


def _state_getter(env: Any) -> Any:
    emulator = getattr(env, "em", None)
    getter = getattr(emulator, "get_state", None)
    if callable(getter):
        return getter
    getter = getattr(env, "get_state", None)
    if callable(getter):
        return getter
    raise TypeError(
        "environments must expose get_state() or em.get_state() for snapshots"
    )


def _state_setter(env: Any) -> Any:
    emulator = getattr(env, "em", None)
    setter = getattr(emulator, "set_state", None)
    if callable(setter):
        return setter
    setter = getattr(env, "set_state", None)
    if callable(setter):
        return setter
    raise TypeError(
        "environments must expose set_state() or em.set_state() for snapshots"
    )


def get_emulator_state(env: Any) -> Any:
    """Return an independent copy of the raw emulator state payload."""

    return deepcopy(_state_getter(env)())


def set_emulator_state(env: Any, state: Any) -> None:
    """Load an independent copy of a raw emulator state payload."""

    _state_setter(env)(deepcopy(state))


@dataclass(frozen=True, slots=True)
class EmulatorOnlyAdapter:
    """Uncertified adapter: emulator state only, no wrapper restoration.

    Use this (or omit a full-env adapter) when only ``get_state``/``set_state``
    matter.  Envelopes are marked :attr:`SnapshotCertification.EMULATOR_ONLY`
    and are **not** full-environment certified.
    """

    core_digest: str = "unknown-core"
    game_digest: str = "unknown-game"
    adapter_id: str = EMULATOR_ONLY_ADAPTER_ID
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    @property
    def certification(self) -> SnapshotCertification:
        return SnapshotCertification.EMULATOR_ONLY

    def core_identity_digest(self, env: Any) -> str:
        del env
        return self.core_digest

    def game_identity_digest(self, env: Any) -> str:
        del env
        return self.game_digest

    def capture_adapter_state(self, env: Any) -> None:
        del env
        return None

    def restore_adapter_state(self, env: Any, state: Any | None) -> None:
        del env, state
        return None


@dataclass(frozen=True, slots=True)
class AttributeSnapshotAdapter:
    """Full-env adapter that deep-copies named attributes on the env.

    Suitable for thin wrappers that keep counters, caches, and RNG objects as
    ordinary attributes.  RNG objects are restored via ``getstate``/``setstate``
    when those methods exist; otherwise the attribute is deep-copied.
    """

    adapter_id: str
    attributes: tuple[str, ...]
    core_digest: str = "unknown-core"
    game_digest: str = "unknown-game"
    schema_version: int = SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "adapter_id", require_nonempty(self.adapter_id, "adapter_id")
        )
        if not self.attributes:
            raise SnapshotError("AttributeSnapshotAdapter requires attributes")
        object.__setattr__(self, "attributes", tuple(self.attributes))
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise SnapshotError("schema_version must be a positive int")

    @property
    def certification(self) -> SnapshotCertification:
        return SnapshotCertification.CERTIFIED_FULL_ENV

    def core_identity_digest(self, env: Any) -> str:
        del env
        return self.core_digest

    def game_identity_digest(self, env: Any) -> str:
        del env
        return self.game_digest

    def capture_adapter_state(self, env: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name in self.attributes:
            if not hasattr(env, name):
                raise SnapshotError(
                    f"adapter attribute {name!r} missing on {type(env).__name__}"
                )
            value = getattr(env, name)
            getstate = getattr(value, "getstate", None)
            if callable(getstate):
                payload[name] = {"__rng_state__": deepcopy(getstate())}
            else:
                payload[name] = deepcopy(value)
        return payload

    def restore_adapter_state(self, env: Any, state: Any | None) -> None:
        if not isinstance(state, dict):
            raise SnapshotError("AttributeSnapshotAdapter state must be a dict")
        missing = [name for name in self.attributes if name not in state]
        if missing:
            raise SnapshotError(
                f"adapter state missing attributes: {', '.join(missing)}"
            )
        for name in self.attributes:
            raw = state[name]
            if isinstance(raw, dict) and set(raw.keys()) == {"__rng_state__"}:
                current = getattr(env, name, None)
                setstate = getattr(current, "setstate", None)
                if not callable(setstate):
                    raise SnapshotError(
                        f"attribute {name!r} has no setstate for RNG restore"
                    )
                setstate(deepcopy(raw["__rng_state__"]))
            else:
                setattr(env, name, deepcopy(raw))


def identity_for(env: Any, adapter: SnapshotAdapter) -> SnapshotIdentity:
    """Build a :class:`SnapshotIdentity` from ``env`` and ``adapter``."""

    if not isinstance(adapter.schema_version, int) or isinstance(
        adapter.schema_version, bool
    ):
        raise SnapshotError("adapter schema_version must be an int")
    return SnapshotIdentity(
        adapter_id=require_nonempty(adapter.adapter_id, "adapter_id"),
        schema_version=adapter.schema_version,
        core_identity_digest=require_nonempty(
            adapter.core_identity_digest(env), "core_identity_digest"
        ),
        game_identity_digest=require_nonempty(
            adapter.game_identity_digest(env), "game_identity_digest"
        ),
    )


def capture_envelope(env: Any, adapter: SnapshotAdapter) -> SnapshotEnvelope:
    """Capture one envelope for ``env`` using ``adapter``."""

    certification = SnapshotCertification.from_value(adapter.certification)
    adapter_state = adapter.capture_adapter_state(env)
    if certification is SnapshotCertification.EMULATOR_ONLY:
        adapter_state = None
    elif adapter_state is None:
        raise SnapshotError(
            "CERTIFIED_FULL_ENV adapters must return non-None adapter_state"
        )
    return SnapshotEnvelope(
        certification=certification,
        identity=identity_for(env, adapter),
        emulator_state=get_emulator_state(env),
        adapter_state=adapter_state,
    )


def assert_envelope_compatible(
    env: Any,
    envelope: SnapshotEnvelope,
    adapter: SnapshotAdapter,
) -> SnapshotIdentity:
    """Validate identity; raise :class:`SnapshotIdentityMismatch` if not.

    Does not mutate ``env``.
    """

    if not isinstance(envelope, SnapshotEnvelope):
        raise TypeError("envelope must be a SnapshotEnvelope")
    expected = identity_for(env, adapter)
    if envelope.identity.adapter_id != expected.adapter_id:
        raise SnapshotIdentityMismatch(
            "snapshot adapter_id mismatch: "
            f"envelope={envelope.identity.adapter_id!r} "
            f"live={expected.adapter_id!r}"
        )
    if envelope.identity.schema_version != expected.schema_version:
        raise SnapshotIdentityMismatch(
            "snapshot schema_version mismatch: "
            f"envelope={envelope.identity.schema_version} "
            f"live={expected.schema_version}"
        )
    if envelope.identity.core_identity_digest != expected.core_identity_digest:
        raise SnapshotIdentityMismatch(
            "snapshot core identity mismatch: "
            f"envelope={envelope.identity.core_identity_digest!r} "
            f"live={expected.core_identity_digest!r}"
        )
    if envelope.identity.game_identity_digest != expected.game_identity_digest:
        raise SnapshotIdentityMismatch(
            "snapshot game identity mismatch: "
            f"envelope={envelope.identity.game_identity_digest!r} "
            f"live={expected.game_identity_digest!r}"
        )
    adapter_cert = SnapshotCertification.from_value(adapter.certification)
    if envelope.certification is not adapter_cert:
        raise SnapshotIdentityMismatch(
            "snapshot certification mismatch: "
            f"envelope={envelope.certification.value} "
            f"adapter={adapter_cert.value}"
        )
    return expected


def restore_envelope(
    env: Any,
    envelope: SnapshotEnvelope,
    adapter: SnapshotAdapter,
) -> None:
    """Restore ``envelope`` onto ``env`` after identity checks.

    Identity is verified before any ``set_state`` or adapter restore so a
    mismatch leaves the environment untouched.
    """

    assert_envelope_compatible(env, envelope, adapter)
    # Emulator first, then adapter, so wrapper logic sees restored core.
    set_emulator_state(env, envelope.emulator_state)
    if envelope.certification is SnapshotCertification.CERTIFIED_FULL_ENV:
        adapter.restore_adapter_state(env, deepcopy(envelope.adapter_state))
    else:
        # Explicit no-op path for uncertified envelopes.
        adapter.restore_adapter_state(env, None)


__all__ = [
    "AttributeSnapshotAdapter",
    "EMULATOR_ONLY_ADAPTER_ID",
    "EmulatorOnlyAdapter",
    "PoolSnapshot",
    "SNAPSHOT_SCHEMA_VERSION",
    "SnapshotAdapter",
    "SnapshotCertification",
    "SnapshotEnvelope",
    "SnapshotError",
    "SnapshotIdentity",
    "SnapshotIdentityMismatch",
    "assert_envelope_compatible",
    "capture_envelope",
    "get_emulator_state",
    "identity_for",
    "restore_envelope",
    "set_emulator_state",
]
