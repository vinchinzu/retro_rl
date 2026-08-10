"""Owned intervention counters and typed audit evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from retro_harness.identity import require_nonempty


class RuntimeObservationClass(str, Enum):
    """What the policy may observe while an attempt is active."""

    GOLD = "Gold"
    SILVER = "Silver"
    BRONZE = "Bronze"

    @classmethod
    def _missing_(cls, value: object) -> "RuntimeObservationClass | None":
        if isinstance(value, str):
            normalized = value.strip().casefold()
            for member in cls:
                if member.value.casefold() == normalized:
                    return member
        return None

    @classmethod
    def from_value(cls, value: Any) -> "RuntimeObservationClass":
        if isinstance(value, cls):
            return value
        if isinstance(value, Enum):
            value = value.value
        if not isinstance(value, str):
            raise TypeError(
                "runtime observation class must be an enum or string"
            )
        normalized = value.strip().casefold().replace("_", "-").replace(" ", "-")
        values = {member.value.casefold(): member for member in cls}
        if normalized in values:
            return values[normalized]
        raise ValueError(
            f"invalid runtime observation class {value!r}; "
            "expected Gold, Silver, or Bronze"
        )


class InterventionClass(str, Enum):
    """What an attempt may mutate, independent of runtime observations."""

    CLEAN = "Clean"
    ASSISTED = "Assisted"
    SURVIVAL_ASSISTED = "Survival-assisted"
    RESOURCE_ASSISTED = "Resource-assisted"
    PROTECTION_ASSISTED = "Protection-assisted"
    PROGRESSION_ASSISTED = "Progression-assisted"

    @classmethod
    def _missing_(cls, value: object) -> "InterventionClass | None":
        if isinstance(value, str):
            normalized = value.strip().casefold().replace("_", "-").replace(" ", "-")
            for member in cls:
                if member.value.casefold() == normalized:
                    return member
        return None

    @classmethod
    def from_value(cls, value: Any) -> "InterventionClass":
        if isinstance(value, cls):
            return value
        if isinstance(value, Enum):
            value = value.value
        if not isinstance(value, str):
            raise TypeError("intervention class must be an enum or string")
        normalized = value.strip().casefold().replace("_", "-").replace(" ", "-")
        values = {member.value.casefold(): member for member in cls}
        aliases = {
            "clean": cls.CLEAN,
            "assisted": cls.ASSISTED,
            "survival": cls.SURVIVAL_ASSISTED,
            "resource": cls.RESOURCE_ASSISTED,
            "protection": cls.PROTECTION_ASSISTED,
            "progression": cls.PROGRESSION_ASSISTED,
        }
        if normalized in values:
            return values[normalized]
        if normalized in aliases:
            return aliases[normalized]
        parts = [part for part in normalized.split("+") if part]
        known_assists = {
            "survival-assisted",
            "resource-assisted",
            "protection-assisted",
            "progression-assisted",
        }
        if len(parts) > 1 and all(part in known_assists for part in parts):
            return cls.ASSISTED
        raise ValueError(
            f"invalid intervention class {value!r}; "
            "expected Clean or a known assisted class"
        )

    @property
    def is_clean(self) -> bool:
        return self is type(self).CLEAN


def normalize_assists(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if isinstance(value, bool):
        return {"assist": int(value)} if value else {}
    if isinstance(value, int):
        if value < 0:
            raise ValueError("assist count must be non-negative")
        return {"assist": value} if value else {}
    if not isinstance(value, Mapping):
        raise TypeError("assists must be a mapping, integer, or bool")
    normalized: dict[str, int] = {}
    for key, count in value.items():
        name = require_nonempty(key, "assist name")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("assist counts must be non-negative integers")
        if count:
            normalized[name] = count
    return dict(sorted(normalized.items()))


def _optional_event_count(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer or None")
    return value


def _positive_increment(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} count must be a positive integer")
    return value


@dataclass(frozen=True)
class AuditCapabilities:
    """Proof that an audit provider can observe every intervention channel."""

    provider: str
    observes_ram_writes: bool
    observes_mid_run_loads: bool
    observes_assists: bool
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider", require_nonempty(self.provider, "audit capability provider")
        )
        if self.schema_version != 1:
            raise ValueError("unsupported audit capability schema_version")
        for field_name in (
            "observes_ram_writes",
            "observes_mid_run_loads",
            "observes_assists",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")

    @property
    def complete(self) -> bool:
        return (
            self.observes_ram_writes
            and self.observes_mid_run_loads
            and self.observes_assists
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "observes_ram_writes": self.observes_ram_writes,
            "observes_mid_run_loads": self.observes_mid_run_loads,
            "observes_assists": self.observes_assists,
        }

    @classmethod
    def all(cls, provider: str) -> "AuditCapabilities":
        return cls(provider, True, True, True)

    @classmethod
    def from_value(cls, value: Any) -> "AuditCapabilities | None":
        if value is None or isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("audit_capabilities must be a mapping")
        return cls(
            provider=value.get("provider", ""),
            observes_ram_writes=value.get("observes_ram_writes"),
            observes_mid_run_loads=value.get("observes_mid_run_loads"),
            observes_assists=value.get("observes_assists"),
            schema_version=value.get("schema_version", 1),
        )


@dataclass(frozen=True)
class AttemptAudit:
    """Observed interventions and identity evidence for one attempt."""

    ram_writes: int | bool | None = None
    mid_run_loads: int | bool | None = None
    assists: Mapping[str, int] | int | bool | None = None
    start_identity_digest: str | None = None
    policy_identity_digest: str | None = None
    runtime_observation_class: RuntimeObservationClass | str | None = None
    intervention_class: InterventionClass | str | None = None
    capabilities: AuditCapabilities | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ram_writes", _optional_event_count(self.ram_writes, "ram_writes")
        )
        object.__setattr__(
            self,
            "mid_run_loads",
            _optional_event_count(self.mid_run_loads, "mid_run_loads"),
        )
        object.__setattr__(
            self,
            "assists",
            None if self.assists is None else normalize_assists(self.assists),
        )
        object.__setattr__(
            self, "capabilities", AuditCapabilities.from_value(self.capabilities)
        )
        for field_name in ("start_identity_digest", "policy_identity_digest"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, require_nonempty(value, field_name))
        if self.runtime_observation_class is not None:
            object.__setattr__(
                self,
                "runtime_observation_class",
                RuntimeObservationClass.from_value(self.runtime_observation_class),
            )
        if self.intervention_class is not None:
            object.__setattr__(
                self,
                "intervention_class",
                InterventionClass.from_value(self.intervention_class),
            )

    @property
    def assist_count(self) -> int:
        return sum((self.assists or {}).values())

    @property
    def has_interventions(self) -> bool:
        return bool(self.ram_writes or self.mid_run_loads or self.assist_count)

    @property
    def has_complete_instrumentation(self) -> bool:
        return bool(
            self.capabilities is not None
            and self.capabilities.complete
            and self.ram_writes is not None
            and self.mid_run_loads is not None
            and self.assists is not None
        )

    @classmethod
    def from_info(cls, info: Mapping[str, Any] | None) -> "AttemptAudit":
        values = info if isinstance(info, Mapping) else {}
        return cls(
            ram_writes=values.get("ram_writes", values.get("ram_write_count")),
            mid_run_loads=values.get(
                "mid_run_loads",
                values.get("mid_run_load_count", values.get("save_state_loads")),
            ),
            assists=values.get("assists"),
            start_identity_digest=values.get("start_identity_digest"),
            policy_identity_digest=values.get("policy_identity_digest"),
            runtime_observation_class=values.get("runtime_observation_class"),
            intervention_class=values.get("intervention_class"),
            capabilities=values.get("audit_capabilities"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "ram_writes": self.ram_writes,
            "mid_run_loads": self.mid_run_loads,
            "assists": dict(self.assists) if self.assists is not None else None,
            "runtime_observation_class": (
                self.runtime_observation_class.value
                if isinstance(self.runtime_observation_class, Enum)
                else self.runtime_observation_class
            ),
            "intervention_class": (
                self.intervention_class.value
                if isinstance(self.intervention_class, Enum)
                else self.intervention_class
            ),
            "start_identity_digest": self.start_identity_digest,
            "policy_identity_digest": self.policy_identity_digest,
            "audit_capabilities": (
                self.capabilities.to_record() if self.capabilities is not None else None
            ),
        }


class _AuditedData:
    def __init__(self, data: Any, owner: "AuditedEnv") -> None:
        self._data = data
        self._owner = owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._data, name)

    def set_value(self, key: str, value: Any) -> Any:
        result = self._data.set_value(key, value)
        self._owner.record_ram_write()
        self._owner.record_assist("data.set_value")
        return result


class _AuditedEmulator:
    def __init__(self, emulator: Any, owner: "AuditedEnv") -> None:
        self._emulator = emulator
        self._owner = owner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._emulator, name)

    def set_state(self, state: bytes) -> Any:
        result = self._emulator.set_state(state)
        self._owner.record_state_load()
        return result


class AuditedEnv:
    """Environment boundary that owns RAM-write, load, and assist counters."""

    def __init__(self, env: Any, *, capabilities: AuditCapabilities):
        if not isinstance(capabilities, AuditCapabilities):
            raise TypeError("capabilities must be an AuditCapabilities")
        self.env = env
        self.audit_capabilities = capabilities
        self._data = _AuditedData(env.data, self) if hasattr(env, "data") else None
        self._emulator = (
            _AuditedEmulator(env.em, self) if hasattr(env, "em") else None
        )
        self._attempt_active = False
        self.begin_attempt()

    @property
    def data(self) -> Any:
        if self._data is None:
            raise AttributeError("wrapped environment has no data interface")
        return self._data

    @property
    def em(self) -> Any:
        if self._emulator is None:
            raise AttributeError("wrapped environment has no emulator interface")
        return self._emulator

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def begin_attempt(
        self,
        *,
        start_identity_digest: str | None = None,
        policy_identity_digest: str | None = None,
        runtime_observation_class: RuntimeObservationClass | str | None = None,
        intervention_class: InterventionClass | str | None = None,
    ) -> None:
        self._ram_writes = 0
        self._mid_run_loads = 0
        self._assists: dict[str, int] = {}
        self._context = {
            "start_identity_digest": start_identity_digest,
            "policy_identity_digest": policy_identity_digest,
            "runtime_observation_class": runtime_observation_class,
            "intervention_class": intervention_class,
        }
        self._attempt_active = True

    def load_start_state(self, state: bytes, **attempt_identity: Any) -> None:
        """Restore a benchmark start before opening the audited attempt."""
        if self._emulator is None:
            raise AttributeError("wrapped environment has no emulator interface")
        self._attempt_active = False
        self.em.set_state(state)
        self.begin_attempt(**attempt_identity)

    def _augment_info(self, info: Mapping[str, Any] | None) -> dict[str, Any]:
        values = dict(info or {})
        values.update(
            {
                "ram_writes": self._ram_writes,
                "mid_run_loads": self._mid_run_loads,
                "assists": dict(sorted(self._assists.items())),
                "audit_capabilities": self.audit_capabilities.to_record(),
                **self._context,
            }
        )
        return values

    def reset(self, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._attempt_active = False
        result = self.env.reset(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("AuditedEnv requires a Gymnasium-style reset result")
        observation, info = result
        self.begin_attempt()
        return observation, self._augment_info(info)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self.env.step(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise TypeError("AuditedEnv requires a Gymnasium-style step result")
        observation, reward, terminated, truncated, info = result
        return observation, reward, terminated, truncated, self._augment_info(info)

    def record_ram_write(self, count: int = 1) -> None:
        self._ram_writes += _positive_increment(count, "RAM write")

    def record_state_load(self, count: int = 1) -> None:
        increment = _positive_increment(count, "state load")
        if self._attempt_active:
            self._mid_run_loads += increment

    def record_assist(self, name: str, count: int = 1) -> None:
        normalized = require_nonempty(name, "assist name")
        self._assists[normalized] = self._assists.get(normalized, 0) + (
            _positive_increment(count, "assist")
        )

    def audit(self) -> AttemptAudit:
        return AttemptAudit.from_info(self._augment_info({}))

    def close(self) -> None:
        self.env.close()


__all__ = [
    "AuditCapabilities",
    "AuditedEnv",
    "AttemptAudit",
    "InterventionClass",
    "RuntimeObservationClass",
    "normalize_assists",
]
