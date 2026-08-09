"""Versioned, canonical environment/model compatibility contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class ContractError(ValueError):
    """Base error for malformed or incompatible contract records."""


class ContractMismatchError(ContractError):
    """Raised when a checkpoint/runtime contract comparison fails closed."""


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    return value.strip()


def _jsonable(value: Any, path: str = "value") -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ContractError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path} mapping keys must be strings")
            out[key] = _jsonable(item, f"{path}.{key}")
        return dict(sorted(out.items()))
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, f"{path}[]") for item in value]
    raise ContractError(f"{path} contains unsupported {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def contract_digest(kind: str, record: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json({"kind": _nonempty(kind, "kind"), **dict(record)}).encode(
            "utf-8"
        )
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise ContractError(f"identity file does not exist: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_digest(kind: str, value: str) -> str:
    return contract_digest("identity-v1", {"identity_kind": kind, "value": value})


@dataclass(frozen=True, slots=True)
class ObservationField:
    name: str
    dtype: str
    shape: tuple[int, ...] = ()
    semantic: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "field name"))
        object.__setattr__(self, "dtype", _nonempty(self.dtype, "field dtype"))
        shape = tuple(self.shape)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in shape
        ):
            raise ContractError("observation field dimensions must be positive integers")
        object.__setattr__(self, "shape", shape)
        if not isinstance(self.semantic, str):
            raise ContractError("field semantic must be a string")

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "semantic": self.semantic,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ObservationField":
        return cls(
            name=record["name"],
            dtype=record["dtype"],
            shape=tuple(record.get("shape", ())),
            semantic=record.get("semantic", ""),
        )


@dataclass(frozen=True, slots=True)
class ObservationContract:
    fields: tuple[ObservationField, ...]
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1"

    def __post_init__(self) -> None:
        values = tuple(self.fields)
        if not values:
            raise ContractError("observation contract requires at least one field")
        if not all(isinstance(value, ObservationField) for value in values):
            raise ContractError("observation fields must be ObservationField values")
        names = [value.name for value in values]
        if len(names) != len(set(names)):
            raise ContractError("observation field names must be unique")
        object.__setattr__(self, "fields", values)
        object.__setattr__(
            self, "preprocessing", _jsonable(dict(self.preprocessing), "preprocessing")
        )
        object.__setattr__(self, "version", _nonempty(self.version, "version"))

    def identity_record(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "fields": [value.to_record() for value in self.fields],
            "preprocessing": dict(self.preprocessing),
        }

    @property
    def identity_digest(self) -> str:
        return contract_digest("observation-contract-v1", self.identity_record())

    def to_record(self) -> dict[str, Any]:
        return {**self.identity_record(), "identity_digest": self.identity_digest}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ObservationContract":
        value = cls(
            fields=tuple(
                ObservationField.from_record(item) for item in record["fields"]
            ),
            preprocessing=record.get("preprocessing", {}),
            version=record.get("version", "1"),
        )
        _verify_published_digest(value.identity_digest, record)
        return value


@dataclass(frozen=True, slots=True)
class ActionEntry:
    action_id: str
    buttons: tuple[int, ...]
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _nonempty(self.action_id, "action_id"))
        buttons = tuple(self.buttons)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1)
            for value in buttons
        ):
            raise ContractError("action buttons must contain integer 0/1 values")
        object.__setattr__(self, "buttons", buttons)
        if not isinstance(self.label, str):
            raise ContractError("action label must be a string")

    def to_record(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "buttons": list(self.buttons),
            "label": self.label,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ActionEntry":
        return cls(
            action_id=record["action_id"],
            buttons=tuple(record["buttons"]),
            label=record.get("label", ""),
        )


@dataclass(frozen=True, slots=True)
class ActionContract:
    controller_buttons: tuple[str, ...]
    entries: tuple[ActionEntry, ...]
    version: str = "1"

    def __post_init__(self) -> None:
        buttons = tuple(_nonempty(value, "controller button") for value in self.controller_buttons)
        if not buttons or len(buttons) != len(set(buttons)):
            raise ContractError("controller button names must be non-empty and unique")
        entries = tuple(self.entries)
        if not entries:
            raise ContractError("action contract requires at least one entry")
        if not all(isinstance(value, ActionEntry) for value in entries):
            raise ContractError("action entries must be ActionEntry values")
        if any(len(value.buttons) != len(buttons) for value in entries):
            raise ContractError("action entry width does not match controller buttons")
        ids = [value.action_id for value in entries]
        if len(ids) != len(set(ids)):
            raise ContractError("action IDs must be unique")
        object.__setattr__(self, "controller_buttons", buttons)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "version", _nonempty(self.version, "version"))

    @property
    def action_count(self) -> int:
        return len(self.entries)

    def identity_record(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "controller_buttons": list(self.controller_buttons),
            "entries": [entry.to_record() for entry in self.entries],
        }

    @property
    def identity_digest(self) -> str:
        return contract_digest("action-contract-v1", self.identity_record())

    def to_record(self) -> dict[str, Any]:
        return {**self.identity_record(), "identity_digest": self.identity_digest}

    @classmethod
    def from_button_rows(
        cls,
        rows: Sequence[Sequence[int]],
        *,
        controller_buttons: Sequence[str],
        labels: Sequence[str] = (),
        version: str = "1",
    ) -> "ActionContract":
        label_values = tuple(labels)
        return cls(
            controller_buttons=tuple(controller_buttons),
            entries=tuple(
                ActionEntry(
                    action_id=str(index),
                    buttons=tuple(row),
                    label=label_values[index] if index < len(label_values) else "",
                )
                for index, row in enumerate(rows)
            ),
            version=version,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ActionContract":
        value = cls(
            controller_buttons=tuple(record["controller_buttons"]),
            entries=tuple(ActionEntry.from_record(item) for item in record["entries"]),
            version=record.get("version", "1"),
        )
        _verify_published_digest(value.identity_digest, record)
        return value


@dataclass(frozen=True, slots=True)
class RewardComponent:
    name: str
    weight: float
    semantic: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "reward name"))
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise ContractError("reward weight must be numeric")
        value = float(self.weight)
        if value != value or value in (float("inf"), float("-inf")):
            raise ContractError("reward weight must be finite")
        object.__setattr__(self, "weight", value)
        object.__setattr__(self, "semantic", _nonempty(self.semantic, "reward semantic"))

    def to_record(self) -> dict[str, Any]:
        return {"name": self.name, "weight": self.weight, "semantic": self.semantic}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "RewardComponent":
        return cls(record["name"], record["weight"], record["semantic"])


@dataclass(frozen=True, slots=True)
class RewardContract:
    components: tuple[RewardComponent, ...]
    aggregation: str = "sum"
    version: str = "1"

    def __post_init__(self) -> None:
        values = tuple(self.components)
        if not values or not all(isinstance(value, RewardComponent) for value in values):
            raise ContractError("reward contract requires RewardComponent values")
        names = [value.name for value in values]
        if len(names) != len(set(names)):
            raise ContractError("reward component names must be unique")
        object.__setattr__(self, "components", values)
        object.__setattr__(self, "aggregation", _nonempty(self.aggregation, "aggregation"))
        object.__setattr__(self, "version", _nonempty(self.version, "version"))

    def identity_record(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "aggregation": self.aggregation,
            "components": [value.to_record() for value in self.components],
        }

    @property
    def identity_digest(self) -> str:
        return contract_digest("reward-contract-v1", self.identity_record())

    def to_record(self) -> dict[str, Any]:
        return {**self.identity_record(), "identity_digest": self.identity_digest}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "RewardContract":
        value = cls(
            components=tuple(
                RewardComponent.from_record(item) for item in record["components"]
            ),
            aggregation=record.get("aggregation", "sum"),
            version=record.get("version", "1"),
        )
        _verify_published_digest(value.identity_digest, record)
        return value


@dataclass(frozen=True, slots=True)
class WrapperSpec:
    name: str
    config: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "wrapper name"))
        object.__setattr__(self, "version", _nonempty(self.version, "version"))
        object.__setattr__(self, "config", _jsonable(dict(self.config), "wrapper config"))

    def to_record(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "config": dict(self.config)}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "WrapperSpec":
        return cls(record["name"], record.get("config", {}), record.get("version", "1"))


@dataclass(frozen=True, slots=True)
class WrapperContract:
    stack: tuple[WrapperSpec, ...]
    version: str = "1"

    def __post_init__(self) -> None:
        values = tuple(self.stack)
        if not values or not all(isinstance(value, WrapperSpec) for value in values):
            raise ContractError("wrapper contract requires an ordered non-empty stack")
        object.__setattr__(self, "stack", values)
        object.__setattr__(self, "version", _nonempty(self.version, "version"))

    def identity_record(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "stack": [value.to_record() for value in self.stack],
        }

    @property
    def identity_digest(self) -> str:
        return contract_digest("wrapper-contract-v1", self.identity_record())

    def to_record(self) -> dict[str, Any]:
        return {**self.identity_record(), "identity_digest": self.identity_digest}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "WrapperContract":
        value = cls(
            stack=tuple(WrapperSpec.from_record(item) for item in record["stack"]),
            version=record.get("version", "1"),
        )
        _verify_published_digest(value.identity_digest, record)
        return value


@dataclass(frozen=True, slots=True)
class EnvironmentContract:
    game_id: str
    state_id: str
    action_space_size: int
    frame_skip: int
    rom_identity_digest: str
    state_identity_digest: str
    core_identity_digest: str
    players: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = "1"

    def __post_init__(self) -> None:
        for name in (
            "game_id",
            "state_id",
            "rom_identity_digest",
            "state_identity_digest",
            "core_identity_digest",
            "version",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in ("action_space_size", "frame_skip", "players"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractError(f"{name} must be a positive integer")
        object.__setattr__(self, "metadata", _jsonable(dict(self.metadata), "metadata"))

    def identity_record(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "game_id": self.game_id,
            "state_id": self.state_id,
            "action_space_size": self.action_space_size,
            "frame_skip": self.frame_skip,
            "players": self.players,
            "rom_identity_digest": self.rom_identity_digest,
            "state_identity_digest": self.state_identity_digest,
            "core_identity_digest": self.core_identity_digest,
            "metadata": dict(self.metadata),
        }

    @property
    def identity_digest(self) -> str:
        return contract_digest("environment-contract-v1", self.identity_record())

    def to_record(self) -> dict[str, Any]:
        return {**self.identity_record(), "identity_digest": self.identity_digest}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "EnvironmentContract":
        value = cls(
            game_id=record["game_id"],
            state_id=record["state_id"],
            action_space_size=record["action_space_size"],
            frame_skip=record["frame_skip"],
            players=record.get("players", 1),
            rom_identity_digest=record["rom_identity_digest"],
            state_identity_digest=record["state_identity_digest"],
            core_identity_digest=record["core_identity_digest"],
            metadata=record.get("metadata", {}),
            version=record.get("version", "1"),
        )
        _verify_published_digest(value.identity_digest, record)
        return value


@dataclass(frozen=True, slots=True)
class ContractBundle:
    environment: EnvironmentContract
    observation: ObservationContract
    action: ActionContract
    reward: RewardContract
    wrappers: WrapperContract
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ContractError("unsupported ContractBundle schema_version")
        expected = (
            (self.environment, EnvironmentContract),
            (self.observation, ObservationContract),
            (self.action, ActionContract),
            (self.reward, RewardContract),
            (self.wrappers, WrapperContract),
        )
        if any(not isinstance(value, kind) for value, kind in expected):
            raise ContractError("contract bundle contains an invalid component")
        if self.environment.action_space_size != self.action.action_count:
            raise ContractError("environment action size does not match action contract")

    @property
    def schema_digests(self) -> dict[str, str]:
        return {
            "observation": self.observation.identity_digest,
            "action": self.action.identity_digest,
            "reward": self.reward.identity_digest,
            "wrapper": self.wrappers.identity_digest,
        }

    @property
    def environment_identity_digests(self) -> dict[str, str]:
        return {
            "rom": self.environment.rom_identity_digest,
            "state": self.environment.state_identity_digest,
            "core": self.environment.core_identity_digest,
        }

    def identity_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "environment": self.environment.to_record(),
            "observation": self.observation.to_record(),
            "action": self.action.to_record(),
            "reward": self.reward.to_record(),
            "wrappers": self.wrappers.to_record(),
        }

    @property
    def identity_digest(self) -> str:
        return contract_digest("contract-bundle-v1", self.identity_record())

    def to_record(self) -> dict[str, Any]:
        return {**self.identity_record(), "identity_digest": self.identity_digest}

    def assert_compatible(self, other: "ContractBundle") -> None:
        if not isinstance(other, ContractBundle):
            raise TypeError("other must be a ContractBundle")
        mismatches: list[str] = []
        if self.environment.identity_digest != other.environment.identity_digest:
            mismatches.append("environment")
        for name, digest in self.schema_digests.items():
            if digest != other.schema_digests[name]:
                mismatches.append(name)
        if mismatches:
            raise ContractMismatchError(
                "contract mismatch: " + ", ".join(mismatches)
            )

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_record(), allow_nan=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        return output

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ContractBundle":
        value = cls(
            environment=EnvironmentContract.from_record(record["environment"]),
            observation=ObservationContract.from_record(record["observation"]),
            action=ActionContract.from_record(record["action"]),
            reward=RewardContract.from_record(record["reward"]),
            wrappers=WrapperContract.from_record(record["wrappers"]),
            schema_version=record.get("schema_version", 1),
        )
        _verify_published_digest(value.identity_digest, record)
        return value

    @classmethod
    def load(cls, path: str | Path) -> "ContractBundle":
        return cls.from_record(json.loads(Path(path).read_text(encoding="utf-8")))


def _verify_published_digest(actual: str, record: Mapping[str, Any]) -> None:
    if record.get("identity_digest") != actual:
        raise ContractMismatchError("published contract identity digest mismatch")


SNES_BUTTONS = (
    "B",
    "Y",
    "SELECT",
    "START",
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "A",
    "X",
    "L",
    "R",
)


def action_rows_from_sparse_maps(
    values: Iterable[Mapping[int, int]],
    *,
    width: int = 12,
) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = []
    for mapping in values:
        row = [0] * width
        for index, enabled in mapping.items():
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < width:
                raise ContractError("sparse action index is out of range")
            if enabled not in (0, 1):
                raise ContractError("sparse action values must be 0 or 1")
            row[index] = int(enabled)
        rows.append(tuple(row))
    return tuple(rows)


__all__ = [
    "ActionContract",
    "ActionEntry",
    "ContractBundle",
    "ContractError",
    "ContractMismatchError",
    "EnvironmentContract",
    "ObservationContract",
    "ObservationField",
    "RewardComponent",
    "RewardContract",
    "SNES_BUTTONS",
    "WrapperContract",
    "WrapperSpec",
    "action_rows_from_sparse_maps",
    "canonical_json",
    "contract_digest",
    "identity_digest",
    "sha256_file",
]
