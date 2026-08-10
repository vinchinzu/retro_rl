"""Natural-entry emulator state corpora with leakage-safe train/eval splits."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from retro_harness.identity import (
    canonical_json,
    require_nonempty as _nonempty,
    sha256_bytes,
)


class EntryStateError(ValueError):
    """Raised when a corpus or retained state fails integrity checks."""


def _read_state(path: Path) -> bytes:
    value = path.read_bytes()
    return gzip.decompress(value) if value[:2] == b"\x1f\x8b" else value


@dataclass(frozen=True, slots=True)
class EntryStateRecord:
    state_digest: str
    ram_snapshot_digest: str
    state_path: str
    source_skill_id: str
    source_segment_id: str
    source_trajectory_digest: str
    frame: int
    observation_schema_digest: str
    contract_bundle_digest: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "state_digest",
            "ram_snapshot_digest",
            "state_path",
            "source_skill_id",
            "source_segment_id",
            "source_trajectory_digest",
            "observation_schema_digest",
            "contract_bundle_digest",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        if Path(self.state_path).is_absolute() or ".." in Path(self.state_path).parts:
            raise EntryStateError("state_path must be a contained relative path")
        if isinstance(self.frame, bool) or not isinstance(self.frame, int) or self.frame < 0:
            raise EntryStateError("frame must be a non-negative integer")
        normalized = json.loads(canonical_json(dict(self.metadata)))
        object.__setattr__(self, "metadata", normalized)

    def to_record(self) -> dict[str, Any]:
        return {
            "state_digest": self.state_digest,
            "ram_snapshot_digest": self.ram_snapshot_digest,
            "state_path": self.state_path,
            "source_skill_id": self.source_skill_id,
            "source_segment_id": self.source_segment_id,
            "source_trajectory_digest": self.source_trajectory_digest,
            "frame": self.frame,
            "frame_parity": self.frame % 2,
            "observation_schema_digest": self.observation_schema_digest,
            "contract_bundle_digest": self.contract_bundle_digest,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "EntryStateRecord":
        return cls(
            state_digest=record["state_digest"],
            ram_snapshot_digest=record["ram_snapshot_digest"],
            state_path=record["state_path"],
            source_skill_id=record["source_skill_id"],
            source_segment_id=record["source_segment_id"],
            source_trajectory_digest=record["source_trajectory_digest"],
            frame=record["frame"],
            observation_schema_digest=record["observation_schema_digest"],
            contract_bundle_digest=record["contract_bundle_digest"],
            metadata=record.get("metadata", {}),
        )


class SplitStrategy(str, Enum):
    SOURCE_TRAJECTORY = "source_trajectory"
    HASH_BUCKET = "hash_bucket"


@dataclass(frozen=True, slots=True)
class EntryStateSplit:
    train: tuple[EntryStateRecord, ...]
    eval: tuple[EntryStateRecord, ...]
    strategy: SplitStrategy
    train_fraction: float
    salt: str

    def __post_init__(self) -> None:
        train_digests = {record.state_digest for record in self.train}
        eval_digests = {record.state_digest for record in self.eval}
        if train_digests & eval_digests:
            raise EntryStateError("train/eval state leakage")
        if self.strategy is SplitStrategy.SOURCE_TRAJECTORY:
            train_sources = {
                record.source_trajectory_digest for record in self.train
            }
            eval_sources = {
                record.source_trajectory_digest for record in self.eval
            }
            if train_sources & eval_sources:
                raise EntryStateError("source trajectory leaks across split")

    def partition(self, name: str) -> tuple[EntryStateRecord, ...]:
        if name == "train":
            return self.train
        if name == "eval":
            return self.eval
        raise ValueError("partition must be 'train' or 'eval'")

    def to_record(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "train_fraction": self.train_fraction,
            "salt": self.salt,
            "train_state_digests": [record.state_digest for record in self.train],
            "eval_state_digests": [record.state_digest for record in self.eval],
        }


@dataclass(frozen=True, slots=True)
class EntryStateCorpus:
    corpus_id: str
    game_id: str
    contract_bundle_digest: str
    observation_schema_digest: str
    records: tuple[EntryStateRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise EntryStateError("unsupported EntryStateCorpus schema_version")
        for name in (
            "corpus_id",
            "game_id",
            "contract_bundle_digest",
            "observation_schema_digest",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        records = tuple(self.records)
        if not records:
            raise EntryStateError("entry-state corpus cannot be empty")
        state_digests = [record.state_digest for record in records]
        if len(state_digests) != len(set(state_digests)):
            raise EntryStateError("entry-state corpus contains duplicate states")
        for record in records:
            if record.contract_bundle_digest != self.contract_bundle_digest:
                raise EntryStateError("record contract bundle mismatch")
            if record.observation_schema_digest != self.observation_schema_digest:
                raise EntryStateError("record observation schema mismatch")
        object.__setattr__(
            self, "records", tuple(sorted(records, key=lambda value: value.state_digest))
        )
        object.__setattr__(
            self,
            "metadata",
            json.loads(canonical_json(dict(self.metadata))),
        )

    @property
    def identity_digest(self) -> str:
        return sha256_bytes(canonical_json(self.identity_record()).encode("utf-8"))

    def identity_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "corpus_id": self.corpus_id,
            "game_id": self.game_id,
            "contract_bundle_digest": self.contract_bundle_digest,
            "observation_schema_digest": self.observation_schema_digest,
            "records": [record.to_record() for record in self.records],
            "metadata": dict(self.metadata),
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.identity_record(), "identity_digest": self.identity_digest}

    def filter(
        self,
        *,
        source_skill_id: str | None = None,
        source_segment_id: str | None = None,
        predicate: Callable[[EntryStateRecord], bool] | None = None,
    ) -> tuple[EntryStateRecord, ...]:
        return tuple(
            record
            for record in self.records
            if (source_skill_id is None or record.source_skill_id == source_skill_id)
            and (
                source_segment_id is None
                or record.source_segment_id == source_segment_id
            )
            and (predicate is None or predicate(record))
        )

    def split(
        self,
        *,
        train_fraction: float = 0.8,
        strategy: SplitStrategy = SplitStrategy.HASH_BUCKET,
        salt: str = "entry-state-split-v1",
        require_nonempty: bool = True,
    ) -> EntryStateSplit:
        if not isinstance(strategy, SplitStrategy):
            raise TypeError("strategy must be a SplitStrategy")
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between zero and one")
        salt = _nonempty(salt, "salt")
        threshold = int(train_fraction * 10_000)
        train: list[EntryStateRecord] = []
        evaluation: list[EntryStateRecord] = []
        for record in self.records:
            split_key = (
                record.source_trajectory_digest
                if strategy is SplitStrategy.SOURCE_TRAJECTORY
                else record.state_digest
            )
            bucket = int(
                sha256_bytes(f"{salt}:{split_key}".encode("utf-8"))[:8],
                16,
            ) % 10_000
            (train if bucket < threshold else evaluation).append(record)
        if require_nonempty and (not train or not evaluation):
            raise EntryStateError("split produced an empty train or eval partition")
        return EntryStateSplit(
            tuple(train),
            tuple(evaluation),
            strategy,
            train_fraction,
            salt,
        )

    def state_bytes(
        self,
        record: EntryStateRecord,
        *,
        root: str | Path,
    ) -> bytes:
        root_path = Path(root).resolve()
        path = (root_path / record.state_path).resolve()
        try:
            path.relative_to(root_path)
        except ValueError as exc:
            raise EntryStateError("state path escapes corpus root") from exc
        state = _read_state(path)
        if sha256_bytes(state) != record.state_digest:
            raise EntryStateError(f"state digest mismatch: {record.state_path}")
        return state

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
    def from_record(cls, record: Mapping[str, Any]) -> "EntryStateCorpus":
        value = cls(
            corpus_id=record["corpus_id"],
            game_id=record["game_id"],
            contract_bundle_digest=record["contract_bundle_digest"],
            observation_schema_digest=record["observation_schema_digest"],
            records=tuple(
                EntryStateRecord.from_record(item) for item in record["records"]
            ),
            metadata=record.get("metadata", {}),
            schema_version=record.get("schema_version", 1),
        )
        if record.get("identity_digest") != value.identity_digest:
            raise EntryStateError("EntryStateCorpus identity digest mismatch")
        return value

    @classmethod
    def load(cls, path: str | Path) -> "EntryStateCorpus":
        return cls.from_record(json.loads(Path(path).read_text(encoding="utf-8")))


class EntryStateCorpusBuilder:
    """Deduplicating builder used by game-owned predecessor harvesters."""

    def __init__(
        self,
        *,
        corpus_id: str,
        game_id: str,
        contract_bundle_digest: str,
        observation_schema_digest: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.corpus_id = corpus_id
        self.game_id = game_id
        self.contract_bundle_digest = contract_bundle_digest
        self.observation_schema_digest = observation_schema_digest
        self.metadata = dict(metadata or {})
        self._records: dict[str, EntryStateRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def add(
        self,
        *,
        state_bytes: bytes,
        ram_snapshot: bytes,
        state_path: str,
        source_skill_id: str,
        source_segment_id: str,
        source_trajectory_digest: str,
        frame: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> EntryStateRecord:
        state_digest = sha256_bytes(state_bytes)
        if state_digest in self._records:
            raise EntryStateError("duplicate emulator state in corpus harvest")
        record = EntryStateRecord(
            state_digest=state_digest,
            ram_snapshot_digest=sha256_bytes(ram_snapshot),
            state_path=state_path,
            source_skill_id=source_skill_id,
            source_segment_id=source_segment_id,
            source_trajectory_digest=source_trajectory_digest,
            frame=frame,
            observation_schema_digest=self.observation_schema_digest,
            contract_bundle_digest=self.contract_bundle_digest,
            metadata=dict(metadata or {}),
        )
        self._records[state_digest] = record
        return record

    def build(self) -> EntryStateCorpus:
        return EntryStateCorpus(
            corpus_id=self.corpus_id,
            game_id=self.game_id,
            contract_bundle_digest=self.contract_bundle_digest,
            observation_schema_digest=self.observation_schema_digest,
            records=tuple(self._records.values()),
            metadata=self.metadata,
        )


__all__ = [
    "EntryStateCorpus",
    "EntryStateCorpusBuilder",
    "EntryStateError",
    "EntryStateRecord",
    "EntryStateSplit",
    "SplitStrategy",
]
