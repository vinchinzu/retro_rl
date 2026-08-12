"""Typed, offline TAS provenance records for the SMB2 first-level scaffold.

The planned checkpoint names in :func:`planned_level1_checkpoints` are evidence
slots, not generated emulator states.  This module deliberately does not
download movies, invoke BizHawk, or check that state files exist.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Self
from urllib.parse import urlparse

from smb2.paths import STATE_ARTIFACTS_RELATIVE_DIR


class HashAlgorithm(StrEnum):
    """Hash algorithms accepted by the evidence record."""

    SHA1 = "sha1"
    SHA256 = "sha256"


class MovieFormat(StrEnum):
    """Input movie formats that can be recorded in the manifest."""

    FM2 = "fm2"
    BK2 = "bk2"
    LSMV = "lsmv"


class BizHawkValidationStatus(StrEnum):
    """Honest state of the optional BizHawk replay validation step."""

    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"


class CheckpointStatus(StrEnum):
    """Materialization state of one named state artifact slot."""

    PLANNED = "planned"
    MATERIALIZED = "materialized"
    VALIDATED = "validated"


_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def _text(value: object, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} must not be empty")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    return result


def _hash(value: object, algorithm: HashAlgorithm, field_name: str) -> str:
    result = _text(value, field_name)
    expected_length = 40 if algorithm is HashAlgorithm.SHA1 else 64
    if len(result) != expected_length or _HEX_RE.fullmatch(result) is None:
        raise ValueError(
            f"{field_name} must be a {expected_length}-character "
            f"{algorithm.value} digest"
        )
    return result.lower()


@dataclass(frozen=True, slots=True)
class CheckpointEvidence:
    """A named state artifact path, possibly still only planned."""

    name: str
    state_artifact_path: str
    frame: int | None = None
    status: CheckpointStatus = CheckpointStatus.PLANNED
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "checkpoint name"))
        object.__setattr__(
            self,
            "state_artifact_path",
            _text(self.state_artifact_path, "state_artifact_path"),
        )
        try:
            status = CheckpointStatus(self.status)
        except ValueError as exc:
            raise ValueError(f"invalid checkpoint status: {self.status!r}") from exc
        object.__setattr__(self, "status", status)
        if self.frame is not None:
            if (
                not isinstance(self.frame, int)
                or isinstance(self.frame, bool)
                or self.frame < 0
            ):
                raise ValueError("checkpoint frame must be a non-negative integer")
        object.__setattr__(self, "description", str(self.description))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible checkpoint mapping."""
        return {
            "name": self.name,
            "state_artifact_path": self.state_artifact_path,
            "frame": self.frame,
            "status": self.status.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Self:
        """Build a checkpoint from a decoded JSON mapping."""
        try:
            name = payload["name"]
            state_artifact_path = payload["state_artifact_path"]
        except KeyError as exc:
            raise ValueError(
                f"checkpoint missing required field: {exc.args[0]}"
            ) from exc
        return cls(
            name=str(name),
            state_artifact_path=str(state_artifact_path),
            frame=payload.get("frame"),  # type: ignore[arg-type]
            status=payload.get("status", CheckpointStatus.PLANNED),  # type: ignore[arg-type]
            description=str(payload.get("description", "")),
        )

    @property
    def state_path(self) -> str:
        """Compatibility spelling for callers that call the artifact a state."""
        return self.state_artifact_path


@dataclass(frozen=True, slots=True)
class TASEvidenceManifest:
    """Provenance and checkpoint metadata for one SMB2 TAS source."""

    source_url: str
    movie_hash: str
    movie_format: MovieFormat
    rom_hash: str
    source_emulator: str
    source_core: str
    checkpoints: tuple[CheckpointEvidence, ...]
    bizhawk_validation_status: BizHawkValidationStatus = BizHawkValidationStatus.NOT_RUN
    movie_hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    rom_hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    movie_path: str | None = None
    rom_path: str | None = None
    game: str = "Super Mario Bros. 2 (NES)"
    level: str = "1-1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        parsed_url = urlparse(_text(self.source_url, "source_url"))
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("source_url must be an http(s) URL")
        object.__setattr__(self, "source_url", self.source_url.strip())

        try:
            movie_format = MovieFormat(self.movie_format)
        except ValueError as exc:
            raise ValueError(
                f"unsupported movie format: {self.movie_format!r}"
            ) from exc
        object.__setattr__(self, "movie_format", movie_format)

        try:
            movie_algorithm = HashAlgorithm(self.movie_hash_algorithm)
            rom_algorithm = HashAlgorithm(self.rom_hash_algorithm)
        except ValueError as exc:
            raise ValueError("unsupported hash algorithm") from exc
        object.__setattr__(self, "movie_hash_algorithm", movie_algorithm)
        object.__setattr__(self, "rom_hash_algorithm", rom_algorithm)
        object.__setattr__(
            self,
            "movie_hash",
            _hash(self.movie_hash, movie_algorithm, "movie_hash"),
        )
        object.__setattr__(
            self,
            "rom_hash",
            _hash(self.rom_hash, rom_algorithm, "rom_hash"),
        )

        object.__setattr__(
            self,
            "source_emulator",
            _text(self.source_emulator, "source_emulator"),
        )
        object.__setattr__(self, "source_core", _text(self.source_core, "source_core"))
        try:
            validation = BizHawkValidationStatus(self.bizhawk_validation_status)
        except ValueError as exc:
            raise ValueError(
                f"invalid BizHawk validation status: {self.bizhawk_validation_status!r}"
            ) from exc
        object.__setattr__(self, "bizhawk_validation_status", validation)

        try:
            checkpoints = tuple(
                checkpoint
                if isinstance(checkpoint, CheckpointEvidence)
                else CheckpointEvidence.from_dict(checkpoint)
                for checkpoint in self.checkpoints
            )
        except TypeError as exc:
            raise ValueError(
                "checkpoints must be an iterable of checkpoint records"
            ) from exc
        if not checkpoints:
            raise ValueError("at least one named checkpoint is required")
        names = [checkpoint.name for checkpoint in checkpoints]
        if len(names) != len(set(names)):
            raise ValueError("checkpoint names must be unique")
        object.__setattr__(self, "checkpoints", checkpoints)

        for field_name in ("movie_path", "rom_path"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _text(value, field_name))
        object.__setattr__(self, "game", _text(self.game, "game"))
        object.__setattr__(self, "level", _text(self.level, "level"))
        if self.schema_version != 1:
            raise ValueError("unsupported manifest schema_version")

    @property
    def bizhawk_validation(self) -> BizHawkValidationStatus:
        """Short alias for the serialized BizHawk status field."""
        return self.bizhawk_validation_status

    @property
    def movie_sha256(self) -> str:
        """Return the movie digest when this manifest uses SHA-256."""
        if self.movie_hash_algorithm is not HashAlgorithm.SHA256:
            raise ValueError("movie hash is not SHA-256")
        return self.movie_hash

    @property
    def rom_sha256(self) -> str:
        """Return the ROM digest when this manifest uses SHA-256."""
        if self.rom_hash_algorithm is not HashAlgorithm.SHA256:
            raise ValueError("ROM hash is not SHA-256")
        return self.rom_hash

    def validate(self) -> None:
        """Re-run constructor validation for callers accepting external data."""
        type(self)(**self.to_dict())

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible manifest mapping."""
        return {
            "schema_version": self.schema_version,
            "game": self.game,
            "level": self.level,
            "source_url": self.source_url,
            "movie_path": self.movie_path,
            "movie_hash": self.movie_hash,
            "movie_hash_algorithm": self.movie_hash_algorithm.value,
            "movie_format": self.movie_format.value,
            "rom_path": self.rom_path,
            "rom_hash": self.rom_hash,
            "rom_hash_algorithm": self.rom_hash_algorithm.value,
            "source_emulator": self.source_emulator,
            "source_core": self.source_core,
            "bizhawk_validation_status": self.bizhawk_validation_status.value,
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
        }

    def to_json(self) -> str:
        """Serialize the manifest without touching any emulator artifacts."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def write_json(self, path: Path | str) -> Path:
        """Write this manifest to *path* and return the resulting path."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json(), encoding="utf-8")
        return output

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Validate and build a manifest from a decoded JSON mapping."""
        required = (
            "source_url",
            "movie_hash",
            "movie_format",
            "rom_hash",
            "source_emulator",
            "source_core",
            "checkpoints",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"manifest missing required fields: {', '.join(missing)}")
        status = payload.get(
            "bizhawk_validation_status",
            payload.get("bizhawk_validation", BizHawkValidationStatus.NOT_RUN),
        )
        return cls(
            source_url=str(payload["source_url"]),
            movie_hash=str(payload["movie_hash"]),
            movie_format=payload["movie_format"],  # type: ignore[arg-type]
            rom_hash=str(payload["rom_hash"]),
            source_emulator=str(payload["source_emulator"]),
            source_core=str(payload["source_core"]),
            checkpoints=tuple(
                CheckpointEvidence.from_dict(item) for item in payload["checkpoints"]
            ),
            bizhawk_validation_status=status,  # type: ignore[arg-type]
            movie_hash_algorithm=payload.get(
                "movie_hash_algorithm", HashAlgorithm.SHA256
            ),  # type: ignore[arg-type]
            rom_hash_algorithm=payload.get("rom_hash_algorithm", HashAlgorithm.SHA256),  # type: ignore[arg-type]
            movie_path=(
                str(payload["movie_path"])
                if payload.get("movie_path") is not None
                else None
            ),
            rom_path=(
                str(payload["rom_path"])
                if payload.get("rom_path") is not None
                else None
            ),
            game=str(payload.get("game", "Super Mario Bros. 2 (NES)")),
            level=str(payload.get("level", "1-1")),
            schema_version=int(payload.get("schema_version", 1)),
        )

    @classmethod
    def from_json(cls, source: Path | str) -> Self:
        """Read JSON text or a JSON file path and validate it."""
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        else:
            stripped = source.lstrip()
            if stripped.startswith("{") or stripped.startswith("["):
                text = source
            else:
                text = Path(source).read_text(encoding="utf-8")
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("manifest JSON root must be an object")
        return cls.from_dict(payload)


def planned_level1_checkpoints() -> tuple[CheckpointEvidence, ...]:
    """Return named, unmaterialized first-level checkpoint evidence slots."""
    names = ("level1_start", "level1_control", "level1_goal")
    return tuple(
        CheckpointEvidence(
            name=name,
            state_artifact_path=str(STATE_ARTIFACTS_RELATIVE_DIR / f"{name}.state"),
            description="Planned slot; no emulator state has been generated.",
        )
        for name in names
    )


def make_scaffold_manifest(
    *,
    source_url: str,
    movie_hash: str,
    movie_format: MovieFormat,
    rom_hash: str,
    source_emulator: str,
    source_core: str,
    movie_path: str | None = None,
    rom_path: str | None = None,
) -> TASEvidenceManifest:
    """Build a first-level manifest with planned checkpoints and no validation claim."""
    return TASEvidenceManifest(
        source_url=source_url,
        movie_hash=movie_hash,
        movie_format=movie_format,
        rom_hash=rom_hash,
        source_emulator=source_emulator,
        source_core=source_core,
        checkpoints=planned_level1_checkpoints(),
        movie_path=movie_path,
        rom_path=rom_path,
    )


TasEvidenceManifest = TASEvidenceManifest
EvidenceManifest = TASEvidenceManifest
Checkpoint = CheckpointEvidence

__all__ = [
    "BizHawkValidationStatus",
    "Checkpoint",
    "CheckpointEvidence",
    "CheckpointStatus",
    "EvidenceManifest",
    "HashAlgorithm",
    "MovieFormat",
    "TASEvidenceManifest",
    "TasEvidenceManifest",
    "make_scaffold_manifest",
    "planned_level1_checkpoints",
]
