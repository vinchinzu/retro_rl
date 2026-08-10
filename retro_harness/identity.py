"""Canonical JSON and SHA-256 primitives for identity-bearing records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class IdentityError(ValueError):
    """Raised when a value cannot participate in a stable identity."""


def require_nonempty(value: Any, name: str) -> str:
    """Return a stripped identity string or reject it consistently."""
    if not isinstance(value, str) or not value.strip():
        raise IdentityError(f"{name} must be a non-empty string")
    return value.strip()


def jsonable(value: Any, path: str = "value") -> Any:
    """Normalize the deliberately small JSON subset accepted for identities."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise IdentityError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise IdentityError(f"{path} mapping keys must be strings")
            normalized[key] = jsonable(item, f"{path}.{key}")
        return dict(sorted(normalized.items()))
    if isinstance(value, (list, tuple)):
        return [jsonable(item, f"{path}[]") for item in value]
    raise IdentityError(f"{path} contains unsupported {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize an identity value with one fail-closed canonical dialect."""
    return json.dumps(
        jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def digest_record(kind: str, record: Mapping[str, Any]) -> str:
    """Hash a typed canonical record."""
    payload = {"kind": require_nonempty(kind, "kind"), **dict(record)}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise IdentityError(f"identity file does not exist: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "IdentityError",
    "canonical_json",
    "digest_record",
    "jsonable",
    "require_nonempty",
    "sha256_bytes",
    "sha256_file",
]
