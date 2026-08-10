"""Small shared hashing helpers for route / map artifacts."""

from __future__ import annotations

from pathlib import Path

from retro_harness.identity import sha256_file as _sha256_file


def sha256_file(path: Path) -> str:
    return _sha256_file(path)
