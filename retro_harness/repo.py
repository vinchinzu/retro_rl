"""Monorepo layout helpers for games nested under ``snes/`` and ``nes/``.

Game packages keep their import names (``alttp``, ``super_metroid``, …).
Those packages live under console folders, so callers must put both the
monorepo root and the console folder on ``sys.path``.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

# Nested packages where the import root is the game project dir, not snes/nes.
# e.g. ``import harvest`` resolves from ``snes/harvest/`` (package at harvest/).
NESTED_PACKAGE_ROOTS: dict[str, str] = {
    "harvest": "snes/harvest",
    "hals_golf": "snes/hals_golf",
}


@lru_cache(maxsize=1)
def monorepo_root() -> Path:
    """Return the retro_rl monorepo root (directory containing ``retro_harness/``)."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "retro_harness").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("could not locate monorepo root from retro_harness.repo")


def snes_dir() -> Path:
    return monorepo_root() / "snes"


def nes_dir() -> Path:
    return monorepo_root() / "nes"


def resolve_game_dir(slug: str, *, root: Path | None = None) -> Path:
    """Resolve a game workspace directory by slug.

    Prefers ``snes/<slug>`` and ``nes/<slug>``, then a top-level ``<slug>``
    for any games not yet relocated.
    """
    base = root if root is not None else monorepo_root()
    nested = NESTED_PACKAGE_ROOTS.get(slug)
    if nested is not None:
        path = base / nested
        if path.is_dir():
            return path
    for candidate in (base / "snes" / slug, base / "nes" / slug, base / slug):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"game directory for {slug!r} not found under snes/, nes/, or monorepo root"
    )


def import_path_entries(*, root: Path | None = None) -> list[Path]:
    """Paths that should be on ``sys.path`` for monorepo + game imports."""
    base = root if root is not None else monorepo_root()
    entries = [
        base,
        base / "snes",
        base / "nes",
        base / "snes" / "harvest",
        base / "snes" / "hals_golf",
    ]
    return [p for p in entries if p.is_dir()]


def ensure_import_paths(*, root: Path | None = None) -> list[Path]:
    """Insert monorepo/console paths at the front of ``sys.path`` if missing."""
    inserted: list[Path] = []
    for path in reversed(import_path_entries(root=root)):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
            inserted.append(path)
    return inserted
