"""Monorepo layout helpers for games nested under ``snes/`` and ``nes/``.

Game packages keep their import names (``alttp``, ``super_metroid``, …).
Those packages live under console folders, so callers must put both the
monorepo root and the console folder on ``sys.path``.

A few games use a *nested package root*: the importable package sits one
level deeper than the game workspace (``snes/harvest/harvest/``,
``snes/hals_golf/hals_golf/``) because the workspace root already owns
non-package dirs that would collide (``tasks/``, ``maps/``, …). Those game
dirs are discovered by layout (``<console>/<slug>/<slug>/__init__.py``),
not listed as policy.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path


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
    for candidate in (base / "snes" / slug, base / "nes" / slug, base / slug):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"game directory for {slug!r} not found under snes/, nes/, or monorepo root"
    )


def discover_nested_package_roots(*, root: Path | None = None) -> list[Path]:
    """Game dirs that must be on ``sys.path`` for ``import <slug>``.

    Layout signal: ``snes|nes/<slug>/<slug>/__init__.py`` (package nested under
    the game workspace). Empty when every game package sits at console root.
    """
    base = root if root is not None else monorepo_root()
    found: list[Path] = []
    for console in ("snes", "nes"):
        console_dir = base / console
        if not console_dir.is_dir():
            continue
        for game_dir in sorted(console_dir.iterdir()):
            if not game_dir.is_dir() or game_dir.name.startswith("."):
                continue
            nested = game_dir / game_dir.name / "__init__.py"
            if nested.is_file():
                found.append(game_dir)
    return found


def import_path_entries(*, root: Path | None = None) -> list[Path]:
    """Paths that should be on ``sys.path`` for monorepo + game imports."""
    base = root if root is not None else monorepo_root()
    entries = [
        base,
        base / "snes",
        base / "nes",
        *discover_nested_package_roots(root=base),
    ]
    # Preserve order, drop missing / duplicates.
    seen: set[Path] = set()
    out: list[Path] = []
    for path in entries:
        resolved = path.resolve() if path.exists() else path
        if not path.is_dir():
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def ensure_import_paths(*, root: Path | None = None) -> list[Path]:
    """Insert monorepo/console paths at the front of ``sys.path`` if missing."""
    inserted: list[Path] = []
    for path in reversed(import_path_entries(root=root)):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
            inserted.append(path)
    return inserted
