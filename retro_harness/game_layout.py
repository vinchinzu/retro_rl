"""Standard per-game filesystem layout (console folder games).

Most ``snes/<game>/paths.py`` / ``nes/<game>/paths.py`` files only need
integration name + a handful of path constants. Use this instead of copying
the same six lines into every game.

Nested packages (``snes/harvest/harvest/``) should pass
``package_file=…/harvest/paths.py`` and ``workspace_parent=True`` so
``game_dir`` is the workspace root (parent of the package dir).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GamePaths:
    """Resolved paths for one game workspace under ``snes/`` or ``nes/``."""

    game_dir: Path
    repo_root: Path
    integration: str
    integration_dir: Path
    recordings_dir: Path
    roms_dir: Path
    docs_dir: Path

    @property
    def game(self) -> str:
        """stable-retro integration id (alias of :attr:`integration`)."""
        return self.integration


def game_paths(
    package_file: str | Path,
    integration: str,
    *,
    workspace_parent: bool = False,
) -> GamePaths:
    """Build standard layout from a package module's ``__file__``.

    Parameters
    ----------
    package_file:
        Usually ``__file__`` of the game's ``paths.py``.
    integration:
        stable-retro game id, e.g. ``FinalFight-Snes``.
    workspace_parent:
        When the importable package is nested (``…/harvest/harvest/paths.py``),
        set True so ``game_dir`` is the outer workspace (``…/harvest/``).
    """
    package_dir = Path(package_file).resolve().parent
    game_dir = package_dir.parent if workspace_parent else package_dir
    # snes/<game> or nes/<game> → monorepo root is two parents up from game_dir.
    repo_root = game_dir.parent.parent
    return GamePaths(
        game_dir=game_dir,
        repo_root=repo_root,
        integration=integration,
        integration_dir=game_dir / "custom_integrations" / integration,
        recordings_dir=game_dir / "recordings",
        roms_dir=game_dir / "roms",
        docs_dir=game_dir / "docs",
    )
