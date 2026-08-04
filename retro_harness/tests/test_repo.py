"""Monorepo layout helpers (nested package discovery, path wiring)."""

from __future__ import annotations

from pathlib import Path

from retro_harness.repo import (
    discover_nested_package_roots,
    import_path_entries,
    monorepo_root,
    resolve_game_dir,
)


def test_resolve_game_dir_console_layout() -> None:
    root = monorepo_root()
    harvest = resolve_game_dir("harvest", root=root)
    assert harvest == root / "snes" / "harvest"
    smb = resolve_game_dir("smb", root=root)
    assert smb == root / "nes" / "smb"


def test_nested_package_roots_are_layout_discovered() -> None:
    """harvest / hals_golf use snes/<slug>/<slug>/; no hardcoded slug map."""
    root = monorepo_root()
    nested = discover_nested_package_roots(root=root)
    names = {p.name for p in nested}
    assert "harvest" in names
    assert "hals_golf" in names
    for path in nested:
        assert (path / path.name / "__init__.py").is_file()
    # Flat packages (package == game dir) must not appear.
    assert "super_metroid" not in names
    assert "smb" not in names


def test_import_path_entries_include_nested_without_duplicates() -> None:
    root = monorepo_root()
    entries = import_path_entries(root=root)
    assert root in entries
    assert root / "snes" in entries
    assert root / "nes" in entries
    assert root / "snes" / "harvest" in entries
    assert root / "snes" / "hals_golf" in entries
    assert len(entries) == len(set(Path(p).resolve() for p in entries))
