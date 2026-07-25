"""Tests for snes_oneshot.setup_all_roms."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from snes_oneshot.ladder import LadderEntry, LadderStatus
from snes_oneshot.setup_all_roms import main, setup_entry, setup_slugs


def _write_fake_rom_zip(zip_path: Path, rom_name: str = "game.sfc") -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(rom_name, b"SNESROMFAKE")


def test_setup_entry_extracts_and_links(tmp_path: Path) -> None:
    entry = LadderEntry(
        rank=99,
        slug="demo_game",
        title="Demo",
        rom_zip="Demo.zip",
        integration="Demo-Snes",
        tier=0,
        status=LadderStatus.SCAFFOLDED,
    )
    _write_fake_rom_zip(tmp_path / "roms" / "Super Nintendo" / entry.rom_zip)
    (tmp_path / entry.slug / "custom_integrations" / entry.integration).mkdir(
        parents=True
    )

    result = setup_entry(entry, repo_root=tmp_path)

    assert result.ok
    assert result.rom_path is not None
    assert result.rom_path.is_file()
    link = (
        tmp_path
        / entry.slug
        / "custom_integrations"
        / entry.integration
        / "rom.sfc"
    )
    assert link.is_symlink()
    assert (link.parent / "rom.sha").is_file()


def test_setup_entry_missing_zip(tmp_path: Path) -> None:
    entry = LadderEntry(
        rank=99,
        slug="missing_game",
        title="Missing",
        rom_zip="DoesNotExist.zip",
        integration="Missing-Snes",
        tier=0,
    )
    result = setup_entry(entry, repo_root=tmp_path)
    assert not result.ok
    assert result.error is not None
    assert "missing zip" in result.error


def test_main_list_exits_zero() -> None:
    assert main(["--list"]) == 0


def test_setup_slugs_unknown_raises() -> None:
    with pytest.raises(KeyError):
        setup_slugs(["not_a_real_slug"])
