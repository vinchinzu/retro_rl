"""Seed package fixture round-trip."""

from __future__ import annotations

from pathlib import Path

from alttp_rando.seed import SeedPackage, write_fixture_seed


def test_fixture_write_load(tmp_path: Path) -> None:
    pkg = write_fixture_seed(
        seed_number="7",
        name="unit_fixture",
        directory=tmp_path / "unit_fixture",
    )
    loaded = SeedPackage.load(pkg.directory)
    assert loaded.seed_number == "7"
    assert any(loc.get("location") == "Link's Uncle" for loc in loaded.locations)
