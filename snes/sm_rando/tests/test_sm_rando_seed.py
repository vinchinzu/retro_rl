"""Seed package fixture round-trip."""

from __future__ import annotations

from pathlib import Path

from sm_rando.paths import DEMO_SEED_DIR, FIRST_PLAY_STATE
from sm_rando.seed import SeedPackage, ensure_demo_seed, write_demo_seed, write_fixture_seed


def test_fixture_write_load(tmp_path: Path) -> None:
    pkg = write_fixture_seed(
        seed_number="42",
        name="unit_fixture",
        directory=tmp_path / "unit_fixture",
    )
    assert pkg.meta_path.is_file()
    loaded = SeedPackage.load(pkg.directory)
    assert loaded.seed_number == "42"
    assert loaded.locations
    assert loaded.settings.get("logic") == "vanilla"


def test_demo_seed_write_load(tmp_path: Path) -> None:
    pkg = write_demo_seed(directory=tmp_path / "demo_seed")
    assert pkg.meta_path.is_file()
    loaded = SeedPackage.load(pkg.directory)
    assert loaded.name == "demo_seed"
    assert loaded.source == "demo_fixture"
    assert loaded.locations
    note = str(loaded.meta.get("note", ""))
    assert "vanilla" in note.lower() or "FirstPlay" in note
    assert loaded.meta.get("first_play_state") == FIRST_PLAY_STATE
    assert loaded.meta.get("expected_room_id_hex") == "0xDF45"


def test_ensure_demo_seed_idempotent() -> None:
    a = ensure_demo_seed()
    b = ensure_demo_seed()
    assert a.directory == DEMO_SEED_DIR
    assert b.directory == DEMO_SEED_DIR
    assert (DEMO_SEED_DIR / "meta.json").is_file()
