"""Seed package load/save round-trip (offline, no API)."""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

from smz3.seed import SeedPackage


def test_seed_package_write_load(tmp_path: Path) -> None:
    # Tiny synthetic seed patch (not a real seed).
    patch = struct.pack("<IH", 0x100, 4) + b"TEST"
    patch_b64 = base64.b64encode(patch).decode("ascii")
    pkg = SeedPackage(
        name="fixture",
        directory=tmp_path / "fixture",
        seed_number="99",
        hash_code="A B C D",
        url="https://example.test/seed/x",
        guid="00000000-0000-0000-0000-000000000099",
        game_version="11.3.2",
        settings={"smlogic": "normal", "seed": "99"},
        spoiler=[{"Sphere 0": {"Morphing Ball": "Morphing Ball"}}],
        locations=[{"locationId": 1, "itemId": 2}],
        patch_b64=patch_b64,
        meta={"created_at": "2026-07-29T00:00:00Z", "source": "test"},
    )
    pkg.write()

    assert pkg.meta_path.is_file()
    assert pkg.patch_path.read_bytes() == patch
    meta = json.loads(pkg.meta_path.read_text(encoding="utf-8"))
    assert meta["seed_number"] == "99"
    assert meta["location_count"] == 1

    loaded = SeedPackage.load(pkg.directory)
    assert loaded.seed_number == "99"
    assert loaded.hash_code == "A B C D"
    assert loaded.patch_bytes() == patch
    assert loaded.spoiler[0]["Sphere 0"]["Morphing Ball"] == "Morphing Ball"


def test_race_plan_scaffold(tmp_path: Path) -> None:
    from smz3.race import plan_race

    patch = struct.pack("<IH", 0, 1) + b"\x00"
    pkg = SeedPackage(
        name="r",
        directory=tmp_path / "r",
        seed_number="1",
        hash_code="X",
        url="",
        guid="",
        game_version="11.3.2",
        settings={},
        spoiler=[],
        locations=[],
        patch_b64=base64.b64encode(patch).decode("ascii"),
    )
    pkg.write()
    plan = plan_race(pkg, bot_count=2)
    d = plan.to_dict()
    assert d["bot_count"] == 2
    assert d["room_timeout_multiplier"] == 3.0
    assert d["hooks"]["status"] == "scaffold"
