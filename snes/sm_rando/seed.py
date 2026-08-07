"""Seed package schema for SM randomizer (offline-first).

Generation backends (VARIA CLI, web API, hand-placed fixtures) plug in later.
Early work uses fixture packages under ``seeds/`` plus vanilla ROM for skill
play until patched ROM wiring lands.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sm_rando.paths import (
    DEMO_SEED_DIR,
    DEMO_SEED_NUMBER,
    SEEDS_DIR,
    SHARED_SM_ROM,
    TEST_SEED_DIR,
    TEST_SEED_NUMBER,
)

DEFAULT_SETTINGS: dict[str, str] = {
    "logic": "vanilla",
    "goal": "beat_the_game",
    "area_randomization": "false",
    "boss_randomization": "false",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_") or "seed"


@dataclass
class SeedPackage:
    """On-disk seed artifact set under ``sm_rando/seeds/<name>/``."""

    name: str
    directory: Path
    seed_number: str
    settings: dict[str, str] = field(default_factory=dict)
    spoiler: list[Any] = field(default_factory=list)
    locations: list[dict[str, Any]] = field(default_factory=list)
    # Optional patch / ROM paths once generator is wired.
    patch_path: str | None = None
    rom_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    source: str = "fixture"

    @property
    def meta_path(self) -> Path:
        return self.directory / "meta.json"

    @property
    def spoiler_path(self) -> Path:
        return self.directory / "spoiler.json"

    @property
    def locations_path(self) -> Path:
        return self.directory / "locations.json"

    def write(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "seed_number": self.seed_number,
            "settings": self.settings,
            "location_count": len(self.locations),
            "spoiler_entries": len(self.spoiler),
            "patch_path": self.patch_path,
            "rom_path": self.rom_path,
            "source": self.source,
            "created_at": self.meta.get("created_at", _utc_now()),
            **{k: v for k, v in self.meta.items() if k != "created_at"},
        }
        self.meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        self.spoiler_path.write_text(
            json.dumps(self.spoiler, indent=2) + "\n", encoding="utf-8"
        )
        self.locations_path.write_text(
            json.dumps(self.locations, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path | str) -> SeedPackage:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        spoiler_path = directory / "spoiler.json"
        locations_path = directory / "locations.json"
        spoiler = (
            json.loads(spoiler_path.read_text(encoding="utf-8"))
            if spoiler_path.is_file()
            else []
        )
        locations = (
            json.loads(locations_path.read_text(encoding="utf-8"))
            if locations_path.is_file()
            else []
        )
        return cls(
            name=str(meta.get("name", directory.name)),
            directory=directory,
            seed_number=str(meta.get("seed_number", "")),
            settings=dict(meta.get("settings") or {}),
            spoiler=spoiler,
            locations=locations,
            patch_path=meta.get("patch_path"),
            rom_path=meta.get("rom_path"),
            meta={k: v for k, v in meta.items() if k not in {
                "name", "seed_number", "settings", "location_count",
                "spoiler_entries", "patch_path", "rom_path", "source",
            }},
            source=str(meta.get("source", "fixture")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["directory"] = str(self.directory)
        return data


def write_fixture_seed(
    *,
    seed_number: str = TEST_SEED_NUMBER,
    name: str | None = None,
    settings: Mapping[str, str] | None = None,
    directory: Path | None = None,
) -> SeedPackage:
    """Write a deterministic offline fixture (no real item shuffle yet)."""
    name = name or f"fixture_{_slugify(seed_number)}"
    directory = directory or (SEEDS_DIR / name)
    # Minimal spoiler: morph in original location — logic solver practice data.
    locations = [
        {"location": "Morphing Ball", "item": "Morphing Ball", "area": "Crateria"},
        {"location": "Missile (blue Brinstar middle)", "item": "Missile", "area": "Brinstar"},
        {"location": "Bombs", "item": "Bombs", "area": "Crateria"},
    ]
    spoiler = [{"Sphere 0": {loc["location"]: loc["item"] for loc in locations[:1]}}]
    pkg = SeedPackage(
        name=name,
        directory=directory,
        seed_number=str(seed_number),
        settings={**DEFAULT_SETTINGS, **dict(settings or {})},
        spoiler=spoiler,
        locations=locations,
        meta={"created_at": _utc_now(), "note": "offline fixture for logic graph"},
        source="fixture",
    )
    pkg.write()
    return pkg


def ensure_test_seed() -> SeedPackage:
    if (TEST_SEED_DIR / "meta.json").is_file():
        return SeedPackage.load(TEST_SEED_DIR)
    return write_fixture_seed(seed_number=TEST_SEED_NUMBER, name="test_seed", directory=TEST_SEED_DIR)


def write_demo_seed(
    *,
    directory: Path | None = None,
    seed_number: str = DEMO_SEED_NUMBER,
) -> SeedPackage:
    """Write the playable demo seed package (vanilla SM FirstPlay until rando gen).

    This is **not** a shuffled ROM — it points at the shared vanilla Super
    Metroid dump and documents logic-fixture locations for early graph work.
    Real generator patches replace ``rom_path`` / ``patch_path`` later.
    """
    directory = directory or DEMO_SEED_DIR
    locations = [
        {"location": "Morphing Ball", "item": "Morphing Ball", "area": "Crateria"},
        {"location": "Missile (blue Brinstar middle)", "item": "Missile", "area": "Brinstar"},
        {"location": "Bombs", "item": "Bombs", "area": "Crateria"},
        {"location": "Energy Tank (Crateria surface)", "item": "Energy Tank", "area": "Crateria"},
    ]
    spoiler = [
        {
            "Sphere 0": {
                "Morphing Ball": "Morphing Ball",
                "note": "vanilla placement — demo only",
            }
        }
    ]
    rom_rel = None
    if SHARED_SM_ROM.is_file():
        try:
            rom_rel = str(SHARED_SM_ROM.resolve())
        except OSError:
            rom_rel = str(SHARED_SM_ROM)

    pkg = SeedPackage(
        name="demo_seed",
        directory=directory,
        seed_number=str(seed_number),
        settings={**DEFAULT_SETTINGS, "demo": "true", "start": "ceres"},
        spoiler=spoiler,
        locations=locations,
        patch_path=None,
        rom_path=rom_rel,
        meta={
            "created_at": _utc_now(),
            "note": (
                "Playable demo: vanilla Super Metroid ROM + FirstPlay state "
                "(Ceres elevator). Not a shuffled seed until a rando generator "
                "is wired. Use ./play or python -m sm_rando.scripts.play."
            ),
            "first_play_state": "FirstPlay",
            "expected_room_id_hex": "0xDF45",
            "kind": "vanilla_first_play_demo",
        },
        source="demo_fixture",
    )
    pkg.write()
    return pkg


def ensure_demo_seed() -> SeedPackage:
    """Load or create ``seeds/demo_seed/`` for playable vanilla FirstPlay demos."""
    if (DEMO_SEED_DIR / "meta.json").is_file():
        return SeedPackage.load(DEMO_SEED_DIR)
    return write_demo_seed(directory=DEMO_SEED_DIR)
