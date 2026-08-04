"""SMZ3 seed generation and on-disk seed packages.

Primary path: public samus.link API via ``pyz3r`` (optional dependency).
Produces a seed directory with metadata, spoiler log, raw patch, and (when
vanilla ROMs + base IPS are available) a playable combo ``.sfc``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from smz3.paths import SEEDS_DIR, TEST_SEED_DIR, TEST_SEED_NUMBER
from smz3.rom_builder import (
    build_combo_rom,
    decode_seed_patch_b64,
    write_combo_rom,
)

DEFAULT_SETTINGS: dict[str, str] = {
    "smlogic": "normal",
    "goal": "defeatboth",
    "swordlocation": "uncle",
    "morphlocation": "original",
    "race": "false",
    "gamemode": "normal",
    "players": "1",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_") or "seed"


@dataclass
class SeedPackage:
    """On-disk seed artifact set under ``smz3/seeds/<name>/``."""

    name: str
    directory: Path
    seed_number: str
    hash_code: str
    url: str
    guid: str
    game_version: str
    settings: dict[str, str]
    spoiler: list[Any]
    locations: list[dict[str, Any]]
    patch_b64: str
    rom_path: Path | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def patch_path(self) -> Path:
        return self.directory / "seed_patch.bin"

    @property
    def meta_path(self) -> Path:
        return self.directory / "meta.json"

    @property
    def spoiler_path(self) -> Path:
        return self.directory / "spoiler.json"

    def patch_bytes(self) -> bytes:
        if self.patch_path.is_file():
            return self.patch_path.read_bytes()
        return decode_seed_patch_b64(self.patch_b64)

    def to_meta(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed_number": self.seed_number,
            "hash_code": self.hash_code,
            "url": self.url,
            "guid": self.guid,
            "game_version": self.game_version,
            "settings": self.settings,
            "rom_path": str(self.rom_path) if self.rom_path else None,
            "created_at": self.meta.get("created_at", _utc_now()),
            "source": self.meta.get("source", "samus.link"),
            "location_count": len(self.locations),
            "spoiler_spheres": len(self.spoiler),
            **{k: v for k, v in self.meta.items() if k not in {"created_at", "source"}},
        }

    def write(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.patch_path.write_bytes(decode_seed_patch_b64(self.patch_b64))
        self.spoiler_path.write_text(
            json.dumps(self.spoiler, indent=2) + "\n", encoding="utf-8"
        )
        (self.directory / "locations.json").write_text(
            json.dumps(self.locations, indent=2) + "\n", encoding="utf-8"
        )
        self.meta_path.write_text(
            json.dumps(self.to_meta(), indent=2) + "\n", encoding="utf-8"
        )
        # Keep base64 for rebuilds without re-encoding.
        (self.directory / "seed_patch.b64").write_text(
            self.patch_b64 + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> SeedPackage:
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        spoiler = json.loads((directory / "spoiler.json").read_text(encoding="utf-8"))
        locations_path = directory / "locations.json"
        locations = (
            json.loads(locations_path.read_text(encoding="utf-8"))
            if locations_path.is_file()
            else []
        )
        b64_path = directory / "seed_patch.b64"
        if b64_path.is_file():
            patch_b64 = b64_path.read_text(encoding="utf-8").strip()
        else:
            patch_b64 = base64.b64encode(
                (directory / "seed_patch.bin").read_bytes()
            ).decode("ascii")
        rom_path = directory / "smz3.sfc"
        return cls(
            name=meta.get("name", directory.name),
            directory=directory,
            seed_number=str(meta.get("seed_number", "")),
            hash_code=str(meta.get("hash_code", "")),
            url=str(meta.get("url", "")),
            guid=str(meta.get("guid", "")),
            game_version=str(meta.get("game_version", "")),
            settings=dict(meta.get("settings") or {}),
            spoiler=spoiler,
            locations=locations,
            patch_b64=patch_b64,
            rom_path=rom_path if rom_path.is_file() else None,
            meta={k: v for k, v in meta.items() if k not in {
                "name", "seed_number", "hash_code", "url", "guid",
                "game_version", "settings", "rom_path", "location_count",
                "spoiler_spheres",
            }},
        )

    def build_rom(self, *, out: Path | None = None) -> Path:
        """Materialize combo ROM under the seed directory (or ``out``)."""
        rom = build_combo_rom(self.patch_bytes())
        target = out or (self.directory / "smz3.sfc")
        write_combo_rom(target, rom)
        self.rom_path = target
        # Refresh meta with rom path.
        meta = self.to_meta()
        meta["rom_path"] = str(target)
        meta["rom_size"] = len(rom)
        self.meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return target


async def generate_seed_async(
    *,
    seed: str | None = None,
    settings: Mapping[str, str] | None = None,
    name: str | None = None,
    out_dir: Path | None = None,
    build_rom: bool = True,
    baseurl: str = "https://samus.link",
) -> SeedPackage:
    """Roll a seed via samus.link and write a seed package.

    Requires ``pyz3r`` (and network). Does not add pyz3r to core deps — install
    with ``uv pip install pyz3r`` when generating seeds.
    """
    try:
        from pyz3r.sm import sm as sm_api
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyz3r is required for online seed generation. "
            "Install with: uv pip install pyz3r"
        ) from exc

    cfg = dict(DEFAULT_SETTINGS)
    if settings:
        cfg.update({str(k): str(v) for k, v in settings.items()})
    if seed is not None:
        cfg["seed"] = str(seed)

    api_seed = await sm_api(settings=cfg, randomizer="smz3", baseurl=baseurl)
    data = api_seed.data
    world = data["worlds"][0]
    patch_b64 = world["patch"]
    spoiler_raw = data.get("spoiler", "[]")
    if isinstance(spoiler_raw, str):
        spoiler = json.loads(spoiler_raw)
    else:
        spoiler = list(spoiler_raw)

    seed_number = str(data.get("seedNumber", cfg.get("seed", "")))
    hash_code = str(data.get("hash", api_seed.code))
    pkg_name = name or _slugify(f"seed_{seed_number}_{hash_code.replace(' ', '_')}")
    directory = Path(out_dir) if out_dir else SEEDS_DIR / pkg_name

    package = SeedPackage(
        name=pkg_name,
        directory=directory,
        seed_number=seed_number,
        hash_code=hash_code,
        url=str(api_seed.url),
        guid=str(data.get("guid", "")),
        game_version=str(data.get("gameVersion", "")),
        settings=cfg,
        spoiler=spoiler,
        locations=list(world.get("locations") or []),
        patch_b64=patch_b64,
        meta={
            "created_at": _utc_now(),
            "source": baseurl,
            "game_id": data.get("gameId"),
            "game_name": data.get("gameName"),
            "mode": data.get("mode"),
            "world_settings": world.get("settings"),
        },
    )
    package.write()
    if build_rom:
        try:
            package.build_rom()
        except FileNotFoundError as exc:
            package.meta["rom_build_error"] = str(exc)
            package.write()
    return package


def generate_seed(
    *,
    seed: str | None = None,
    settings: Mapping[str, str] | None = None,
    name: str | None = None,
    out_dir: Path | None = None,
    build_rom: bool = True,
) -> SeedPackage:
    """Sync wrapper around :func:`generate_seed_async`."""
    return asyncio.run(
        generate_seed_async(
            seed=seed,
            settings=settings,
            name=name,
            out_dir=out_dir,
            build_rom=build_rom,
        )
    )


def generate_test_seed(*, force: bool = False, build_rom: bool = True) -> SeedPackage:
    """Generate or reload the pinned test seed (seed number 1337, uncle sword)."""
    if TEST_SEED_DIR.is_dir() and (TEST_SEED_DIR / "meta.json").is_file() and not force:
        pkg = SeedPackage.load(TEST_SEED_DIR)
        if build_rom and pkg.rom_path is None:
            try:
                pkg.build_rom()
            except FileNotFoundError:
                pass
        return pkg
    return generate_seed(
        seed=TEST_SEED_NUMBER,
        settings={
            **DEFAULT_SETTINGS,
            "seed": TEST_SEED_NUMBER,
            "swordlocation": "uncle",
            "morphlocation": "original",
        },
        name="test_seed",
        out_dir=TEST_SEED_DIR,
        build_rom=build_rom,
    )


def load_seed(path: str | Path) -> SeedPackage:
    return SeedPackage.load(Path(path))
