"""Stable-retro setup for the Hal's Hole in One Golf integration."""

from __future__ import annotations

import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

from hals_golf.paths import (
    CUSTOM_INTEGRATIONS_DIR,
    GAME,
    GAME_DIR,
    PROJECT_DIR,
    SHARED_ROMS_DIR,
)

ROM_LINK = GAME_DIR / "rom.sfc"
ROM_SHA_PATH = GAME_DIR / "rom.sha"
MUTABLE_STATE_NAMES = frozenset({"latest", "current", "QuickSave"})

_ROM_ENV_VARS = ("HALS_GOLF_ROM", "HALS_HOLE_IN_ONE_ROM")
_ROM_FILENAMES = (
    "HalsHoleInOneGolf.smc",
    "HalsHoleInOneGolf.sfc",
    "Hal's Hole in One Golf.smc",
    "Hal's Hole in One Golf.sfc",
    "rom.smc",
    "rom.sfc",
)


def _expected_rom_sha() -> str | None:
    try:
        text = ROM_SHA_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text.split()[0].lower() if text else None


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dedupe(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = os.fspath(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def golf_rom_candidates() -> list[Path]:
    """Return local ROM paths worth trying, in priority order."""
    env_paths = [
        Path(value).expanduser()
        for name in _ROM_ENV_VARS
        if (value := os.getenv(name))
    ]
    rom_dirs = [
        PROJECT_DIR / "roms",
        SHARED_ROMS_DIR,
    ]
    file_paths = [
        directory / filename
        for directory in rom_dirs
        for filename in _ROM_FILENAMES
    ]
    # Also accept the original zip-extracted name under shared roms.
    zip_extract = SHARED_ROMS_DIR / "Hal's Hole in One Golf.smc"
    return _dedupe([*env_paths, *file_paths, zip_extract])


def find_golf_rom() -> Path | None:
    """Find a local ROM matching ``rom.sha`` when possible."""
    expected = _expected_rom_sha()
    existing = [path for path in golf_rom_candidates() if path.is_file()]
    if not existing:
        return None
    if expected is None:
        return existing[0]
    for path in existing:
        try:
            if _sha1(path).lower() == expected:
                return path
        except OSError:
            continue
    return None


def ensure_golf_rom(*, required: bool = True, quiet: bool = False) -> Path | None:
    """Ensure ``HalsHoleInOne-Snes/rom.sfc`` resolves to a real ROM."""
    expected = _expected_rom_sha()

    if ROM_LINK.exists():
        if expected is None or _sha1(ROM_LINK).lower() == expected:
            return ROM_LINK

    if ROM_LINK.is_symlink():
        ROM_LINK.unlink()

    candidate = find_golf_rom()
    if candidate is not None:
        GAME_DIR.mkdir(parents=True, exist_ok=True)
        if ROM_LINK.exists() and not ROM_LINK.is_symlink():
            raise FileExistsError(
                f"{ROM_LINK} exists but does not match {ROM_SHA_PATH}; "
                "move it aside or set HALS_GOLF_ROM."
            )
        if ROM_LINK.exists() or ROM_LINK.is_symlink():
            ROM_LINK.unlink()
        relative_target = os.path.relpath(candidate.resolve(), ROM_LINK.parent)
        ROM_LINK.symlink_to(relative_target)
        if not quiet:
            print(f"[RETRO] Linked {ROM_LINK} -> {relative_target}")
        return ROM_LINK

    if required:
        searched = ", ".join(str(p) for p in golf_rom_candidates())
        raise FileNotFoundError(
            f"Could not find Hal's Hole in One Golf ROM. Searched: {searched}"
        )
    return None


def register_golf_integration(retro: object, *, quiet: bool = False) -> Path:
    """Register the custom integration and repair the ROM symlink."""
    ensure_golf_rom(required=True, quiet=quiet)
    integrations = getattr(retro, "data").Integrations
    integrations._init()
    path = str(CUSTOM_INTEGRATIONS_DIR.resolve())
    if path not in integrations.CUSTOM_PATHS:
        integrations.add_custom_path(path)
        if not quiet:
            print(f"[RETRO] Registered {GAME} at {path}")
    return GAME_DIR


def backup_state(state_name: str, *, label: str) -> Path | None:
    """Copy a state before a deliberate checkpoint refresh."""
    src = GAME_DIR / f"{state_name}.state"
    if not src.is_file():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = GAME_DIR / f"{state_name}_backup_{label}_{stamp}.state"
    shutil.copy2(src, dest)
    return dest


def backup_mutable_start_state(state_name: str, *, label: str) -> Path | None:
    """Copy mutable start states before recording overwrites them."""
    if state_name not in MUTABLE_STATE_NAMES:
        return None
    return backup_state(state_name, label=label)
