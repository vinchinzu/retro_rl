"""Helpers to wire shared ROM zips into per-game integrations."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

ROM_EXTENSIONS = {".sfc", ".smc", ".fig", ".swc", ".nes"}
SNES_EXTENSIONS = {".sfc", ".smc", ".fig", ".swc"}
NES_EXTENSIONS = {".nes"}


def sha1_file(path: Path) -> str:
    """Return hex SHA1 of a file."""
    h = hashlib.sha1()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def extract_rom_from_zip(
    zip_path: Path,
    dest_dir: Path,
    *,
    extensions: set[str] | None = None,
) -> Path:
    """Extract the first matching ROM member from a zip into dest_dir.

    Returns:
        Path to the extracted ROM file.
    """
    allowed = extensions if extensions is not None else ROM_EXTENSIONS
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            name
            for name in zf.namelist()
            if Path(name).suffix.lower() in allowed
            and not name.endswith("/")
        ]
        if not members:
            raise FileNotFoundError(f"No ROM with {sorted(allowed)} in zip: {zip_path}")
        member = members[0]
        out_name = Path(member).name
        out_path = dest_dir / out_name
        with zf.open(member) as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return out_path


def integration_link_name(rom_path: Path) -> str:
    """Return the stable-retro ``rom.<ext>`` name for an extracted ROM."""
    suffix = rom_path.suffix.lower()
    if suffix in SNES_EXTENSIONS:
        return "rom.sfc"
    if suffix in NES_EXTENSIONS:
        return "rom.nes"
    raise ValueError(f"unsupported ROM extension for integration link: {rom_path}")


def link_rom_into_integration(
    rom_path: Path,
    integration_dir: Path,
    *,
    link_name: str | None = None,
) -> tuple[Path, Path]:
    """Symlink ROM into an integration dir and write rom.sha.

    Returns:
        (link_path, sha_path)
    """
    integration_dir.mkdir(parents=True, exist_ok=True)
    resolved_name = link_name or integration_link_name(rom_path)
    link_path = integration_dir / resolved_name
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    # Drop stale alternate-platform links so only one rom.* remains.
    for stale in integration_dir.glob("rom.*"):
        if stale.name == "rom.sha":
            continue
        if stale != link_path and (stale.is_symlink() or stale.is_file()):
            stale.unlink()
    link_path.symlink_to(rom_path.resolve())
    sha_path = integration_dir / "rom.sha"
    sha_path.write_text(sha1_file(rom_path) + "\n", encoding="utf-8")
    return link_path, sha_path


def setup_game_rom(
    *,
    shared_zip: Path,
    game_dir: Path,
    integration_name: str,
    extensions: set[str] | None = None,
) -> Path:
    """Extract shared zip ROM into game_dir/roms and wire integration.

    Returns:
        Path to the extracted ROM.
    """
    roms_dir = game_dir / "roms"
    integration_dir = game_dir / "custom_integrations" / integration_name
    rom_path = extract_rom_from_zip(shared_zip, roms_dir, extensions=extensions)
    link_rom_into_integration(rom_path, integration_dir)
    return rom_path
