"""Extract and wire the locally owned Super Mario Bros. 2 NES ROM."""

from __future__ import annotations

import shutil
from pathlib import Path

from retro_harness.env import (
    NES_EXTENSIONS,
    extract_rom_from_zip,
    link_rom_into_integration,
    sha1_file,
)
from smb2.paths import (
    GAME,
    INTEGRATION_DIR,
    ROM_PATH,
    ROM_SHA1_PATH,
    ROMS_DIR,
    SHARED_ROM_ZIP,
)

def setup_rom() -> Path:
    """Copy the first NES ROM from the local zip to the canonical SMB2 path."""
    extracted = extract_rom_from_zip(
        SHARED_ROM_ZIP,
        ROMS_DIR,
        extensions=NES_EXTENSIONS,
    )
    if extracted != ROM_PATH:
        ROM_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(extracted, ROM_PATH)
        extracted.unlink()

    _, hash_path = link_rom_into_integration(ROM_PATH, INTEGRATION_DIR)
    if hash_path != ROM_SHA1_PATH:
        raise RuntimeError(f"unexpected ROM hash path for {GAME}: {hash_path}")
    return ROM_PATH

def main() -> None:
    """Prepare the local ROM and print its recorded SHA-1."""
    rom = setup_rom()
    print(f"ROM ready: {rom}")
    print(f"SHA1: {sha1_file(rom)}")

if __name__ == "__main__":
    main()
