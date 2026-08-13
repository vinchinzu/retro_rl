#!/usr/bin/env python3
"""Build Super Metroid practice-hack ROMs into repo ``roms/``.

Patches vanilla ``roms/SuperMetroid.sfc`` (NTSC, unheadered, 3 MiB,
SHA1 da957f0d…) with the community practice hack IPS from
https://smpractice.speedga.me/ (tewtal/sm_practice_hack).

Outputs (gitignored under ``roms/``):

- ``SuperMetroid_Practice.sfc`` — emulator InfoHUD + presets (no tinystates)
- ``SuperMetroid_Practice_tinystates.sfc`` — same + in-ROM savestates (Snes9x-ish)

Product continuous / pure evidence still uses vanilla ``SuperMetroid.sfc``.
Practice ROM is for human repertoire (preset menus) + InfoHUD practice.

```bash
uv run python snes/super_metroid/scripts/setup_practice_rom.py
uv run python snes/super_metroid/scripts/setup_practice_rom.py --status
```
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SNES_IMPORT_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _SNES_IMPORT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from super_metroid.paths import (  # noqa: E402
    SHARED_PRACTICE_ROM,
    SHARED_PRACTICE_ROM_TINYSTATES,
    SHARED_ROM,
    VANILLA_ROM_SHA1,
)

IPS_EMULATOR = "https://smpractice.speedga.me/patches/emulator-ntsc.ips"
IPS_TINYSTATES = "https://smpractice.speedga.me/patches/tinystates-ntsc.ips"
PRACTICE_ROM_SIZE = 4_194_304  # practice hack expands 3 MiB → 4 MiB


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_ips(rom: bytes, patch: bytes, min_size: int | None = None) -> bytes:
    """Apply a binary IPS (PATCH…EOF) to ``rom``, expanding as needed."""
    if not patch.startswith(b"PATCH"):
        raise ValueError("not an IPS patch (missing PATCH header)")
    data = bytearray(rom)
    if min_size is not None and len(data) < min_size:
        data.extend(bytes(min_size - len(data)))
    i = 5
    while i + 3 <= len(patch):
        if patch[i : i + 3] == b"EOF":
            break
        offset = (patch[i] << 16) | (patch[i + 1] << 8) | patch[i + 2]
        i += 3
        size = (patch[i] << 8) | patch[i + 1]
        i += 2
        if size == 0:
            rle_size = (patch[i] << 8) | patch[i + 1]
            value = patch[i + 2]
            i += 3
            end = offset + rle_size
            if end > len(data):
                data.extend(bytes(end - len(data)))
            for j in range(offset, end):
                data[j] = value
        else:
            end = offset + size
            if end > len(data):
                data.extend(bytes(end - len(data)))
            data[offset:end] = patch[i : i + size]
            i += size
    return bytes(data)


def _fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def build_practice_roms(
    *,
    cache_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Path]:
    if not SHARED_ROM.is_file():
        raise FileNotFoundError(
            f"vanilla ROM missing: {SHARED_ROM}\n"
            "Place NTSC Super Metroid (unheadered .sfc) at roms/SuperMetroid.sfc"
        )
    vanilla_sha = _sha1(SHARED_ROM)
    if vanilla_sha.lower() != VANILLA_ROM_SHA1.lower():
        print(
            f"warning: vanilla SHA1 {vanilla_sha} != expected {VANILLA_ROM_SHA1} "
            "(practice patch may fail or desync)"
        )
    vanilla = SHARED_ROM.read_bytes()
    if len(vanilla) not in (3_145_728, 4_194_304):
        raise ValueError(f"unexpected vanilla size {len(vanilla)} (want 3 MiB unheadered)")

    cache = cache_dir or (_REPO_ROOT / "roms" / "patches")
    cache.mkdir(parents=True, exist_ok=True)
    outs: dict[str, Path] = {}
    for label, url, dest in (
        ("emulator", IPS_EMULATOR, SHARED_PRACTICE_ROM),
        ("tinystates", IPS_TINYSTATES, SHARED_PRACTICE_ROM_TINYSTATES),
    ):
        if dest.is_file() and not force and dest.stat().st_size == PRACTICE_ROM_SIZE:
            print(f"ready: {dest} ({dest.stat().st_size} bytes)")
            outs[label] = dest
            continue
        ips_path = cache / f"{label}-ntsc.ips"
        if force or not ips_path.is_file():
            _fetch(url, ips_path)
        patched = apply_ips(vanilla, ips_path.read_bytes(), min_size=PRACTICE_ROM_SIZE)
        dest.write_bytes(patched)
        print(f"wrote {dest} ({len(patched)} bytes) sha1={hashlib.sha1(patched).hexdigest()}")
        outs[label] = dest
    return outs


def status() -> int:
    rows = [
        ("vanilla", SHARED_ROM, VANILLA_ROM_SHA1),
        ("practice", SHARED_PRACTICE_ROM, None),
        ("practice tinystates", SHARED_PRACTICE_ROM_TINYSTATES, None),
    ]
    ok = True
    for name, path, expect in rows:
        if not path.is_file():
            print(f"MISSING  {name}: {path}")
            ok = False
            continue
        sha = _sha1(path)
        size = path.stat().st_size
        extra = f" expected={expect}" if expect and sha.lower() != expect.lower() else ""
        if expect and sha.lower() != expect.lower():
            ok = False
        print(f"OK       {name}: {path} size={size} sha1={sha}{extra}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--status", action="store_true", help="print ROM presence + hashes")
    p.add_argument("--force", action="store_true", help="re-download IPS and rebuild")
    args = p.parse_args(argv)
    if args.status:
        return status()
    build_practice_roms(force=args.force)
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
