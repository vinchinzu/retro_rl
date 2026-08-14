#!/usr/bin/env python3
"""Wire Japanese 1.0 ALttP into alttp_rando integration.

**JP 1.0 only** — refuse the USA dump (``roms/zelda3.sfc``) as primary.

Expected shared ROM::

    roms/zelda3_jp.sfc

samus.link xxHash32 (seed SMZ3): ``0x8AC8FD15``
Internal title: ``ZELDANODENSETSU`` (not ``THE LEGEND OF ZELDA``).

Usage::

    uv run python -m alttp_rando.scripts.setup_rom
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from alttp_rando.paths import (  # noqa: E402
    GAME_DIR,
    INTEGRATION_DIR,
    LOCAL_Z3_ROM,
    REPO_ROOT,
    ROMS_DIR,
    SHARED_Z3_JP_ROM,
    SHARED_Z3_US_ROM,
    VANILLA_DIR,
    Z3_JP_SHA1,
    Z3_JP_XXH32,
)

_Z3_JP_CANDIDATES = (
    REPO_ROOT
    / "roms"
    / "zelda-no-densetsu-kamigami-no-triforce-j-v-1.0"
    / "Zelda no Densetsu - Kamigami no Triforce (J) (V1.0).smc",
    REPO_ROOT / "roms" / "Zelda no Densetsu - Kamigami no Triforce (J) (V1.0).smc",
    REPO_ROOT / "roms" / "Zelda no Densetsu - Kamigami no Triforce (Japan).sfc",
)

_SMZ3_XXH_SEED = 0x534D5A33
_Z3_SIZE = 0x100000
_US_TITLE_PREFIX = b"THE LEGEND OF ZELDA"

def _strip_smc_header(data: bytes) -> bytes:
    if len(data) % 1024 == 512:
        return data[512:]
    return data

def _rom_digest(data: bytes) -> int:
    """samus.link-compatible xxHash32 (seed SMZ3)."""
    body = _strip_smc_header(data)
    try:
        from smz3.rom_builder import smz3_rom_digest

        return smz3_rom_digest(body)
    except Exception:
        pass
    try:
        import xxhash  # type: ignore

        return int(xxhash.xxh32(body, seed=_SMZ3_XXH_SEED).intdigest()) & 0xFFFFFFFF
    except Exception as exc:
        raise RuntimeError(
            "Cannot validate JP ROM hash (need smz3.rom_builder or xxhash)"
        ) from exc

def _lorom_title(data: bytes) -> bytes:
    body = _strip_smc_header(data)
    if len(body) < 0x7FC0 + 21:
        return b""
    return body[0x7FC0 : 0x7FC0 + 21]

def validate_z3_jp(data: bytes, *, path: Path | None = None) -> bytes:
    """Return unheadered JP 1.0 bytes or raise ValueError."""
    body = _strip_smc_header(data)
    label = f" ({path})" if path else ""
    if len(body) < _Z3_SIZE:
        raise ValueError(f"Zelda 3 ROM too small: {len(body)} < {_Z3_SIZE}{label}")
    body = body[:_Z3_SIZE]
    digest = _rom_digest(body)
    if digest == Z3_JP_XXH32:
        return body
    title = _lorom_title(body)
    if title.startswith(_US_TITLE_PREFIX):
        raise ValueError(
            "ALttP ROM is the **USA** dump (title THE LEGEND OF ZELDA). "
            "alttp_rando requires **Japanese 1.0** (ZELDANODENSETSU, "
            f"xxh32 0x{Z3_JP_XXH32:08X}). Place it at roms/zelda3_jp.sfc.{label}"
        )
    raise ValueError(
        f"ALttP ROM failed JP 1.0 hash: got 0x{digest:08X}, "
        f"expected 0x{Z3_JP_XXH32:08X} (title={title!r}){label}"
    )

def _ensure_shared_jp() -> Path:
    if SHARED_Z3_JP_ROM.is_file() or SHARED_Z3_JP_ROM.is_symlink():
        try:
            validate_z3_jp(SHARED_Z3_JP_ROM.read_bytes(), path=SHARED_Z3_JP_ROM)
            return SHARED_Z3_JP_ROM
        except (ValueError, OSError) as exc:
            print(f"WARN: {SHARED_Z3_JP_ROM} invalid ({exc}); searching…")
            if SHARED_Z3_JP_ROM.exists() or SHARED_Z3_JP_ROM.is_symlink():
                SHARED_Z3_JP_ROM.unlink()

    for cand in _Z3_JP_CANDIDATES:
        if not cand.is_file():
            continue
        try:
            validate_z3_jp(cand.read_bytes(), path=cand)
        except ValueError as exc:
            print(f"skip {cand}: {exc}")
            continue
        rel = cand.relative_to(SHARED_Z3_JP_ROM.parent)
        SHARED_Z3_JP_ROM.symlink_to(rel)
        print(f"Linked: {SHARED_Z3_JP_ROM} -> {rel}")
        return SHARED_Z3_JP_ROM

    msg = f"Missing ALttP JP 1.0: {SHARED_Z3_JP_ROM}"
    if SHARED_Z3_US_ROM.is_file():
        try:
            us_digest = _rom_digest(SHARED_Z3_US_ROM.read_bytes())
            msg += (
                f"\n  Note: {SHARED_Z3_US_ROM} is the USA dump "
                f"(xxh32=0x{us_digest:08X}); do not symlink it here."
            )
        except Exception:
            msg += f"\n  Note: {SHARED_Z3_US_ROM} exists but is USA-only for alttp/."
    raise FileNotFoundError(msg)

def _link(shared: Path, local: Path) -> None:
    ROMS_DIR.mkdir(parents=True, exist_ok=True)
    if local.exists() or local.is_symlink():
        if local.resolve() == shared.resolve():
            print(f"OK: {local} -> {shared}")
            return
        local.unlink()
    try:
        rel = shared.resolve().relative_to(local.parent.resolve())
        local.symlink_to(rel)
    except ValueError:
        local.symlink_to(shared)
    print(f"Linked: {local} -> {shared}")

def _wire_integration(shared: Path) -> str:
    INTEGRATION_DIR.mkdir(parents=True, exist_ok=True)

    rom_link = INTEGRATION_DIR / "rom.sfc"
    if rom_link.exists() or rom_link.is_symlink():
        rom_link.unlink()
    rom_link.symlink_to(Path("../../roms/zelda3_jp.sfc"))

    raw = shared.read_bytes()
    validate_z3_jp(raw, path=shared)
    digest = hashlib.sha1(raw).hexdigest()
    if digest != Z3_JP_SHA1:
        print(f"WARN: sha1 {digest} != documented {Z3_JP_SHA1} (still wiring)")

    (INTEGRATION_DIR / "rom.sha").write_text(f"{digest}\n", encoding="utf-8")

    data_src = VANILLA_DIR / "custom_integrations" / "Zelda3-Snes" / "data.json"
    data_dst = INTEGRATION_DIR / "data.json"
    if data_src.is_file():
        data_dst.write_text(data_src.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Copied data.json from {data_src}")
    elif not data_dst.is_file() or data_dst.stat().st_size < 20:
        data_dst.write_text(
            json.dumps(
                {
                    "info": {
                        "health": {"address": 3894, "type": "|u1"},
                        "max_health": {"address": 3895, "type": "|u1"},
                        "game_mode": {"address": 16, "type": "|u1"},
                        "submodule": {"address": 17, "type": "|u1"},
                        "room_id": {"address": 160, "type": "|u2"},
                        "indoors": {"address": 27, "type": "|u1"},
                        "link_y": {"address": 32, "type": "|u2"},
                        "link_x": {"address": 34, "type": "|u2"},
                        "link_direction": {"address": 47, "type": "|u1"},
                        "link_action": {"address": 93, "type": "|u1"},
                        "camera_y": {"address": 224, "type": "|u2"},
                        "camera_x": {"address": 226, "type": "|u2"},
                    }
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote fallback data.json -> {data_dst}")

    meta = {
        "default_state": "FirstPlay",
        "default_player_state": "FirstPlay",
        "whitelist": {"data.json": ["*"]},
    }
    (INTEGRATION_DIR / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    scenario = INTEGRATION_DIR / "scenario.json"
    if not scenario.is_file():
        scenario.write_text(
            json.dumps(
                {
                    "done": {"condition": "never"},
                    "reward": {"variables": {}},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    stale = ROMS_DIR / "zelda3.sfc"
    if stale.exists() or stale.is_symlink():
        try:
            title = _lorom_title(stale.read_bytes())
        except OSError:
            title = b""
        if title.startswith(_US_TITLE_PREFIX) or stale.resolve() != shared.resolve():
            stale.unlink()
            print(f"Removed non-JP roms entry: {stale}")

    return digest

def main() -> int:
    try:
        shared = _ensure_shared_jp()
    except FileNotFoundError as exc:
        print(f"setup_rom FAILED: {exc}", file=sys.stderr)
        return 1

    try:
        body = validate_z3_jp(shared.read_bytes(), path=shared)
        digest = _rom_digest(body)
        print(f"Z3 JP OK  xxh32=0x{digest:08X}  {shared.resolve()}")
    except ValueError as exc:
        print(f"setup_rom FAILED: {exc}", file=sys.stderr)
        return 1

    _link(shared, LOCAL_Z3_ROM)
    sha1 = _wire_integration(shared)
    print(f"ROM ready: {INTEGRATION_DIR / 'rom.sfc'} -> {LOCAL_Z3_ROM}")
    print(f"rom.sha: {sha1}")
    print(f"Game dir: {GAME_DIR}")
    print("JP 1.0 only — USA zelda3.sfc is not used by this package.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
