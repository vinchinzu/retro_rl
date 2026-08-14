"""Link vanilla Super Metroid + ALttP **JP 1.0** into smz3/roms/.

SMZ3 (samus.link / tewtal V11) rejects the USA ALttP dump used by ``alttp/``.
Place the Japanese 1.0 unheadered ROM at::

    roms/zelda3_jp.sfc

Internal title should be ``ZELDANODENSETSU`` (not ``THE LEGEND OF ZELDA``).
Official samus.link xxHash32 (seed ``SMZ3``): SM ``0xCADB4883``, Z3 ``0x8AC8FD15``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from smz3.paths import (  # noqa: E402
    LOCAL_SM_ROM,
    LOCAL_Z3_ROM,
    REPO_ROOT,
    ROMS_DIR,
    SHARED_SM_ROM,
    SHARED_Z3_JP_ROM,
    SMZ3_SM_XXH32,
    SMZ3_Z3_XXH32,
)
from smz3.rom_builder import smz3_rom_digest, validate_sm_rom, validate_z3_jp_rom  # noqa: E402

# Known local dump locations (Internet Archive folder name, etc.).
_Z3_JP_CANDIDATES = (
    REPO_ROOT
    / "roms"
    / "zelda-no-densetsu-kamigami-no-triforce-j-v-1.0"
    / "Zelda no Densetsu - Kamigami no Triforce (J) (V1.0).smc",
    REPO_ROOT / "roms" / "Zelda no Densetsu - Kamigami no Triforce (J) (V1.0).smc",
    REPO_ROOT / "roms" / "Zelda no Densetsu - Kamigami no Triforce (Japan).sfc",
)

def _link(shared: Path, local: Path) -> None:
    if not shared.is_file():
        raise FileNotFoundError(f"Missing shared ROM: {shared}")
    ROMS_DIR.mkdir(parents=True, exist_ok=True)
    if local.exists() or local.is_symlink():
        if local.resolve() == shared.resolve():
            print(f"OK: {local} -> {shared}")
            return
        local.unlink()
    local.symlink_to(shared)
    print(f"Linked: {local} -> {shared}")

def _ensure_shared_z3_jp() -> Path | None:
    """Return path to validated JP ROM, creating ``roms/zelda3_jp.sfc`` if needed."""
    if SHARED_Z3_JP_ROM.is_file() or SHARED_Z3_JP_ROM.is_symlink():
        try:
            validate_z3_jp_rom(SHARED_Z3_JP_ROM.read_bytes(), path=SHARED_Z3_JP_ROM)
            return SHARED_Z3_JP_ROM
        except (ValueError, OSError) as exc:
            print(f"WARN: {SHARED_Z3_JP_ROM} invalid ({exc}); searching candidates…")
            if SHARED_Z3_JP_ROM.is_symlink() or SHARED_Z3_JP_ROM.is_file():
                SHARED_Z3_JP_ROM.unlink()

    for cand in _Z3_JP_CANDIDATES:
        if not cand.is_file():
            continue
        try:
            validate_z3_jp_rom(cand.read_bytes(), path=cand)
        except ValueError as exc:
            print(f"skip {cand}: {exc}")
            continue
        # Relative symlink under roms/ for portability.
        rel = cand.relative_to(SHARED_Z3_JP_ROM.parent)
        SHARED_Z3_JP_ROM.symlink_to(rel)
        print(f"Linked: {SHARED_Z3_JP_ROM} -> {rel}")
        return SHARED_Z3_JP_ROM
    return None

def main() -> int:
    us_hint = _REPO_ROOT / "roms" / "zelda3.sfc"
    errors: list[str] = []

    if not SHARED_SM_ROM.is_file():
        errors.append(f"Missing Super Metroid: {SHARED_SM_ROM}")
    else:
        try:
            validate_sm_rom(SHARED_SM_ROM.read_bytes(), path=SHARED_SM_ROM)
            print(f"SM OK  xxh32=0x{SMZ3_SM_XXH32:08X}  {SHARED_SM_ROM}")
        except ValueError as exc:
            errors.append(str(exc))

    z3_path = _ensure_shared_z3_jp()
    if z3_path is None:
        msg = f"Missing ALttP JP 1.0: {SHARED_Z3_JP_ROM}"
        if us_hint.is_file():
            digest = smz3_rom_digest(us_hint.read_bytes())
            msg += (
                f"\n  Note: {us_hint} exists but is the USA dump "
                f"(xxh32=0x{digest:08X}); SMZ3 needs JP 1.0 "
                f"(0x{SMZ3_Z3_XXH32:08X}). Do not symlink the USA file."
            )
        msg += (
            "\n  Or place archive dump under "
            "roms/zelda-no-densetsu-kamigami-no-triforce-j-v-1.0/"
        )
        errors.append(msg)
    else:
        print(f"Z3 OK  xxh32=0x{SMZ3_Z3_XXH32:08X}  {z3_path.resolve()}")

    if errors:
        print("SMZ3 vanilla ROM setup FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        print(
            "\nObtain Super Metroid (JU) + Zelda no Densetsu JP 1.0 unheadered,\n"
            "place JP ALttP at roms/zelda3_jp.sfc, then re-run this script and\n"
            "  uv run python smz3/scripts/generate_seed.py --test\n"
            "  uv run python smz3/scripts/wire_integration_rom.py",
            file=sys.stderr,
        )
        return 1

    _link(SHARED_SM_ROM, LOCAL_SM_ROM)
    _link(SHARED_Z3_JP_ROM, LOCAL_Z3_ROM)
    # Remove stale US symlink if present from older setup.
    stale = ROMS_DIR / "zelda3.sfc"
    if stale.is_symlink() or stale.is_file():
        try:
            digest = smz3_rom_digest(stale.read_bytes())
        except OSError:
            digest = 0
        if digest != SMZ3_Z3_XXH32:
            stale.unlink()
            print(f"Removed stale non-JP link: {stale}")
    print("SMZ3 vanilla ROMs ready (combo ROM built per seed).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
