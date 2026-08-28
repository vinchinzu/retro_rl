"""Capture practice-hack category presets as gs=8 stable-retro states.

Practice-only contractor pins. Product evidence still never loads this ROM.
Default category is KPDR Early Ice (``kpdr25``). Trigger is the hack's own
loader: open the practice menu (Start+Select), write the 24-bit preset
pointer to ``$7E:FD5C``, set ``ram_cm_leave``, and wait for gs=8 to match
the fingerprint. Gameplay writes alone do not load.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from retro_harness.actions import idle_action, snes_action
from retro_harness.env import make_env, reset_obs, write_state_bytes
from super_metroid.paths import (
    GAME_DIR,
    INTEGRATION_DIR,
    PRACTICE_CONTRACTOR_STATE_DIR,
    PRACTICE_GAME,
    PRACTICE_INTEGRATION_DIR,
    SHARED_PRACTICE_ROM,
)
from super_metroid.practice_repertoire.catalog import PRODUCT_CATEGORY, load_catalog
from super_metroid.practice_repertoire.rom_map import (
    map_preset_addresses,
    write_address_map,
)
from super_metroid.ram import (
    GS_ORDINARY,
    parse_env_state,
    read_wram_u16,
    write_wram_u8,
    write_wram_u16,
)

# !ram_load_preset_low_word = !WRAM_START+$5C → $7EFD5C (3-byte pointer).
ADDR_LOAD_PRESET = 0xFD5C
# !WRAM_MENU_START = $7EFE00
ADDR_CM_MENU_BANK = 0xFE26
ADDR_CM_LEAVE = 0xFE2A
BOOT_LIMIT = 8_000
MENU_LIMIT = 400
LOAD_LIMIT = 1_200
XY_BAND = 8
MENU_PULSE = 3
MENU_SETTLE = 8
MENU_OPEN_ATTEMPTS = 3


def _symlink_points_at(link: Path, target: Path) -> bool:
    if not link.is_symlink():
        return False
    try:
        return link.resolve() == target.resolve()
    except OSError:
        return False


def ensure_practice_integration() -> Path:
    """Create the practice-ROM custom integration if missing (symlink ROM)."""

    dest = PRACTICE_INTEGRATION_DIR
    dest.mkdir(parents=True, exist_ok=True)
    rom_link = dest / "rom.sfc"
    want = SHARED_PRACTICE_ROM.resolve()
    if not _symlink_points_at(rom_link, want):
        if rom_link.exists() or rom_link.is_symlink():
            rom_link.unlink()
        rom_link.symlink_to(want)
    for name in ("data.json", "metadata.json", "scenario.json"):
        src = INTEGRATION_DIR / name
        if src.is_file() and not (dest / name).is_file():
            shutil.copy2(src, dest / name)
    return dest


def _write_preset_pointer(env: Any, snes: int) -> None:
    write_wram_u16(env, ADDR_LOAD_PRESET, snes & 0xFFFF)
    write_wram_u8(env, ADDR_LOAD_PRESET + 2, (snes >> 16) & 0xFF)


def _fingerprint_ok(state: Any, rec: dict[str, Any]) -> bool:
    if int(state.game_state) != GS_ORDINARY:
        return False
    if rec.get("room_id") is not None and int(state.room_id) != int(rec["room_id"]):
        return False
    if rec.get("x") is not None and abs(int(state.samus_x) - int(rec["x"])) > XY_BAND:
        return False
    if rec.get("y") is not None and abs(int(state.samus_y) - int(rec["y"])) > XY_BAND:
        return False
    return True


def boot_to_gameplay(env: Any, *, limit: int = BOOT_LIMIT) -> Any:
    """Mash START from power-on until ordinary gameplay."""

    reset_obs(env)
    start = snes_action("START")
    idle = idle_action()
    state = parse_env_state(env, mode="nav")
    for frame in range(limit):
        env.step(start if (frame % 40) < 8 else idle)
        state = parse_env_state(env, frame=frame, mode="nav")
        if (
            state.game_state == GS_ORDINARY
            and state.door_transition == 0
            and frame > 40
        ):
            return state
    raise RuntimeError(
        f"practice ROM never reached gs=8 (last gs={state.game_state} "
        f"room=0x{state.room_id:04X})"
    )


def _menu_bank(env: Any) -> int:
    return int(read_wram_u16(env, ADDR_CM_MENU_BANK))


def _wait_gameplay(env: Any, *, limit: int = MENU_LIMIT) -> None:
    """Idle until gs=8 with no door flag so Start+Select can fire the menu shortcut."""

    idle = idle_action()
    for _ in range(limit):
        state = parse_env_state(env, mode="nav")
        if state.game_state == GS_ORDINARY and state.door_transition == 0:
            return
        env.step(idle)


def _open_practice_menu(env: Any, *, limit: int = MENU_LIMIT) -> None:
    """Start+Select opens the InfoHUD practice menu. ``cm_init`` clears the load pointer.

    ``ram_cm_menu_bank`` stays nonzero after the menu closes, so a leftover bank is
    not "menu is open". Clear it and wait for ``cm_init`` to write it again.
    """

    combo = snes_action("START", "SELECT")
    idle = idle_action()
    _wait_gameplay(env)
    for attempt in range(MENU_OPEN_ATTEMPTS):
        if _menu_bank(env):
            write_wram_u16(env, ADDR_CM_MENU_BANK, 0)
            for _ in range(MENU_SETTLE):
                env.step(idle)
        for _ in range(MENU_PULSE):
            env.step(combo)
        for _ in range(limit):
            env.step(idle)
            if _menu_bank(env):
                for _ in range(MENU_SETTLE):
                    env.step(idle)
                return
        for _ in range(30):
            env.step(idle)
    raise RuntimeError("practice menu did not open (Start+Select)")


def load_category_preset(
    env: Any,
    snes: int,
    rec: dict[str, Any],
    *,
    limit: int = LOAD_LIMIT,
    attempts: int = 2,
) -> Any:
    """Load via menu exit, the only path that JSL ``preset_load``."""

    idle = idle_action()
    state = parse_env_state(env, mode="nav")
    last_error = "preset did not settle"
    for _attempt in range(attempts):
        try:
            _open_practice_menu(env)
        except RuntimeError as exc:
            last_error = str(exc)
            continue
        _write_preset_pointer(env, snes)
        write_wram_u16(env, ADDR_CM_LEAVE, 1)
        for frame in range(limit):
            env.step(idle)
            state = parse_env_state(env, frame=frame, mode="nav")
            if frame < 8:
                continue
            if _fingerprint_ok(state, rec):
                for _ in range(MENU_SETTLE):
                    env.step(idle)
                return state
        last_error = (
            f"preset {rec.get('id')} did not settle "
            f"(gs={state.game_state} room=0x{state.room_id:04X} "
            f"xy=({state.samus_x},{state.samus_y}) want room="
            f"{rec.get('room_hex')} xy=({rec.get('x')},{rec.get('y')}))"
        )
    raise RuntimeError(last_error)


def _session_rows(
    *,
    category: str | None,
    limit: int | None,
    ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    catalog = load_catalog()
    want = set(ids) if ids else None
    rows = [
        rec
        for rec in catalog["sessions"]
        if rec.get("kind", "category_preset") == "category_preset"
        and (category is None or rec.get("category") == category)
        and (want is None or rec.get("id") in want)
    ]
    if limit is not None:
        rows = rows[:limit]
    return rows


def capture_sessions(
    *,
    category: str | None = PRODUCT_CATEGORY,
    limit: int | None = None,
    ids: list[str] | None = None,
    skip_existing: bool = True,
    out_dir: Path = PRACTICE_CONTRACTOR_STATE_DIR,
) -> dict[str, Any]:
    """Boot the practice ROM once and dump one .state per matching session."""

    if not SHARED_PRACTICE_ROM.is_file():
        raise FileNotFoundError(f"practice ROM missing: {SHARED_PRACTICE_ROM}")
    report = map_preset_addresses(SHARED_PRACTICE_ROM)
    write_address_map(report)
    presets = report["presets"]
    rows = _session_rows(category=category, limit=limit, ids=ids)
    ensure_practice_integration()
    env = make_env(PRACTICE_GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    captured = 0
    skipped = 0
    failed: list[dict[str, str]] = []
    try:
        boot_to_gameplay(env)
        for rec in rows:
            dest = out_dir / f"{rec['id']}.state"
            if skip_existing and dest.is_file():
                skipped += 1
                continue
            label = rec.get("data_label")
            snes = rec.get("snes")
            if snes is None and label:
                loc = presets.get(label)
                snes = None if loc is None else loc.get("snes")
            if snes is None:
                failed.append({"id": rec["id"], "error": f"unmapped {label}"})
                continue
            try:
                load_category_preset(env, int(snes), rec)
            except RuntimeError as exc:
                failed.append({"id": rec["id"], "error": str(exc)})
                print(f"FAIL {rec['id']}: {exc}", flush=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            write_state_bytes(dest, env.em.get_state())
            captured += 1
            if captured % 25 == 0 or captured == 1:
                print(
                    f"captured {captured} skipped {skipped} failed {len(failed)} "
                    f"last={rec['id']}",
                    flush=True,
                )
    finally:
        env.close()
    summary = {
        "captured": captured,
        "skipped": skipped,
        "failed": failed,
        "mapped": report["mapped"],
        "map_missing": len(report["missing"]),
        "requested": len(rows),
        "out_dir": str(out_dir),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        default=PRODUCT_CATEGORY,
        help=f"preset category (default {PRODUCT_CATEGORY}, the KPDR Early Ice run)",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="capture every category preset (slow; not needed for KPDR)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="only these session ids (e.g. kpdr25/crateria/morph)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--map-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.map_only:
        report = map_preset_addresses()
        path = write_address_map(report)
        print(
            f"mapped {report['mapped']} missing {len(report['missing'])} → {path}"
        )
        if report["missing"][:8]:
            print("missing sample:", ", ".join(report["missing"][:8]))
        return 0 if not report["missing"] else 1
    category = None if args.all_categories else args.category
    summary = capture_sessions(
        category=category,
        limit=args.limit,
        ids=args.ids,
        skip_existing=not args.overwrite,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "failed"}, indent=2))
    if summary["failed"]:
        print(f"failed {len(summary['failed'])}: {summary['failed'][:5]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
