"""Replay a tape and pin the first settled frame of each room.

Settled is module ``0x07`` or ``0x09`` and submodule 0, and not the item
hold-up pose. The first such frame in a new room or overworld screen is
saved as ``Scan_<room>.state`` with provenance ``scan``, plus the
``pin_probe`` text. Boot matches ``replay_tape`` (blank SRAM once, before
the audit). No RAM writes after that.

    cd snes && SDL_VIDEODRIVER=dummy uv run python alttp/scripts/scan_tape.py \\
        alttp/recordings/verified_tip.tape.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from alttp.paths import BOOT_STATE, RECORDINGS_DIR
from alttp.ram import read_snapshot
from alttp.room_timer import location_key
from alttp.scripts.pin_probe import describe
from alttp.spine.tape import (
    LiveRun, load_tape, ram_crc, read_pin_provenance, write_pin_provenance,
)
from alttp.startup import build_blank_sram, build_boot_env, inject_sram_into_env
from retro_harness.env import write_state_bytes
from retro_harness.segment_runner import configure_headless

SCAN_DIR = RECORDINGS_DIR / "scan"


def pin_name(key: tuple[Any, ...]) -> str:
    """``Scan_0x50`` indoors, ``Scan_ow_0x1B`` / ``Scan_dw_0x1B`` outdoors."""
    if key[0] == "in":
        room = int(key[1])
        text = f"0x{room:04X}" if room > 0xFF else f"0x{room:02X}"
        return f"Scan_{text}"
    world = "dw" if key[1] else "ow"
    return f"Scan_{world}_0x{int(key[2]):02X}"


def _frame_crc(path: Path) -> np.ndarray | None:
    with np.load(path, allow_pickle=False) as data:
        if "frame_crc" not in data.files:
            return None
        return np.asarray(data["frame_crc"], dtype=np.uint32)


def _save_pin(env: object, out_dir: Path, name: str, provenance: dict[str, Any]) -> None:
    state = bytes(env.em.get_state())  # type: ignore[attr-defined]
    path = write_state_bytes(out_dir / f"{name}.state", state)
    (out_dir / f"{name}.txt").write_text("\n".join(describe(env)) + "\n")
    sidecar = write_pin_provenance(
        path,
        source_run=f"scan:{provenance['tape']}",
        frame=provenance["frame"],
        kind="scan",
    )
    payload = read_pin_provenance(path)
    assert payload is not None
    payload.update({key: provenance[key] for key in ("tape", "location", "pin")})
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def scan_tape(path: Path, out_dir: Path | None = None) -> dict[str, Any]:
    """Replay ``path`` from power-on. One pin per settled room or screen."""
    tape = load_tape(path)
    if tape.boot_state != BOOT_STATE:
        raise SystemExit(
            f"tape boot_state {tape.boot_state!r} is not power-on {BOOT_STATE}"
        )
    if tape.meta.get("resumed_from"):
        raise SystemExit(
            f"tape resumed from {tape.meta['resumed_from']!r}; scan needs power-on"
        )
    crcs = _frame_crc(path)
    frames = int(tape.buttons.shape[0])
    if crcs is None or len(crcs) != frames:
        raise SystemExit(f"{path} has no per-frame sync CRC")
    out = SCAN_DIR if out_dir is None else Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    blank = bool(tape.meta.get("sram_blank_at_boot"))
    env = build_boot_env()
    pins: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    visited: dict[tuple[Any, ...], dict[str, Any]] = {}
    last: dict[str, Any] | None = None
    desynced = 0
    first_desync: int | None = None
    try:
        env.reset()
        if blank:
            inject_sram_into_env(env, build_blank_sram())
        live = LiveRun(env, provider="alttp.scan_tape")
        wrapped = live.env
        for index, vec in enumerate(tape.buttons):
            wrapped.step(np.asarray(vec, dtype=np.int8))
            if ram_crc(wrapped.get_ram()) != int(crcs[index]):
                desynced += 1
                if first_desync is None:
                    first_desync = index
                break
            snap = read_snapshot(np.asarray(wrapped.get_ram(), dtype=np.uint8))
            key_now = location_key(
                indoors=snap.indoors,
                room=snap.room_id,
                dark_world=snap.dark_world,
                screen=snap.screen_id,
            )
            if key_now not in visited:
                visited[key_now] = {
                    "location": list(key_now),
                    "frame": index,
                    "module": snap.game_mode,
                    "submodule": snap.submodule,
                    "settled": bool(snap.has_control and not snap.is_hold_up_item),
                }
            last = {
                "location": list(key_now),
                "frame": index,
                "module": snap.game_mode,
                "submodule": snap.submodule,
                "link": [snap.link_x, snap.link_y],
                "settled": bool(snap.has_control and not snap.is_hold_up_item),
            }
            if snap.has_control and not snap.is_hold_up_item:
                visited[key_now]["settled"] = True
            if not snap.has_control or snap.is_hold_up_item:
                continue
            key = key_now
            if key in seen:
                continue
            seen.add(key)
            name = pin_name(key)
            provenance = {
                "tape": path.name,
                "frame": index,
                "location": list(key),
                "pin": name,
            }
            _save_pin(wrapped, out, name, provenance)
            pins.append(
                {
                    "pin": name,
                    "frame": index,
                    "location": list(key),
                    "indoors": bool(snap.indoors),
                    "room": f"0x{snap.room_id:04X}",
                    "screen": f"0x{snap.screen_id:02X}",
                }
            )
        audit = live.report()
    finally:
        env.close()

    loads = int(audit.set_state)
    writes = int(audit.writes)
    report = {
        "ok": desynced == 0 and loads == 0 and writes == 0 and len(pins) == len(seen),
        "tape": str(path),
        "out": str(out),
        "frames": frames,
        "played": first_desync if first_desync is not None else frames,
        "desynced": desynced,
        "first_desync_frame": first_desync,
        "loads": loads,
        "writes": writes,
        "visited": list(visited.values()),
        "unsettled": [row for row in visited.values() if not row["settled"]],
        "last": last,
        "pins": pins,
    }
    (out / "index.json").write_text(json.dumps(report, indent=2) + "\n")
    names = ", ".join(row["pin"] for row in pins)
    print(
        f"scan {path.name}: ok={report['ok']} pins={len(pins)} "
        f"desynced={desynced} frames={frames} loads={loads} writes={writes}"
    )
    print(names)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tape", type=Path, help="a .tape.npz, replayed from power-on")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="pin directory (default recordings/scan)",
    )
    args = parser.parse_args(argv)
    configure_headless()
    report = scan_tape(args.tape, args.out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
