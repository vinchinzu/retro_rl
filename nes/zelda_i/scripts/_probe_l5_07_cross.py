"""Scan which Y lets Link cross 0x07 pit from left mouth to right."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import read_snapshot

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ytgt in (140, 149, 157, 160, 165, 169, 173, 177, 181, 189):
        env, assist, _ = w.open_env("Level5Entered07")
        total = [1]
        try:
            idle(env, assist, total, 8)
            w.walk_axis(env, assist, total, "y", ytgt, max_f=300)
            snap = read_snapshot(env.get_ram())
            at = [snap.link_x, snap.link_y]
            w.walk_axis(env, assist, total, "x", 192, max_f=400)
            snap = read_snapshot(env.get_ram())
            rec = {"ytgt": ytgt, "after_y": at, "after_x": [snap.link_x, snap.link_y], "mode": snap.mode, "room": f"0x{snap.screen:02x}", "crossed": snap.link_x >= 160}
            print("Y", rec, flush=True)
            rows.append(rec)
        finally:
            env.close()
    w.write_dump("l5_07_cross", {"rows": rows, "pokes": False, "status_claim": None})
    return {"rows": rows}


if __name__ == "__main__":
    print(main())
