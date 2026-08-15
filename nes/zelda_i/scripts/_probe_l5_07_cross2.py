"""Raw-step scan: which Y crosses 0x07 pit. walk_axis bails on mode 9."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from retro_harness.nes import nes_action
from retro_harness.segment_runner import configure_headless
from zelda_i.dungeon_ops import idle
from zelda_i.paths import RECORDINGS_DIR
from zelda_i.ram import read_snapshot

HERE = Path(__file__).resolve()
_spec = importlib.util.spec_from_file_location("l5whistle", HERE.parent / "_probe_l5_whistle_path.py")
w = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(w)


def raw_axis(env, assist, total, axis, tgt, max_f=400) -> list:
    last = None
    stall = 0
    for _ in range(max_f):
        snap = read_snapshot(env.get_ram())
        if axis == "x":
            if abs(snap.link_x - tgt) <= 1:
                return [snap.link_x, snap.link_y]
            w.step(env, assist, total, nes_action("RIGHT" if snap.link_x < tgt else "LEFT"))
        else:
            if abs(snap.link_y - tgt) <= 1:
                return [snap.link_x, snap.link_y]
            w.step(env, assist, total, nes_action("DOWN" if snap.link_y < tgt else "UP"))
        snap2 = read_snapshot(env.get_ram())
        pos = (snap2.link_x, snap2.link_y)
        if pos == last:
            stall += 1
            if stall >= 30:
                return [snap2.link_x, snap2.link_y]
        else:
            stall = 0
        last = pos
    snap = read_snapshot(env.get_ram())
    return [snap.link_x, snap.link_y]


def main() -> dict:
    configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ytgt in (120, 140, 149, 157, 160, 165, 169, 173, 177, 181):
        env, assist, _ = w.open_env("Level5Entered07")
        total = [1]
        try:
            idle(env, assist, total, 6)
            after_y = raw_axis(env, assist, total, "y", ytgt)
            after_x = raw_axis(env, assist, total, "x", 192)
            snap = read_snapshot(env.get_ram())
            rec = {"ytgt": ytgt, "after_y": after_y, "after_x": after_x, "mode": snap.mode, "room": f"0x{snap.screen:02x}", "crossed": after_x[0] >= 160}
            print("Y", rec, flush=True)
            rows.append(rec)
        finally:
            env.close()
    w.write_dump("l5_07_cross2", {"rows": rows, "pokes": False, "status_claim": None})
    return rows


if __name__ == "__main__":
    main()
