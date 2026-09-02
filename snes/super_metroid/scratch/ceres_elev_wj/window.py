"""Scratch: how wide is the entry->475 wall-jump window, in frames and in x?

Sweeps the three tunable spans of the shipped recipe off the cached entry
snapshot and reports, per variant, whether pose 132 ever latched and where the
climb seated. This is the measurement behind any "N-frame window" claim.

    PYTHONPATH=snes uv run python -m super_metroid.scratch.ceres_elev_wj.window rise
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import parse_state

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entry_state import ENTRY_STATE, open_entry, snap  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "window.json"

BASE = {"rise": 16, "release": 2, "kick": 8, "ride": 34}
GRID = {
    "rise": list(range(6, 31)),
    "release": list(range(0, 9)),
    "kick": list(range(1, 13)),
    "ride": list(range(20, 51, 2)),
}


def run(env, session, blob: bytes, p: dict) -> dict:
    env.em.set_state(blob)
    session.state = parse_state(env.get_ram(), frame=session.frame)
    rows = []

    def go(names, n):
        for _ in range(n):
            session.step(buttons(*names) if names else idle_action(), "wj")
            rows.append(snap(session.state))

    go(("RIGHT", "A"), p["rise"])
    contact_x = rows[-1]["x"] if rows else None
    go(("LEFT",), p["release"])
    go(("LEFT", "A"), p["kick"])
    go(("A",), p["ride"])
    latched = any(r["pose"] == 132 for r in rows)
    apex = min(r["y"] for r in rows)
    for _ in range(60):
        session.step(idle_action(), "wj")
        rows.append(snap(session.state))
        r = rows[-1]
        if r["mt"] in (0, 1) and r["vy"] == 0:
            break
    seat = rows[-1]
    return {
        "params": dict(p),
        "latched": latched,
        "contact_x": contact_x,
        "apex": apex,
        "seat": (seat["x"], seat["y"]),
        "ok": latched and abs(seat["y"] - 475) <= 4,
    }


def main() -> None:
    keys = sys.argv[1:] or list(GRID)
    env, session = open_entry()
    blob = ENTRY_STATE.read_bytes()
    out = []
    try:
        for key in keys:
            print(f"--- {key} (base {BASE}) ---", flush=True)
            for v in GRID[key]:
                got = run(env, session, blob, dict(BASE, **{key: v}))
                out.append(got)
                mark = "OK " if got["ok"] else ("latch" if got["latched"] else "  .")
                print(
                    f"{key}={v:3d} {mark} contact_x={got['contact_x']} "
                    f"apex={got['apex']} seat={got['seat']}",
                    flush=True,
                )
    finally:
        env.close()
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
