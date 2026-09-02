"""Scratch: from the reactive 171 seat, map the route to the Ceres ship pad.

Runs the tuned climb once, snapshots the y=171 seat, then replays walk/jump
grids off that snapshot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import parse_state

sys.path.insert(0, str(Path(__file__).resolve().parent))
import climb as C  # noqa: E402
from entry_state import ENTRY_STATE, open_entry, snap  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "top_map.json"


def main() -> None:
    env, session = open_entry()
    try:
        env.em.set_state(ENTRY_STATE.read_bytes())
        session.state = parse_state(env.get_ram(), frame=session.frame)
        C.TRACE.clear()
        got = C.climb(session, dict(C.BASE, x267=144, d171="LEFT"))
        print("climb", got, "frames", len(C.TRACE), flush=True)
        if not got.get("171"):
            raise SystemExit("no 171 seat")
        top = env.em.get_state()
        seat = snap(session.state)
        print("171 seat", seat, flush=True)

        runs = []
        for dirn in ("LEFT", "RIGHT"):
            for w in range(0, 40, 4):
                for tag, launch in (
                    ("jR", (("RIGHT",), 2, ("RIGHT", "A"), 110)),
                    ("jU", ((), 0, ("A",), 110)),
                    ("jL", (("LEFT",), 2, ("LEFT", "A"), 110)),
                    ("walk", ((), 0, (dirn,), 110)),
                ):
                    env.em.set_state(top)
                    session.state = parse_state(env.get_ram(), frame=session.frame)
                    C.TRACE.clear()
                    for _ in range(w):
                        C.step(session, dirn)
                    t0, n0, t1, n1 = launch
                    for _ in range(n0):
                        C.step(session, *t0)
                    for _ in range(n1):
                        row = C.step(session, *t1)
                        if int(session.state.game_state) != 8:
                            break
                    rows = list(C.TRACE)
                    seats = []
                    for r in rows:
                        if C.grounded(r) and (not seats or seats[-1][1] != r["y"]):
                            seats.append((r["x"], r["y"]))
                    end = rows[-1]
                    top_y = min(r["y"] for r in rows)
                    print(
                        f"{dirn[0]}{w:2d}_{tag:4s} apexy{top_y:4d} "
                        f"end=({end['x']},{end['y']})p{end['pose']} gs{end['gs']} "
                        f"seats={seats[-3:]}",
                        flush=True,
                    )
                    runs.append({"name": f"{dirn[0]}{w}_{tag}", "rows": rows})
    finally:
        env.close()
    OUT.write_text(json.dumps({"seat": seat, "runs": runs}, indent=1) + "\n")
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
