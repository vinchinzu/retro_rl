"""Scratch: replay input tapes off the cached elevator fast-entry snapshot.

Edit ``variants()`` and rerun; every tape replays from the same snapshot, so
a pass is seconds. Prints apex y, x range, first wall-latch pose and the
final seat per tape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import parse_state

sys.path.insert(0, str(Path(__file__).resolve().parent))
from entry_state import ENTRY_STATE, open_entry, snap  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "sweep.json"
Tape = tuple[tuple[str, ...], ...]


def spans(*pairs) -> Tape:
    out: list[tuple[str, ...]] = []
    for names, n in pairs:
        out.extend([tuple(names)] * int(n))
    return tuple(out)


# Entry -> 475 (product shape), settled on the ledge at (156, 475).
ENTRY_TO_475 = (
    spans((("RIGHT", "A"), 16))
    + spans((("LEFT",), 2))
    + spans((("LEFT", "A"), 8))
    + spans((("A",), 34))
    + spans(((), 18))
)
PREFIX = ENTRY_TO_475
LAUNCH = len(PREFIX)


def variants() -> list[tuple[str, Tape]]:
    """Chimney wall jump: the x=155 pinch (walls both sides, y 371-404)."""
    out: list[tuple[str, Tape]] = []
    for launch_x in (145, 147, 149, 151, 153):
        walk = spans((("LEFT",), max(0, (156 - launch_x) * 2 // 3 + 4)))
        for jump in ("LEFT", "RIGHT"):
            for k in (10, 12, 14, 16, 18, 20, 22):
                for away in ("LEFT", "RIGHT"):
                    tape = (
                        PREFIX
                        + walk
                        + spans(((jump, "A"), 2))
                        + spans((("LEFT", "RIGHT", "A"), k))
                        + spans(((away,), 2))
                        + spans(((away, "A"), 60))
                    )
                    out.append((f"x{launch_x}_{jump[0]}{k}_{away[0]}", tape))
    return out


def run_tape(env, session, blob: bytes, tape: Tape) -> list[dict]:
    env.em.set_state(blob)
    session.state = parse_state(env.get_ram(), frame=session.frame)
    rows = []
    for i, names in enumerate(tape):
        session.step(buttons(*names) if names else idle_action(), "sweep")
        rows.append(snap(session.state, f=i + 1, act=" ".join(names) or "-"))
    return rows


def main() -> None:
    env, session = open_entry()
    runs = []
    try:
        entry = snap(session.state)
        print("entry", entry, flush=True)
        blob = ENTRY_STATE.read_bytes()
        for name, tape in variants():
            rows = run_tape(env, session, blob, tape)
            top = min(rows, key=lambda r: r["y"])
            latch = next((r for r in rows if r["pose"] in (131, 132)), None)
            end = rows[-1]
            launch_x = rows[LAUNCH]["x"]
            tag = "-" if latch is None else "f{}p{}".format(latch["f"], latch["pose"])
            seats = []
            for r in rows:
                if r["mt"] in (0, 1) and (not seats or seats[-1][1] != r["y"]):
                    seats.append((r["x"], r["y"]))
            xs = [r["x"] for r in rows]
            print(
                f"{name:14s} apex y{top['y']:4d}@f{top['f']:3d} x{top['x']:4d} "
                f"x[{min(xs)},{max(xs)}] latch={tag} "
                f"lx{launch_x} end=({end['x']},{end['y']})p{end['pose']} seats={seats[-2:]}",
                flush=True,
            )
            runs.append({"name": name, "rows": rows})
            if len(sys.argv) > 1 and sys.argv[1] == "-v":
                for r in rows[LAUNCH:]:
                    print(
                        f"   f{r['f']:3d} {r['act']:12s} x{r['x']:4d} y{r['y']:4d} "
                        f"p{r['pose']:4d} mx{r['mx']:3d} vy{r['vy']:3d} mt{r['mt']:3d}"
                    )
    finally:
        env.close()
    OUT.write_text(json.dumps({"entry": entry, "runs": runs}, indent=1) + "\n")
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
