"""Scratch: reactive elevator climb candidate, tuned off the entry snapshot.

Shape under test (ported to magnet.py once green):

* entry -> 475  one precise wall jump off the right wall (pose 132)
* 475 -> 363    walk to a launch x, turn, ground spin jump
* 363 -> 267    same
* 267 -> 171    same

Named-face kicks (``363`` / ``475side``) dump ``latch_<face>.json`` in the
tas_wram slice schema. Occupancy already named the faces; this file does not
remap the shaft.
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
from tas_wram import grab  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "climb.json"
TRACE: list[dict] = []
LATCH_POSES = {131, 132}


def step(session, *names: str) -> dict:
    session.step(buttons(*names) if names else idle_action(), "climb")
    row = snap(session.state, f=len(TRACE) + 1, act=" ".join(names) or "-")
    TRACE.append(row)
    return row


def grounded(row: dict) -> bool:
    return row["mt"] in (0, 1) and row["vy"] == 0


def walk_to(session, target_x: int, *, limit: int = 90) -> dict:
    row = TRACE[-1]
    for _ in range(limit):
        if abs(row["x"] - target_x) <= 1:
            break
        row = step(session, "RIGHT" if row["x"] < target_x else "LEFT")
    return row


def ledge_jump(
    session,
    dirn: str,
    target_y: int,
    *,
    turn: int = 2,
    slack: int = 4,
    limit: int = 120,
) -> bool:
    for _ in range(turn):
        step(session, dirn)
    airborne = False
    for _ in range(limit):
        row = step(session, dirn, "A")
        airborne = airborne or not grounded(row)
        if airborne and grounded(row):
            return abs(row["y"] - target_y) <= slack
    return False


def entry_to_475(session, *, rise: int = 16, ride: int = 34) -> bool:
    for _ in range(rise):
        step(session, "RIGHT", "A")
    for _ in range(2):
        step(session, "LEFT")
    latched = False
    for _ in range(8):
        latched = step(session, "LEFT", "A")["pose"] == 132 or latched
    for _ in range(ride):
        latched = step(session, "A")["pose"] == 132 or latched
    for _ in range(40):
        row = step(session)
        if grounded(row):
            return latched and abs(row["y"] - 475) <= 4
    return False


STAMPS: list[tuple[str, int]] = []


def stamp(label: str) -> None:
    STAMPS.append((label, len(TRACE)))


def climb(session, params: dict) -> dict:
    STAMPS.clear()
    out = {"entry": False, "475": False, "363": False, "267": False, "171": False}
    out["475"] = entry_to_475(session, rise=params["rise"], ride=params["ride"])
    stamp("475")
    if not out["475"]:
        return out
    walk_to(session, params["x475"])
    stamp("walk475")
    out["363"] = ledge_jump(session, params["d363"], 363)
    stamp("363")
    if not out["363"]:
        return out
    walk_to(session, params["x363"])
    stamp("walk363")
    out["267"] = ledge_jump(session, params["d267"], 267)
    stamp("267")
    if not out["267"]:
        return out
    walk_to(session, params["x267"])
    stamp("walk267")
    out["171"] = ledge_jump(session, params["d171"], 171)
    stamp("171")
    walk_to(session, params["x171"])
    stamp("walk171")
    for _ in range(2):
        step(session, params["dship"])
    for _ in range(120):
        row = step(session, params["dship"], "A")
        if row["gs"] != 8:
            break
    out["ship"] = TRACE[-1]["gs"] == 32
    out["seat"] = (TRACE[-1]["x"], TRACE[-1]["y"])
    return out


BASE = {
    "rise": 16,
    "ride": 34,
    "x475": 137,
    "d363": "RIGHT",
    "x363": 191,
    "d267": "LEFT",
    "x267": 144,
    "d171": "LEFT",
    "x171": 48,
    "dship": "RIGHT",
}

# 363 left face x=160 y=384-399, kick LEFT. Launch 137 is the product hop
# that already threads wj_x 147-155 at that y (climb.json f112-118).
FACE_363 = {
    "name": "363",
    "face_x": 160,
    "y0": 384,
    "y1": 399,
    "into": "RIGHT",
    "kick": "LEFT",
    "wj_x": (147, 155),
    "launch_x": 137,
    "approach": "up",
}

# 475 box sides sit below the stand point (floor 496). Drop off the lip,
# then the same 2f A-off kick. Right side is 4px from the 475 plant.
FACE_475_RIGHT = {
    "name": "475right",
    "face_x": 160,
    "y0": 496,
    "y1": 511,
    "into": "LEFT",
    "kick": "RIGHT",
    "wj_x": (165, 173),
    "launch_x": 156,
    "approach": "down",
    "drop": True,
}

FACE_475_LEFT = {
    "name": "475left",
    "face_x": 96,
    "y0": 496,
    "y1": 511,
    "into": "RIGHT",
    "kick": "LEFT",
    "wj_x": (83, 91),
    "launch_x": 104,
    "approach": "down",
    "drop": True,
}


def _wram_act(session, rows: list[dict], *names: str) -> dict:
    session.step(buttons(*names) if names else idle_action(), "face")
    ram = session.env.get_ram()  # type: ignore[attr-defined]
    rows.append({"f": len(rows), "btn": list(names), "ram": grab(ram)})
    return snap(session.state)


def kick_named_face(session, face: dict) -> dict:
    """One kick off a named occupancy face. Claim before the live A-press.

    Ride into the wall with A until the first frame in the face's (x,y)
    band, 2f away with A off, then away+A. Does not sweep ride counts.
    """
    rows: list[dict] = []
    claim = (
        f"first frame y in [{face['y0']},{face['y1']}] x in "
        f"{face['wj_x']} then 2f {face['kick']} A-off + {face['kick']}+A "
        f"latches pose 131/132 on face {face['face_x']}"
    )
    walk_to(session, face["launch_x"])
    into, away = face["into"], face["kick"]
    x0, x1 = face["wj_x"]
    if face.get("drop"):
        for _ in range(24):
            st = _wram_act(session, rows, away)
            off = st["x"] >= x0 if away == "RIGHT" else st["x"] <= x1
            if off and not grounded(st):
                break
    for _ in range(2):
        _wram_act(session, rows, into)
    in_band = False
    rising = face.get("approach", "up") == "up"
    for _ in range(40):
        st = _wram_act(session, rows, into, "A")
        if face["y0"] <= st["y"] <= face["y1"] and x0 <= st["x"] <= x1:
            in_band = True
            break
        if rising and st["y"] < face["y0"]:
            break
        if not rising and st["y"] > face["y1"]:
            break
    for _ in range(2):
        _wram_act(session, rows, away)
    latch = None
    for _ in range(10):
        st = _wram_act(session, rows, away, "A")
        if st["pose"] in LATCH_POSES:
            latch = {"f": rows[-1]["f"], **st}
            break
    for _ in range(24):
        _wram_act(session, rows, "A")
        if latch is None and snap(session.state)["pose"] in LATCH_POSES:
            latch = {"f": rows[-1]["f"], **snap(session.state)}
    report = {
        "face": face["name"],
        "face_x": face["face_x"],
        "claim": claim,
        "in_band": in_band,
        "latched": latch is not None,
        "latch": latch,
        "end": snap(session.state),
        "rows": rows,
    }
    path = HERE / f"latch_{face['name']}.json"
    path.write_text(json.dumps(report) + "\n")
    report["path"] = str(path)
    return report


def run_face(face: dict) -> dict:
    env, session = open_entry()
    try:
        TRACE.clear()
        planted = entry_to_475(session)
        if not planted:
            raise SystemExit(f"475 plant missed: {snap(session.state)}")
        got = kick_named_face(session, face)
    finally:
        env.close()
    latch = got["latch"]
    print(
        f"face={got['face']} in_band={got['in_band']} "
        f"latched={got['latched']} latch={latch} end={got['end']}",
        flush=True,
    )
    print(f"claim: {got['claim']}", flush=True)
    print(f"report: {got['path']}", flush=True)
    return got


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg in ("363", "363face", "face=363"):
        run_face(FACE_363)
        return
    if arg in ("475right", "475side", "face=475right"):
        run_face(FACE_475_RIGHT)
        return
    if arg in ("475left", "face=475left"):
        run_face(FACE_475_LEFT)
        return
    env, session = open_entry()
    blob = ENTRY_STATE.read_bytes()
    results = []
    try:
        key, _, raw = (arg or "x267=142").partition("=")
        values = [int(v) for v in raw.split(",")] if raw else [BASE[key]]
        for value in values:
            TRACE.clear()
            env.em.set_state(blob)
            session.state = parse_state(env.get_ram(), frame=session.frame)
            params = dict(BASE, **{key: value})
            got = climb(session, params)
            seats = []
            for r in TRACE:
                if grounded(r) and (not seats or seats[-1][1] != r["y"]):
                    seats.append((r["x"], r["y"]))
            print(
                f"{key}={value:4d} {got} "
                f"frames={len(TRACE)} seats={seats[-3:]} stamps={STAMPS}",
                flush=True,
            )
            results.append({"params": params, "got": got, "frames": len(TRACE),
                            "seats": seats, "trace": list(TRACE)})
    finally:
        env.close()
    OUT.write_text(json.dumps(results, indent=1) + "\n")
    print(f"report: {OUT}")


if __name__ == "__main__":
    main()
