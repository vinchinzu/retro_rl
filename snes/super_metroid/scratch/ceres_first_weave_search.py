"""Search Ceres 1 moonfall weave after TAS-like initiate. Scratch only."""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.ram import parse_env_state, set_moonwalk

GAME_DIR = Path(__file__).resolve().parents[1]
PIN = GAME_DIR / "scratch" / "ceres_first_control.state"
OUT = GAME_DIR / "scratch" / "ceres_first_weave_search.json"

# TAS pad recipe from settled y=72.
INIT = [
    (("RIGHT", "B"), 1),
    (("RIGHT", "A", "B"), 1),
    (("RIGHT", "B"), 10),
    (("LEFT", "B"), 1),
    (("L", "B"), 1),
    (("B",), 1),
    (("RIGHT", "X", "B"), 1),
    (("RIGHT", "A", "B"), 1),
    (("B",), 1),
]

LEDGES = (171, 267, 363, 475, 571)


def _row(st, f: int, names: tuple[str, ...]) -> dict:
    return {
        "f": f,
        "x": int(st.samus_x),
        "y": int(st.samus_y),
        "pose": int(st.pose),
        "vd": int(st.vertical_direction),
        "vy": int(st.velocity_y),
        "mt": int(st.movement_type),
        "air": int(st.movement_type) in (2, 3, 6, 23),
        "gs": int(st.game_state),
        "room": f"0x{int(st.room_id):04X}",
        "in": list(names),
    }


def _corridor(x: int, y: int) -> str | None:
    if y < 90:
        lo, hi = 150, 168
    elif y < 180:
        lo, hi = 152, 166
    elif y < 280:
        lo, hi = 158, 170
    elif y < 380:
        lo, hi = 154, 166
    elif y < 500:
        lo, hi = 154, 172
    else:
        lo, hi = 170, 200
    if x < lo:
        return "RIGHT"
    if x > hi:
        return "LEFT"
    return None


def _near_ledge(y: int, band: int = 12) -> bool:
    return any(abs(y - ledge) <= band for ledge in LEDGES)


def _run(name: str, after: str, n: int = 360) -> dict:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        assist.attach_env(env)
        boot_from_state(env, PIN, settle_frames=0)
        set_moonwalk(env, True)
        f = 0
        trace: list[dict] = []

        def step(names: tuple[str, ...]) -> object:
            nonlocal f
            env.step(buttons(*names) if names else idle_action())
            f += 1
            st = parse_env_state(env, frame=f, mode="nav")
            assist.apply(env.data, st)
            trace.append(_row(st, f, names))
            return st

        for names, hold in INIT:
            for _ in range(hold):
                step(names)

        st = parse_env_state(env, frame=f, mode="nav")
        for _ in range(n - f):
            x, y = int(st.samus_x), int(st.samus_y)
            room = int(st.room_id)
            gs = int(st.game_state)
            air = int(st.movement_type) in (2, 3, 6, 23)
            if room != 0xDF45 and gs == 8:
                break
            if room == 0xDF45 and gs == 8 and (not air) and y >= 640:
                # run to door
                names = ("RIGHT", "B") if x < 236 else ("RIGHT",)
                st = step(names)
                continue
            if after == "idle":
                names = ()
            elif after == "freeze":
                if _near_ledge(y) and air:
                    names = ()
                else:
                    steer = _corridor(x, y)
                    names = (steer,) if steer else ()
            elif after == "freeze8":
                if _near_ledge(y, 8) and air:
                    names = ()
                else:
                    steer = _corridor(x, y)
                    names = (steer,) if steer else ()
            elif after == "tas_rest":
                # leftover TAS weave from rel 119, compressed as idle-bias
                names = ()
            else:
                names = ()
            st = step(names)

        grounded = [r for r in trace if (not r["air"]) and r["y"] >= 80 and r["gs"] == 8]
        vd0 = [r for r in trace if r["vd"] == 0 and r["air"]]
        vy_abs = max((abs(r["vy"]) for r in vd0), default=0)
        land_ys = sorted({r["y"] for r in grounded})
        return {
            "name": name,
            "frames": f,
            "end": trace[-1] if trace else None,
            "vd0": len(vd0),
            "max_abs_vy": vy_abs,
            "first_ground_y80": grounded[0] if grounded else None,
            "ground_ys": land_ys[:12],
            "skipped_267": not any(abs(r["y"] - 267) <= 8 and not r["air"] for r in trace),
            "door": trace[-1]["room"] != "0xDF45" if trace else False,
            "y651": next((r for r in trace if r["y"] >= 640), None),
        }
    finally:
        env.close()


def main() -> None:
    rows = [
        _run("idle", "idle"),
        _run("freeze12", "freeze"),
        _run("freeze8", "freeze8"),
    ]
    OUT.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
