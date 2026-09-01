"""Replay exact Sniq Ceres 1 pad inputs from the settled pin. Scratch only."""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from retro_harness.controls import SNES_BUTTON_NAMES
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.ram import parse_env_state, set_moonwalk
from super_metroid.tas.lsmv import parse_lsmv

GAME_DIR = Path(__file__).resolve().parents[1]
PIN = GAME_DIR / "scratch" / "ceres_first_control.state"
LSMV = GAME_DIR / "tas" / "ref" / "sniq_100_4010M.lsmv"
OUT = GAME_DIR / "scratch" / "ceres_first_tas_replay.json"
TAS_CTRL = 8538
# First Ceres 1 input is rel 101 (B+RIGHT) at pad.
TAS_PAD = 8639
TAS_DOOR = 8789  # gs=9 at x=238


def _row(st, f: int, names: list[str]) -> dict:
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
        "in": names,
    }


def main() -> None:
    movie = parse_lsmv(LSMV)
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        assist.attach_env(env)
        boot_from_state(env, PIN, settle_frames=0)
        set_moonwalk(env, True)
        f = 0
        trace: list[dict] = []
        # Replay pad→door TAS buttons, then RIGHT+B until Falling gs=8.
        end = TAS_DOOR + 40
        for i in range(TAS_PAD, min(end, movie.num_frames)):
            names = [
                SNES_BUTTON_NAMES[k]
                for k, v in enumerate(movie.frames[i])
                if v
            ]
            env.step(buttons(*names) if names else idle_action())
            f += 1
            st = parse_env_state(env, frame=f, mode="nav")
            assist.apply(env.data, st)
            trace.append(_row(st, f, names))
            if int(st.room_id) == 0xDF8D and int(st.game_state) == 8:
                break
        else:
            for _ in range(200):
                env.step(buttons("RIGHT", "B"))
                f += 1
                st = parse_env_state(env, frame=f, mode="nav")
                assist.apply(env.data, st)
                trace.append(_row(st, f, ["RIGHT", "B"]))
                if int(st.room_id) == 0xDF8D and int(st.game_state) == 8:
                    break
        grounded = [r for r in trace if (not r["air"]) and r["y"] >= 80 and r["gs"] == 8]
        vd0 = [r for r in trace if r["vd"] == 0 and r["air"]]
        report = {
            "frames": f,
            "end": trace[-1],
            "vd0": len(vd0),
            "max_abs_vy_vd0": max((abs(r["vy"]) for r in vd0), default=0),
            "first_ground_y80": grounded[0] if grounded else None,
            "skipped_171": not any(abs(r["y"] - 171) <= 6 and not r["air"] for r in trace),
            "skipped_267": not any(abs(r["y"] - 267) <= 8 and not r["air"] for r in trace),
            "door_gs9": next((r for r in trace if r["gs"] == 9), None),
            "falling8": next(
                (r for r in trace if r["room"] == "0xDF8D" and r["gs"] == 8), None
            ),
            "sel": [
                r
                for r in trace
                if r["f"] <= 30
                or r["f"] % 15 == 0
                or r["y"] in range(165, 178)
                or r["y"] in range(260, 275)
                or r["y"] >= 640
                or (r["vd"] == 0 and r["air"] and r["f"] < 40)
            ][:80],
        }
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {k: report[k] for k in report if k != "sel"},
                indent=2,
            )
        )
        print("--- sel ---")
        for r in report["sel"]:
            print(
                r["f"],
                r["x"],
                r["y"],
                "p",
                r["pose"],
                "vd",
                r["vd"],
                "vy",
                r["vy"],
                "air",
                r["air"],
                r["in"],
                r["room"],
                r["gs"],
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
