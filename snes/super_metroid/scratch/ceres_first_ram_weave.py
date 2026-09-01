"""RAM corridor weave after TAS-like Ceres 1 initiate. Scratch only."""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.ram import parse_env_state, set_moonwalk
from super_metroid.routes.skills.moonfall import is_moonfalling

GAME_DIR = Path(__file__).resolve().parents[1]
PIN = GAME_DIR / "scratch" / "ceres_first_control.state"
OUT = GAME_DIR / "scratch" / "ceres_first_ram_weave.json"

INIT = [
    (("RIGHT", "B"), 1),
    (("RIGHT", "A", "B", "Y"), 1),
    (("RIGHT", "B"), 10),
    (("LEFT", "B"), 1),
    (("L", "B"), 1),
    (("B",), 1),
    (("RIGHT", "X", "B"), 1),
    (("RIGHT", "A", "B"), 1),
    (("B",), 1),
]


def _steer(x: int, y: int) -> tuple[str, ...]:
    """Idle-bias TAS corridor. 171 is right-wall; 267 is left-mid."""
    if y < 90:
        if x < 158:
            return ("RIGHT",)
        if x > 168:
            return ("LEFT",)
        return ()
    if y < 190:
        # Pass 171 in the shaft (~x154). Never RIGHT onto x211.
        if x > 158:
            return ("LEFT",)
        if x < 152:
            return ("RIGHT",)
        return ()
    if y < 280:
        # Pass 267 around x166.
        if x < 160:
            return ("RIGHT",)
        if x > 170:
            return ("LEFT",)
        return ()
    if y < 380:
        if x < 154:
            return ("RIGHT",)
        if x > 166:
            return ("LEFT",)
        return ()
    if y < 500:
        if x < 154:
            return ("RIGHT",)
        if x > 172:
            return ("LEFT",)
        return ()
    if y < 630:
        if x < 170:
            return ("RIGHT",)
        if x > 190:
            return ("LEFT",)
        return ()
    return ("RIGHT",)


def main() -> None:
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        assist.attach_env(env)
        boot_from_state(env, PIN, settle_frames=0)
        set_moonwalk(env, True)
        f = 0
        trace = []

        def step(names):
            nonlocal f
            env.step(buttons(*names) if names else idle_action())
            f += 1
            st = parse_env_state(env, frame=f, mode="nav")
            assist.apply(env.data, st)
            trace.append(
                {
                    "f": f,
                    "x": int(st.samus_x),
                    "y": int(st.samus_y),
                    "pose": int(st.pose),
                    "vd": int(st.vertical_direction),
                    "vy": int(st.velocity_y),
                    "mt": int(st.movement_type),
                    "air": int(st.movement_type) in (2, 3, 6, 23),
                    "mf": bool(is_moonfalling(st)),
                    "gs": int(st.game_state),
                    "room": f"0x{int(st.room_id):04X}",
                    "in": list(names),
                }
            )
            return st

        for names, hold in INIT:
            for _ in range(hold):
                step(names)
        st = parse_env_state(env, frame=f, mode="nav")
        for _ in range(700):
            if int(st.room_id) == 0xDF8D and int(st.game_state) == 8:
                break
            x, y = int(st.samus_x), int(st.samus_y)
            air = int(st.movement_type) in (2, 3, 6, 23)
            if int(st.room_id) != 0xDF45:
                st = step(("RIGHT", "B"))
                continue
            if (not air) and y >= 640:
                names = ("RIGHT", "B") if x < 236 else ("RIGHT",)
                st = step(names)
                continue
            if (not air) and y < 120:
                # still on first platform — keep moonfall takeoff
                st = step(("RIGHT", "B"))
                continue
            st = step(_steer(x, y))
        grounded = [r for r in trace if (not r["air"]) and r["y"] >= 80 and r["gs"] == 8]
        vd0 = [r for r in trace if r["mf"]]
        report = {
            "frames": f,
            "end": trace[-1],
            "vd0": len(vd0),
            "max_abs_vy": max((abs(r["vy"]) for r in vd0), default=0),
            "first_ground_y80": grounded[0] if grounded else None,
            "skipped_171": not any(abs(r["y"] - 171) <= 6 and not r["air"] for r in trace),
            "skipped_267": not any(abs(r["y"] - 267) <= 8 and not r["air"] for r in trace),
            "door_gs9": next((r for r in trace if r["gs"] == 9), None),
            "falling8": next(
                (r for r in trace if r["room"] == "0xDF8D" and r["gs"] == 8), None
            ),
        }
        OUT.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
