"""Try fall policies after exact TAS Ceres 1 initiate. Scratch only."""

from __future__ import annotations

import json
from pathlib import Path

from retro_harness.actions import buttons, idle_action
from super_metroid.assist import UnlimitedResourcesAssist
from super_metroid.dev.common import boot_from_state, make_dev_env
from super_metroid.ram import parse_env_state, set_moonwalk

GAME_DIR = Path(__file__).resolve().parents[1]
PIN = GAME_DIR / "scratch" / "ceres_first_control.state"

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
    (("RIGHT", "B"), 7),
    (("RIGHT",), 7),
]


def wide(x, y):
    if y >= 630:
        return ("RIGHT",)
    if x < 150:
        return ("RIGHT",)
    if x > 172:
        return ("LEFT",)
    if 330 <= y < 375 and x >= 157:
        return ("LEFT",)
    return ()


def tas_y(x, y):
    if y >= 630:
        return ("RIGHT",)
    if y < 90:
        return ("RIGHT",) if x < 164 else ()
    if y < 145:
        return ("LEFT",) if x > 158 else ()
    if y < 250:
        return ()
    if y < 280:
        return ("LEFT",) if x > 168 else ()
    if y < 330:
        return ()
    if y < 375:
        return ("LEFT",) if x >= 157 else ()
    if y < 500:
        return ("RIGHT",) if x < 154 else ()
    return ("RIGHT",) if x < 175 else ()


POLICIES = {
    "wide": wide,
    "tas_y": tas_y,
}


def _run(name, fn):
    env = make_dev_env()
    assist = UnlimitedResourcesAssist()
    try:
        assist.attach_env(env)
        boot_from_state(env, PIN, settle_frames=0)
        set_moonwalk(env, True)
        f = 0

        def step(names):
            nonlocal f
            env.step(buttons(*names) if names else idle_action())
            f += 1
            st = parse_env_state(env, frame=f, mode="nav")
            assist.apply(env.data, st)
            return st

        for names, hold in INIT:
            for _ in range(hold):
                st = step(names)
        for _ in range(700):
            if int(st.room_id) == 0xDF8D and int(st.game_state) == 8:
                break
            x, y = int(st.samus_x), int(st.samus_y)
            air = int(st.movement_type) in (2, 3, 6, 23)
            if int(st.room_id) != 0xDF45:
                st = step(("RIGHT", "B"))
                continue
            if (not air) and y >= 640:
                st = step(("RIGHT", "B") if x < 236 else ("RIGHT",))
                continue
            st = step(fn(x, y))
        end = parse_env_state(env, frame=f, mode="nav")
        return {
            "name": name,
            "frames": f,
            "x": int(end.samus_x),
            "y": int(end.samus_y),
            "pose": int(end.pose),
            "room": f"0x{int(end.room_id):04X}",
            "gs": int(end.game_state),
        }
    finally:
        env.close()


def main():
    rows = [_run(n, fn) for n, fn in POLICIES.items()]
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
