"""Recon: from Level7InteriorReconFixture (settled 0x6B, goriyas clear, Food 1),
walk one cardinal exit of 0x6B and observe the destination room.

0x6B layout (recon): west mouth (16,141); a central X of diamond blocks walls
the y=141 centre band at x~96; the y=93 and y=109 bands are clear x=32..208.
So the east (RIGHT) traverse rides the y=109 band east, drops the east column
to y=141, and pushes the OPEN east doorway.  UP exit is at the top-centre.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room6b_onward.py --dir RIGHT --tag 6b_right_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_KEYS,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist

ROOM_6B = 0x6B
BAND_Y = 109
DOOR_Y = 141
EAST_COL_X = 200
WEST_COL_X = 40
TOP_Y = 61
Y_TOL = 4
X_TOL = 4


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    types = sorted(
        {
            int(o.type_id)
            for o in s.objects
            if 1 <= o.slot <= 12 and int(o.type_id) not in (0, 0xFF)
        }
    )
    return {
        "screen": f"0x{s.screen:02x}",
        "screen_int": int(s.screen),
        "level": int(s.level),
        "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "tile": int(s.colliding_tile),
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "room_item_id": int(s.room_item_id),
        "room_all_dead": int(s.room_all_dead),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "food": int(read_u8(ram, ADDR_FOOD)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "triforce": int(s.triforce),
    }


def _step(env, assist, btn, frame):
    env.step(nes_action(btn) if btn else nes_idle_action())
    if assist is not None:
        assist.apply_env(env, frame=frame)


def _route_right(s):
    """Waypoint micro: ride y=109 east, drop east column to y=141, push RIGHT."""
    x, y = int(s.link_x), int(s.link_y)
    if x < EAST_COL_X - X_TOL:
        if abs(y - BAND_Y) > Y_TOL:
            return "UP" if y > BAND_Y else "DOWN"
        return "RIGHT"
    if abs(y - DOOR_Y) > Y_TOL:
        return "UP" if y > DOOR_Y else "DOWN"
    return "RIGHT"


def _route_left(s):
    x, y = int(s.link_x), int(s.link_y)
    if x > WEST_COL_X + X_TOL:
        if abs(y - BAND_Y) > Y_TOL:
            return "UP" if y > BAND_Y else "DOWN"
        return "LEFT"
    if abs(y - DOOR_Y) > Y_TOL:
        return "UP" if y > DOOR_Y else "DOWN"
    return "LEFT"


def _route_up(s):
    x, y = int(s.link_x), int(s.link_y)
    CX = 128
    if abs(x - CX) > X_TOL and y > BAND_Y - Y_TOL:
        return "LEFT" if x > CX else "RIGHT"
    return "UP"


def _route_down(s):
    x, y = int(s.link_x), int(s.link_y)
    CX = 128
    if abs(x - CX) > X_TOL and y < DOOR_Y + Y_TOL:
        return "LEFT" if x > CX else "RIGHT"
    return "DOWN"


ROUTES = {"RIGHT": _route_right, "LEFT": _route_left, "UP": _route_up, "DOWN": _route_down}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="RIGHT", choices=list(ROUTES))
    ap.add_argument("--tag", default="6b_onward_v1")
    ap.add_argument("--from-state", default="Level7InteriorReconFixture")
    ap.add_argument("--max-frames", type=int, default=2500)
    args = ap.parse_args()
    configure_headless()
    assist = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"dir": args.dir, "from_state": args.from_state, "route_eligible": False}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        start = _glance(env)
        out["start"] = start
        print("START", start)
        if start["screen_int"] != ROOM_6B or start["mode"] != PLAY_MODE:
            out["success"] = False
            out["failed"] = "start_pin_mismatch"
            (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
            print("FAIL start pin", start)
            return
        route = ROUTES[args.dir]
        samples = []
        last_key = (start["level"], start["mode"], start["screen_int"])
        arrived_frame = None
        for frame in range(args.max_frames):
            s = read_snapshot(env.get_ram())
            key = (int(s.level), int(s.mode), int(s.screen))
            if int(s.screen) != ROOM_6B and int(s.mode) == PLAY_MODE and not s.transitioning:
                arrived_frame = frame
                break
            if s.transitioning:
                _step(env, assist, args.dir, frame)
                continue
            btn = route(s)
            _step(env, assist, args.dir if s.transitioning else btn, frame)
            if frame % 60 == 0 or key != last_key:
                samples.append({"f": frame, "key": [hex(k) for k in key],
                                "xy": [int(s.link_x), int(s.link_y)],
                                "tile": int(s.colliding_tile)})
                last_key = key
        end = _glance(env)
        end["deaths"] = int(assist.telemetry.deaths)
        end["progression_writes"] = int(assist.telemetry.progression_writes)
        end["capacity_writes"] = int(assist.telemetry.capacity_writes)
        out["end"] = end
        out["arrived_frame"] = arrived_frame
        out["samples"] = samples[-40:]
        # settle a bit in the dest room and re-glance
        if arrived_frame is not None:
            for f in range(120):
                _step(env, assist, None, arrived_frame + f)
            settled = _glance(env)
            out["settled"] = settled
            print("SETTLED", settled)
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        ok = (
            arrived_frame is not None
            and end["level"] == 7
            and end["deaths"] == 0
            and end["progression_writes"] == 0
            and end["capacity_writes"] == 0
        )
        out["success"] = bool(ok)
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print(f"success={ok} arrived_frame={arrived_frame} dest={end['screen']} "
              f"xy={end['xy']} deaths={end['deaths']}")
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
