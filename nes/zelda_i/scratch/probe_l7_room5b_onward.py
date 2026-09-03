"""Recon: fixture -> 0x6B UP -> 0x5B, then map 0x5B interior + exits.

0x5B is dark (Candle 0), south mouth (120,205), census bubble 0x40 + 0x50.
Grid-probe: for each (band y, x) try to reach it and push a cardinal; record
transitions.  Priority UP (mainline north).

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room5b_onward.py --tag 5b_v1
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
    ADDR_BOMBS, ADDR_CANDLE, ADDR_FOOD, ADDR_KEYS, PLAY_MODE,
    read_snapshot, read_u8,
)
from zelda_i.runner import make_assist

ROOM_6B, ROOM_5B = 0x6B, 0x5B
NOTCH_X, BAND_Y, MID_Y = 118, 93, 109


def _s(env):
    return read_snapshot(env.get_ram())


def _step(env, a, btn, f):
    env.step(nes_action(btn) if btn else nes_idle_action())
    if a:
        a.apply_env(env, frame=f)


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    types = sorted({int(o.type_id) for o in s.objects
                    if 1 <= o.slot <= 12 and int(o.type_id) not in (0, 0xFF)})
    return {
        "screen": f"0x{int(s.screen):02x}", "screen_int": int(s.screen),
        "mode": int(s.mode), "xy": [int(s.link_x), int(s.link_y)],
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "room_item_id": int(s.room_item_id), "room_all_dead": int(s.room_all_dead),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "food": int(read_u8(ram, ADDR_FOOD)), "candle": int(read_u8(ram, ADDR_CANDLE)),
        "keys": int(read_u8(ram, ADDR_KEYS)), "bombs": int(read_u8(ram, ADDR_BOMBS)),
    }


def _drive_to_5b(env, a, f):
    # 0x6B: ride y=109 to x=118, up to y=93, up through notch
    for _ in range(300):
        s = _s(env)
        if int(s.screen) != ROOM_6B:
            break
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - NOTCH_X) > 2 and abs(y - MID_Y) <= 8:
            btn = "LEFT" if x > NOTCH_X else "RIGHT"
        elif y > BAND_Y + 3:
            btn = "UP"
        else:
            btn = "UP"
        _step(env, a, btn, f); f += 1
    for _ in range(200):
        s = _s(env)
        if int(s.screen) == ROOM_5B and int(s.mode) == PLAY_MODE and not s.transitioning:
            return f, True
        _step(env, a, "UP", f); f += 1
    return f, False


def _reach(env, a, tx, ty, f, budget=260):
    """Greedy reach (x then y then x), dark-room tolerant."""
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM_5B:
            return f, False, int(s.screen)
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - tx) <= 3 and abs(y - ty) <= 3:
            return f, True, ROOM_5B
        if abs(x - tx) > 3:
            _step(env, a, "LEFT" if x > tx else "RIGHT", f)
        elif abs(y - ty) > 3:
            _step(env, a, "UP" if y > ty else "DOWN", f)
        f += 1
    return f, False, ROOM_5B


def _push(env, a, btn, f, budget=120):
    start = _s(env)
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM_5B and int(s.mode) == PLAY_MODE and not s.transitioning:
            return f, {"dir": btn, "result": f"0x{int(s.screen):02x}",
                       "from": [int(start.link_x), int(start.link_y)],
                       "at": [int(s.link_x), int(s.link_y)]}
        _step(env, a, btn, f); f += 1
    e = _s(env)
    return f, {"dir": btn, "result": "blocked",
               "from": [int(start.link_x), int(start.link_y)],
               "to": [int(e.link_x), int(e.link_y)], "tile": int(e.colliding_tile)}


def _back_to_5b(env, a, btn, f):
    opp = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}[btn]
    for _ in range(160):
        s = _s(env)
        if int(s.screen) == ROOM_5B and int(s.mode) == PLAY_MODE and not s.transitioning:
            return f
        _step(env, a, opp, f); f += 1
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="5b_v1")
    ap.add_argument("--from-state", default="Level7InteriorReconFixture")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"route_eligible": False, "probes": []}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        f = 0
        f, ok = _drive_to_5b(env, a, f)
        out["reached_5b"] = ok
        if not ok:
            out["end"] = _glance(env)
            (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
            print("FAIL 5b", out["end"]); return
        for _ in range(120):
            _step(env, a, None, f); f += 1
        out["dest_5b"] = _glance(env)
        print("0x5B", out["dest_5b"])
        # probe grid: try to slip past the mid wall on each side band, push UP;
        # also test RIGHT/LEFT/DOWN from the door row.
        plan = [
            ("UP", 48, 141), ("UP", 48, 109), ("UP", 48, 77),
            ("UP", 208, 141), ("UP", 208, 109), ("UP", 208, 77),
            ("UP", 118, 77), ("UP", 128, 77),
            ("RIGHT", 200, 77), ("LEFT", 40, 77),
            ("RIGHT", 200, 141), ("LEFT", 40, 141),
        ]
        transitioned = None
        for btn, tx, ty in plan:
            f, reached, scr = _reach(env, a, tx, ty, f)
            if scr != ROOM_5B:
                out["probes"].append({"target": [tx, ty], "dir": btn,
                                      "result": f"0x{scr:02x}_on_reach"})
                transitioned = (btn, scr)
                break
            f, res = _push(env, a, btn, f)
            res["target"] = [tx, ty]
            out["probes"].append(res)
            print(res)
            if res["result"].startswith("0x"):
                transitioned = (btn, int(res["result"], 16))
                for _ in range(150):
                    _step(env, a, None, f); f += 1
                out["dest_settled"] = _glance(env)
                print("DEST", out["dest_settled"])
                break
            f = _back_to_5b(env, a, btn, f) if False else f
        out["transitioned"] = ([transitioned[0], f"0x{transitioned[1]:02x}"]
                               if transitioned else None)
        end = _glance(env)
        end["deaths"] = int(a.telemetry.deaths)
        out["end"] = end
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("transitioned", out["transitioned"], "end", end)
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
