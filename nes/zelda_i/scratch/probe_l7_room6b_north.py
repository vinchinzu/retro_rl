"""Recon: 0x6B NORTH open door -> 0x5B, then scan 0x5B exits.

0x6B north door notch is at x~118-120 on the y=93 top band (NOT x=128 —
that is solid).  Route: from the west-mouth / fixture pose, ride to
(118, 93), push UP through the notch to live $EB=0x5B.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room6b_north.py --tag 6b_north_dest_v1
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
NOTCH_X = 118
BAND_Y = 93
MID_Y = 109


def _s(env):
    return read_snapshot(env.get_ram())


def _step(env, assist, btn, f):
    env.step(nes_action(btn) if btn else nes_idle_action())
    if assist:
        assist.apply_env(env, frame=f)


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    types = sorted({int(o.type_id) for o in s.objects
                    if 1 <= o.slot <= 12 and int(o.type_id) not in (0, 0xFF)})
    return {
        "screen": f"0x{int(s.screen):02x}", "screen_int": int(s.screen),
        "level": int(s.level), "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "room_item_id": int(s.room_item_id), "room_all_dead": int(s.room_all_dead),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "food": int(read_u8(ram, ADDR_FOOD)), "candle": int(read_u8(ram, ADDR_CANDLE)),
        "keys": int(read_u8(ram, ADDR_KEYS)), "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "triforce": int(s.triforce),
    }


def _goto(env, assist, tx, ty, f, budget=240):
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM_6B:
            return f, False
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - tx) <= 2 and abs(y - ty) <= 3:
            return f, True
        if abs(x - tx) > 2 and abs(y - MID_Y) <= 6:
            btn = "LEFT" if x > tx else "RIGHT"
        elif abs(y - ty) > 3:
            btn = "UP" if y > ty else "DOWN"
        else:
            btn = "LEFT" if x > tx else "RIGHT"
        _step(env, assist, btn, f); f += 1
    return f, False


def _scan_dest(env, assist, f):
    """From 0x5B south-mouth, probe each cardinal for a room change."""
    probes = {}
    for btn, (wx, wy) in {"UP": (120, None), "RIGHT": (None, 141),
                          "LEFT": (None, 141), "DOWN": (120, None)}.items():
        # re-center
        for _ in range(120):
            s = _s(env)
            if int(s.screen) != ROOM_5B:
                break
            x, y = int(s.link_x), int(s.link_y)
            if wx is not None and abs(x - wx) > 3:
                _step(env, assist, "LEFT" if x > wx else "RIGHT", f)
            elif wy is not None and abs(y - wy) > 3:
                _step(env, assist, "UP" if y > wy else "DOWN", f)
            else:
                break
            f += 1
        start = _s(env)
        res = {"from": [int(start.link_x), int(start.link_y)]}
        for _ in range(150):
            s = _s(env)
            if int(s.screen) != ROOM_5B and int(s.mode) == PLAY_MODE and not s.transitioning:
                res["result"] = f"0x{int(s.screen):02x}"
                res["at"] = [int(s.link_x), int(s.link_y)]
                break
            _step(env, assist, btn, f); f += 1
        else:
            e = _s(env)
            res["result"] = "blocked"
            res["to"] = [int(e.link_x), int(e.link_y)]
            res["tile"] = int(e.colliding_tile)
        probes[btn] = res
        # step back into 0x5B if we left
        if res.get("result", "").startswith("0x"):
            opp = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}[btn]
            for _ in range(150):
                s = _s(env)
                if int(s.screen) == ROOM_5B and int(s.mode) == PLAY_MODE and not s.transitioning:
                    break
                _step(env, assist, opp, f); f += 1
    return f, probes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="6b_north_dest_v1")
    ap.add_argument("--from-state", default="Level7InteriorReconFixture")
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()
    configure_headless()
    assist = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"route_eligible": False}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        f = 0
        f, at_notch = _goto(env, assist, NOTCH_X, BAND_Y, f)
        out["at_notch"] = at_notch
        arrived = None
        for _ in range(200):
            s = _s(env)
            if int(s.screen) == ROOM_5B and int(s.mode) == PLAY_MODE and not s.transitioning:
                arrived = f
                break
            _step(env, assist, "UP", f); f += 1
        out["arrived_frame"] = arrived
        if arrived is None:
            out["success"] = False
            out["failed"] = "no_5b"
            out["end"] = _glance(env)
            (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
            print("FAIL", out["end"]); return
        for _ in range(150):
            _step(env, assist, None, f); f += 1
        out["dest_5b"] = _glance(env)
        print("DEST 0x5B", out["dest_5b"])
        if args.scan:
            f, probes = _scan_dest(env, assist, f)
            out["scan"] = probes
            for k, v in probes.items():
                print(k, v)
        end = _glance(env)
        end["deaths"] = int(assist.telemetry.deaths)
        end["progression_writes"] = int(assist.telemetry.progression_writes)
        end["capacity_writes"] = int(assist.telemetry.capacity_writes)
        out["end"] = end
        out["success"] = (arrived is not None and end["deaths"] == 0
                          and end["progression_writes"] == 0 and end["capacity_writes"] == 0)
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("success", out["success"], "arrived", arrived)
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
