"""Recon: fixture -> 0x6B east (Room6BEastController) -> 0x6C, then probe 0x6C
exits (priority RIGHT: source mainline 0x6C -> STALFOS_KEY).

0x6C (DIGDOGGER_1) entry (16,141) west mouth; census 0x38 digdogger-family +
0x55 statue projectile.  Whistle is owned (shrinks digdogger) but the room
may just be traversable along a band.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room6c_onward.py --dir RIGHT --tag 6c_right_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name
from zelda_i.level7.path import Room6BEastController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS, ADDR_CANDLE, ADDR_FOOD, ADDR_KEYS, PLAY_MODE,
    read_snapshot, read_u8,
)
from zelda_i.runner import make_assist

ROOM_6B, ROOM_6C = 0x6B, 0x6C
BANDS = [141, 109, 93, 125, 77, 157]


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


def _to_6c(env, a, f):
    ctl = Room6BEastController()
    for _ in range(ctl.max_frames):
        s = _s(env)
        if int(s.screen) == ROOM_6C and int(s.mode) == PLAY_MODE and not s.transitioning:
            return f, True
        act = ctl.step(s)
        env.step(act.action)
        if a:
            a.apply_env(env, frame=f)
        f += 1
        if ctl.failed:
            return f, False
    return f, False


def _probe(env, a, btn, f, bands):
    """For each y band: reach it near the far side, hold btn, watch for room change."""
    results = []
    far_x = 208 if btn == "RIGHT" else (16 if btn == "LEFT" else 120)
    for by in bands:
        # center on band at mid-x
        for _ in range(150):
            s = _s(env)
            if int(s.screen) != ROOM_6C:
                break
            x, y = int(s.link_x), int(s.link_y)
            if abs(y - by) > 3 and abs(x - 120) < 60:
                _step(env, a, "UP" if y > by else "DOWN", f)
            elif btn in ("RIGHT", "LEFT") and abs(x - 120) > 8:
                _step(env, a, "LEFT" if x > 120 else "RIGHT", f)
            else:
                break
            f += 1
        start = _s(env)
        if int(start.screen) != ROOM_6C:
            results.append({"band": by, "result": f"0x{int(start.screen):02x}_early",
                            "at": [int(start.link_x), int(start.link_y)]})
            return f, results, int(start.screen)
        hit = None
        for _ in range(170):
            s = _s(env)
            if int(s.screen) != ROOM_6C and int(s.mode) == PLAY_MODE and not s.transitioning:
                hit = (int(s.screen), [int(s.link_x), int(s.link_y)])
                break
            _step(env, a, btn, f); f += 1
        e = _s(env)
        rec = {"band": by, "from": [int(start.link_x), int(start.link_y)]}
        if hit:
            rec["result"] = f"0x{hit[0]:02x}"
            rec["at"] = hit[1]
            results.append(rec)
            return f, results, hit[0]
        rec["result"] = "blocked"
        rec["to"] = [int(e.link_x), int(e.link_y)]
        rec["tile"] = int(e.colliding_tile)
        results.append(rec)
        # walk back toward opposite side
        opp = {"RIGHT": "LEFT", "LEFT": "RIGHT", "UP": "DOWN", "DOWN": "UP"}[btn]
        for _ in range(120):
            s = _s(env)
            if btn in ("RIGHT", "LEFT") and abs(int(s.link_x) - 120) < 10:
                break
            _step(env, a, opp, f); f += 1
    return f, results, ROOM_6C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="RIGHT")
    ap.add_argument("--tag", default="6c_right_v1")
    ap.add_argument("--from-state", default="Level7InteriorReconFixture")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"dir": args.dir, "route_eligible": False}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        f = 0
        f, ok = _to_6c(env, a, f)
        out["reached_6c"] = ok
        if not ok:
            out["end"] = _glance(env)
            (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
            print("FAIL 6c", out["end"]); return
        for _ in range(90):
            _step(env, a, None, f); f += 1
        out["at_6c"] = _glance(env)
        print("0x6C", out["at_6c"])
        f, res, scr = _probe(env, a, args.dir, f, BANDS)
        out["probe"] = res
        for r in res:
            print(r)
        if scr != ROOM_6C:
            for _ in range(150):
                _step(env, a, None, f); f += 1
            out["dest_settled"] = _glance(env)
            print("DEST", out["dest_settled"])
        end = _glance(env)
        end["deaths"] = int(a.telemetry.deaths)
        end["progression_writes"] = int(a.telemetry.progression_writes)
        end["capacity_writes"] = int(a.telemetry.capacity_writes)
        out["end"] = end
        out["success"] = scr != ROOM_6C and end["deaths"] == 0
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("success", out["success"], "dest", f"0x{scr:02x}")
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
