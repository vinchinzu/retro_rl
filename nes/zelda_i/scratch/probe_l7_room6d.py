"""Recon: fixture -> 0x6B east -> 0x6C east -> 0x6D (STALFOS_KEY), then scan
0x6D's four cardinals (source says dead-end / backtrack only -- verify).

0x6D: stalfos 0x2a, room_item_id 0x19 (small_key), entry (16,141) west mouth.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room6d.py --tag 6d_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name, room_item_name
from zelda_i.level7.path import Room6BEastController
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS, ADDR_CANDLE, ADDR_FOOD, ADDR_KEYS, PLAY_MODE,
    read_snapshot, read_u8,
)
from zelda_i.runner import make_assist

ROOM_6B, ROOM_6C, ROOM_6D = 0x6B, 0x6C, 0x6D
OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


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
        "room_item_id": int(s.room_item_id),
        "room_item_name": room_item_name(int(s.room_item_id)),
        "room_all_dead": int(s.room_all_dead),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "food": int(read_u8(ram, ADDR_FOOD)), "candle": int(read_u8(ram, ADDR_CANDLE)),
        "keys": int(read_u8(ram, ADDR_KEYS)), "bombs": int(read_u8(ram, ADDR_BOMBS)),
    }


def _east_band_traverse(env, a, room, next_room, f, budget=1400):
    """Generic W->E: ride y=141, on block bump UP a few frames then RIGHT again."""
    stuck = 0
    up_ticks = 0
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) == next_room and int(s.mode) == PLAY_MODE and not s.transitioning:
            return f, True
        if int(s.screen) not in (room, next_room):
            return f, False
        if s.transitioning:
            _step(env, a, "RIGHT", f); f += 1; continue
        x, y = int(s.link_x), int(s.link_y)
        px = x
        if up_ticks > 0:
            btn = "UP"; up_ticks -= 1
        elif y > 141 + 4:
            btn = "UP"
        elif y < 141 - 4:
            btn = "DOWN"
        else:
            btn = "RIGHT"
        _step(env, a, btn, f); f += 1
        s2 = _s(env)
        if int(s2.link_x) == px and btn == "RIGHT":
            stuck += 1
            if stuck > 6:
                up_ticks = 10
                stuck = 0
        else:
            stuck = 0
    return f, False


def _scan_card(env, a, room, btn, f, want_xy):
    wx, wy = want_xy
    for _ in range(160):
        s = _s(env)
        if int(s.screen) != room:
            break
        x, y = int(s.link_x), int(s.link_y)
        if wx is not None and abs(x - wx) > 3:
            _step(env, a, "LEFT" if x > wx else "RIGHT", f)
        elif wy is not None and abs(y - wy) > 3:
            _step(env, a, "UP" if y > wy else "DOWN", f)
        else:
            break
        f += 1
    start = _s(env)
    hit = None
    for _ in range(170):
        s = _s(env)
        if int(s.screen) != room and int(s.mode) == PLAY_MODE and not s.transitioning:
            hit = (int(s.screen), [int(s.link_x), int(s.link_y)])
            break
        _step(env, a, btn, f); f += 1
    e = _s(env)
    rec = {"dir": btn, "from": [int(start.link_x), int(start.link_y)]}
    if hit:
        rec["result"] = f"0x{hit[0]:02x}"; rec["at"] = hit[1]
    else:
        rec["result"] = "blocked"; rec["to"] = [int(e.link_x), int(e.link_y)]
        rec["tile"] = int(e.colliding_tile)
    # return into room
    if hit:
        for _ in range(170):
            s = _s(env)
            if int(s.screen) == room and int(s.mode) == PLAY_MODE and not s.transitioning:
                break
            _step(env, a, OPP[btn], f); f += 1
    return f, rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="6d_v1")
    ap.add_argument("--from-state", default="Level7InteriorReconFixture")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"route_eligible": False}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        f = 0
        ctl = Room6BEastController()
        for _ in range(ctl.max_frames):
            s = _s(env)
            if int(s.screen) == ROOM_6C and int(s.mode) == PLAY_MODE and not s.transitioning:
                break
            env.step(ctl.step(s).action)
            if a:
                a.apply_env(env, frame=f)
            f += 1
            if ctl.failed:
                break
        f, ok6c = (f, int(_s(env).screen) == ROOM_6C)
        f, ok = _east_band_traverse(env, a, ROOM_6C, ROOM_6D, f)
        out["reached_6d"] = ok
        if not ok:
            out["end"] = _glance(env)
            (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
            print("FAIL 6d", out["end"]); return
        for _ in range(120):
            _step(env, a, None, f); f += 1
        out["at_6d"] = _glance(env)
        print("0x6D", out["at_6d"])
        scans = []
        for btn, wxy in [("RIGHT", (None, 141)), ("UP", (120, None)),
                         ("DOWN", (120, None)), ("LEFT", (None, 141))]:
            f, rec = _scan_card(env, a, ROOM_6D, btn, f, wxy)
            scans.append(rec)
            print(rec)
        out["scans"] = scans
        end = _glance(env)
        end["deaths"] = int(a.telemetry.deaths)
        end["progression_writes"] = int(a.telemetry.progression_writes)
        end["capacity_writes"] = int(a.telemetry.capacity_writes)
        out["end"] = end
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("END", end)
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
