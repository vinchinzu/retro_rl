"""Recon: from Level7Interior58ReconFixture, blind-map 0x58 (DODONGOS_UPGRADE):
for each y band walk E then W recording reach + any room transition; then for
each detected edge, hold the cardinal and record dest $EB.

0x58: dark, obj 0x31 (dodongo-family), room_item_id 0x0f, Link spawns
(120,205) bottom.  Source: RIGHT -> GORIYA_COMPASS (mainline),
UP (KEY) -> BOMB_UPGRADE (0x48), DOWN -> KEESE_TRAPS (0x68).

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room58_map.py --tag 58_map_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name, room_item_name
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS, ADDR_KEYS, PLAY_MODE, read_snapshot, read_u8,
)
from zelda_i.runner import make_assist

ROOM_58 = 0x58
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
        "screen": f"0x{int(s.screen):02x}", "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "room_item_id": int(s.room_item_id),
        "room_item_name": room_item_name(int(s.room_item_id)),
        "open_doorway_mask": int(s.open_doorway_mask),
        "cur_opened_doors": int(s.cur_opened_doors),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "keys": int(read_u8(ram, ADDR_KEYS)), "bombs": int(read_u8(ram, ADDR_BOMBS)),
    }


def _reach_y(env, a, ty, f, budget=200):
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM_58:
            return f, False
        y = int(s.link_y)
        if abs(y - ty) <= 3:
            return f, True
        _step(env, a, "UP" if y > ty else "DOWN", f); f += 1
    return f, False


def _sweep(env, a, ty, direction, f, budget=110):
    xs = []
    trans = None
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM_58:
            trans = (int(s.screen), [int(s.link_x), int(s.link_y)])
            break
        xs.append(int(s.link_x))
        _step(env, a, direction, f); f += 1
    e = _s(env)
    return f, {"y": ty, "dir": direction,
               "x_min": min(xs) if xs else None, "x_max": max(xs) if xs else None,
               "end": [int(e.link_x), int(e.link_y)],
               "trans": f"0x{trans[0]:02x}" if trans else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="58_map_v1")
    ap.add_argument("--from-state", default="Level7Interior58ReconFixture")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"route_eligible": False, "bands": []}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        f = 0
        for ty in range(61, 214, 16):
            f, ok = _reach_y(env, a, ty, f)
            if not ok:
                s = _s(env)
                out["bands"].append({"y": ty, "left_room_to": f"0x{int(s.screen):02x}"})
                break
            # go to far west first
            for _ in range(70):
                s = _s(env)
                if int(s.screen) != ROOM_58 or int(s.link_x) <= 20:
                    break
                _step(env, a, "LEFT", f); f += 1
            if int(_s(env).screen) != ROOM_58:
                break
            f, rec = _sweep(env, a, ty, "RIGHT", f)
            out["bands"].append(rec)
            print(rec)
            if rec["trans"]:
                for _ in range(140):
                    _step(env, a, None, f); f += 1
                out["east_dest"] = _glance(env)
                print("EAST DEST", out["east_dest"])
                # step back
                for _ in range(160):
                    s = _s(env)
                    if int(s.screen) == ROOM_58 and int(s.mode) == PLAY_MODE and not s.transitioning:
                        break
                    _step(env, a, "LEFT", f); f += 1
        out["end"] = _glance(env)
        out["end"]["deaths"] = int(a.telemetry.deaths)
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("END", out["end"])
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
