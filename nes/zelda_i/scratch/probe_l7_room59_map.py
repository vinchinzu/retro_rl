"""Recon: from Level7Interior59ReconFixture, clear the 0x59 goriya 0x05/0x06,
then blind-band-map 0x59 (GORIYA_COMPASS) to find the route around the central
obstacle to the UP door (x~120) and the RIGHT door (x~224, dead-end COMPASS).

Source: 0x59 UP -> GORIYA_BUBBLE (mainline), RIGHT (KILL_CLEAR) -> COMPASS.
The naive clear boxes Link at (48,125); this maps every y band W->E and
records reach + transitions, then tries a perimeter waypoint route UP.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room59_map.py --tag 59_map_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.combat import nearest_enemy
from zelda_i.dungeon.behaviors import EnemyKind, engagement_hint
from zelda_i.dungeon.ids import object_name, room_item_name
from zelda_i.level7.path import live_goriyas
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS, ADDR_CANDLE, ADDR_FOOD, ADDR_KEYS, PLAY_MODE,
    read_snapshot, read_u8,
)
from zelda_i.runner import make_assist

ROOM_59 = 0x59


def _s(env):
    return read_snapshot(env.get_ram())


def _step(env, a, btn, f):
    if btn is None:
        env.step(nes_idle_action())
    elif isinstance(btn, tuple):
        env.step(nes_action(*btn))
    else:
        env.step(nes_action(btn))
    if a:
        a.apply_env(env, frame=f)


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    types = sorted({int(o.type_id) for o in s.objects
                    if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)})
    return {
        "screen": f"0x{int(s.screen):02x}", "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "room_all_dead": int(s.room_all_dead),
        "room_item_id": int(s.room_item_id),
        "room_item_name": room_item_name(int(s.room_item_id)),
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "keys": int(read_u8(ram, ADDR_KEYS)), "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)), "food": int(read_u8(ram, ADDR_FOOD)),
    }


def _clear(env, a, f, budget=2600):
    saw = False
    for i in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM_59:
            return f, "left"
        live = live_goriyas(s)
        if live:
            saw = True
        if not live:
            if saw and i > 60:
                return f, "clear"
            _step(env, a, "DOWN", f); f += 1; continue
        tgt = nearest_enemy(s.link_x, s.link_y, live)
        hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
        _step(env, a, (hint.face, "A") if (i % 8) < 4 else hint.face, f); f += 1
    return f, "timeout"


def _reach_y(env, a, ty, f, budget=220):
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM_59:
            return f, False
        y = int(s.link_y)
        if abs(y - ty) <= 3:
            return f, True
        _step(env, a, "UP" if y > ty else "DOWN", f); f += 1
    return f, False


def _go_x(env, a, tx, f, budget=90):
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM_59:
            return f, False
        x = int(s.link_x)
        if abs(x - tx) <= 3:
            return f, True
        _step(env, a, "LEFT" if x > tx else "RIGHT", f); f += 1
    return f, True


def _sweep(env, a, ty, direction, f, budget=120):
    xs, ys = [], []
    trans = None
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM_59:
            trans = (int(s.screen), [int(s.link_x), int(s.link_y)])
            break
        xs.append(int(s.link_x)); ys.append(int(s.link_y))
        _step(env, a, direction, f); f += 1
    e = _s(env)
    return f, {"y": ty, "dir": direction,
               "x_min": min(xs) if xs else None, "x_max": max(xs) if xs else None,
               "y_seen": [min(ys), max(ys)] if ys else None,
               "end": [int(e.link_x), int(e.link_y)],
               "trans": f"0x{trans[0]:02x}" if trans else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="59_map_v1")
    ap.add_argument("--from-state", default="Level7Interior59ReconFixture")
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
        print("START", out["start"])
        f = 0
        f, status = _clear(env, a, f)
        out["clear_status"] = status
        out["after_clear"] = _glance(env)
        print("AFTER CLEAR", status, out["after_clear"])
        # band map
        for ty in range(64, 200, 12):
            f, ok = _reach_y(env, a, ty, f)
            if not ok:
                out["bands"].append({"y": ty, "reach": False})
                continue
            # far west
            for _ in range(80):
                s = _s(env)
                if int(s.screen) != ROOM_59 or int(s.link_x) <= 18:
                    break
                _step(env, a, "LEFT", f); f += 1
            if int(_s(env).screen) != ROOM_59:
                out["bands"].append({"y": ty, "left_going_west": True})
                break
            f, rec = _sweep(env, a, ty, "RIGHT", f)
            out["bands"].append(rec)
            print("BAND", rec)
            if rec["trans"]:
                for _ in range(150):
                    _step(env, a, None, f); f += 1
                out[f"east_dest_y{ty}"] = _glance(env)
                print("EAST DEST", out[f"east_dest_y{ty}"])
                break
        # perimeter UP-door attempt: go to top-west, cross to x=120, push UP
        out["up_attempt"] = {}
        f, _ = _reach_y(env, a, 72, f)
        f, _ = _go_x(env, a, 40, f)
        f, _ = _reach_y(env, a, 64, f)
        f, okx = _go_x(env, a, 120, f)
        s = _s(env)
        out["up_attempt"]["pre_push"] = [int(s.link_x), int(s.link_y)]
        hit = None
        for _ in range(200):
            s = _s(env)
            if int(s.screen) != ROOM_59 and int(s.mode) == PLAY_MODE and not s.transitioning:
                hit = (int(s.screen), f)
                break
            x = int(s.link_x)
            b = "UP" if abs(x - 120) <= 4 else ("LEFT" if x > 120 else "RIGHT")
            _step(env, a, b, f); f += 1
        if hit:
            out["up_attempt"]["result"] = f"0x{hit[0]:02x}"
            for _ in range(160):
                _step(env, a, None, f); f += 1
            out["up_attempt"]["dest"] = _glance(env)
            print("UP DEST", out["up_attempt"]["dest"])
        else:
            e = _s(env)
            out["up_attempt"]["result"] = "blocked"
            out["up_attempt"]["end"] = [int(e.link_x), int(e.link_y)]
            out["up_attempt"]["tile"] = int(e.colliding_tile)
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
