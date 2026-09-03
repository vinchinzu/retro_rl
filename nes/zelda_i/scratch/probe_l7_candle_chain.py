"""Recon driver for the L7 candle chain past 0x59.

From a Level7Interior<room>ReconFixture: optionally clear goriyas, band-map the
room (W->E each y), then take a perimeter/waypoint route to a chosen door and
record the live dest.  Byte-level glance is dumped for 2/2 comparison.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_candle_chain.py \\
        --room 0x49 --from-state Level7Interior49ReconFixture \\
        --dir UP --door-x 120 --wp "y:100,x:120" --tag 49_up_v1 \\
        [--clear-goriya] [--save-fixture Level7Interior<dst>ReconFixture]

--wp is a ; or space separated list of waypoints; each is comma-parts of
"y:<n>" and/or "x:<n>" applied in order (y first then x within one waypoint).
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.combat import nearest_enemy
from zelda_i.dungeon.behaviors import EnemyKind, engagement_hint
from zelda_i.dungeon.ids import object_name, room_item_name
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.level7.path import live_goriyas
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS, ADDR_CANDLE, ADDR_FOOD, ADDR_KEYS, PLAY_MODE,
    read_snapshot, read_u8,
)
from zelda_i.runner import make_assist

OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


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


def _clear(env, a, room, f, budget=3000):
    saw = False
    for i in range(budget):
        s = _s(env)
        if int(s.screen) != room:
            return f, "left"
        live = live_goriyas(s)
        if live:
            saw = True
        elif saw and i > 60:
            return f, "clear"
        else:
            _step(env, a, "DOWN", f); f += 1; continue
        tgt = nearest_enemy(s.link_x, s.link_y, live)
        hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
        _step(env, a, (hint.face, "A") if (i % 8) < 4 else hint.face, f); f += 1
    return f, "timeout"


def _reach_y(env, a, room, ty, f, budget=260):
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != room:
            return f, False
        if abs(int(s.link_y) - ty) <= 3:
            return f, True
        _step(env, a, "UP" if int(s.link_y) > ty else "DOWN", f); f += 1
    return f, False


def _go_x(env, a, room, tx, f, budget=160):
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != room:
            return f, False
        if abs(int(s.link_x) - tx) <= 3:
            return f, True
        _step(env, a, "LEFT" if int(s.link_x) > tx else "RIGHT", f); f += 1
    return f, True


def _sweep(env, a, room, ty, direction, f, budget=130):
    xs, ys = [], []
    trans = None
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != room:
            trans = int(s.screen)
            break
        xs.append(int(s.link_x)); ys.append(int(s.link_y))
        _step(env, a, direction, f); f += 1
    return f, {"y": ty, "dir": direction,
               "x_min": min(xs) if xs else None, "x_max": max(xs) if xs else None,
               "y_seen": [min(ys), max(ys)] if ys else None,
               "trans": f"0x{trans:02x}" if trans else None}


def _parse_wps(txt):
    wps = []
    for chunk in txt.replace(";", " ").split():
        wy = wx = None
        for part in chunk.split(","):
            part = part.strip()
            if part.startswith("y:"):
                wy = int(part[2:])
            elif part.startswith("x:"):
                wx = int(part[2:])
        wps.append((wx, wy))
    return wps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True)
    ap.add_argument("--from-state", required=True)
    ap.add_argument("--dir", default="UP", choices=["UP", "DOWN", "LEFT", "RIGHT"])
    ap.add_argument("--door-x", type=int, default=120)
    ap.add_argument("--door-y", type=int, default=141)
    ap.add_argument("--wp", default="")
    ap.add_argument("--tag", default="cc_v1")
    ap.add_argument("--clear-goriya", action="store_true")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--save-fixture", default="")
    args = ap.parse_args()
    room = int(args.room, 16)
    vert = args.dir in ("UP", "DOWN")
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"route_eligible": False, "room": args.room, "dir": args.dir, "bands": []}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        print("START", out["start"])
        f = 0
        if args.clear_goriya:
            f, status = _clear(env, a, room, f)
            out["clear_status"] = status
            out["after_clear"] = _glance(env)
            print("AFTER CLEAR", status, out["after_clear"])
        if not args.no_map:
            for ty in range(64, 200, 12):
                f, ok = _reach_y(env, a, room, ty, f)
                if not ok:
                    out["bands"].append({"y": ty, "reach": False,
                                         "now": f"0x{int(_s(env).screen):02x}"})
                    if int(_s(env).screen) != room:
                        break
                    continue
                for _ in range(90):
                    s = _s(env)
                    if int(s.screen) != room or int(s.link_x) <= 18:
                        break
                    _step(env, a, "LEFT", f); f += 1
                if int(_s(env).screen) != room:
                    break
                f, rec = _sweep(env, a, room, ty, "RIGHT", f)
                out["bands"].append(rec)
                print("BAND", rec)
                if rec["trans"]:
                    for _ in range(150):
                        _step(env, a, None, f); f += 1
                    out[f"trans_dest_y{ty}"] = _glance(env)
                    print("TRANS DEST", out[f"trans_dest_y{ty}"])
                    break
        # waypoint door route
        for (wx, wy) in _parse_wps(args.wp):
            if wy is not None:
                f, _ = _reach_y(env, a, room, wy, f)
            if wx is not None:
                f, _ = _go_x(env, a, room, wx, f)
        s = _s(env)
        out["pre_push"] = [int(s.link_x), int(s.link_y)]
        print("PRE PUSH", out["pre_push"])
        hit = None
        for _ in range(240):
            s = _s(env)
            if int(s.screen) != room and int(s.mode) == PLAY_MODE and not s.transitioning:
                hit = (int(s.screen), f)
                break
            if vert:
                x = int(s.link_x)
                b = args.dir if abs(x - args.door_x) <= 4 else ("LEFT" if x > args.door_x else "RIGHT")
            else:
                y = int(s.link_y)
                b = args.dir if abs(y - args.door_y) <= 4 else ("UP" if y > args.door_y else "DOWN")
            _step(env, a, b, f); f += 1
        if hit is None:
            e = _s(env)
            out["result"] = "blocked"
            out["end_xy"] = [int(e.link_x), int(e.link_y)]
            out["tile"] = int(e.colliding_tile)
        else:
            out["result"] = f"0x{hit[0]:02x}"
            out["arrived_frame"] = hit[1]
            for _ in range(170):
                _step(env, a, None, f); f += 1
            out["dest"] = _glance(env)
            print("DEST", out["dest"])
            if args.save_fixture:
                path = save_state(env, GAME_DIR, GAME, args.save_fixture)
                src = state_path(GAME_DIR, GAME, args.from_state)
                write_state_provenance(
                    path, source_state_path=src if src.exists() else None,
                    request={"bead": "rr-8t4.2",
                             "phase": f"level7_interior_{out['result'][2:]}_recon",
                             "track": "recon_fixture", "route_eligible": False,
                             "fixture_only": True, "natural_entry": False,
                             "development_only": True, "fixture_writes": [],
                             "notes": [f"Derived from {args.from_state}: {args.room} "
                                       f"{args.dir} door -> {out['result']}. No set_state, "
                                       "no Candle/Food/TF/door/key/bomb writes.",
                                       "UnlimitedHealthAssist traversal aid only."]},
                    selected_trial={"ok": True, "state": compact_snapshot(_s(env)),
                                    "glance": out["dest"]},
                    natural_entry=False)
                out["saved_fixture"] = str(path)
                print("saved", path)
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
