"""Recon: HIDDEN_RUPEES 0x08 bomb-east (source GORIYA_POST_RUPEE).

Pin: Level7Interior08ReconFixture — L7 play 0x08 (120,189) S mouth, mode 5,
diamond cross, 0x35 cluster, bombs 6 / keys 3 / Candle 0.

Hypothesis: east wall at door-row y=141 is BOMB. Stand analog of
L7_ROOM69_WEST_BOMB (44,141) LEFT: (208,141) face RIGHT. Waypoint around
the diamond cross: south band RIGHT, east column UP.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room08_onward.py --tag 08_be_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name, room_item_name
from zelda_i.dungeon.ops import ensure_bomb
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_KEYS,
    ADDR_LADDER,
    ADDR_SELECTED_ITEM,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist

ROOM = 0x08
STAND = (208, 141)
FACE = "RIGHT"
OPP = "LEFT"
WPS = ((200, 189), (200, 141), (208, 141))


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
    types = sorted(
        {
            int(o.type_id)
            for o in s.objects
            if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)
        }
    )
    return {
        "screen": f"0x{int(s.screen):02x}",
        "screen_int": int(s.screen),
        "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "colliding_tile": int(s.colliding_tile),
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "room_all_dead": int(s.room_all_dead),
        "room_item_id": int(s.room_item_id),
        "room_item_name": room_item_name(int(s.room_item_id)),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "food": int(read_u8(ram, ADDR_FOOD)),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "ladder": int(read_u8(ram, ADDR_LADDER)),
        "selected": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "triforce": int(s.triforce),
    }


def _reach(env, a, room, tx, ty, f, budget=360, tag=""):
    last = None
    stuck = 0
    for i in range(budget):
        s = _s(env)
        if int(s.screen) != room:
            return f, False, [int(s.link_x), int(s.link_y)]
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - tx) <= 3 and abs(y - ty) <= 3:
            return f, True, [x, y]
        xy = (x, y)
        if xy == last:
            stuck += 1
            if stuck >= 50:
                return f, False, [x, y]
        else:
            stuck = 0
            last = xy
        # y-first on south band / east column so we do not walk the diamonds.
        if abs(y - ty) > 3:
            btn = "UP" if y > ty else "DOWN"
        else:
            btn = "LEFT" if x > tx else "RIGHT"
        _step(env, a, btn, f)
        f += 1
        if tag and i > 0 and i % 250 == 0:
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{tag}_stuck_{i}_{x}_{y}.png")
    e = _s(env)
    return f, False, [int(e.link_x), int(e.link_y)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="08_be_v1")
    ap.add_argument("--from-state", default="Level7Interior08ReconFixture")
    ap.add_argument("--room", default="0x08")
    ap.add_argument("--stand-x", type=int, default=STAND[0])
    ap.add_argument("--stand-y", type=int, default=STAND[1])
    ap.add_argument("--wp", default="")
    ap.add_argument("--save-fixture", default="")
    args = ap.parse_args()
    stand = (args.stand_x, args.stand_y)
    room = int(args.room, 16)
    wps = WPS
    if args.wp:
        parsed = []
        for chunk in args.wp.replace(";", " ").split():
            wx = wy = None
            for part in chunk.split(","):
                if part.startswith("x:"):
                    wx = int(part[2:])
                elif part.startswith("y:"):
                    wy = int(part[2:])
            parsed.append((wx if wx is not None else stand[0],
                           wy if wy is not None else stand[1]))
        wps = tuple(parsed)
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "route_eligible": False,
        "stand": list(stand),
        "face": FACE,
        "wps": [list(w) for w in wps],
        "room": args.room,
    }
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        print("START", out["start"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_start.png")
        f = 0
        wp_log = []
        for wx, wy in wps:
            f, ok, xy = _reach(env, a, room, wx, wy, f, tag=args.tag)
            rec = {"target": [wx, wy], "ok": ok, "xy": xy, "tile": int(_s(env).colliding_tile)}
            wp_log.append(rec)
            print("WP", rec)
            if int(_s(env).screen) != room:
                out["result"] = f"walked_into_0x{int(_s(env).screen):02x}"
                break
        out["waypoints"] = wp_log
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_stand.png")
        out["at_stand"] = _glance(env)
        print("STAND", out["at_stand"])
        if int(_s(env).screen) != room:
            out["end"] = _glance(env)
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
            (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
            print("WALKED OUT", out["end"])
            return

        for _ in range(4):
            ensure_bomb(env)
            _step(env, a, FACE, f)
            f += 1
        ensure_bomb(env)
        bombs_before = int(read_u8(env.get_ram(), ADDR_BOMBS))
        env.step(nes_action(FACE, "B"))
        if a:
            a.apply_env(env, frame=f)
        f += 1
        for _ in range(7):
            _step(env, a, OPP, f)
            f += 1
        for _ in range(110):
            _step(env, a, None, f)
            f += 1
        out["after_blast"] = _glance(env)
        out["bombs_before"] = bombs_before
        print("BLAST", out["after_blast"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_blast.png")

        hit = None
        samples = []
        for i in range(220):
            s = _s(env)
            if int(s.screen) != room:
                hit = (int(s.screen), f)
                save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_trans.png")
                break
            y = int(s.link_y)
            b = FACE if abs(y - stand[1]) <= 4 else ("UP" if y > stand[1] else "DOWN")
            if i % 40 == 0:
                rec = {"f": f, "xy": [int(s.link_x), int(s.link_y)],
                       "tile": int(s.colliding_tile), "doors": int(s.cur_opened_doors)}
                samples.append(rec)
                print("PUSH", rec)
            _step(env, a, b, f)
            f += 1
        out["push_samples"] = samples
        if hit is None:
            out["result"] = "blocked"
            e = _s(env)
            out["end_xy"] = [int(e.link_x), int(e.link_y)]
            out["tile"] = int(e.colliding_tile)
            print("BLOCKED", _glance(env))
        else:
            out["result"] = f"0x{hit[0]:02x}"
            out["arrived_frame"] = hit[1]
            for _ in range(280):
                s = _s(env)
                if int(s.mode) == PLAY_MODE and not s.transitioning:
                    break
                _step(env, a, None, f)
                f += 1
            out["dest"] = _glance(env)
            print("DEST", out["dest"])
            if args.save_fixture:
                path = save_state(env, GAME_DIR, GAME, args.save_fixture)
                src = state_path(GAME_DIR, GAME, args.from_state)
                write_state_provenance(
                    path,
                    source_state_path=src if src.exists() else None,
                    request={
                        "bead": "rr-8t4.2",
                        "phase": f"level7_interior_{out['result'][2:]}_recon",
                        "track": "recon_fixture",
                        "route_eligible": False,
                        "fixture_only": True,
                        "natural_entry": False,
                        "development_only": True,
                        "fixture_writes": [],
                        "notes": [
                            f"Derived from {args.from_state}: 0x08 bomb-east "
                            f"stand {list(stand)} face RIGHT -> {out['result']}. "
                            "No Candle/Food/TF/door/key writes.",
                            "UnlimitedHealthAssist traversal aid only.",
                        ],
                    },
                    selected_trial={
                        "ok": True,
                        "state": compact_snapshot(_s(env)),
                        "glance": out["dest"],
                    },
                    natural_entry=False,
                )
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
