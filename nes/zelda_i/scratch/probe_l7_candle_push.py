"""Recon: 0x1A CANDLE_PUSH — push left 0x68, stairs to Red Candle cellar.

Pin: Level7Interior1AReconFixture — L7 play 0x1A (32,141) W mouth, 4-block
square around a dark center, goriya 0x05/0x06, 0x68 pushable, Candle 0.

Hypothesis: leftmost 0x68 is the "left block"; stand to its right and push
LEFT. Stairs appear in the center. Walk onto them → cellar. ADDR_CANDLE
0→2 must be NATURAL (walk onto the item). Never poke ADDR_CANDLE.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_candle_push.py --tag 1a_push_v1
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
from zelda_i.level9.stairs import PUSHABLE_BLOCK
from zelda_i.walk.physics import OccupancyWalker
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

ROOM = 0x1A
CELLAR_MODES = {9, 10, 11}


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


def _blocks(s):
    return [
        {"slot": int(o.slot), "x": int(o.x), "y": int(o.y), "hp": int(o.hp)}
        for o in s.objects
        if 1 <= int(o.slot) <= 12 and int(o.type_id) == PUSHABLE_BLOCK
    ]


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
        "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "colliding_tile": int(s.colliding_tile),
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "room_item_id": int(s.room_item_id),
        "room_item_name": room_item_name(int(s.room_item_id)),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "objects": [
            {
                "slot": int(o.slot),
                "type": f"0x{int(o.type_id):02x}",
                "hp": int(o.hp),
                "xy": [int(o.x), int(o.y)],
            }
            for o in s.objects
            if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)
        ],
        "room_all_dead": int(s.room_all_dead),
        "blocks": _blocks(s),
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "food": int(read_u8(ram, ADDR_FOOD)),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "ladder": int(read_u8(ram, ADDR_LADDER)),
        "selected": int(read_u8(ram, ADDR_SELECTED_ITEM)),
        "triforce": int(s.triforce),
    }


def _reach(env, a, tx, ty, f, budget=360):
    last = None
    stuck = 0
    for _ in range(budget):
        s = _s(env)
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
        if abs(y - ty) > 3:
            btn = "UP" if y > ty else "DOWN"
        else:
            btn = "LEFT" if x > tx else "RIGHT"
        _step(env, a, btn, f)
        f += 1
    e = _s(env)
    return f, False, [int(e.link_x), int(e.link_y)]


def _occ_to(env, a, tx, ty, f, budget=1800, tag=""):
    """OccupancyWalker to (tx,ty). Miss → block cell → replan; no path → stand."""
    walker = OccupancyWalker(goal=(tx, ty))
    stood = 0
    samples = []
    for i in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM:
            return f, False, [int(s.link_x), int(s.link_y)], walker.misses, samples
        if int(s.mode) in CELLAR_MODES:
            return f, True, [int(s.link_x), int(s.link_y)], walker.misses, samples
        xy = (int(s.link_x), int(s.link_y))
        walker.observe(xy)
        if abs(xy[0] - tx) <= 3 and abs(xy[1] - ty) <= 3:
            return f, True, list(xy), walker.misses, samples
        direction = walker.next_dir(xy)
        if direction is None:
            stood += 1
            if stood >= 20:
                samples.append({"f": f, "xy": list(xy), "stand": True,
                                "misses": walker.misses, "blocked": len(walker.grid.blocked)})
                return f, False, list(xy), walker.misses, samples
            _step(env, a, None, f)
        else:
            stood = 0
            _step(env, a, direction, f)
        f += 1
        if tag and i > 0 and i % 250 == 0:
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{tag}_occ_{i}_{xy[0]}_{xy[1]}.png")
            samples.append({"f": f, "xy": list(xy), "misses": walker.misses})
    e = _s(env)
    return f, False, [int(e.link_x), int(e.link_y)], walker.misses, samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1a_push_v1")
    ap.add_argument("--from-state", default="Level7Interior1AReconFixture")
    ap.add_argument("--push", default="LEFT")
    ap.add_argument("--save-fixture", default="")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"route_eligible": False, "push": args.push}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        print("START", out["start"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_start.png")
        f = 0
        candle0 = int(read_u8(env.get_ram(), ADDR_CANDLE))
        # v7: stand south of 0x68 was correct but the block never moved —
        # goriya hitstun resets the push timer. Clear first.
        saw = False
        for i in range(3000):
            s = _s(env)
            live = live_goriyas(s)
            if live:
                saw = True
            elif saw and i > 60:
                break
            else:
                _step(env, a, "DOWN", f)
                f += 1
                continue
            tgt = nearest_enemy(s.link_x, s.link_y, live)
            hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
            _step(env, a, (hint.face, "A") if (i % 8) < 4 else hint.face, f)
            f += 1
        out["after_clear"] = _glance(env)
        print("AFTER CLEAR", out["after_clear"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_clear.png")
        blocks = _glance(env)["blocks"]
        if not blocks:
            out["result"] = "no_block"
            out["end"] = _glance(env)
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
            (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
            print("NO BLOCK", out["end"])
            return
        left = min(blocks, key=lambda b: (b["x"], b["y"]))
        out["left_block"] = left
        print("LEFT BLOCK", left)
        print("ROOM_ALL_DEAD", out["after_clear"].get("room_all_dead"),
              "objects", out["after_clear"].get("objects"))
        # v13: one 0x05 HP80 survived at ~(160,100) NE of the plus; clear
        # boxed at (112,165). North-around, finish the kill, then L5-style
        # south-face UP (retest once room_all_dead can flip).
        if live_goriyas(_s(env)):
            for wx, wy in ((32, 189), (192, 189), (192, 93), (160, 93)):
                f, ok, xy = _reach(env, a, wx, wy, f)
                print("HUNT WP", [wx, wy], ok, xy)
                if int(_s(env).screen) != ROOM:
                    break
            for i in range(2500):
                s = _s(env)
                live = live_goriyas(s)
                if not live:
                    break
                tgt = nearest_enemy(s.link_x, s.link_y, live)
                hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
                _step(env, a, (hint.face, "A") if (i % 8) < 4 else hint.face, f)
                f += 1
            out["after_hunt"] = _glance(env)
            print("AFTER HUNT", out["after_hunt"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_hunt.png")
        stand_x, stand_y = int(left["x"]), int(left["y"]) + 16
        for wx, wy in ((192, 189), (96, 189), (stand_x, stand_y)):
            f, ok, xy = _reach(env, a, wx, wy, f)
            print("WP", [wx, wy], ok, xy)
            if int(_s(env).screen) != ROOM:
                break
        out["at_push_stand"] = {"ok": ok, "xy": xy, "glance": _glance(env)}
        print("PUSH STAND", out["at_push_stand"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_stand.png")
        if not ok or int(_s(env).screen) != ROOM:
            out["result"] = "no_stand"
            end = _glance(env)
            end["deaths"] = int(a.telemetry.deaths)
            end["progression_writes"] = int(a.telemetry.progression_writes)
            end["capacity_writes"] = int(a.telemetry.capacity_writes)
            out["end"] = end
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
            (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
            print("NO STAND", end)
            return
        for i in range(160):
            s = _s(env)
            if int(s.screen) != ROOM:
                break
            if int(s.mode) in CELLAR_MODES:
                break
            _step(env, a, args.push, f)
            f += 1
            if i % 30 == 0:
                print("PUSHING", _glance(env)["xy"], "blocks", _blocks(s))
        out["after_push"] = _glance(env)
        print("AFTER PUSH", out["after_push"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_pushed.png")

        # Walk the center / block-rest tiles looking for stairs (mode 9).
        stair_hits = []
        for tx, ty in (
            (112, 141),
            (128, 141),
            (120, 141),
            (96, 133),
            (120, 125),
            (104, 141),
            (136, 141),
            (128, 133),
        ):
            if int(_s(env).mode) in CELLAR_MODES:
                break
            f, _, xy = _reach(env, a, tx, ty, f, budget=200)
            for _ in range(80):
                s = _s(env)
                if int(s.mode) in CELLAR_MODES or (
                    int(s.screen) != ROOM and int(s.mode) == PLAY_MODE
                ):
                    stair_hits.append({"xy": [int(s.link_x), int(s.link_y)],
                                       "mode": int(s.mode),
                                       "screen": f"0x{int(s.screen):02x}",
                                       "tile": int(s.colliding_tile)})
                    break
                _step(env, a, None, f)
                f += 1
            rec = {"target": [tx, ty], "xy": xy, "mode": int(_s(env).mode),
                   "screen": f"0x{int(_s(env).screen):02x}"}
            stair_hits.append(rec)
            print("STAIR HUNT", rec)
        out["stair_hunt"] = stair_hits
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_stairs.png")

        # If in cellar, walk onto the item (center-ish).
        s = _s(env)
        if int(s.mode) in CELLAR_MODES:
            out["cellar_enter"] = _glance(env)
            print("CELLAR", out["cellar_enter"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_cellar.png")
            # Two-ladder item cellar: drop to floor, east ladder, climb,
            # walk onto the Red Candle on the center pad.
            for tx, ty in (
                (128, 189),
                (176, 189),
                (176, 141),
                (176, 125),
                (120, 125),
                (120, 141),
                (112, 141),
                (128, 141),
            ):
                if int(read_u8(env.get_ram(), ADDR_CANDLE)) >= 2:
                    break
                f, ok, xy = _reach(env, a, tx, ty, f, budget=280)
                print("CELLAR WP", [tx, ty], ok, xy, "candle",
                      int(read_u8(env.get_ram(), ADDR_CANDLE)),
                      "mode", int(_s(env).mode))
                for _ in range(30):
                    if int(read_u8(env.get_ram(), ADDR_CANDLE)) >= 2:
                        break
                    _step(env, a, None, f)
                    f += 1
            out["after_item_walk"] = _glance(env)
            print("ITEM WALK", out["after_item_walk"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_item.png")
            if int(read_u8(env.get_ram(), ADDR_CANDLE)) >= 2:
                # v17: pad does not walk RIGHT to the east ladder (tile 243).
                # Drop to the floor, west-ladder UP back to 0x1A.
                for wx, wy in ((128, 189), (48, 189), (48, 93)):
                    f, ok, xy = _reach(env, a, wx, wy, f, budget=240)
                    print("RETURN WP", [wx, wy], ok, xy, "mode", int(_s(env).mode),
                          "screen", f"0x{int(_s(env).screen):02x}")
                for _ in range(240):
                    s = _s(env)
                    if int(s.mode) == PLAY_MODE and int(s.screen) == ROOM:
                        break
                    _step(env, a, "UP", f)
                    f += 1
                for _ in range(170):
                    s = _s(env)
                    if int(s.mode) == PLAY_MODE and not s.transitioning:
                        break
                    _step(env, a, None, f)
                    f += 1
                out["after_return"] = _glance(env)
                print("RETURN", out["after_return"])
                save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_return.png")

        candle = int(read_u8(env.get_ram(), ADDR_CANDLE))
        out["candle0"] = candle0
        out["candle"] = candle
        out["result"] = (
            "candle2" if candle >= 2 else
            ("cellar_no_candle" if int(_s(env).mode) in CELLAR_MODES else "no_stairs")
        )
        end = _glance(env)
        end["deaths"] = int(a.telemetry.deaths)
        end["progression_writes"] = int(a.telemetry.progression_writes)
        end["capacity_writes"] = int(a.telemetry.capacity_writes)
        out["end"] = end
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        if candle >= 2 and args.save_fixture:
            path = save_state(env, GAME_DIR, GAME, args.save_fixture)
            src = state_path(GAME_DIR, GAME, args.from_state)
            write_state_provenance(
                path,
                source_state_path=src if src.exists() else None,
                request={
                    "bead": "rr-8t4.2",
                    "phase": "level7_red_candle_cellar_recon",
                    "track": "recon_fixture",
                    "route_eligible": False,
                    "fixture_only": True,
                    "natural_entry": False,
                    "development_only": True,
                    "fixture_writes": [],
                    "notes": [
                        f"Derived from {args.from_state}: 0x1A left-block push "
                        f"+ stairs. ADDR_CANDLE {candle0}->{candle} NATURAL "
                        "(walked onto cellar item). No Candle poke.",
                        "UnlimitedHealthAssist traversal aid only.",
                    ],
                },
                selected_trial={
                    "ok": True,
                    "state": compact_snapshot(_s(env)),
                    "glance": end,
                },
                natural_entry=False,
            )
            out["saved_fixture"] = str(path)
            print("saved", path)
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("END", end)
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
