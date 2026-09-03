"""Recon: MAP 0x18 bomb-north at the hypothesized centre stand.

Pin: Level7Interior18ReconFixture — L7 play 0x18 (120,189), dark, map
room_item 0x17, goriya+keese+bubble, keys 3 / bombs 7 / Candle 0.

Hypothesis: north wall at x=120 is BOMB (source HIDDEN_RUPEES). Stand is
the L7 north-door column analog of L7_ROOM69_WEST_BOMB (44,141) LEFT:
(120, 93) face UP. One trial. If the wall does not open, halt with glance
+ PNG; do not poke doors / ADDR_CANDLE / max_bombs.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room18_bomb_north.py --tag 18_bn_v1
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

ROOM = 0x18
STAND = (120, 93)
FACE = "UP"
OPP = "DOWN"


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


def _reach(env, a, room, tx, ty, f, budget=400):
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
            if stuck >= 40:
                return f, False, [x, y]
        else:
            stuck = 0
            last = xy
        if abs(x - tx) > 3:
            btn = "LEFT" if x > tx else "RIGHT"
        else:
            btn = "UP" if y > ty else "DOWN"
        _step(env, a, btn, f)
        f += 1
        if i > 0 and i % 250 == 0:
            save_rgb_png(
                env.render(),
                RECORDINGS_DIR / f"18_bn_stuck_{i}_{x}_{y}.png",
            )
    e = _s(env)
    return f, False, [int(e.link_x), int(e.link_y)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="18_bn_v1")
    ap.add_argument("--from-state", default="Level7Interior18ReconFixture")
    ap.add_argument("--stand-x", type=int, default=STAND[0])
    ap.add_argument("--stand-y", type=int, default=STAND[1])
    ap.add_argument("--save-fixture", default="")
    args = ap.parse_args()
    stand = (args.stand_x, args.stand_y)
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "route_eligible": False,
        "stand": list(stand),
        "face": FACE,
        "hypothesis": "0x18 north wall at x=120 is BOMB -> HIDDEN_RUPEES",
    }
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        print("START", out["start"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_start.png")
        f = 0
        bombs0 = int(read_u8(env.get_ram(), ADDR_BOMBS))
        f, ok, xy = _reach(env, a, ROOM, stand[0], stand[1], f)
        out["at_stand"] = {
            "ok": ok,
            "xy": xy,
            "glance": _glance(env),
        }
        print("STAND", out["at_stand"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_stand.png")
        if int(_s(env).screen) != ROOM:
            out["result"] = f"walked_into_0x{int(_s(env).screen):02x}"
            out["end"] = _glance(env)
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
            (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
            print("WALKED OUT", out["end"])
            return

        # Face + place. B-slot select of already-owned bombs only.
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
        out["bombs_after_blast"] = out["after_blast"]["bombs"]
        print("BLAST", out["after_blast"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_blast.png")

        hit = None
        samples = []
        for i in range(220):
            s = _s(env)
            # v1 leftover 0x08 mode 4 (scroll) at (120,221) — wait play, not
            # require PLAY_MODE on the first screen-change frame.
            if int(s.screen) != ROOM:
                hit = (int(s.screen), f)
                save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_trans.png")
                break
            x = int(s.link_x)
            b = FACE if abs(x - stand[0]) <= 3 else ("LEFT" if x > stand[0] else "RIGHT")
            if i % 40 == 0:
                rec = {"f": f, "xy": [int(s.link_x), int(s.link_y)],
                       "tile": int(s.colliding_tile), "doors": int(s.cur_opened_doors)}
                samples.append(rec)
                print("PUSH", rec)
            _step(env, a, b, f)
            f += 1
        out["push_samples"] = samples
        if hit is None:
            e = _s(env)
            out["result"] = "blocked"
            out["end_xy"] = [int(e.link_x), int(e.link_y)]
            out["tile"] = int(e.colliding_tile)
            print("BLOCKED", _glance(env))
        else:
            out["result"] = f"0x{hit[0]:02x}"
            out["arrived_frame"] = hit[1]
            for _ in range(170):
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
                            f"Derived from {args.from_state}: 0x18 bomb-north "
                            f"stand {list(stand)} face UP -> {out['result']}. "
                            "No set_state, no Candle/Food/TF/door/key writes. "
                            "ensure_bomb is B-slot select of already-owned bombs.",
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
        end["bombs0"] = bombs0
        out["end"] = end
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("END", end)
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
