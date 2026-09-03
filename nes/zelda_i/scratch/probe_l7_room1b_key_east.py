"""Recon: 0x1B GORIYA_PRE_DIG KEY-east -> live dest (hyp FORCED_DIGDOGGER).

Pin: Level7Interior1BReconFixture — L7 play 0x1B (32,141) W mouth, goriya
0x05, keys 3, bombs 7, candle 2. East door is a diamond KEY lock.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=.:nes:snes uv run python \\
        nes/zelda_i/scratch/probe_l7_room1b_key_east.py --tag 1b_ke_v2
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name, room_item_name
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_KEYS,
    ADDR_LADDER,
    ADDR_WHISTLE,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist

ROOM = 0x1B
DOOR_Y = 141


def _s(env):
    return read_snapshot(env.get_ram())


def _step(env, a, btn, f):
    env.step(nes_action(btn) if btn else nes_idle_action())
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
        "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "colliding_tile": int(s.colliding_tile),
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "room_all_dead": int(s.room_all_dead),
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
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "food": int(read_u8(ram, ADDR_FOOD)),
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "ladder": int(read_u8(ram, ADDR_LADDER)),
        "triforce": int(s.triforce),
        "level": int(s.level),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1b_ke_v2")
    ap.add_argument("--from-state", default="Level7Interior1BReconFixture")
    ap.add_argument("--save-fixture", default="")
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
        out["start"] = _glance(env)
        print("START", out["start"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_start.png")
        f = 0
        hit = None
        for i in range(400):
            s = _s(env)
            if int(s.screen) != ROOM:
                hit = (int(s.screen), f)
                save_rgb_png(
                    env.render(), RECORDINGS_DIR / f"{args.tag}_trans.png"
                )
                print("TRANS", _glance(env))
                break
            y = int(s.link_y)
            btn = "RIGHT" if abs(y - DOOR_Y) <= 4 else (
                "UP" if y > DOOR_Y else "DOWN"
            )
            if i % 40 == 0:
                print("PUSH", {"f": f, "xy": [int(s.link_x), int(s.link_y)],
                               "tile": int(s.colliding_tile),
                               "keys": int(s.keys)})
            _step(env, a, btn, f)
            f += 1
        if hit is None:
            out["result"] = "blocked"
            print("BLOCKED", _glance(env))
        else:
            for _ in range(220):
                s = _s(env)
                if int(s.mode) == PLAY_MODE and not s.transitioning:
                    break
                _step(env, a, None, f)
                f += 1
            out["result"] = f"0x{int(_s(env).screen):02x}"
            out["arrived_frame"] = hit[1]
            out["dest"] = _glance(env)
            print("DEST", out["dest"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_dest.png")
            if args.save_fixture:
                path = save_state(env, GAME_DIR, GAME, args.save_fixture)
                src = state_path(GAME_DIR, GAME, args.from_state)
                write_state_provenance(
                    path,
                    source_state_path=src if src.exists() else None,
                    request={
                        "bead": "rr-8t4.3",
                        "phase": "level7_interior_1c_recon",
                        "track": "recon_fixture",
                        "route_eligible": False,
                        "fixture_only": True,
                        "natural_entry": False,
                        "development_only": True,
                        "fixture_writes": [],
                        "notes": [
                            f"Derived from {args.from_state}: 0x1B KEY-east "
                            f"-> {out['result']}. Natural key spend. No "
                            "Candle/TF/door writes.",
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
