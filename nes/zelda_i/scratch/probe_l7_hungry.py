"""Feed the Hungry Goriya in 0x28. Equip already-owned Bait (B-slot 6), walk
UP to the NPC, press B. Food 1->0 must be NATURAL. Do not poke ADDR_FOOD.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=nes:. uv run python \\
        nes/zelda_i/scratch/probe_l7_hungry.py --tag 28_feed_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name, room_item_name
from zelda_i.dungeon.ops import mem_write
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

ROOM = 0x28
FOOD_B_SLOT = 6  # after whistle=5; already-owned item select
DOOR_X = 120


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
        "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "room_all_dead": int(s.room_all_dead),
        "room_item_id": int(s.room_item_id),
        "room_item_name": room_item_name(int(s.room_item_id)),
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "colliding_tile": int(s.colliding_tile),
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="28_feed_v1")
    ap.add_argument("--from-state", default="Level7Interior28ReconFixture")
    ap.add_argument("--save-fixture", default="")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"route_eligible": False, "samples": []}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        print("START", out["start"])
        food0 = int(read_u8(env.get_ram(), ADDR_FOOD))
        sel0 = int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM))
        if food0 < 1:
            raise SystemExit("Food already 0 — cannot feed")
        # Already-owned B-slot select (ASSIST_CONTRACT). Not a Food poke.
        mem_write(env, ADDR_SELECTED_ITEM, FOOD_B_SLOT)
        out["equip"] = {
            "selected_from": sel0,
            "selected_to": int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)),
            "food_unchanged": int(read_u8(env.get_ram(), ADDR_FOOD)) == food0,
        }
        print("EQUIP", out["equip"])
        f = 0
        fed = False
        hit = None
        for i in range(1200):
            s = _s(env)
            food = int(read_u8(env.get_ram(), ADDR_FOOD))
            if i % 40 == 0:
                rec = {
                    "f": f,
                    "xy": [int(s.link_x), int(s.link_y)],
                    "food": food,
                    "doors": int(s.cur_opened_doors),
                    "screen": f"0x{int(s.screen):02x}",
                }
                out["samples"].append(rec)
                print("SAMPLE", rec)
            if food == 0 and food0 == 1 and not fed:
                fed = True
                out["fed_frame"] = f
                out["after_feed"] = _glance(env)
                print("FED", out["after_feed"])
                save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_fed.png")
            if int(s.screen) != ROOM and int(s.mode) == PLAY_MODE and not s.transitioning:
                hit = (int(s.screen), f)
                save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_trans.png")
                break
            x, y = int(s.link_x), int(s.link_y)
            if not fed:
                # Approach the goriya (center-north of the black pad) and tap B.
                if abs(x - DOOR_X) > 4:
                    btn = "LEFT" if x > DOOR_X else "RIGHT"
                    _step(env, a, btn, f)
                elif y > 141:
                    _step(env, a, "UP", f)
                elif i % 16 < 6:
                    _step(env, a, "B", f)
                else:
                    _step(env, a, "UP", f)
            else:
                if abs(x - DOOR_X) > 4:
                    btn = "LEFT" if x > DOOR_X else "RIGHT"
                    _step(env, a, btn, f)
                else:
                    _step(env, a, "UP", f)
            f += 1
        if hit is None:
            e = _s(env)
            out["result"] = "blocked" if not fed else "fed_no_exit"
            out["end_xy"] = [int(e.link_x), int(e.link_y)]
            print("END STATE", out["result"], _glance(env))
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
                        "fixture_writes": [
                            {
                                "field": "selected_item",
                                "address": ADDR_SELECTED_ITEM,
                                "from": sel0,
                                "to": FOOD_B_SLOT,
                            }
                        ],
                        "notes": [
                            f"Derived from {args.from_state}: Hungry Goriya "
                            f"natural Food 1->0 (Bait on B), then UP -> "
                            f"{out['result']}. ADDR_FOOD was not poked. "
                            "selected_item poke of already-owned Bait only.",
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
