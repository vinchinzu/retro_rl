"""Recon: from Level7Interior38ReconFixture, kill-clear 0x38 goriyas then
KEY-UP (source: HUNGRY_GORIYA).  Food stays 1 until the hungry room.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=nes:. uv run python \\
        nes/zelda_i/scratch/probe_l7_room38_up.py --tag 38_up_v1 [--save-fixture]
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

ROOM = 0x38
DOOR_X = 120
NORTH_PLANE_Y = 93


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
        "triforce": int(s.triforce),
    }


def _clear(env, a, f, budget=4000):
    saw = False
    for i in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM:
            return f, "left"
        live = live_goriyas(s)
        if live:
            saw = True
        elif saw and i > 60:
            return f, "clear"
        else:
            _step(env, a, "LEFT", f)
            f += 1
            continue
        tgt = nearest_enemy(s.link_x, s.link_y, live)
        hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
        _step(env, a, (hint.face, "A") if (i % 8) < 4 else hint.face, f)
        f += 1
    return f, "timeout"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="38_up_v1")
    ap.add_argument("--from-state", default="Level7Interior38ReconFixture")
    ap.add_argument("--save-fixture", default="")
    ap.add_argument("--no-clear", action="store_true")
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
        f = 0
        if not args.no_clear:
            f, status = _clear(env, a, f)
            out["clear_status"] = status
            out["after_clear"] = _glance(env)
            print("AFTER CLEAR", status, out["after_clear"])
            save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_after_clear.png")
        hit = None
        for i in range(900):
            s = _s(env)
            if i % 40 == 0:
                rec = {
                    "f": f,
                    "xy": [int(s.link_x), int(s.link_y)],
                    "tile": int(s.colliding_tile),
                    "doors": int(s.cur_opened_doors),
                    "keys": int(read_u8(env.get_ram(), ADDR_KEYS)),
                    "screen": f"0x{int(s.screen):02x}",
                }
                out["samples"].append(rec)
                print("SAMPLE", rec)
            if int(s.screen) != ROOM and int(s.mode) == PLAY_MODE and not s.transitioning:
                hit = (int(s.screen), f)
                save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_trans.png")
                break
            x, y = int(s.link_x), int(s.link_y)
            # v5 pinned UP at (200,149). The east mouth itself is (208,141) —
            # push all the way into the door pocket before rising.
            EAST_COL = 208
            if x < EAST_COL - 4 and y > NORTH_PLANE_Y + 12:
                btn = "RIGHT"
            elif y > NORTH_PLANE_Y + 4:
                btn = "UP"
            elif abs(x - DOOR_X) > 4:
                btn = "LEFT" if x > DOOR_X else "RIGHT"
            else:
                btn = "UP"
            _step(env, a, btn, f)
            f += 1
        if hit is None:
            e = _s(env)
            out["result"] = "blocked"
            out["end_xy"] = [int(e.link_x), int(e.link_y)]
            out["tile"] = int(e.colliding_tile)
            out["doors"] = int(e.cur_opened_doors)
            print("BLOCKED", out["end_xy"], "tile", out["tile"], "doors", out["doors"])
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
                            f"Derived from {args.from_state}: 0x38 goriya clear "
                            f"+ KEY-UP -> {out['result']}. No Candle/Food/TF/"
                            "door pokes. Key consume is natural.",
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
