"""Recon: 0x49 (GORIYA_BUBBLE) has a horizontal water moat across the middle
(~y120).  UP door (KILL_CLEAR, opens after goriya) is at x~120 top.  Find the
x where Link can walk NORTH across the moat from y~136 to the top band, then
push UP -> live $EB (source: DIGDOGGER_2).

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room49_cross.py --tag 49_cross_v1 [--save-fixture]
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

ROOM = 0x49
DEST_FIXTURE = "Level7Interior4BReconFixture"


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
        "room_all_dead": int(s.room_all_dead), "room_item_id": int(s.room_item_id),
        "room_item_name": room_item_name(int(s.room_item_id)),
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "keys": int(read_u8(ram, ADDR_KEYS)), "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)), "food": int(read_u8(ram, ADDR_FOOD)),
    }


def _clear(env, a, f, budget=3000):
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
            _step(env, a, "DOWN", f); f += 1; continue
        tgt = nearest_enemy(s.link_x, s.link_y, live)
        hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
        _step(env, a, (hint.face, "A") if (i % 8) < 4 else hint.face, f); f += 1
    return f, "timeout"


def _reach_y(env, a, ty, f, budget=200):
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM:
            return f, False
        if abs(int(s.link_y) - ty) <= 3:
            return f, True
        _step(env, a, "UP" if int(s.link_y) > ty else "DOWN", f); f += 1
    return f, False


def _go_x(env, a, tx, f, budget=160):
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM:
            return f, False
        if abs(int(s.link_x) - tx) <= 2:
            return f, True
        _step(env, a, "LEFT" if int(s.link_x) > tx else "RIGHT", f); f += 1
    return f, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="49_cross_v1")
    ap.add_argument("--from-state", default="Level7Interior49ReconFixture")
    ap.add_argument("--save-fixture", action="store_true")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"route_eligible": False, "probes": []}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        f = 0
        f, status = _clear(env, a, f)
        out["clear_status"] = status
        out["after_clear"] = _glance(env)
        print("AFTER CLEAR", status, out["after_clear"])
        crossed_x = None
        for tx in list(range(112, 129, 2)) + [110, 108, 130, 132, 100, 140, 88, 152, 40, 200]:
            f, _ = _reach_y(env, a, 138, f)
            f, _ = _go_x(env, a, tx, f)
            s0 = _s(env)
            top_y = None
            for _ in range(60):
                s = _s(env)
                if int(s.screen) != ROOM:
                    break
                y = int(s.link_y)
                if y <= 112:
                    top_y = y
                    break
                _step(env, a, "UP", f); f += 1
            rec = {"x": tx, "from": [int(s0.link_x), int(s0.link_y)],
                   "reached_y": top_y, "now": [int(_s(env).link_x), int(_s(env).link_y)]}
            out["probes"].append(rec)
            print("CROSS", rec)
            if int(_s(env).screen) != ROOM:
                out["left_during_cross"] = _glance(env)
                break
            if top_y is not None:
                crossed_x = tx
                break
        out["crossed_x"] = crossed_x
        hit = None
        if crossed_x is not None:
            f, _ = _reach_y(env, a, 90, f)
            f, _ = _go_x(env, a, 120, f)
            out["pre_push"] = [int(_s(env).link_x), int(_s(env).link_y)]
            print("PRE PUSH", out["pre_push"])
            for _ in range(240):
                s = _s(env)
                if int(s.screen) != ROOM and int(s.mode) == PLAY_MODE and not s.transitioning:
                    hit = (int(s.screen), f)
                    break
                x = int(s.link_x)
                b = "UP" if abs(x - 120) <= 4 else ("LEFT" if x > 120 else "RIGHT")
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
                path = save_state(env, GAME_DIR, GAME, DEST_FIXTURE)
                src = state_path(GAME_DIR, GAME, args.from_state)
                write_state_provenance(
                    path, source_state_path=src if src.exists() else None,
                    request={"bead": "rr-8t4.2",
                             "phase": f"level7_interior_{out['result'][2:]}_recon",
                             "track": "recon_fixture", "route_eligible": False,
                             "fixture_only": True, "natural_entry": False,
                             "development_only": True, "fixture_writes": [],
                             "notes": [f"Derived from {args.from_state}: 0x49 goriya clear "
                                       f"+ moat crossing at x={crossed_x} + UP door -> "
                                       f"{out['result']}. No set_state, no "
                                       "Candle/Food/TF/door/key/bomb writes.",
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
    finally:
        env.close()


if __name__ == "__main__":
    main()
