"""Recon: from Level7Interior59ReconFixture (0x59 GORIYA_COMPASS, goriya
0x05/0x06, entry (16,141) W mouth), clear/dodge the goriyas then probe exits.

Source: 0x59 UP -> GORIYA_BUBBLE (mainline candle chain), RIGHT (KILL_CLEAR)
-> COMPASS (dead-end pickup), LEFT -> 0x58 (back).

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room59_onward.py --dir UP --tag 59_up_v1 [--save-fixture]
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

ROOM_59 = 0x59
OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
DEST_NAME = {"UP": "Level7Interior49ReconFixture"}


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
            _step(env, a, ("RIGHT", "A"), f); f += 1; continue
        tgt = nearest_enemy(s.link_x, s.link_y, live)
        hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
        _step(env, a, (hint.face, "A") if (i % 8) < 4 else hint.face, f); f += 1
    return f, "timeout"


def _to_band(env, a, tx, ty, f, budget=300):
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM_59:
            return f, False
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - tx) <= 4 and abs(y - ty) <= 4:
            return f, True
        if abs(y - ty) > 4 and abs(y - ty) >= abs(x - tx):
            _step(env, a, "UP" if y > ty else "DOWN", f)
        elif abs(x - tx) > 4:
            _step(env, a, "LEFT" if x > tx else "RIGHT", f)
        else:
            _step(env, a, "UP" if y > ty else "DOWN", f)
        f += 1
    return f, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="UP", choices=["UP", "RIGHT"])
    ap.add_argument("--tag", default="59_up_v1")
    ap.add_argument("--from-state", default="Level7Interior59ReconFixture")
    ap.add_argument("--save-fixture", action="store_true")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"dir": args.dir, "route_eligible": False}
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
        if status not in ("clear", "left"):
            (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
            return
        # to the door approach
        if args.dir == "UP":
            targets = [(120, 141), (120, 100)]
        else:
            targets = [(200, 141)]
        for (tx, ty) in targets:
            f, _ = _to_band(env, a, tx, ty, f)
        start = _s(env)
        hit = None
        for _ in range(240):
            s = _s(env)
            if int(s.screen) != ROOM_59 and int(s.mode) == PLAY_MODE and not s.transitioning:
                hit = (int(s.screen), [int(s.link_x), int(s.link_y)], f)
                break
            x = int(s.link_x)
            b = args.dir
            if args.dir == "UP" and abs(x - 120) > 4:
                b = "LEFT" if x > 120 else "RIGHT"
            _step(env, a, b, f); f += 1
        out["push_from"] = [int(start.link_x), int(start.link_y)]
        if hit is None:
            e = _s(env)
            out["result"] = "blocked"
            out["end_xy"] = [int(e.link_x), int(e.link_y)]
            out["tile"] = int(e.colliding_tile)
        else:
            out["result"] = f"0x{hit[0]:02x}"
            out["arrived_frame"] = hit[2]
            for _ in range(150):
                _step(env, a, None, f); f += 1
            out["dest"] = _glance(env)
            print("DEST", out["dest"])
            if args.save_fixture and args.dir in DEST_NAME:
                path = save_state(env, GAME_DIR, GAME, DEST_NAME[args.dir])
                src = state_path(GAME_DIR, GAME, args.from_state)
                write_state_provenance(
                    path, source_state_path=src if src.exists() else None,
                    request={"bead": "rr-8t4.2", "phase": "level7_interior_49_recon",
                             "track": "recon_fixture", "route_eligible": False,
                             "fixture_only": True, "natural_entry": False, "fixture_writes": [],
                             "notes": [f"Derived from {args.from_state}: 0x59 cleared goriya "
                                       f"+ UP -> {out['result']}. No set_state.",
                                       "UnlimitedHealthAssist traversal aid; no Candle/TF/door writes."]},
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
