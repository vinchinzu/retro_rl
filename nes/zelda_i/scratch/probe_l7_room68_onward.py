"""Recon: fixture -> 0x6B LEFT -> 0x6A LEFT -> 0x69 -> west BOMB -> 0x68
(KEESE_TRAPS), settle + glance, then scan 0x68's exits.

Source: 0x68 UP -> DODONGOS_UPGRADE (0x58), DOWN -> ROPES_KEY, RIGHT -> 0x69.
Optionally --save-fixture writes Level7Interior68ReconFixture (+ provenance,
disclosed, route_eligible=false) for downstream recon.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room68_onward.py --tag 68_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs, save_state, state_path
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name, room_item_name
from zelda_i.dungeon.ops import ensure_bomb, poke_bombs
from zelda_i.dungeon.trace import compact_snapshot, write_state_provenance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS, ADDR_CANDLE, ADDR_FOOD, ADDR_KEYS, ADDR_TRIFORCE,
    ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8,
)
from zelda_i.runner import make_assist

ROOM_69, ROOM_6A, ROOM_6B, ROOM_68 = 0x69, 0x6A, 0x6B, 0x68
OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def _s(env):
    return read_snapshot(env.get_ram())


def _step(env, a, btn, f):
    env.step(nes_action(btn) if btn else nes_idle_action())
    if a:
        a.apply_env(env, frame=f)


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    types = sorted({int(o.type_id) for o in s.objects
                    if 1 <= o.slot <= 12 and int(o.type_id) not in (0, 0xFF)})
    return {
        "screen": f"0x{int(s.screen):02x}", "screen_int": int(s.screen),
        "mode": int(s.mode), "xy": [int(s.link_x), int(s.link_y)],
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "room_item_id": int(s.room_item_id),
        "room_item_name": room_item_name(int(s.room_item_id)),
        "room_all_dead": int(s.room_all_dead),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "food": int(read_u8(ram, ADDR_FOOD)), "candle": int(read_u8(ram, ADDR_CANDLE)),
        "keys": int(read_u8(ram, ADDR_KEYS)), "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "triforce": int(s.triforce), "whistle": int(read_u8(ram, ADDR_WHISTLE)),
    }


def _west_traverse(env, a, room, next_room, f, budget=2400):
    """W exit mirror of Room6AEastController."""
    TOP_Y, DOOR_Y = 93, 141
    EAST_COL, WEST_COL = 200, 36
    phase = "off_mouth"
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) == next_room and int(s.mode) == PLAY_MODE and not s.transitioning:
            return f, True
        if int(s.screen) not in (room, next_room):
            return f, False
        if s.transitioning:
            _step(env, a, "LEFT", f); f += 1; continue
        x, y = int(s.link_x), int(s.link_y)
        if phase == "off_mouth":
            btn = "LEFT" if x > EAST_COL + 8 else "UP"
            if x <= EAST_COL + 8:
                phase = "rise"
        elif phase == "rise":
            btn = "UP" if y > TOP_Y + 4 else "LEFT"
            if y <= TOP_Y + 4:
                phase = "cross"
        elif phase == "cross":
            if x > WEST_COL + 4:
                btn = "LEFT" if abs(y - TOP_Y) <= 6 else ("UP" if y > TOP_Y else "DOWN")
            else:
                phase = "drop"; btn = "DOWN"
        elif phase == "drop":
            btn = ("DOWN" if y < DOOR_Y else "UP") if abs(y - DOOR_Y) > 4 else "LEFT"
            if abs(y - DOOR_Y) <= 4:
                phase = "push"
        else:
            btn = "LEFT" if abs(y - DOOR_Y) <= 6 else ("UP" if y > DOOR_Y else "DOWN")
        _step(env, a, btn, f); f += 1
    return f, False


def _reach(env, a, room, tx, ty, f, budget=240):
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != room:
            return f, False
        x, y = int(s.link_x), int(s.link_y)
        if (tx is None or abs(x - tx) <= 3) and (ty is None or abs(y - ty) <= 3):
            return f, True
        if ty is not None and abs(y - ty) > 3 and (tx is None or abs(x - tx) <= 10):
            _step(env, a, "UP" if y > ty else "DOWN", f)
        elif tx is not None and abs(x - tx) > 3:
            _step(env, a, "LEFT" if x > tx else "RIGHT", f)
        elif ty is not None and abs(y - ty) > 3:
            _step(env, a, "UP" if y > ty else "DOWN", f)
        else:
            return f, True
        f += 1
    return f, False


def _bomb_west(env, a, f):
    f, _ = _reach(env, a, ROOM_69, 44, 141, f)
    for _ in range(4):
        ensure_bomb(env); _step(env, a, "LEFT", f); f += 1
    ensure_bomb(env)
    env.step(nes_action("LEFT", "B"))
    if a:
        a.apply_env(env, frame=f)
    f += 1
    for _ in range(7):
        _step(env, a, "RIGHT", f); f += 1
    for _ in range(110):
        _step(env, a, None, f); f += 1
    for _ in range(180):
        s = _s(env)
        if int(s.screen) == ROOM_68 and int(s.mode) == PLAY_MODE and not s.transitioning:
            return f, True
        _step(env, a, "LEFT", f); f += 1
    return f, int(_s(env).screen) == ROOM_68


def _push(env, a, room, btn, f, budget=170):
    start = _s(env)
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != room and int(s.mode) == PLAY_MODE and not s.transitioning:
            return f, {"dir": btn, "from": [int(start.link_x), int(start.link_y)],
                       "result": f"0x{int(s.screen):02x}", "at": [int(s.link_x), int(s.link_y)]}
        _step(env, a, btn, f); f += 1
    e = _s(env)
    return f, {"dir": btn, "from": [int(start.link_x), int(start.link_y)],
               "result": "blocked", "to": [int(e.link_x), int(e.link_y)], "tile": int(e.colliding_tile)}


def _back(env, a, room, btn, f):
    for _ in range(200):
        s = _s(env)
        if int(s.screen) == room and int(s.mode) == PLAY_MODE and not s.transitioning:
            return f
        _step(env, a, OPP[btn], f); f += 1
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="68_v1")
    ap.add_argument("--from-state", default="Level7InteriorReconFixture")
    ap.add_argument("--save-fixture", action="store_true")
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
        f = 0
        f, ok = _west_traverse(env, a, ROOM_6B, ROOM_6A, f)
        if ok:
            for _ in range(30):
                _step(env, a, None, f); f += 1
            f, ok = _west_traverse(env, a, ROOM_6A, ROOM_69, f)
        out["reached_69"] = ok
        if ok:
            for _ in range(60):
                _step(env, a, None, f); f += 1
            poke_bombs(env, 8)
            f, ok = _bomb_west(env, a, f)
        out["reached_68"] = ok
        if not ok:
            out["end"] = _glance(env)
            (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
            print("FAIL 68", out["end"]); return
        for _ in range(150):
            _step(env, a, None, f); f += 1
        out["at_68"] = _glance(env)
        print("0x68", out["at_68"])

        if args.save_fixture:
            path = save_state(env, GAME_DIR, GAME, "Level7Interior68ReconFixture")
            src = state_path(GAME_DIR, GAME, args.from_state)
            g = _glance(env)
            write_state_provenance(
                path,
                source_state_path=src if src.exists() else None,
                request={
                    "bead": "rr-8t4.2",
                    "phase": "level7_interior_68_recon",
                    "track": "recon_fixture",
                    "route_eligible": False, "fixture_only": True, "natural_entry": False,
                    "fixture_writes": [
                        {"name": "bombs_topup_for_recon", "address": ADDR_BOMBS,
                         "address_hex": "0x0658", "note": "poke_bombs(8) before the "
                         "0x69 west bomb; count only, no max_bombs write"},
                    ],
                    "notes": [
                        "Derived from Level7InteriorReconFixture by WALKING "
                        "0x6B LEFT -> 0x6A LEFT -> 0x69 then bombing the 0x69 "
                        "west wall -> 0x68 (KEESE_TRAPS). No set_state/teleport.",
                        "Traverse under UnlimitedHealthAssist (traversal aid). "
                        "Food 1 / keys 4 / bombs 8 carried from the parent fixture.",
                        "No Candle/Whistle/TF/door/room/health/heart-capacity writes.",
                    ],
                },
                selected_trial={"ok": True, "state": compact_snapshot(_s(env)),
                                "glance": g},
                natural_entry=False,
            )
            out["saved_fixture"] = str(path)
            print("saved", path)

        scans = []
        for btn, wxy in [("UP", (120, None)), ("DOWN", (120, None)),
                         ("LEFT", (None, 141)), ("RIGHT", (None, 141))]:
            f, _ = _reach(env, a, ROOM_68, wxy[0], wxy[1], f)
            f, rec = _push(env, a, ROOM_68, btn, f)
            scans.append(rec); print("push", rec)
            if rec["result"].startswith("0x"):
                for _ in range(140):
                    _step(env, a, None, f); f += 1
                out[f"dest_{btn}"] = _glance(env)
                print("  DEST", out[f"dest_{btn}"])
                f = _back(env, a, ROOM_68, btn, f)
        out["scans"] = scans
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
