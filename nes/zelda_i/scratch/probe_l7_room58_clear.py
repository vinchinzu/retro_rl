"""Recon: from Level7Interior58ReconFixture, identify + kill the 0x31 enemy,
collect room_item_id 0x0f, then open + walk the EAST door (mainline ->
GORIYA_COMPASS).

0x31 is not the catalogued DODONGO_TYPE (0x32) -- observe it, then try
bomb-in-mouth (facing-based) + sword.  Track hp / room_all_dead.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room58_clear.py --tag 58_clear_v1
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
    ADDR_BOMBS, ADDR_CANDLE, ADDR_FOOD, ADDR_KEYS, PLAY_MODE,
    read_snapshot, read_u8,
)
from zelda_i.runner import make_assist

ROOM_58 = 0x58
ENEMY = 0x31
FACE_E, FACE_W, FACE_S, FACE_N = 1, 2, 4, 8


def _s(env):
    return read_snapshot(env.get_ram())


def _step(env, a, btn, f):
    env.step(nes_action(btn) if isinstance(btn, str) else (nes_action(*btn) if btn else nes_idle_action()))
    if a:
        a.apply_env(env, frame=f)


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    objs = [{"slot": int(o.slot), "t": f"0x{int(o.type_id):02x}", "x": int(o.x),
             "y": int(o.y), "hp": int(o.hp), "facing": int(o.facing)}
            for o in s.objects if 1 <= int(o.slot) <= 12 and int(o.type_id) not in (0, 0xFF)]
    return {
        "screen": f"0x{int(s.screen):02x}", "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "room_all_dead": int(s.room_all_dead),
        "room_item_id": int(s.room_item_id),
        "room_item_name": room_item_name(int(s.room_item_id)),
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "objs": objs,
        "keys": int(read_u8(ram, ADDR_KEYS)), "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "candle": int(read_u8(ram, ADDR_CANDLE)), "food": int(read_u8(ram, ADDR_FOOD)),
    }


def _enemies(s):
    return [o for o in s.objects if int(o.type_id) == ENEMY and 1 <= int(o.slot) <= 12 and int(o.hp) > 0]


def _mouth(o, off=14):
    f = int(o.facing)
    if f & FACE_E:
        return int(o.x) + off, int(o.y), "LEFT"
    if f & FACE_W:
        return int(o.x) - off, int(o.y), "RIGHT"
    if f & FACE_S:
        return int(o.x), int(o.y) + off, "UP"
    if f & FACE_N:
        return int(o.x), int(o.y) - off, "DOWN"
    return int(o.x), int(o.y) - off, "DOWN"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="58_clear_v1")
    ap.add_argument("--from-state", default="Level7Interior58ReconFixture")
    ap.add_argument("--save-fixture", action="store_true")
    ap.add_argument("--fight-frames", type=int, default=5000)
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
        f = 0
        obs = []
        place_cd = 0
        place_face = "UP"
        cleared_frame = None
        for i in range(args.fight_frames):
            s = _s(env)
            if int(s.screen) != ROOM_58:
                break
            en = _enemies(s)
            if i % 100 == 0:
                obs.append({"f": i, "dead": int(s.room_all_dead),
                            "n": len(en), "item": int(s.room_item_id),
                            "hp": [int(o.hp) for o in en],
                            "face": [int(o.facing) for o in en]})
            if not en and int(s.room_all_dead) >= 20:
                cleared_frame = i
                break
            if not en:
                _step(env, a, ("RIGHT", "A"), f); f += 1; continue
            if place_cd > 0:
                place_cd -= 1
                back = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}[place_face]
                _step(env, a, back if place_cd > 40 else None, f); f += 1
                continue
            o = min(en, key=lambda o: abs(int(o.x) - int(s.link_x)) + abs(int(o.y) - int(s.link_y)))
            tx, ty, face = _mouth(o)
            lx, ly = int(s.link_x), int(s.link_y)
            if abs(lx - tx) > 6:
                _step(env, a, "LEFT" if lx > tx else "RIGHT", f); f += 1; continue
            if abs(ly - ty) > 6:
                _step(env, a, "UP" if ly > ty else "DOWN", f); f += 1; continue
            # in position: drop a bomb, also swing sword every other cycle
            ensure_bomb(env)
            env.step(nes_action(face, "B"))
            if a:
                a.apply_env(env, frame=f)
            f += 1
            place_cd = 90
            place_face = face
        out["cleared_frame"] = cleared_frame
        out["obs"] = obs
        out["after_fight"] = _glance(env)
        print("AFTER FIGHT", out["after_fight"])

        if cleared_frame is not None:
            # collect the room item: sweep the room
            got = None
            for _ in range(900):
                s = _s(env)
                if int(s.room_item_id) in (0, 3):
                    got = int(s.room_item_id)
                    break
                d = ("UP", "RIGHT", "DOWN", "LEFT", "RIGHT", "UP")[(f // 30) % 6]
                _step(env, a, d, f); f += 1
            out["item_collected_marker"] = got
            out["after_item"] = _glance(env)
            print("AFTER ITEM", out["after_item"])

            if args.save_fixture:
                path = save_state(env, GAME_DIR, GAME, "Level7Interior58ClearReconFixture")
                src = state_path(GAME_DIR, GAME, args.from_state)
                write_state_provenance(
                    path, source_state_path=src if src.exists() else None,
                    request={"bead": "rr-8t4.2", "phase": "level7_interior_58_cleared",
                             "track": "recon_fixture", "route_eligible": False,
                             "fixture_only": True, "natural_entry": False,
                             "fixture_writes": [{"name": "bombs_topup", "address": ADDR_BOMBS,
                                                 "address_hex": "0x0658", "note": "count only"}],
                             "notes": ["Derived from Level7Interior58ReconFixture by killing "
                                       "the 0x31 enemy + collecting room_item 0x0f. No set_state.",
                                       "UnlimitedHealthAssist traversal aid; no Candle/TF/door writes."]},
                    selected_trial={"ok": True, "state": compact_snapshot(_s(env)),
                                    "glance": _glance(env)},
                    natural_entry=False)
                out["saved_fixture"] = str(path)
                print("saved", path)

            # probe EAST on y=125 and y=141
            east = []
            for by in (125, 141, 109):
                for _ in range(200):
                    s = _s(env)
                    if int(s.screen) != ROOM_58:
                        break
                    x, y = int(s.link_x), int(s.link_y)
                    if abs(y - by) > 3:
                        _step(env, a, "UP" if y > by else "DOWN", f)
                    elif x < 200:
                        _step(env, a, "RIGHT", f)
                    else:
                        break
                    f += 1
                start = _s(env)
                hit = None
                for _ in range(150):
                    s = _s(env)
                    if int(s.screen) != ROOM_58 and int(s.mode) == PLAY_MODE and not s.transitioning:
                        hit = (int(s.screen), [int(s.link_x), int(s.link_y)])
                        break
                    _step(env, a, "RIGHT", f); f += 1
                e = _s(env)
                rec = {"band": by, "from": [int(start.link_x), int(start.link_y)],
                       "result": f"0x{hit[0]:02x}" if hit else "blocked",
                       "to": [int(e.link_x), int(e.link_y)], "tile": int(e.colliding_tile)}
                east.append(rec); print("EAST", rec)
                if hit:
                    for _ in range(150):
                        _step(env, a, None, f); f += 1
                    out["east_dest"] = _glance(env)
                    print("EAST DEST", out["east_dest"])
                    break
                for _ in range(90):
                    _step(env, a, "LEFT", f); f += 1
            out["east_probe"] = east
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
