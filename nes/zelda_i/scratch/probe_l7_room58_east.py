"""Recon: from Level7Interior58ReconFixture, reach the 0x58 EAST edge and open
the door (the 3x 0x31 are invulnerable roamers -- not clearable; the sweep
saw keys 4->3 + cur_opened_doors RIGHT, so EAST is a LOCKED KEY door).

Navigate to the east edge on a mid band, push RIGHT persistently; if a key
is consumed and the screen changes, that is the door.  If not, bomb the
east wall as a fallback.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room58_east.py --tag 58_east_v1 [--save-fixture]
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


def _wp(env, a, waypoints, f, budget_each=300):
    """Walk a waypoint list (x,y); y-first when far, then x.  0x31 knockback
    is soaked by assist -- just persist."""
    for (wx, wy) in waypoints:
        for _ in range(budget_each):
            s = _s(env)
            if int(s.screen) != ROOM_58:
                return f, False
            x, y = int(s.link_x), int(s.link_y)
            if abs(x - wx) <= 5 and abs(y - wy) <= 5:
                break
            if abs(y - wy) > 5 and abs(y - wy) >= abs(x - wx):
                _step(env, a, "UP" if y > wy else "DOWN", f)
            elif abs(x - wx) > 5:
                _step(env, a, "LEFT" if x > wx else "RIGHT", f)
            else:
                _step(env, a, "UP" if y > wy else "DOWN", f)
            f += 1
    return f, True


def _reach(env, a, tx, ty, f, budget=400):
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM_58:
            return f, False
        x, y = int(s.link_x), int(s.link_y)
        if (tx is None or abs(x - tx) <= 4) and (ty is None or abs(y - ty) <= 4):
            return f, True
        if ty is not None and abs(y - ty) > 4 and (tx is None or abs(y - ty) >= abs(x - tx)):
            _step(env, a, "UP" if y > ty else "DOWN", f)
        elif tx is not None and abs(x - tx) > 4:
            _step(env, a, "LEFT" if x > tx else "RIGHT", f)
        else:
            _step(env, a, "UP" if y > ty else "DOWN", f)
        f += 1
    return f, False


def _push_east(env, a, f, budget=260):
    """Persistently push RIGHT at the east edge; keep y near band."""
    start = _s(env)
    keys0 = int(read_u8(env.get_ram(), ADDR_KEYS))
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM_58 and int(s.mode) == PLAY_MODE and not s.transitioning:
            return f, {"result": f"0x{int(s.screen):02x}",
                       "at": [int(s.link_x), int(s.link_y)],
                       "keys_before": keys0,
                       "keys_after": int(read_u8(env.get_ram(), ADDR_KEYS))}
        y = int(s.link_y)
        by = int(start.link_y)
        if abs(y - by) > 8:
            _step(env, a, "UP" if y > by else "DOWN", f)
        else:
            _step(env, a, "RIGHT", f)
        f += 1
    e = _s(env)
    return f, {"result": "blocked", "to": [int(e.link_x), int(e.link_y)],
               "tile": int(e.colliding_tile), "keys_before": keys0,
               "keys_after": int(read_u8(env.get_ram(), ADDR_KEYS)),
               "cur_opened_doors": int(e.cur_opened_doors)}


def _bomb_east(env, a, sy, f):
    f, _ = _reach(env, a, 196, sy, f)
    for _ in range(4):
        ensure_bomb(env); _step(env, a, "RIGHT", f); f += 1
    ensure_bomb(env)
    env.step(nes_action("RIGHT", "B"))
    if a:
        a.apply_env(env, frame=f)
    f += 1
    for _ in range(7):
        _step(env, a, "LEFT", f); f += 1
    for _ in range(110):
        _step(env, a, None, f); f += 1
    for _ in range(150):
        s = _s(env)
        if int(s.screen) != ROOM_58 and int(s.mode) == PLAY_MODE and not s.transitioning:
            return f, f"0x{int(s.screen):02x}"
        _step(env, a, "RIGHT", f); f += 1
    return f, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="58_east_v1")
    ap.add_argument("--from-state", default="Level7Interior58ReconFixture")
    ap.add_argument("--save-fixture", action="store_true")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"route_eligible": False, "tries": []}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        f = 0
        dest = None
        for by in (141, 125, 157, 109):
            # multi-waypoint: up the east-open column, then to the band's east edge
            f, _ = _wp(env, a, [(120, 165), (200, 165), (200, by)], f)
            f, rec = _push_east(env, a, f)
            rec["band"] = by
            out["tries"].append(rec)
            print("push", rec)
            if rec["result"].startswith("0x"):
                dest = rec["result"]
                for _ in range(150):
                    _step(env, a, None, f); f += 1
                out["east_dest"] = _glance(env)
                print("EAST DEST", out["east_dest"])
                break
            # back to center
            for _ in range(80):
                _step(env, a, "LEFT", f); f += 1
        if dest is None:
            poke_bombs(env, 8)
            for by in (141, 125):
                if int(_s(env).screen) != ROOM_58:
                    break
                f, d = _bomb_east(env, a, by, f)
                out["tries"].append({"bomb_band": by, "result": d})
                print("bomb", by, d)
                if d:
                    dest = d
                    for _ in range(150):
                        _step(env, a, None, f); f += 1
                    out["east_dest"] = _glance(env)
                    print("EAST DEST", out["east_dest"])
                    break
                for _ in range(80):
                    _step(env, a, "LEFT", f); f += 1

        if dest is not None and args.save_fixture:
            path = save_state(env, GAME_DIR, GAME, "Level7Interior59ReconFixture")
            src = state_path(GAME_DIR, GAME, args.from_state)
            write_state_provenance(
                path, source_state_path=src if src.exists() else None,
                request={"bead": "rr-8t4.2", "phase": "level7_interior_59_recon",
                         "track": "recon_fixture", "route_eligible": False,
                         "fixture_only": True, "natural_entry": False,
                         "fixture_writes": [{"name": "bombs_topup", "address": ADDR_BOMBS,
                                             "address_hex": "0x0658", "note": "count only"}],
                         "notes": [f"Derived from {args.from_state} by opening the 0x58 "
                                   f"east door -> {dest}. No set_state.",
                                   "UnlimitedHealthAssist traversal aid; no Candle/TF/door writes."]},
                selected_trial={"ok": True, "state": compact_snapshot(_s(env)),
                                "glance": _glance(env)},
                natural_entry=False)
            out["saved_fixture"] = str(path)
            print("saved", path)

        end = _glance(env)
        end["deaths"] = int(a.telemetry.deaths)
        end["progression_writes"] = int(a.telemetry.progression_writes)
        end["capacity_writes"] = int(a.telemetry.capacity_writes)
        out["end"] = end
        out["dest"] = dest
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("END", end, "dest", dest)
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
