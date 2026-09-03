"""Recon: Level7Interior58ReconFixture -> 0x58 KEY north -> 0x48 BOMB_UPGRADE.

0x58 has 3x invuln 0x31 (hp 240) — dodge, do not kill.  A central 2-block
mass walls the x=120 column around y=141; east-around then the x=120
channel.  KEY door spends one key.  Dest is a 100-rupee old-man
bomb-capacity dead-end; do not write max_bombs.  2/2 (58_north_v2/v3).

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room58_north.py --tag 58_north_v2
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
    ADDR_MAX_BOMBS,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist

ROOM_58 = 0x58
NORTH_X = 120
# Central 2-block mass walls x=120 around y=141.  East-around (same first
# column as Room58EastController) then back to the x=120 north channel.
EAST_COLUMN_X = 160
MID_Y = 165
TOP_Y = 93


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
        "screen_int": int(s.screen),
        "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)],
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "room_item_id": int(s.room_item_id),
        "room_item_name": room_item_name(int(s.room_item_id)),
        "room_all_dead": int(s.room_all_dead),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "food": int(read_u8(ram, ADDR_FOOD)),
        "candle": int(read_u8(ram, ADDR_CANDLE)),
        "keys": int(read_u8(ram, ADDR_KEYS)),
        "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "max_bombs": int(read_u8(ram, ADDR_MAX_BOMBS)),
        "colliding_tile": int(s.colliding_tile),
    }


def _policy(x: int, y: int) -> str:
    """East-around the 0x58 central blocks, then the x=120 KEY north door.

    v1 OccupancyWalker to (120,93) boxed at (122,165) (25 misses) — the
    central mass sits on the x=120 column.  Dodge 0x31; do not fight.
    """
    if y <= TOP_Y + 4:
        if abs(x - NORTH_X) > 4:
            return "LEFT" if x > NORTH_X else "RIGHT"
        return "UP"
    if y > MID_Y + 4:
        return "UP"
    if x < EAST_COLUMN_X - 4:
        return "RIGHT"
    return "UP"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="58_north_v1")
    ap.add_argument("--from-state", default="Level7Interior58ReconFixture")
    ap.add_argument("--save-fixture", action="store_true")
    ap.add_argument("--budget", type=int, default=2400)
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"route_eligible": False, "samples": [], "shots": []}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        print("START", out["start"])
        f = 0
        last_screen = int(_s(env).screen)
        stuck = 0
        last_xy = None
        arrived = None
        keys_seen = [out["start"]["keys"]]
        for _ in range(args.budget):
            s = _s(env)
            screen = int(s.screen)
            mode = int(s.mode)
            x, y = int(s.link_x), int(s.link_y)
            keys = int(read_u8(env.get_ram(), ADDR_KEYS))
            if keys != keys_seen[-1]:
                keys_seen.append(keys)
            if screen != last_screen:
                shot = RECORDINGS_DIR / f"{args.tag}_f{f}_0x{screen:02x}.png"
                save_rgb_png(env.render(), shot)
                out["shots"].append(str(shot.name))
                last_screen = screen
            if (
                screen != ROOM_58
                and mode == PLAY_MODE
                and not s.transitioning
            ):
                arrived = (screen, [x, y], f)
                break
            if last_xy == (x, y):
                stuck += 1
            else:
                stuck = 0
            last_xy = (x, y)
            if stuck and stuck % 250 == 0:
                shot = RECORDINGS_DIR / f"{args.tag}_stuck_f{f}_{x}_{y}.png"
                save_rgb_png(env.render(), shot)
                out["shots"].append(str(shot.name))
            if f % 20 == 0:
                out["samples"].append(
                    [f, x, y, f"0x{screen:02x}", stuck, keys]
                )
            if s.transitioning or mode != PLAY_MODE:
                _step(env, a, "UP", f)
            else:
                btn = _policy(x, y)
                _step(env, a, btn or None, f)
            f += 1
        out["keys_trace"] = keys_seen
        if arrived is None:
            e = _s(env)
            out["result"] = "blocked"
            out["end_xy"] = [int(e.link_x), int(e.link_y)]
            out["tile"] = int(e.colliding_tile)
            print("BLOCKED", out["end_xy"], "tile", out["tile"])
        else:
            out["result"] = f"0x{arrived[0]:02x}"
            out["arrived_frame"] = arrived[2]
            out["arrived_xy"] = arrived[1]
            for _ in range(160):
                _step(env, a, None, f)
                f += 1
            out["dest"] = _glance(env)
            print("DEST", out["dest"])
            dest_screen = int(out["dest"]["screen_int"])
            if args.save_fixture and dest_screen != ROOM_58:
                name = f"Level7Interior{dest_screen:02X}ReconFixture"
                path = save_state(env, GAME_DIR, GAME, name)
                src = state_path(GAME_DIR, GAME, args.from_state)
                write_state_provenance(
                    path,
                    source_state_path=src if src.exists() else None,
                    request={
                        "bead": "rr-8t4.2",
                        "phase": "level7_interior_bomb_upgrade_recon",
                        "track": "recon_fixture",
                        "route_eligible": False,
                        "fixture_only": True,
                        "natural_entry": False,
                        "development_only": True,
                        "fixture_writes": [],
                        "notes": [
                            f"Derived from {args.from_state} by WALKING "
                            f"0x58 UP x=120 KEY channel -> 0x{dest_screen:02x} "
                            "(BOMB_UPGRADE). No set_state/teleport.",
                            "UnlimitedHealthAssist traversal aid only. "
                            "Key spend is natural (door KEY). "
                            "No Candle/Whistle/TF/door poke/max_bombs writes.",
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
