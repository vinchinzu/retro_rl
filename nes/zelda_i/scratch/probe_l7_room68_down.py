"""Recon: Level7Interior68ReconFixture -> 0x68 DOWN -> ROPES_KEY.

0x68 is dark, 4 blade traps 0x49 (corners) + 4 keese 0x1b.  Naive hold-DOWN
from (120,93) got shoved west on the y=189 trap row (68_down_v1).
OccupancyWalker poisoned the grid (v2 stood (174,149)).  Waypoint: peel
west to x=160, drop y=141, align x=120, push DOWN.  If knocked onto the
y~189 trap row off-x, rise first.  2/2 (68_down_v3/v4).

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room68_down.py --tag 68_down_v3
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

ROOM_68 = 0x68
SOUTH_X = 120
SAFE_X = 160  # off the east trap column (x=208) and west of the south door
MID_Y = 141  # between trap rows y~93 and y~189
SOUTH_DOOR_Y = 205
TRAP_ROW_Y = 189


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
    """Waypoint micro that stays off trap columns until the south door.

    v2 OccupancyWalker poisoned the grid (12 misses) and stood at (174,149).
    Traps sit in the four corners (x~32/208, y~93/189).  Route: peel west
    to x=160 on the entry band, drop to y=141 (between trap rows), align
    x=120, push DOWN.  If knocked onto the y~189 trap row off-x, rise first.
    """
    if abs(x - SOUTH_X) > 4 and abs(y - TRAP_ROW_Y) <= 8:
        return "UP"
    if x > SAFE_X + 4:
        return "LEFT"
    if y < MID_Y - 4:
        return "DOWN"
    if abs(x - SOUTH_X) > 4:
        return "LEFT" if x > SOUTH_X else "RIGHT"
    return "DOWN"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="68_down_v2")
    ap.add_argument("--from-state", default="Level7Interior68ReconFixture")
    ap.add_argument("--save-fixture", action="store_true")
    ap.add_argument("--budget", type=int, default=1800)
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
        for _ in range(args.budget):
            s = _s(env)
            screen = int(s.screen)
            mode = int(s.mode)
            x, y = int(s.link_x), int(s.link_y)
            if screen != last_screen:
                shot = RECORDINGS_DIR / f"{args.tag}_f{f}_0x{screen:02x}.png"
                save_rgb_png(env.render(), shot)
                out["shots"].append(str(shot.name))
                last_screen = screen
            if (
                screen != ROOM_68
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
                out["samples"].append([f, x, y, f"0x{screen:02x}", stuck])
            if s.transitioning or mode != PLAY_MODE:
                _step(env, a, "DOWN", f)
            else:
                btn = _policy(x, y)
                _step(env, a, btn or None, f)
            f += 1
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
            if args.save_fixture and dest_screen != ROOM_68:
                name = f"Level7Interior{dest_screen:02X}ReconFixture"
                path = save_state(env, GAME_DIR, GAME, name)
                src = state_path(GAME_DIR, GAME, args.from_state)
                write_state_provenance(
                    path,
                    source_state_path=src if src.exists() else None,
                    request={
                        "bead": "rr-8t4.2",
                        "phase": "level7_interior_ropes_key_recon",
                        "track": "recon_fixture",
                        "route_eligible": False,
                        "fixture_only": True,
                        "natural_entry": False,
                        "development_only": True,
                        "fixture_writes": [],
                        "notes": [
                            f"Derived from {args.from_state} by WALKING "
                            f"0x68 DOWN -> 0x{dest_screen:02x} (ROPES_KEY). "
                            "No set_state/teleport.",
                            "UnlimitedHealthAssist traversal aid only. "
                            "No Candle/Whistle/TF/door/key/bomb/max_bombs writes.",
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
