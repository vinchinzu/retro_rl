"""Recon: from Level7Interior68ReconFixture (settled 0x68 KEESE_TRAPS, entry
~(208,93)), probe one cardinal cleanly and glance the destination.

0x68: 4 blade traps 0x49 (corners) + 4 keese 0x1b, dark.  Source: UP -> 0x58
DODONGOS_UPGRADE, DOWN -> ROPES_KEY, RIGHT -> 0x69 (bombed).

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room68_exits.py --dir UP --tag 68_up_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name, room_item_name
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS, ADDR_CANDLE, ADDR_FOOD, ADDR_KEYS, PLAY_MODE,
    read_snapshot, read_u8,
)
from zelda_i.runner import make_assist

ROOM_68 = 0x68
# want position to start each push from
WANT = {"UP": (120, 141), "DOWN": (120, 93), "LEFT": (200, 141), "RIGHT": (40, 141)}


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
    }


def _reach(env, a, room, tx, ty, f, budget=400):
    """x-first then y, with small nudges; dark + blade-trap tolerant."""
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != room:
            return f, False
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - tx) <= 3 and abs(y - ty) <= 3:
            return f, True
        if abs(x - tx) > 3:
            _step(env, a, "LEFT" if x > tx else "RIGHT", f)
        else:
            _step(env, a, "UP" if y > ty else "DOWN", f)
        f += 1
    return f, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="UP", choices=list(WANT))
    ap.add_argument("--tag", default="68_up_v1")
    ap.add_argument("--from-state", default="Level7Interior68ReconFixture")
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
        f = 0
        tx, ty = WANT[args.dir]
        f, at = _reach(env, a, ROOM_68, tx, ty, f)
        out["reached_start"] = at
        start = _s(env)
        out["push_from"] = [int(start.link_x), int(start.link_y)]
        arrived = None
        samples = []
        for _ in range(220):
            s = _s(env)
            if int(s.screen) != ROOM_68 and int(s.mode) == PLAY_MODE and not s.transitioning:
                arrived = (int(s.screen), [int(s.link_x), int(s.link_y)], f)
                break
            if f % 20 == 0:
                samples.append([f, int(s.link_x), int(s.link_y), f"0x{int(s.screen):02x}"])
            _step(env, a, args.dir, f); f += 1
        out["samples"] = samples
        if arrived is None:
            e = _s(env)
            out["result"] = "blocked"
            out["end_xy"] = [int(e.link_x), int(e.link_y)]
            out["tile"] = int(e.colliding_tile)
        else:
            out["result"] = f"0x{arrived[0]:02x}"
            out["arrived_frame"] = arrived[2]
            out["arrived_xy"] = arrived[1]
            for _ in range(150):
                _step(env, a, None, f); f += 1
            out["dest"] = _glance(env)
            print("DEST", out["dest"])
        end = _glance(env)
        end["deaths"] = int(a.telemetry.deaths)
        end["progression_writes"] = int(a.telemetry.progression_writes)
        end["capacity_writes"] = int(a.telemetry.capacity_writes)
        out["end"] = end
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print(f"{args.dir} result={out['result']} pushfrom={out['push_from']}")
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
