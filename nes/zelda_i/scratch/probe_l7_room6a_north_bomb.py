"""Recon: from Level7InteriorReconFixture, backtrack 0x6B LEFT -> 0x6A, then
sweep bomb attempts across the 0x6A NORTH wall to find the bombable passage
(mainline: 0x6A UP BOMB -> COMPASS, source).

0x6A is DARK (Candle 0); position reads are RAM so the blind sweep is fine.
For each candidate stand x: walk to it, rise UP until blocked, place a bomb
facing UP, step back, wait for the blast, push UP; if $EB changes, that is
the bomb spot + opens_to.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_room6a_north_bomb.py --tag 6a_north_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.dungeon.ids import object_name
from zelda_i.dungeon.ops import ensure_bomb, poke_bombs
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import (
    ADDR_BOMBS,
    ADDR_CANDLE,
    ADDR_FOOD,
    ADDR_KEYS,
    PLAY_MODE,
    read_snapshot,
    read_u8,
)
from zelda_i.runner import make_assist

ROOM_6A = 0x6A
ROOM_6B = 0x6B
MID_Y = 109
DOOR_Y = 141
WEST_PLANE = 32
X_TOL = 4
Y_TOL = 4
CANDIDATES = [128, 112, 144, 96, 160, 80, 176, 64, 192]


def _glance(env) -> dict:
    ram = env.get_ram()
    s = read_snapshot(ram)
    types = sorted(
        {int(o.type_id) for o in s.objects
         if 1 <= o.slot <= 12 and int(o.type_id) not in (0, 0xFF)}
    )
    return {
        "screen": f"0x{s.screen:02x}", "screen_int": int(s.screen),
        "level": int(s.level), "mode": int(s.mode),
        "xy": [int(s.link_x), int(s.link_y)], "tile": int(s.colliding_tile),
        "cur_opened_doors": int(s.cur_opened_doors),
        "open_doorway_mask": int(s.open_doorway_mask),
        "room_item_id": int(s.room_item_id), "room_all_dead": int(s.room_all_dead),
        "obj_types": [f"0x{t:02x}:{object_name(t)}" for t in types],
        "food": int(read_u8(ram, ADDR_FOOD)), "candle": int(read_u8(ram, ADDR_CANDLE)),
        "keys": int(read_u8(ram, ADDR_KEYS)), "bombs": int(read_u8(ram, ADDR_BOMBS)),
        "triforce": int(s.triforce),
    }


def _walk_to(env, assist, tx, ty, frame, budget=260):
    for _ in range(budget):
        s = read_snapshot(env.get_ram())
        if s.screen != ROOM_6A:
            return frame, False
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - tx) <= X_TOL and abs(y - ty) <= Y_TOL:
            return frame, True
        if abs(y - ty) > Y_TOL and abs(x - tx) <= X_TOL + 6:
            btn = "UP" if y > ty else "DOWN"
        elif abs(x - tx) > X_TOL:
            btn = "LEFT" if x > tx else "RIGHT"
        else:
            btn = "UP" if y > ty else "DOWN"
        env.step(nes_action(btn))
        if assist:
            assist.apply_env(env, frame=frame)
        frame += 1
    return frame, False


def _rise_until_blocked(env, assist, frame, budget=90):
    last_y = None
    for _ in range(budget):
        s = read_snapshot(env.get_ram())
        if s.screen != ROOM_6A:
            return frame, int(s.link_y), True
        y = int(s.link_y)
        if last_y is not None and y == last_y:
            return frame, y, False
        last_y = y
        env.step(nes_action("UP"))
        if assist:
            assist.apply_env(env, frame=frame)
        frame += 1
    return frame, last_y or 0, False


def _bomb_and_push(env, assist, frame, stand_x):
    ensure_bomb(env)
    s = read_snapshot(env.get_ram())
    y_stop = int(s.link_y)
    # place
    for i in range(6):
        ensure_bomb(env)
        env.step(nes_action("UP", "B") if i < 2 else nes_action("UP"))
        if assist:
            assist.apply_env(env, frame=frame)
        frame += 1
    # step back
    for _ in range(8):
        env.step(nes_action("DOWN"))
        if assist:
            assist.apply_env(env, frame=frame)
        frame += 1
    # wait blast
    for _ in range(110):
        env.step(nes_idle_action())
        if assist:
            assist.apply_env(env, frame=frame)
        frame += 1
    # push up
    for _ in range(160):
        s = read_snapshot(env.get_ram())
        if s.screen != ROOM_6A and s.mode == PLAY_MODE and not s.transitioning:
            return frame, int(s.screen), y_stop
        x = int(s.link_x)
        btn = "UP"
        if abs(x - stand_x) > 3:
            btn = "LEFT" if x > stand_x else "RIGHT"
        env.step(nes_action(btn))
        if assist:
            assist.apply_env(env, frame=frame)
        frame += 1
    return frame, ROOM_6A, y_stop


def _route_left_6b(env, assist, frame, budget=900):
    """0x6B -> 0x6A: ride y=109 west, drop y=141, push LEFT west doorway."""
    for _ in range(budget):
        s = read_snapshot(env.get_ram())
        if s.screen == ROOM_6A and s.mode == PLAY_MODE and not s.transitioning:
            return frame, True
        if s.transitioning:
            env.step(nes_action("LEFT"))
        else:
            x, y = int(s.link_x), int(s.link_y)
            if x > WEST_PLANE + X_TOL:
                if abs(y - MID_Y) > Y_TOL:
                    env.step(nes_action("UP" if y > MID_Y else "DOWN"))
                else:
                    env.step(nes_action("LEFT"))
            elif abs(y - DOOR_Y) > Y_TOL:
                env.step(nes_action("UP" if y > DOOR_Y else "DOWN"))
            else:
                env.step(nes_action("LEFT"))
        if assist:
            assist.apply_env(env, frame=frame)
        frame += 1
    return frame, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="6a_north_v1")
    ap.add_argument("--from-state", default="Level7InteriorReconFixture")
    ap.add_argument("--candidates", default="")
    args = ap.parse_args()
    cands = [int(c) for c in args.candidates.split(",")] if args.candidates else CANDIDATES
    configure_headless()
    assist = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"from_state": args.from_state, "route_eligible": False, "attempts": []}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        out["start"] = _glance(env)
        frame, ok = _route_left_6b(env, assist, 0)
        g6a = _glance(env)
        out["at_6a"] = g6a
        print("AT 0x6A", ok, g6a)
        if not ok or g6a["screen_int"] != ROOM_6A:
            out["success"] = False
            out["failed"] = "did_not_reach_0x6a"
            (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
            return
        found = None
        for cx in cands:
            poke_bombs(env, 8)
            frame, at = _walk_to(env, assist, cx, 100, frame)
            frame, y_top, left = _rise_until_blocked(env, assist, frame)
            s = read_snapshot(env.get_ram())
            rec = {"stand_x": cx, "reached_col": at, "y_top": y_top,
                   "xy_before_bomb": [int(s.link_x), int(s.link_y)],
                   "tile": int(s.colliding_tile), "screen": f"0x{int(s.screen):02x}"}
            if s.screen != ROOM_6A:
                rec["result"] = f"walked_into_0x{int(s.screen):02x}"
                out["attempts"].append(rec)
                found = (cx, int(s.screen), "walk")
                break
            frame, dest, y_stop = _bomb_and_push(env, assist, frame, cx)
            rec["y_stop"] = y_stop
            rec["dest"] = f"0x{dest:02x}"
            rec["bombs_after"] = int(read_u8(env.get_ram(), ADDR_BOMBS))
            out["attempts"].append(rec)
            print("ATTEMPT", rec)
            if dest != ROOM_6A:
                found = (cx, dest, "bomb")
                for _ in range(120):
                    env.step(nes_idle_action())
                    if assist:
                        assist.apply_env(env, frame=frame)
                    frame += 1
                out["dest_settled"] = _glance(env)
                break
            # step back down to mid band before next candidate
            frame, _ = _walk_to(env, assist, cx, DOOR_Y, frame, budget=120)
        out["found"] = found
        end = _glance(env)
        end["deaths"] = int(assist.telemetry.deaths)
        end["progression_writes"] = int(assist.telemetry.progression_writes)
        end["capacity_writes"] = int(assist.telemetry.capacity_writes)
        out["end"] = end
        out["success"] = found is not None and end["deaths"] == 0
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("FOUND", found, "end", end)
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
