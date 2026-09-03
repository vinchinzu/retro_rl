"""Recon: precise NORTH-door test for 0x6B and 0x6A from the recon fixture.

Sweeps Link across x on the y=93 top band pushing UP for a fixed window at
each x, recording the exact colliding tile and whether the screen changes.
Also records room_all_dead / open_doorway_mask.  If a north OPEN door exists
its notch is a 1-2 tile gap; a bomb wall shows as uniformly solid.

    QT_QPA_PLATFORM=offscreen uv run python \\
        nes/zelda_i/scratch/probe_l7_6ab_north.py --room 6b --tag 6b_north_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_BOMBS, PLAY_MODE, read_snapshot, read_u8
from zelda_i.runner import make_assist

ROOM_6A, ROOM_6B = 0x6A, 0x6B
BAND_Y = 93
MID_Y = 109
DOOR_Y = 141
WEST_PLANE = 32


def _s(env):
    return read_snapshot(env.get_ram())


def _step(env, assist, btn, f):
    env.step(nes_action(btn) if btn else nes_idle_action())
    if assist:
        assist.apply_env(env, frame=f)


def _to_6a(env, assist, f, budget=900):
    """0x6B -> 0x6A: ride y=109 west, drop y=141, push LEFT."""
    for _ in range(budget):
        s = _s(env)
        if s.screen == ROOM_6A and s.mode == PLAY_MODE and not s.transitioning:
            return f, True
        if s.transitioning:
            _step(env, assist, "LEFT", f); f += 1; continue
        x, y = int(s.link_x), int(s.link_y)
        if x > WEST_PLANE + 4:
            btn = ("UP" if y > MID_Y else "DOWN") if abs(y - MID_Y) > 4 else "LEFT"
        elif abs(y - DOOR_Y) > 4:
            btn = "UP" if y > DOOR_Y else "DOWN"
        else:
            btn = "LEFT"
        _step(env, assist, btn, f); f += 1
    return f, False


def _goto(env, assist, tx, ty, f, budget=200):
    for _ in range(budget):
        s = _s(env)
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - tx) <= 2 and abs(y - ty) <= 3:
            return f, True
        if abs(y - ty) > 3 and abs(x - tx) <= 8:
            btn = "UP" if y > ty else "DOWN"
        elif abs(x - tx) > 2:
            btn = "LEFT" if x > tx else "RIGHT"
        else:
            btn = "UP" if y > ty else "DOWN"
        _step(env, assist, btn, f); f += 1
    return f, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", choices=["6a", "6b"], default="6b")
    ap.add_argument("--tag", default="6b_north_v1")
    ap.add_argument("--from-state", default="Level7InteriorReconFixture")
    args = ap.parse_args()
    room = ROOM_6B if args.room == "6b" else ROOM_6A
    configure_headless()
    assist = make_assist(True)
    env = make_env(GAME, args.from_state, GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {"room": args.room, "route_eligible": False, "scan": []}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        f = 0
        if room == ROOM_6A:
            f, ok = _to_6a(env, assist, f)
            if not ok:
                out["failed"] = "no_6a"
                (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
                print("FAIL no 6a"); return
        s0 = _s(env)
        out["entry"] = {"screen": f"0x{int(s0.screen):02x}", "xy": [int(s0.link_x), int(s0.link_y)],
                        "room_all_dead": int(s0.room_all_dead),
                        "open_doorway_mask": int(s0.open_doorway_mask),
                        "cur_opened_doors": int(s0.cur_opened_doors)}
        print("ENTRY", out["entry"])
        transitioned = None
        for tx in range(112, 149, 2):
            f, _ = _goto(env, assist, tx, BAND_Y, f)
            s = _s(env)
            base_y = int(s.link_y)
            tiles = []
            for _ in range(45):
                s = _s(env)
                if int(s.screen) != room:
                    transitioned = (tx, int(s.screen), [int(s.link_x), int(s.link_y)])
                    break
                tiles.append(int(s.colliding_tile))
                _step(env, assist, "UP", f); f += 1
            s = _s(env)
            rec = {"x": tx, "y_base": base_y, "y_end": int(s.link_y),
                   "tile_end": int(s.colliding_tile),
                   "min_tile": min(tiles) if tiles else None,
                   "screen": f"0x{int(s.screen):02x}"}
            out["scan"].append(rec)
            print(rec)
            if transitioned:
                out["transitioned"] = {"x": transitioned[0], "dest": f"0x{transitioned[1]:02x}",
                                       "xy": transitioned[2]}
                for _ in range(140):
                    _step(env, assist, None, f); f += 1
                s = _s(env)
                out["dest_settled"] = {"screen": f"0x{int(s.screen):02x}",
                                       "xy": [int(s.link_x), int(s.link_y)],
                                       "mode": int(s.mode)}
                break
            # drop back to y=109 before next column
            f, _ = _goto(env, assist, tx, MID_Y, f, budget=80)
        s = _s(env)
        out["end"] = {"screen": f"0x{int(s.screen):02x}", "xy": [int(s.link_x), int(s.link_y)],
                      "deaths": int(assist.telemetry.deaths),
                      "bombs": int(read_u8(env.get_ram(), ADDR_BOMBS))}
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("END", out["end"], "transitioned", out.get("transitioned"))
        print("wrote", RECORDINGS_DIR / f"{args.tag}.json")
    finally:
        env.close()


if __name__ == "__main__":
    main()
