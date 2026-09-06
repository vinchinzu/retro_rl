"""Dump pixel-level $6530 tiles for the 0x1A center diamond + test whether the
0x68 block pushes UP with center goriyas still alive, and whether a bomb near
the center kills a pocket goriya.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=nes:. uv run python \
        nes/zelda_i/scratch/probe_l7_1a_pocket.py --tag 1a_pocket_v1 --mode tiles
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.combat import nearest_enemy
from zelda_i.dungeon.behaviors import EnemyKind, engagement_hint
from zelda_i.dungeon.tilemap import tile_at_screen
from zelda_i.level7.path import live_goriyas
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_BOMBS, ADDR_SELECTED_ITEM, read_snapshot, read_u8
from zelda_i.runner import make_assist

ROOM = 0x1A


def _s(env):
    return read_snapshot(env.get_ram())


def _step(env, a, btn, f):
    env.step(nes_idle_action() if btn is None
             else nes_action(*btn) if isinstance(btn, tuple) else nes_action(btn))
    if a:
        a.apply_env(env, frame=f)


def _tiles(env):
    ram = env.get_ram()
    rows = []
    for y in range(96, 185, 8):
        row = []
        for x in range(80, 177, 8):
            row.append(f"{tile_at_screen(ram, x, y):3d}")
        rows.append(f"y{y:3d} " + " ".join(row))
    return "x    " + " ".join(f"{x:3d}" for x in range(80, 177, 8)) + "\n" + "\n".join(rows)


def _fight_to_center(env, a, f, cap=6000):
    for i in range(cap):
        s = _s(env)
        if int(s.screen) != ROOM:
            break
        live = live_goriyas(s)
        if not live:
            return f, "all_dead"
        outer = [g for g in live
                 if not (104 <= int(g.x) <= 152 and 130 <= int(g.y) <= 154)]
        if not outer and i > 150:
            return f, "centered"
        tgt = nearest_enemy(s.link_x, s.link_y, outer or live)
        hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
        _step(env, a, (hint.face, "A") if (i % 8) < 4 else hint.face, f)
        f += 1
    return f, "cap"


def _reach(env, a, tx, ty, f, budget=400):
    for _ in range(budget):
        s = _s(env)
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - tx) <= 3 and abs(y - ty) <= 3:
            return f, True
        if abs(y - ty) > 3:
            _step(env, a, "UP" if y > ty else "DOWN", f)
        else:
            _step(env, a, "LEFT" if x > tx else "RIGHT", f)
        f += 1
    return f, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1a_pocket_v1")
    ap.add_argument("--mode", default="tiles",
                    choices=["tiles", "push", "bomb"])
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, "Level7Interior1AReconFixture", GAME_DIR,
                   render_mode="rgb_array")
    out: dict = {"mode": args.mode}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        f = 0
        out["tiles_start"] = _tiles(env)
        print(out["tiles_start"])

        if args.mode == "tiles":
            (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
            return

        f, status = _fight_to_center(env, a, f)
        s = _s(env)
        out["fight_status"] = status
        out["after_fight"] = {
            "f": f, "xy": [int(s.link_x), int(s.link_y)],
            "rad": int(s.room_all_dead),
            "g": [[int(g.x), int(g.y), int(g.hp), int(g.state)]
                  for g in live_goriyas(s)],
        }
        print("AFTER FIGHT", out["after_fight"])

        if args.mode == "push":
            f, ok = _reach(env, a, 96, 162, f)
            out["at_stand"] = {"ok": ok, "xy": [int(_s(env).link_x), int(_s(env).link_y)]}
            blk0 = [(int(o.x), int(o.y)) for o in _s(env).objects
                    if int(o.type_id) == 0x68 and 1 <= int(o.slot) <= 12]
            for i in range(200):
                s = _s(env)
                if int(s.mode) in (9, 10, 11) or int(s.screen) != ROOM:
                    break
                _step(env, a, "UP", f)
                f += 1
            s = _s(env)
            blk1 = [(int(o.x), int(o.y)) for o in s.objects
                    if int(o.type_id) == 0x68 and 1 <= int(o.slot) <= 12]
            out["push_result"] = {
                "block_before": blk0, "block_after": blk1,
                "xy": [int(s.link_x), int(s.link_y)], "mode": int(s.mode),
                "scr": f"0x{int(s.screen):02x}", "rad": int(s.room_all_dead),
                "g": [[int(g.x), int(g.y), int(g.hp)] for g in live_goriyas(s)],
            }
            print("PUSH RESULT", out["push_result"])

        elif args.mode == "bomb":
            # Select bombs, stand just outside the pocket, drop a bomb.
            for tx, ty in ((96, 189), (128, 176), (128, 165)):
                f, ok = _reach(env, a, tx, ty, f)
            # ensure bombs selected
            if int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)) != 1:
                env.unwrapped.data.memory.assign(int(ADDR_SELECTED_ITEM), "|u1", 1)
            b0 = int(read_u8(env.get_ram(), ADDR_BOMBS))
            for i in range(8):
                _step(env, a, ("DOWN", "B"), f)
                f += 1
            for i in range(180):
                _step(env, a, None, f)
                f += 1
            s = _s(env)
            out["bomb_result"] = {
                "bombs": [b0, int(read_u8(env.get_ram(), ADDR_BOMBS))],
                "rad": int(s.room_all_dead),
                "xy": [int(s.link_x), int(s.link_y)],
                "g": [[int(g.x), int(g.y), int(g.hp)] for g in live_goriyas(s)],
            }
            print("BOMB RESULT", out["bomb_result"])

        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
    finally:
        env.close()


if __name__ == "__main__":
    main()
