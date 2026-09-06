"""Can a bomb kill the born-trapped center goriyas in 0x1A?
Kill the outer goriyas (or not), then stand at the top/bottom of the center
column and drop bombs facing the pocket.  Report goriya HP + room_all_dead.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=nes:. uv run python \
        nes/zelda_i/scratch/probe_l7_1a_bomb_center.py --tag 1a_bombc_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.combat import nearest_enemy
from zelda_i.dungeon.behaviors import EnemyKind, engagement_hint
from zelda_i.level7.path import live_goriyas
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_BOMBS, ADDR_SELECTED_ITEM, read_snapshot, read_u8
from zelda_i.runner import make_assist

ROOM = 0x1A
CELLAR_MODES = {9, 10, 11}


def _s(env):
    return read_snapshot(env.get_ram())


def _step(env, a, btn, f):
    env.step(nes_idle_action() if btn is None
             else nes_action(*btn) if isinstance(btn, tuple) else nes_action(btn))
    if a:
        a.apply_env(env, frame=f)


def _pocket(o):
    return 104 <= int(o.x) <= 152 and 118 <= int(o.y) <= 170


def _g(s):
    return [[int(o.x), int(o.y), int(o.hp), int(o.state)] for o in live_goriyas(s)]


def _reach(env, a, tx, ty, f, budget=500):
    stuck, last = 0, None
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM or int(s.mode) in CELLAR_MODES:
            return f, False
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - tx) <= 3 and abs(y - ty) <= 3:
            return f, True
        if (x, y) == last:
            stuck += 1
            if stuck > 40:
                return f, False
        else:
            stuck, last = 0, (x, y)
        if abs(x - tx) > 3 and abs(y - ty) <= 6:
            _step(env, a, "LEFT" if x > tx else "RIGHT", f)
        elif abs(y - ty) > 3:
            _step(env, a, "UP" if y > ty else "DOWN", f)
        else:
            _step(env, a, "LEFT" if x > tx else "RIGHT", f)
        f += 1
    return f, False


def _drop_bombs(env, a, face, n, f):
    if int(read_u8(env.get_ram(), ADDR_SELECTED_ITEM)) != 1:
        env.unwrapped.data.memory.assign(int(ADDR_SELECTED_ITEM), "|u1", 1)
    b0 = int(read_u8(env.get_ram(), ADDR_BOMBS))
    for _ in range(n):
        for _ in range(6):
            _step(env, a, (face, "B"), f)
            f += 1
        for _ in range(90):
            _step(env, a, face, f)
            f += 1
    b1 = int(read_u8(env.get_ram(), ADDR_BOMBS))
    return f, b0, b1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1a_bombc_v1")
    ap.add_argument("--clear-outer", action="store_true")
    args = ap.parse_args()
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, "Level7Interior1AReconFixture", GAME_DIR,
                   render_mode="rgb_array")
    out: dict = {}
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        f = 0
        out["start_g"] = _g(_s(env))
        if args.clear_outer:
            for i in range(9000):
                s = _s(env)
                live = live_goriyas(s)
                outer = [o for o in live if not _pocket(o)]
                if not live or not outer:
                    break
                tgt = nearest_enemy(s.link_x, s.link_y, outer)
                hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
                _step(env, a, (hint.face, "A") if (i % 6) < 3 else hint.face, f)
                f += 1
            out["after_outer"] = {"f": f, "rad": int(_s(env).room_all_dead),
                                  "g": _g(_s(env))}
            print("AFTER OUTER", out["after_outer"])

        rounds = []
        for (tx, ty, face) in ((128, 107, "DOWN"), (128, 177, "UP"),
                               (128, 107, "DOWN"), (128, 177, "UP")):
            s = _s(env)
            if int(s.mode) in CELLAR_MODES or not live_goriyas(s):
                break
            f, ok = _reach(env, a, tx, ty, f)
            f, b0, b1 = _drop_bombs(env, a, face, 2, f)
            s = _s(env)
            rec = {"stand": [tx, ty, face], "reached": ok,
                   "xy": [int(s.link_x), int(s.link_y)],
                   "bombs": [b0, b1], "rad": int(s.room_all_dead),
                   "g": _g(s), "mode": int(s.mode)}
            rounds.append(rec)
            print("ROUND", rec)
        out["rounds"] = rounds
        s = _s(env)
        out["end"] = {"f": f, "rad": int(s.room_all_dead), "g": _g(s),
                      "mode": int(s.mode), "scr": f"0x{int(s.screen):02x}",
                      "xy": [int(s.link_x), int(s.link_y)]}
        print("END", out["end"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
    finally:
        env.close()


if __name__ == "__main__":
    main()
