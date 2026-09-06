"""Kill only the goriyas Link can physically reach (outside the sealed center
cross), then check room_all_dead and try the 0x68 UP push -> stairs -> candle.
Tests whether the 2 born-trapped center goriyas gate room clear.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=nes:. uv run python \
        nes/zelda_i/scratch/probe_l7_1a_outer_only.py --tag 1a_outer_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.combat import manhattan, nearest_enemy
from zelda_i.dungeon.behaviors import EnemyKind, engagement_hint
from zelda_i.level7.path import live_goriyas
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_CANDLE, read_snapshot, read_u8
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


def _pocket(o) -> bool:
    return 104 <= int(o.x) <= 152 and 118 <= int(o.y) <= 170


def _g(s):
    return [[int(o.x), int(o.y), int(o.hp), int(o.state)] for o in live_goriyas(s)]


def _reach(env, a, tx, ty, f, budget=500):
    stuck = 0
    last = None
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
        if abs(y - ty) > 3:
            _step(env, a, "UP" if y > ty else "DOWN", f)
        else:
            _step(env, a, "LEFT" if x > tx else "RIGHT", f)
        f += 1
    return f, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1a_outer_v1")
    ap.add_argument("--budget", type=int, default=14000)
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
        # Aggressive melee on the nearest NON-pocket goriya.
        no_outer_since = None
        for i in range(args.budget):
            s = _s(env)
            if int(s.screen) != ROOM:
                break
            live = live_goriyas(s)
            outer = [o for o in live if not _pocket(o)]
            if not live:
                out["clear"] = "all_dead"
                break
            if not outer:
                if no_outer_since is None:
                    no_outer_since = i
                elif i - no_outer_since > 300:
                    out["clear"] = "only_pocket_left"
                    break
                _step(env, a, None, f)
                f += 1
                continue
            no_outer_since = None
            tgt = nearest_enemy(s.link_x, s.link_y, outer)
            hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
            d = manhattan(s.link_x, s.link_y, tgt.x, tgt.y)
            # Always drive toward the target; swing on a cadence when close.
            if d <= 22:
                btn = (hint.face, "A") if (i % 6) < 3 else hint.face
            else:
                btn = hint.face
            _step(env, a, btn, f)
            f += 1
        s = _s(env)
        out["after_fight"] = {
            "f": f, "xy": [int(s.link_x), int(s.link_y)],
            "rad": int(s.room_all_dead), "g": _g(s),
        }
        print("AFTER FIGHT", out.get("clear"), out["after_fight"])

        # Try the UP push from (96,162).
        f, ok = _reach(env, a, 96, 162, f)
        out["at_stand"] = {"ok": ok, "xy": [int(_s(env).link_x), int(_s(env).link_y)]}
        blk0 = [(int(o.x), int(o.y)) for o in _s(env).objects
                if int(o.type_id) == 0x68 and 1 <= int(o.slot) <= 12]
        pushed = False
        for i in range(240):
            s = _s(env)
            if int(s.mode) in CELLAR_MODES or int(s.screen) != ROOM:
                break
            blk = [(int(o.x), int(o.y)) for o in s.objects
                   if int(o.type_id) == 0x68 and 1 <= int(o.slot) <= 12]
            if blk and blk[0][1] <= 132:
                pushed = True
                break
            _step(env, a, "UP", f)
            f += 1
        s = _s(env)
        blk1 = [(int(o.x), int(o.y)) for o in s.objects
                if int(o.type_id) == 0x68 and 1 <= int(o.slot) <= 12]
        out["push"] = {"block_before": blk0, "block_after": blk1, "pushed": pushed,
                       "xy": [int(s.link_x), int(s.link_y)], "mode": int(s.mode),
                       "scr": f"0x{int(s.screen):02x}", "rad": int(s.room_all_dead)}
        print("PUSH", out["push"])

        # If pushed, walk east into the pocket and onto the stairs.
        if pushed:
            for tx, ty in ((96, 144), (112, 141), (128, 141)):
                if int(_s(env).mode) in CELLAR_MODES:
                    break
                f, ok = _reach(env, a, tx, ty, f, budget=300)
                print("STAIR WP", [tx, ty], ok,
                      [int(_s(env).link_x), int(_s(env).link_y)],
                      "mode", int(_s(env).mode))
            for _ in range(120):
                s = _s(env)
                if int(s.mode) in CELLAR_MODES:
                    break
                _step(env, a, "RIGHT", f)
                f += 1
            s = _s(env)
            out["stairs"] = {"mode": int(s.mode), "scr": f"0x{int(s.screen):02x}",
                             "xy": [int(s.link_x), int(s.link_y)]}
            print("STAIRS", out["stairs"])
            if int(s.mode) in CELLAR_MODES:
                # walk toward center to grab the candle
                for tx, ty in ((128, 189), (176, 189), (176, 141), (120, 141)):
                    if int(read_u8(env.get_ram(), ADDR_CANDLE)) >= 2:
                        break
                    f, ok = _reach(env, a, tx, ty, f, budget=300)
                out["candle"] = int(read_u8(env.get_ram(), ADDR_CANDLE))
                out["cellar_glance"] = {
                    "scr": f"0x{int(_s(env).screen):02x}",
                    "xy": [int(_s(env).link_x), int(_s(env).link_y)]}
                print("CANDLE", out["candle"], out["cellar_glance"])

        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
    finally:
        env.close()


if __name__ == "__main__":
    main()
