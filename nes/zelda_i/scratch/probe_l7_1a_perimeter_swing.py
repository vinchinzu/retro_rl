"""From Level7Interior1AReconFixture: let the 2 center goriyas get trapped in
the sealed cross, then park Link at each cross-perimeter stand facing inward
and swing on a cadence for a while.  Report goriya HP over time + any beam /
projectile objects Link spawns.  Tests whether sword (beam) or melee-on-surface
can kill the trapped pair.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=nes:. uv run python \
        nes/zelda_i/scratch/probe_l7_1a_perimeter_swing.py --tag 1a_ps_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.combat import nearest_enemy
from zelda_i.dungeon.behaviors import EnemyKind, engagement_hint
from zelda_i.dungeon.ids import object_name
from zelda_i.level7.path import live_goriyas
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.runner import make_assist

ROOM = 0x1A
CELLAR = {9, 10, 11}


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


def _allobj(s):
    return [{"slot": int(o.slot), "t": f"0x{int(o.type_id) & 0xFF:02x}:"
             f"{object_name(int(o.type_id) & 0xFF)}", "hp": int(o.hp),
             "xy": [int(o.x), int(o.y)]}
            for o in s.objects
            if 1 <= int(o.slot) <= 12 and int(o.type_id) & 0xFF not in (0, 0xFF)]


def _reach(env, a, tx, ty, f, budget=600):
    stuck, last = 0, None
    for _ in range(budget):
        s = _s(env)
        if int(s.screen) != ROOM or int(s.mode) in CELLAR:
            return f, False
        x, y = int(s.link_x), int(s.link_y)
        if abs(x - tx) <= 2 and abs(y - ty) <= 2:
            return f, True
        if (x, y) == last:
            stuck += 1
            if stuck > 60:
                return f, False
        else:
            stuck, last = 0, (x, y)
        # go wide: y to top strip first, then x, then in
        if y > 100 and not (abs(x - tx) <= 2):
            _step(env, a, "UP", f)
        elif abs(x - tx) > 2:
            _step(env, a, "LEFT" if x > tx else "RIGHT", f)
        else:
            _step(env, a, "UP" if y > ty else "DOWN", f)
        f += 1
    return f, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1a_ps_v1")
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
        # Fight outer goriyas until only pocket pair remains (cap 6000).
        for i in range(6000):
            s = _s(env)
            live = live_goriyas(s)
            outer = [o for o in live if not _pocket(o)]
            if not live or (not outer and i > 200):
                break
            tgt = nearest_enemy(s.link_x, s.link_y, outer or live)
            hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
            _step(env, a, (hint.face, "A") if (i % 6) < 3 else hint.face, f)
            f += 1
        s = _s(env)
        out["after_outer"] = {"f": f, "rad": int(s.room_all_dead), "g": _g(s),
                              "hearts_hp": int(s.health) if hasattr(s, "health") else None}
        print("AFTER OUTER", out["after_outer"])
        print("OBJ", _allobj(s))

        stands = [
            (128, 100, "DOWN"), (128, 96, "DOWN"),
            (128, 188, "UP"), (128, 192, "UP"),
            (96, 141, "RIGHT"), (160, 141, "LEFT"),
            (80, 141, "RIGHT"),
        ]
        rounds = []
        for (tx, ty, face) in stands:
            s = _s(env)
            if not live_goriyas(s) or int(s.mode) in CELLAR:
                break
            f, ok = _reach(env, a, tx, ty, f)
            s = _s(env)
            hp0 = sorted(o.hp for o in live_goriyas(s))
            beams = set()
            for i in range(500):
                s = _s(env)
                if not live_goriyas(s) or int(s.mode) in CELLAR:
                    break
                for o in s.objects:
                    t = int(o.type_id) & 0xFF
                    if 1 <= int(o.slot) <= 12 and t not in (0, 0xFF, 0x05, 0x06, 0x68):
                        beams.add(f"0x{t:02x}:{object_name(t)}")
                _step(env, a, (face, "A") if (i % 5) < 3 else face, f)
                f += 1
            s = _s(env)
            hp1 = sorted(o.hp for o in live_goriyas(s))
            rec = {"stand": [tx, ty, face], "reached": ok,
                   "xy": [int(s.link_x), int(s.link_y)],
                   "hp_before": hp0, "hp_after": hp1,
                   "spawned_objs": sorted(beams),
                   "g": _g(s), "rad": int(s.room_all_dead), "mode": int(s.mode)}
            rounds.append(rec)
            print("ROUND", rec)
        out["rounds"] = rounds
        s = _s(env)
        out["end"] = {"f": f, "rad": int(s.room_all_dead), "g": _g(s),
                      "mode": int(s.mode), "scr": f"0x{int(s.screen):02x}"}
        print("END", out["end"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
    finally:
        env.close()


if __name__ == "__main__":
    main()
