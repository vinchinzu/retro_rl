"""Hypothesis: goriyas trapped in the 0x1A center diamond pocket wander back
out if Link stops pressuring them.  Clear the room with the normal fight until
only center-pocket goriyas remain, then park Link in a corner and idle,
sampling goriya positions + room_all_dead.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=nes:. uv run python \
        nes/zelda_i/scratch/probe_l7_1a_center_wander.py --tag 1a_wander_v1
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
from zelda_i.ram import read_snapshot
from zelda_i.runner import make_assist

ROOM = 0x1A


def _s(env):
    return read_snapshot(env.get_ram())


def _centered(g) -> bool:
    return 100 <= int(g.x) <= 156 and 128 <= int(g.y) <= 156


def _step(env, a, btn, f):
    env.step(nes_idle_action() if btn is None else nes_action(*btn)
             if isinstance(btn, tuple) else nes_action(btn))
    if a:
        a.apply_env(env, frame=f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1a_wander_v1")
    ap.add_argument("--corner", default="32,96")
    ap.add_argument("--idle-frames", type=int, default=6000)
    args = ap.parse_args()
    cx, cy = (int(v) for v in args.corner.split(","))
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, "Level7Interior1AReconFixture", GAME_DIR,
                   render_mode="rgb_array")
    out: dict = {"corner": [cx, cy]}
    samples: list[dict] = []
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        f = 0
        # Fight until every live goriya is centered (or none left, or 6000f).
        for i in range(6000):
            s = _s(env)
            if int(s.screen) != ROOM:
                break
            live = live_goriyas(s)
            if not live:
                out["result"] = "all_dead_during_fight"
                break
            outer = [g for g in live if not _centered(g)]
            if not outer and i > 200:
                break
            tgt = nearest_enemy(s.link_x, s.link_y, outer or live)
            hint = engagement_hint(EnemyKind.GORIYA, s, tgt)
            _step(env, a, (hint.face, "A") if (i % 8) < 4 else hint.face, f)
            f += 1
        s = _s(env)
        live = live_goriyas(s)
        out["after_fight"] = {
            "f": f, "xy": [int(s.link_x), int(s.link_y)],
            "rad": int(s.room_all_dead),
            "g": [[int(g.x), int(g.y), int(g.hp), int(g.state)] for g in live],
        }
        print("AFTER FIGHT", out["after_fight"])
        # Walk to the corner.
        for _ in range(400):
            s = _s(env)
            x, y = int(s.link_x), int(s.link_y)
            if abs(x - cx) <= 4 and abs(y - cy) <= 4:
                break
            if abs(y - cy) > 4:
                _step(env, a, "UP" if y > cy else "DOWN", f)
            else:
                _step(env, a, "LEFT" if x > cx else "RIGHT", f)
            f += 1
        # Idle in the corner, sampling.
        for i in range(args.idle_frames):
            _step(env, a, None, f)
            f += 1
            if i % 200 == 0:
                s = _s(env)
                live = live_goriyas(s)
                rec = {
                    "f": f, "xy": [int(s.link_x), int(s.link_y)],
                    "rad": int(s.room_all_dead), "ng": len(live),
                    "g": [[int(g.x), int(g.y)] for g in live],
                    "centered": sum(1 for g in live if _centered(g)),
                }
                samples.append(rec)
                print("IDLE", rec)
                if not live or any(not _centered(g) for g in live):
                    out["escaped"] = True
                    break
        out["samples"] = samples
        s = _s(env)
        out["end"] = {
            "f": f, "xy": [int(s.link_x), int(s.link_y)],
            "rad": int(s.room_all_dead),
            "g": [[int(g.x), int(g.y), int(g.hp)] for g in live_goriyas(s)],
        }
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
        print("END", out["end"])
    finally:
        env.close()


if __name__ == "__main__":
    main()
