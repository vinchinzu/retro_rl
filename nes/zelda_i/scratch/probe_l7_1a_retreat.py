"""Run the real Room1ACandleController until it is stuck in phase=hunt with
pocket goriyas, then override: retreat Link to the bottom row and idle.  Watch
whether the pocket goriyas leave and room_all_dead flips.

    QT_QPA_PLATFORM=offscreen PYTHONPATH=nes:. uv run python \
        nes/zelda_i/scratch/probe_l7_1a_retreat.py --tag 1a_retreat_v1
"""

from __future__ import annotations

import argparse
import json

from retro_harness.env import make_env, reset_obs
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.level7.cellar import Room1ACandleController
from zelda_i.level7.path import live_goriyas
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.runner import make_assist

ROOM = 0x1A


def _s(env):
    return read_snapshot(env.get_ram())


def _step(env, a, btn, f):
    env.step(nes_idle_action() if btn is None
             else nes_action(*btn) if isinstance(btn, tuple) else nes_action(btn))
    if a:
        a.apply_env(env, frame=f)


def _g(s):
    return [[int(o.x), int(o.y), int(o.hp), int(o.state)] for o in live_goriyas(s)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1a_retreat_v1")
    ap.add_argument("--warm", type=int, default=5200)
    ap.add_argument("--retreat", type=int, default=8000)
    ap.add_argument("--park", default="32,189")
    args = ap.parse_args()
    px, py = (int(v) for v in args.park.split(","))
    configure_headless()
    a = make_assist(True)
    env = make_env(GAME, "Level7Interior1AReconFixture", GAME_DIR,
                   render_mode="rgb_array")
    ctl = Room1ACandleController()
    if hasattr(ctl, "bind_env"):
        ctl.bind_env(env)
    out: dict = {"park": [px, py]}
    samples: list[dict] = []
    try:
        reset_obs(env)
        for _ in range(2):
            env.step(nes_idle_action())
        f = 0
        for _ in range(args.warm):
            s = _s(env)
            act = ctl.step(s)
            env.step(act.action)
            a.apply_env(env, frame=f)
            f += 1
            if ctl.success or getattr(ctl, "failed", False):
                break
        s = _s(env)
        out["after_warm"] = {
            "f": f, "phase": getattr(ctl, "_phase", None),
            "hunt_i": getattr(ctl, "_hunt_i", None),
            "xy": [int(s.link_x), int(s.link_y)], "rad": int(s.room_all_dead),
            "g": _g(s), "success": bool(ctl.success),
        }
        print("AFTER WARM", out["after_warm"])

        # Override: walk to park, then idle.
        for i in range(args.retreat):
            s = _s(env)
            if int(s.screen) != ROOM or int(s.mode) in (9, 10, 11):
                break
            x, y = int(s.link_x), int(s.link_y)
            if abs(x - px) <= 4 and abs(y - py) <= 4:
                btn = None
            elif abs(y - py) > 4:
                btn = "UP" if y > py else "DOWN"
            else:
                btn = "LEFT" if x > px else "RIGHT"
            _step(env, a, btn, f)
            f += 1
            if i % 200 == 0:
                live = live_goriyas(s)
                rec = {"f": f, "xy": [x, y], "rad": int(s.room_all_dead),
                       "ng": len(live), "g": _g(s)}
                samples.append(rec)
                print("RETREAT", rec)
                if not live or int(s.room_all_dead) != 0:
                    out["cleared"] = True
                    break
        out["samples"] = samples
        s = _s(env)
        out["end"] = {"f": f, "xy": [int(s.link_x), int(s.link_y)],
                      "rad": int(s.room_all_dead), "g": _g(s),
                      "mode": int(s.mode), "scr": f"0x{int(s.screen):02x}"}
        print("END", out["end"])
        save_rgb_png(env.render(), RECORDINGS_DIR / f"{args.tag}_final.png")
        (RECORDINGS_DIR / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
    finally:
        env.close()


if __name__ == "__main__":
    main()
