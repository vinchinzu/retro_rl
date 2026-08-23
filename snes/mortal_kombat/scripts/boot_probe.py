#!/usr/bin/env python3
"""Power-on → Liu Kang fight-ready, RAM-gated. Writes a screenshot on success."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _p in (_ROOT, _ROOT / "snes"):
    _t = str(_p)
    if _t not in sys.path:
        sys.path.insert(0, _t)

from retro_harness.env import make_env, reset_obs  # noqa: E402
from retro_harness.segment_runner import save_rgb_png  # noqa: E402
from mortal_kombat.boot import boot_to_fight  # noqa: E402
from mortal_kombat.paths import GAME_DIR, GAME_ID, RECORDINGS_DIR  # noqa: E402
from mortal_kombat.ram import char_name, is_fight_ready  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=9000)
    args = parser.parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = make_env(GAME_ID, "NONE", GAME_DIR, render_mode="rgb_array")
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        reset_obs(env)
        try:
            snap = boot_to_fight(env, max_frames=args.max_frames)
        except TimeoutError as exc:
            frame = env.render()
            png = save_rgb_png(frame, RECORDINGS_DIR / "boot_timeout.png")
            print(f"TIMEOUT {exc} png={png}")
            return 1
        ok = is_fight_ready(snap)
        frame = env.render()
        png = save_rgb_png(frame, RECORDINGS_DIR / "boot_liukang_fight.png")
        print(
            f"FIGHT_READY={ok} char={char_name(snap.p1_character)} "
            f"vs={char_name(snap.p2_character)} "
            f"hp={snap.p1_health}/{snap.p2_health} timer={snap.timer} "
            f"match={snap.match_counter} mode={snap.game_mode} "
            f"pos=({snap.p1.x},{snap.p1.y})-({snap.p2.x},{snap.p2.y}) "
            f"png={png}"
        )
        return 0 if ok else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
