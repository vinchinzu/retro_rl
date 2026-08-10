"""Boot Mega Man 2 to Heat Man stage and save a controllable Heat1 state.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python nes/mega_man_2/scripts/boot_heat_probe.py
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NES = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _NES):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mega_man_2.menus import boot_to_heat_man_script
from mega_man_2.paths import GAME, GAME_DIR, RECORDINGS_DIR
from mega_man_2.ram import (
    ADDR_STAGE_CURSOR,
    is_level1_ready,
    parse_game_state,
    read_u8,
)
from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_action, nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png

STABLE_FRAMES = 20
MIN_FRAME = 600
POST_READY_WAIT = 150


def run_probe(*, save_heat1: bool = True, walk_frames: int = 40) -> int:
    """Reach Heat Man stage, verify readiness past READY, save checkpoint."""
    configure_headless()
    out = RECORDINGS_DIR / "heat_boot"
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        frame = 0
        stable = 0
        for scripted in boot_to_heat_man_script():
            obs, *_ = env.step(scripted.action)
            frame += 1
            mean = float(obs.mean())
            if frame >= MIN_FRAME and is_level1_ready(env.get_ram(), obs_mean=mean):
                stable += 1
            else:
                stable = 0
            if stable >= STABLE_FRAMES:
                break
        else:
            png = save_rgb_png(obs, out / "boot_heat1.png")
            print(
                f"HEAT1 frame={frame} ready=False mean={float(obs.mean()):.1f} "
                f"screenshot={png}"
            )
            return 1

        for _ in range(POST_READY_WAIT):
            obs, *_ = env.step(nes_idle_action())
            frame += 1

        before = env.get_ram().copy()
        for _ in range(walk_frames):
            obs, *_ = env.step(nes_action("RIGHT"))
            frame += 1
        after = env.get_ram()
        changed = int((before != after).sum())
        mean = float(obs.mean())
        ready = is_level1_ready(after, obs_mean=mean) and changed >= 3
        state = parse_game_state(after, frame=frame, obs_mean=mean)
        cur = read_u8(after, ADDR_STAGE_CURSOR)
        png = save_rgb_png(obs, out / "boot_heat1.png")
        print(
            f"HEAT1 frame={frame} mode={state.mode.name} ready={ready} "
            f"changed={changed} mean={mean:.1f} cur={cur} "
            f"(expect heat path; leftover cur may be 0) screenshot={png}"
        )
        if save_heat1 and ready:
            path = save_state(env, GAME_DIR, GAME, "Heat1")
            print(f"saved {path}")
        return 0 if ready else 1
    finally:
        env.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--walk-frames", type=int, default=40)
    args = parser.parse_args()
    raise SystemExit(
        run_probe(save_heat1=not args.no_save, walk_frames=args.walk_frames)
    )


if __name__ == "__main__":
    main()
