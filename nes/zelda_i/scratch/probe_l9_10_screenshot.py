"""Screenshot the real power-on Silver Arrows room 0x10 entry (see
pin_l9_room10_entry.py) to inspect its layout.

Live power-on run (rr-sz8.6, 2026-09-06): the room is the same checkered
diagonal-wall diamond-grid pattern as room 0x51's statue diamond (rr-yxy6),
not a plain open floor -- the Silver Arrows item is presumably at the
grid's center, reachable only by threading the collision-free corridor,
same puzzle class as room 0x51. Not yet solved; see LEVEL9_ROUTE.md.

    QT_QPA_PLATFORM=offscreen uv run python \
        nes/zelda_i/scratch/probe_l9_10_screenshot.py
"""
from __future__ import annotations

from pathlib import Path

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, save_rgb_png
from zelda_i.paths import GAME, GAME_DIR

STATE_NAME = "L9Room10EntryReal"


def main() -> int:
    configure_headless()
    env = make_env(GAME, STATE_NAME, GAME_DIR, render_mode="rgb_array")
    obs, _ = reset_obs(env)
    out = Path("nes/zelda_i/recordings/l9_room10_entry.png")
    save_rgb_png(obs, out)
    env.close()
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
