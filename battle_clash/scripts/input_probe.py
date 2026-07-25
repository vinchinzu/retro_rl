"""Confirm the current Battle Clash Super Scope input boundary."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from battle_clash.paths import GAME, GAME_DIR, RECORDINGS_DIR
from retro_harness.env import make_env
from snes_oneshot.actions import buttons, idle_action
from snes_oneshot.segment_runner import configure_headless, save_rgb_png


def run_probe() -> int:
    """Boot the title and report whether cursor/light-gun injection exists."""
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        for frame in range(1, 601):
            slot = frame % 240
            if 20 <= slot < 30:
                action = buttons("START")
            elif 100 <= slot < 108:
                action = buttons("B")
            elif 160 <= slot < 166:
                action = buttons("Y")
            else:
                action = idle_action()
            obs, *_ = env.step(action)

        png = save_rgb_png(obs, RECORDINGS_DIR / "boot_title_input_blocked.png")
        joypad_buttons = tuple(env.unwrapped.buttons)
        emulator_methods = set(dir(env.unwrapped.em))
        cursor_methods = sorted(
            name
            for name in emulator_methods
            if any(token in name.lower() for token in ("cursor", "gun", "mouse"))
        )
        print(
            "INPUT_BLOCKED "
            f"joypad_buttons={joypad_buttons} cursor_api={cursor_methods} "
            f"screenshot={png}"
        )
        return 0 if not cursor_methods else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(run_probe())
