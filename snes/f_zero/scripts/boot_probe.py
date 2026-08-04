"""Boot F-Zero from reset and save a Mute City race-start state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from f_zero.menus import boot_to_mute_city_script
from f_zero.paths import GAME, GAME_DIR, RECORDINGS_DIR
from f_zero.ram import parse_game_state
from retro_harness.env import make_env, save_state
from retro_harness.ram_state import GameMode
from retro_harness.segment_runner import configure_headless, save_rgb_png


def run_probe(*, save_race: bool = True) -> int:
    """Reach the Mute City countdown and optionally save it."""
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        frame = 0
        for scripted in boot_to_mute_city_script():
            obs, *_ = env.step(scripted.action)
            frame += 1
        state = parse_game_state(env.get_ram(), frame=frame)
        png = save_rgb_png(obs, RECORDINGS_DIR / "boot_mute_city.png")
        print(
            f"RACE_READY frame={frame} mode={state.mode.name} "
            f"speed_raw={state.extras['speed_raw']} "
            f"lateral={state.extras['lateral']} screenshot={png}"
        )
        if state.mode is not GameMode.PLAYING:
            return 1
        if save_race:
            path = save_state(env, GAME_DIR, GAME, "MuteCity")
            print(f"saved {path}")
        return 0
    finally:
        env.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run_probe(save_race=not args.no_save))


if __name__ == "__main__":
    main()

