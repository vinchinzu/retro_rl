"""Headless boot probe: mash menus and report RAM / mode transitions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from final_fight.menus import boot_to_stage1_script
from final_fight.paths import GAME, GAME_DIR, INTEGRATION_DIR
from final_fight.ram import (
    ADDR_GAME_STATUS,
    GameStatus,
    parse_game_state,
    read_u8,
)
from retro_harness.env import make_env, save_state
from snes_oneshot.actions import buttons, idle_action
from snes_oneshot.game_state import GameState


def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")


def _is_settled_play(status: int, state: GameState) -> bool:
    return (
        status == GameStatus.ACTIVE_GAMEPLAY
        and bool(state.extras.get("player_active"))
        and state.health > 0
        and state.player_x > 0
        and state.player_y > 0
    )


def run_probe(
    *,
    max_frames: int = 3600,
    save_stage1: bool = True,
) -> int:
    """Boot Final Fight, drive menus, optionally write Stage1.state."""
    _configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        env.reset()
        script = list(boot_to_stage1_script())
        script_i = 0
        reached_play = False
        last_status = -1
        for frame in range(max_frames):
            ram = env.get_ram()
            status = read_u8(ram, ADDR_GAME_STATUS)
            state = parse_game_state(ram, frame=frame)
            if status != last_status:
                print(
                    f"frame={frame} status=0x{status:02X} "
                    f"mode={state.mode.name} "
                    f"hp={state.health} lives={state.lives} "
                    f"pos=({state.player_x},{state.player_y}) "
                    f"cam={state.camera_x} "
                    f"enemies={len(state.living_enemies)}"
                )
                last_status = status
            settled = _is_settled_play(status, state)
            if settled:
                reached_play = True
            # Save when the first wave is on screen (fight-ready Stage1).
            if save_stage1 and settled and state.living_enemies:
                path = save_state(env, GAME_DIR, GAME, "Stage1")
                print(f"FIGHT_READY frame={frame}")
                print(f"saved {path}")
                print(
                    f"snapshot hp={state.health} "
                    f"pos=({state.player_x},{state.player_y}) "
                    f"cam={state.camera_x} "
                    f"enemies={len(state.living_enemies)}"
                )
                return 0
            if script_i < len(script):
                action = script[script_i].action
                script_i += 1
            elif settled:
                action = buttons("RIGHT")
            else:
                action = idle_action()
            env.step(action)
        print(
            "probe finished without Stage1 "
            f"(reached_play={reached_play} "
            f"integration={INTEGRATION_DIR})"
        )
        return 0 if reached_play else 1
    finally:
        env.close()


def main() -> None:
    """CLI entry for the boot probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=3600)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write Stage1.state when gameplay is reached.",
    )
    args = parser.parse_args()
    raise SystemExit(
        run_probe(max_frames=args.max_frames, save_stage1=not args.no_save)
    )


if __name__ == "__main__":
    main()
