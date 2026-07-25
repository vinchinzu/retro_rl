"""Headless boot probe: mash menus and save fight-ready Stage1.state."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env, save_state
from snes_oneshot.actions import buttons, idle_action
from snes_oneshot.game_state import GameMode
from snes_oneshot.segment_runner import configure_headless, save_rgb_png
from tmnt_iv.menus import boot_to_stage1_script
from tmnt_iv.paths import GAME, GAME_DIR, INTEGRATION_DIR, RECORDINGS_DIR
from tmnt_iv.ram import (
    ADDR_EVENT,
    ADDR_LIVES,
    ADDR_MENU,
    MenuId,
    OFF_HP,
    PLAYER_BASE,
    parse_game_state,
    read_u8,
)


def _entered_stage(menu: int, hp: int, lives: int, event: int) -> bool:
    """True once menus are done and Stage 1 gameplay RAM is live."""
    return (
        menu == MenuId.PLAYING
        and 48 <= hp <= 96
        and lives >= 1
        and event >= 0x0A
    )


def run_probe(
    *,
    max_frames: int = 3600,
    save_stage1: bool = True,
    settle_frames: int = 200,
    spawn_walk_frames: int = 60,
) -> int:
    """Boot TMNT IV past intro/char select; save fight-ready Stage1."""
    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        script = list(boot_to_stage1_script())
        script_i = 0
        in_stage = False
        last_key = (-1, -1, -1)
        for frame in range(max_frames):
            ram = env.get_ram()
            menu = read_u8(ram, ADDR_MENU)
            hp = read_u8(ram, PLAYER_BASE + OFF_HP)
            lives = read_u8(ram, ADDR_LIVES)
            event = read_u8(ram, ADDR_EVENT)
            state = parse_game_state(ram, frame=frame)
            key = (menu, hp, event)
            if key != last_key:
                print(
                    f"frame={frame} menu=0x{menu:02X} "
                    f"event=0x{event:02X} mode={state.mode.name} "
                    f"hp={hp} lives={lives} "
                    f"pos=({state.player_x},{state.player_y}) "
                    f"enemies={len(state.living_enemies)}"
                )
                last_key = key

            if not in_stage and _entered_stage(menu, hp, lives, event):
                in_stage = True
                print(f"ENTER_STAGE frame={frame}")
                for _ in range(settle_frames):
                    env.step(idle_action())
                    frame += 1
                # Walk until the first Foot Clan wave spawns (prefer 2+).
                for _ in range(spawn_walk_frames):
                    env.step(buttons("RIGHT"))
                    frame += 1
                    state = parse_game_state(env.get_ram(), frame=frame)
                    if len(state.living_enemies) >= 2:
                        break
                if len(state.living_enemies) < 2:
                    for _ in range(90):
                        env.step(idle_action())
                        frame += 1
                        state = parse_game_state(
                            env.get_ram(), frame=frame
                        )
                        if len(state.living_enemies) >= 2:
                            break
                obs = env.render()
                state = parse_game_state(env.get_ram(), frame=frame)
                if save_stage1 and (
                    state.living_enemies or state.health > 0
                ):
                    path = save_state(env, GAME_DIR, GAME, "Stage1")
                    png = RECORDINGS_DIR / "boot_stage1.png"
                    if obs is not None:
                        save_rgb_png(obs, png)
                    print(f"FIGHT_READY frame={frame}")
                    print(f"saved {path}")
                    print(
                        f"snapshot hp={state.health} lives={state.lives} "
                        f"pos=({state.player_x},{state.player_y}) "
                        f"enemies={len(state.living_enemies)} "
                        f"mode={state.mode.name}"
                    )
                    ok = (
                        state.mode is GameMode.PLAYING
                        and state.health > 0
                    )
                    return 0 if ok else 1
                return 0

            if in_stage:
                action = idle_action()
            elif script_i < len(script):
                action = script[script_i].action
                script_i += 1
            else:
                # Keep advancing menus / intro if the fixed script ends.
                action = (
                    buttons("START") if frame % 40 < 10 else idle_action()
                )
            step = env.step(action)
            obs = step[0]

        print(
            "probe finished without Stage1 "
            f"(in_stage={in_stage} integration={INTEGRATION_DIR})"
        )
        return 1
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
        run_probe(
            max_frames=args.max_frames,
            save_stage1=not args.no_save,
        )
    )


if __name__ == "__main__":
    main()
