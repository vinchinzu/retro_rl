"""Interactive state creator for making start states for new levels.

Play the game normally until reaching the desired starting point,
then press F5 to save a state. The state is saved to both the
current directory and custom_integrations for immediate use.

Usage::

    uv run python -m platformer_common.state_creator \\
        --game DonkeyKongCountry-Snes --game-dir donkey_kong_country \\
        --state "1Player.CongoJungle.JungleHijinks.Level1" \\
        --name JungleHijinks

Controls:
    Normal gameplay: keyboard/controller (see retro_harness.controls)
    TAB: toggle turbo (10x speed for skipping intros)
    F5: save state with configured name
    F7/F8: load last saved state
    ESC: quit
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "x11")

ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from retro_harness.env import make_env, save_state
from retro_harness.play_session import PlaySession


def main():
    parser = argparse.ArgumentParser(description="Interactive State Creator")
    parser.add_argument("--game", required=True, help="Game name (e.g., DonkeyKongCountry-Snes)")
    parser.add_argument("--state", required=True, help="Initial state to load (menu or existing level)")
    parser.add_argument("--game-dir", required=True, help="Game directory name")
    parser.add_argument("--name", required=True, help="Name for the saved state (without .state)")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--ram-addrs", nargs="*", help="RAM addresses to show (hex, e.g., 0x003E 0x0575)")
    args = parser.parse_args()

    game_dir = ROOT_DIR / args.game_dir

    env = make_env(
        game=args.game,
        state=args.state,
        game_dir=game_dir,
        render_mode="rgb_array",
    )

    state_name = args.name
    save_count = 0

    # Parse RAM addresses to monitor
    watch_addrs: list[tuple[str, int]] = []
    if args.ram_addrs:
        for addr_str in args.ram_addrs:
            addr = int(addr_str, 16) if addr_str.startswith("0x") else int(addr_str)
            watch_addrs.append((addr_str, addr))

    def on_hud(info: dict) -> list[str]:
        lines = [
            f"State: {state_name} | Saves: {save_count}",
            "F5:save TAB:turbo ESC:quit",
        ]
        if watch_addrs:
            ram = env.get_ram()
            for label, addr in watch_addrs:
                if addr < len(ram):
                    val = int(ram[addr])
                    # Also show u16 if possible
                    if addr + 1 < len(ram):
                        val16 = int(ram[addr]) | (int(ram[addr + 1]) << 8)
                        lines.append(f"  {label}: u8={val} u16={val16}")
                    else:
                        lines.append(f"  {label}: {val}")
        return lines

    session = PlaySession(
        env,
        game_dir=str(game_dir),
        game=args.game,
        scale=args.scale,
        title=f"State Creator: {state_name}",
    )
    session.on_hud = on_hud

    # Override F5 to save with our custom name
    original_key_handler = session.on_key_down

    def on_key(key):
        import pygame

        nonlocal save_count
        if key == pygame.K_F5:
            path = save_state(env, game_dir, args.game, state_name)
            save_count += 1

            # Show RAM values at save point
            ram = env.get_ram()
            print(f"\n[SAVED] State '{state_name}' -> {path}")
            print(f"  Save #{save_count}")
            if watch_addrs:
                for label, addr in watch_addrs:
                    if addr < len(ram):
                        val = int(ram[addr])
                        if addr + 1 < len(ram):
                            val16 = int(ram[addr]) | (int(ram[addr + 1]) << 8)
                            print(f"  {label}: u8={val} (0x{val:02X}) u16={val16}")
                        else:
                            print(f"  {label}: {val} (0x{val:02X})")
            print()
            return True
        return original_key_handler(key)

    session.on_key_down = on_key
    session.run()


if __name__ == "__main__":
    main()
