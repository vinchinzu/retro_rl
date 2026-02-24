#!/usr/bin/env python3
"""Export Super Mario Bros states from stable-retro built-ins to custom_integrations.

Also copies the ROM if found in ../Mario/ or imports it from stable-retro.

Usage:
    uv run python super_mario_bros/setup_states.py
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
from pathlib import Path

import stable_retro as retro


GAME = "SuperMarioBros-Nes-v0"
SCRIPT_DIR = Path(__file__).resolve().parent
CUSTOM_DIR = SCRIPT_DIR / "custom_integrations" / GAME

# Map our state names → built-in state names
# Built-in states: Level1-1, Level1-4, Level2-1, Level3-1, Level4-1, ...
# We rename Level1-1 → Level1_1 (underscores, our convention)
BUILTIN_STATES = {
    "Level1_1": "Level1-1",
    "Level4_1": "Level4-1",
    "Level8_1": "Level8-1",
}

# States we don't have built-ins for (user will record these manually):
# Level1_2, Level4_2, Level8_2


def find_rom() -> Path | None:
    """Look for the SMB ROM in common locations."""
    candidates = [
        SCRIPT_DIR.parent.parent / "Mario" / "Super Mario Bros..nes",
        SCRIPT_DIR.parent / "Mario" / "Super Mario Bros..nes",
        Path.home() / "roms" / "Super Mario Bros..nes",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def setup_rom() -> bool:
    """Copy ROM and create rom.sha in custom_integrations."""
    rom_dest = CUSTOM_DIR / "rom.nes"
    sha_dest = CUSTOM_DIR / "rom.sha"

    if rom_dest.exists() and sha_dest.exists():
        print(f"  ROM already set up: {rom_dest}")
        return True

    rom_path = find_rom()
    if rom_path is None:
        print("  ERROR: SMB ROM not found!")
        print("  Expected at: ../Mario/Super Mario Bros..nes")
        print("  Copy your ROM to the custom_integrations directory as rom.nes")
        return False

    # Copy ROM
    shutil.copy2(rom_path, rom_dest)
    print(f"  Copied ROM from {rom_path}")

    # Generate SHA
    with open(rom_dest, "rb") as f:
        sha = hashlib.sha1(f.read()).hexdigest()
    sha_dest.write_text(sha)
    print(f"  ROM SHA1: {sha}")
    return True


def export_states() -> int:
    """Export built-in states to custom_integrations."""
    exported = 0

    for our_name, builtin_name in BUILTIN_STATES.items():
        dest = CUSTOM_DIR / f"{our_name}.state"
        if dest.exists():
            print(f"  {our_name}: already exists, skipping")
            exported += 1
            continue

        try:
            # Load from built-in integrations
            env = retro.make(
                GAME,
                state=builtin_name,
                render_mode="rgb_array",
                inttype=retro.data.Integrations.ALL,
            )
            env.reset()
            state_data = env.em.get_state()
            env.close()

            # Save as gzipped state
            with gzip.open(dest, "wb") as f:
                f.write(state_data)
            print(f"  {our_name}: exported from built-in '{builtin_name}'")
            exported += 1
        except Exception as e:
            print(f"  {our_name}: FAILED - {e}")

    return exported


def verify_ram(state_name: str) -> None:
    """Quick RAM readout to verify a state works."""
    from retro_harness.env import make_env

    env = make_env(GAME, state_name, SCRIPT_DIR, render_mode="rgb_array")
    env.reset()
    ram = env.get_ram()

    x_page = ram[0x006D]
    x_offset = ram[0x0086]
    player_x = x_page * 256 + x_offset
    player_y = ram[0x00CE]
    lives = ram[0x075A]
    world = ram[0x075F]
    level = ram[0x0760]

    print(f"    world={world+1}-{level+1}  pos=({player_x}, {player_y})  lives={lives}")
    env.close()


def main():
    print(f"=== Super Mario Bros Setup ===\n")
    print(f"Custom integrations: {CUSTOM_DIR}\n")

    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: ROM
    print("[1] ROM setup")
    if not setup_rom():
        print("\nSetup incomplete - ROM needed. Exiting.")
        return

    # Step 2: Export built-in states
    print(f"\n[2] Exporting built-in states")
    exported = export_states()
    print(f"  {exported}/{len(BUILTIN_STATES)} states exported")

    # Step 3: Verify
    print(f"\n[3] Verifying states")
    for name in sorted(BUILTIN_STATES.keys()):
        state_path = CUSTOM_DIR / f"{name}.state"
        if state_path.exists():
            print(f"  {name}:")
            try:
                verify_ram(name)
            except Exception as e:
                print(f"    VERIFY FAILED: {e}")

    # Step 4: Summary
    all_states = sorted(p.stem for p in CUSTOM_DIR.glob("*.state"))
    print(f"\n[4] All states in custom_integrations:")
    for s in all_states:
        print(f"  - {s}")

    missing = {"Level1_2", "Level4_2", "Level8_2"} - set(all_states)
    if missing:
        print(f"\nStates to record manually: {sorted(missing)}")
        print(f"  Use: uv run python -m platformer_common -l smb_1_1 play")
        print(f"  Then F5 to save state when you reach the desired level")

    print(f"\nDone! Try:")
    print(f"  uv run python -m platformer_common list-levels")
    print(f"  uv run python -m platformer_common -l smb_1_1 selftest")


if __name__ == "__main__":
    main()
