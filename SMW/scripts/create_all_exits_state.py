"""Create an SMW all-exits overworld stable-retro state.

stable-retro does not reliably feed the local ``rom.srm`` into a raw SMW boot
in this workspace, so this script boots to player select, patches the verified
save buffer into WRAM, lets SMW load it through its normal
``LoadSaveBufferData`` path, then saves an overworld state.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from retro_harness.env import make_env, save_state
from SMW.scripts.create_all_exits_sram import build_save_buffer


GAME = "SuperMarioWorld-Snes-v0"
GAME_DIR = "SMW"
WRAM_PAGE_SIZE = 0x2000
INTRO_LEVEL_FLAG = 0x0109
GAME_MODE = 0x0100
EXITS_COMPLETED = 0x1F2E
SAVE_DATA_BUFFER = 0x1F49
SAVE_DATA_BUFFER_EXITS = 0x1FD5


def button_array(env, *names: str) -> np.ndarray:
    action = np.zeros(env.action_space.shape, dtype=np.int8)
    buttons = list(env.unwrapped.buttons)
    for name in names:
        action[buttons.index(name)] = 1
    return action


def patch_save_buffer(env) -> None:
    ram = env.get_ram()
    state = bytearray(env.em.get_state())
    state_ram_base = state.find(bytes(ram[:WRAM_PAGE_SIZE]))
    if state_ram_base < 0:
        raise RuntimeError("could not locate the first WRAM page in emulator state")

    state[state_ram_base + INTRO_LEVEL_FLAG] = 0x00
    save_buffer = build_save_buffer()
    state[
        state_ram_base + SAVE_DATA_BUFFER : state_ram_base + SAVE_DATA_BUFFER + len(save_buffer)
    ] = save_buffer
    env.em.set_state(bytes(state))


def create_state(state_name: str, max_frames: int) -> Path:
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    env.reset()

    zero = np.zeros(env.action_space.shape, dtype=np.int8)
    start = button_array(env, "START")
    patched = False
    try:
        for frame in range(1, max_frames + 1):
            ram = env.get_ram()
            mode = int(ram[GAME_MODE])

            if mode == 0x0A and not patched:
                patch_save_buffer(env)
                patched = True
                action = start
            elif mode in (0x07, 0x08, 0x0A) and frame % 30 < 8:
                action = start
            else:
                action = zero

            env.step(action)
            ram = env.get_ram()
            if int(ram[GAME_MODE]) == 0x0E and int(ram[EXITS_COMPLETED]) == 0x60:
                return save_state(env, GAME_DIR, GAME, state_name)

        ram = env.get_ram()
        raise RuntimeError(
            "did not reach all-exits overworld: "
            f"game_mode=0x{int(ram[GAME_MODE]):02X}, "
            f"exits={int(ram[EXITS_COMPLETED])}, "
            f"savebuf_exits={int(ram[SAVE_DATA_BUFFER_EXITS])}, "
            f"patched={patched}"
        )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-name", default="AllExitsComplete")
    parser.add_argument("--max-frames", type=int, default=5000)
    args = parser.parse_args()

    path = create_state(args.state_name, args.max_frames)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
