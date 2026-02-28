#!/usr/bin/env python3
"""
Replay an existing 1-1 recording on both original and extended ROM.
The recording completes original 1-1. On extended ROM it should reach
page 10 (end of first copy) and then continue into the second copy.
"""
import sys, os, json, shutil, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import stable_retro as retro
from retro_harness.env import make_env

ROM_ORIG = ROOT / "super_mario_bros/custom_integrations/SuperMarioBros-Nes-v0/rom.nes"
ROM_EXT = Path(__file__).resolve().parent / "smb_extended_1_1.nes"
GAME_DIR = ROOT / "super_mario_bros"
GAME_NAME = "SuperMarioBros-Nes-v0"
INTEG_DIR = ROOT / "super_mario_bros/custom_integrations/SuperMarioBros-Nes-v0"
RECORDING = ROOT / "super_mario_bros/optimizer/runs/smb_1_1/hillclimb_iter001000_best.json"

RAM_MARIO_X_PAGE = 0x006D
RAM_MARIO_X_POS = 0x0086

def sha1_file(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()

class RomSwapper:
    def __init__(self, rom_path):
        self.rom_path = Path(rom_path).resolve()
        self.rom_file = INTEG_DIR / "rom.nes"
        self.sha_file = INTEG_DIR / "rom.sha"
        self.need_swap = self.rom_path != self.rom_file.resolve()

    def __enter__(self):
        if self.need_swap:
            self.orig_rom = self.rom_file.read_bytes()
            self.orig_sha = self.sha_file.read_text()
            shutil.copy2(self.rom_path, self.rom_file)
            self.sha_file.write_text(sha1_file(self.rom_file))
        return self

    def __exit__(self, *args):
        if self.need_swap:
            self.rom_file.write_bytes(self.orig_rom)
            self.sha_file.write_text(self.orig_sha)


def load_recording(path):
    """Load action recording (list of action indices)."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "actions" in data:
        return data["actions"]
    raise ValueError(f"Unknown recording format in {path}")

# Load the SMB action table
from platformer_common.levels.smb import SMB_ACTIONS as SMB_NES_ACTIONS
from platformer_common.actions import action_index_to_buttons

def replay_on_rom(rom_path, label, actions):
    """Replay action sequence on ROM, track max progress."""
    with RomSwapper(rom_path):
        env = make_env(
            game=GAME_NAME,
            state="Level1_1",
            game_dir=str(GAME_DIR),
        )
        obs, info = env.reset()
        buttons = env.buttons
        num_buttons = len(buttons)

        max_x = 0
        max_page = 0
        page_history = []

        for i, action_idx in enumerate(actions):
            # Convert action index to button array
            button_array = action_index_to_buttons(action_idx, SMB_NES_ACTIONS)
            # Pad/trim to env's button count
            if len(button_array) < num_buttons:
                button_array = button_array + [0] * (num_buttons - len(button_array))
            elif len(button_array) > num_buttons:
                button_array = button_array[:num_buttons]
            obs, reward, terminated, truncated, info = env.step(button_array)

            ram = env.get_ram()
            page = int(ram[RAM_MARIO_X_PAGE])
            x_pos = int(ram[RAM_MARIO_X_POS])
            total_x = page * 256 + x_pos

            if total_x > max_x:
                max_x = total_x
                max_page = page

            if i % 300 == 0:
                page_history.append((i, page, total_x))

            if terminated or truncated:
                print(f"  {label}: Episode ended at frame {i}, max_x={max_x} (page {max_page})")
                break

        # After recording ends, keep running right for extra frames
        # (on extended ROM, the recording completes page 10 gameplay,
        # but there are 10 more pages of content)
        right_idx = buttons.index('RIGHT')
        b_idx = buttons.index('B')
        a_idx = buttons.index('A')

        extra_frames = 12000  # ~200 seconds at 60fps — enough to traverse second half
        for frame in range(extra_frames):
            action = [0] * num_buttons
            action[right_idx] = 1
            action[b_idx] = 1
            if frame % 55 < 20:
                action[a_idx] = 1

            obs, reward, terminated, truncated, info = env.step(action)
            ram = env.get_ram()
            page = int(ram[RAM_MARIO_X_PAGE])
            x_pos = int(ram[RAM_MARIO_X_POS])
            total_x = page * 256 + x_pos

            if total_x > max_x:
                max_x = total_x
                max_page = page

            idx = len(actions) + frame
            if frame % 300 == 0:
                page_history.append((idx, page, total_x))

            if terminated or truncated:
                print(f"  {label}: Died in extra frames at frame {idx}, max_x={max_x} (page {max_page})")
                break

        env.close()
        return {
            'label': label,
            'max_x': max_x,
            'max_page': max_page,
            'total_frames': len(actions) + min(frame + 1, extra_frames),
            'page_history': page_history,
        }


print("=" * 60)
print("REPLAY EVALUATION: 1-1 Recording on Original vs Extended ROM")
print("=" * 60)

# Load recording
print(f"\nLoading recording: {RECORDING}")
actions = load_recording(RECORDING)
print(f"Recording length: {len(actions)} frames ({len(actions)/60:.1f}s)")

# Replay on original
print("\n--- Original ROM ---")
result_orig = replay_on_rom(ROM_ORIG, "Original", actions)
print(f"  Final: max_x={result_orig['max_x']}, page={result_orig['max_page']}")

# Replay on extended
print("\n--- Extended ROM ---")
result_ext = replay_on_rom(ROM_EXT, "Extended", actions)
print(f"  Final: max_x={result_ext['max_x']}, page={result_ext['max_page']}")

# Compare
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"Original: max page {result_orig['max_page']}, max_x {result_orig['max_x']}")
print(f"Extended: max page {result_ext['max_page']}, max_x {result_ext['max_x']}")

if result_ext['max_page'] > result_orig['max_page']:
    print(f"\nPASS: Extended ROM has MORE playable content!")
    print(f"  Bot reached page {result_ext['max_page']} vs original page {result_orig['max_page']}")
    print(f"  This confirms the second half of the extended level is present and playable")
elif result_ext['max_page'] >= 10:
    print(f"\nPASS: Extended ROM plays through first half correctly")
    print(f"  Both reach page {result_ext['max_page']}")
elif result_ext['max_page'] == result_orig['max_page']:
    print(f"\nINCONCLUSIVE: Same page reached ({result_ext['max_page']})")
    print("  Recording may end at flagpole before reaching extended content")
else:
    print(f"\nFAIL: Extended ROM reaches LESS content")

# Page progression
print("\nPage progression (extended):")
for frame, page, x in result_ext['page_history'][-20:]:
    print(f"  Frame {frame:6d}: page={page:2d}, x={x}")
