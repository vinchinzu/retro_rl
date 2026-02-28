#!/usr/bin/env python3
"""
Render a video of the extended 1-1 level being played.

Uses the optimized 1-1 recording to play through the first copy,
then continues with a simple run-right bot into the second copy.
Outputs MP4 with HUD overlay showing frame count, page, and progress.
"""
from __future__ import annotations

import json
import shutil
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import stable_retro as retro
from retro_harness.env import make_env

# Import after sys.path setup
import platformer_common.levels  # noqa: register all levels
from platformer_common.levels.smb import SMB_ACTIONS
from platformer_common.actions import action_index_to_buttons
from platformer_common.record_video import draw_text

ROM_EXT = Path(__file__).resolve().parent / "smb_extended_1_1.nes"
GAME_DIR = ROOT / "super_mario_bros"
GAME_NAME = "SuperMarioBros-Nes-v0"
INTEG_DIR = ROOT / "super_mario_bros/custom_integrations/SuperMarioBros-Nes-v0"
RECORDING = ROOT / "super_mario_bros/optimizer/runs/smb_1_1/hillclimb_iter001000_best.json"
OUTPUT = Path(__file__).resolve().parent / "extended_1_1.mp4"

# SMB RAM
RAM_X_PAGE = 0x006D
RAM_X_OFFSET = 0x0086
RAM_PLAYER_Y = 0x00CE
RAM_LIVES = 0x075A
RAM_GAME_MODE = 0x0770

# Action labels for SMB_ACTIONS (11 actions)
ACTION_LABELS = [
    "---",     # 0: NOTHING
    "R",       # 1: RIGHT
    "R+B",     # 2: RIGHT+B (run)
    "R+B+A",   # 3: RIGHT+B+A (run+jump)
    "R+A",     # 4: RIGHT+A (walk+jump)
    "A",       # 5: JUMP
    "L",       # 6: LEFT
    "L+B",     # 7: LEFT+B
    "L+B+A",   # 8: LEFT+B+A
    "L+A",     # 9: LEFT+A
    "DOWN",    # 10: DOWN
]


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
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "actions" in data:
        return data["actions"]
    raise ValueError(f"Unknown recording format: {path}")


def render_extended_video(output: str = str(OUTPUT), scale: int = 3):
    actions = load_recording(RECORDING)
    print(f"Recording: {len(actions)} frames ({len(actions)/60:.1f}s)")

    with RomSwapper(ROM_EXT):
        env = make_env(
            game=GAME_NAME,
            state="Level1_1",
            game_dir=str(GAME_DIR),
            render_mode="rgb_array",
        )
        obs, _ = env.reset()
        h, w = obs.shape[:2]
        out_h, out_w = h * scale, w * scale
        buttons = env.buttons
        num_buttons = len(buttons)
        right_idx = buttons.index('RIGHT')
        b_idx = buttons.index('B')
        a_idx = buttons.index('A')

        # Start ffmpeg
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}",
            "-pix_fmt", "rgb24", "-r", "60",
            "-i", "-",
            "-c:v", "libx264", "-preset", "slow", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output,
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        def write_frame(f):
            try:
                proc.stdin.write(f.tobytes())
            except BrokenPipeError:
                pass

        max_x = 0
        max_page = 0
        frame_idx = 0
        phase = "RECORDING"
        died = False

        # Phase 1: Replay the optimized recording
        print(f"Phase 1: Replaying {len(actions)}-frame recording...")
        for action_idx in actions:
            btn = action_index_to_buttons(action_idx, SMB_ACTIONS)
            if len(btn) < num_buttons:
                btn = btn + [0] * (num_buttons - len(btn))
            elif len(btn) > num_buttons:
                btn = btn[:num_buttons]

            obs, reward, terminated, truncated, info = env.step(btn)
            ram = env.get_ram()
            page = int(ram[RAM_X_PAGE])
            x_off = int(ram[RAM_X_OFFSET])
            total_x = page * 256 + x_off
            lives = int(ram[RAM_LIVES])

            if total_x > max_x:
                max_x = total_x
                max_page = page

            # Render frame
            frame = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1).copy()
            secs = frame_idx / 60.0
            label = ACTION_LABELS[action_idx] if action_idx < len(ACTION_LABELS) else str(action_idx)

            draw_text(frame, f"F:{frame_idx}", 4, 4, (255, 255, 255))
            draw_text(frame, f"T:{secs:.1f}S", 4, 12, (200, 200, 200))
            draw_text(frame, f"BTN:{label}", 4, 20, (100, 255, 100))
            draw_text(frame, f"PG:{page} X:{total_x}", 4, 28, (180, 180, 255))

            # Phase indicator
            draw_text(frame, "REPLAY", out_w - 40, 4, (0, 200, 255))

            write_frame(frame)
            frame_idx += 1

            if terminated or truncated:
                died = True
                break

        print(f"  After recording: page={max_page}, max_x={max_x}, died={died}")

        # Phase 2: Run-right bot into the second half
        if not died:
            phase = "BOT"
            extra_frames = 12000  # ~200s at 60fps
            print(f"Phase 2: Bot running right for up to {extra_frames} frames...")
            stuck_frames = 0
            jump_cooldown = 0

            for ef in range(extra_frames):
                action = [0] * num_buttons
                action[right_idx] = 1
                action[b_idx] = 1

                # Smarter jumping
                if jump_cooldown > 0:
                    action[a_idx] = 1
                    jump_cooldown -= 1
                elif stuck_frames > 15:
                    action[a_idx] = 1
                    jump_cooldown = 25
                elif ef % 55 < 20:
                    action[a_idx] = 1

                obs, reward, terminated, truncated, info = env.step(action)
                ram = env.get_ram()
                page = int(ram[RAM_X_PAGE])
                x_off = int(ram[RAM_X_OFFSET])
                total_x = page * 256 + x_off

                if total_x > max_x:
                    max_x = total_x
                    max_page = page
                    stuck_frames = 0
                else:
                    stuck_frames += 1

                # Render
                frame = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1).copy()
                secs = frame_idx / 60.0
                btn_label = "R+B+A" if action[a_idx] else "R+B"

                draw_text(frame, f"F:{frame_idx}", 4, 4, (255, 255, 255))
                draw_text(frame, f"T:{secs:.1f}S", 4, 12, (200, 200, 200))
                draw_text(frame, f"BTN:{btn_label}", 4, 20, (100, 255, 100))
                draw_text(frame, f"PG:{page} X:{total_x}", 4, 28, (180, 180, 255))

                # Phase indicator
                if page > 10:
                    draw_text(frame, "2ND HALF", out_w - 50, 4, (255, 200, 0))
                else:
                    draw_text(frame, "BOT", out_w - 25, 4, (200, 200, 200))

                write_frame(frame)
                frame_idx += 1

                if terminated or truncated:
                    # Death sequence
                    print(f"  Bot died at frame {frame_idx}, page={max_page}, max_x={max_x}")
                    for _ in range(60):
                        obs, *_ = env.step([0] * num_buttons)
                        frame = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1).copy()
                        draw_text(frame, f"F:{frame_idx}", 4, 4, (255, 255, 255))
                        draw_text(frame, f"PG:{max_page} X:{max_x}", 4, 28, (180, 180, 255))
                        draw_text(frame, "DEAD", out_w - 30, 4, (255, 0, 0))
                        write_frame(frame)
                        frame_idx += 1
                    break

                if stuck_frames > 900:
                    print(f"  Bot stuck at frame {frame_idx}, page={max_page}")
                    break

        env.close()

    # Finalize
    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()

    if proc.returncode != 0:
        print(f"ffmpeg exited with code {proc.returncode}")
        sys.exit(1)

    out_path = Path(output)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nDone! {output} ({size_mb:.1f} MB)")
    print(f"Total frames: {frame_idx} ({frame_idx/60:.1f}s)")
    print(f"Max page reached: {max_page}, max_x: {max_x}")

    if size_mb > 10:
        print("Re-encoding with higher CRF...")
        temp = output + ".tmp.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", output,
            "-c:v", "libx264", "-preset", "slow", "-crf", "28",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", temp
        ], check=True, capture_output=True)
        Path(temp).rename(output)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"Re-encoded: {size_mb:.1f} MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render extended 1-1 replay video")
    parser.add_argument("-o", "--output", default=str(OUTPUT), help="Output MP4 path")
    parser.add_argument("--scale", type=int, default=3, help="Pixel scale (default 3)")
    args = parser.parse_args()
    render_extended_video(args.output, args.scale)
