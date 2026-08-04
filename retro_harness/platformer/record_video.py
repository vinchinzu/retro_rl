"""Record an action sequence replay to MP4 with HUD overlay (frame counter + buttons)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# Trigger level registration
import retro_harness.platformer.levels  # noqa: F401

from retro_harness.platformer.actions import action_index_to_buttons, DEFAULT_PLATFORMER_ACTIONS
from retro_harness.platformer.bk2_extract import load_actions
from retro_harness.platformer.level_config import get_level_config
from retro_harness.env import make_env


# Action label lookup for DKC_SPEED_ACTIONS (10 actions)
DKC_SPEED_LABELS = [
    "---",          # 0: NOTHING
    "R+Y",          # 1: RIGHT + Y (run right)
    "R+Y+B",        # 2: RIGHT + Y + B (run + jump)
    "B",            # 3: JUMP
    "L+Y",          # 4: LEFT + Y (run left)
    "L+Y+B",        # 5: LEFT + Y + B (run left + jump)
    "DOWN",         # 6: DOWN
    "UP",           # 7: UP
    "R",            # 8: RIGHT (Y-release)
    "L",            # 9: LEFT (Y-release)
]

DEFAULT_LABELS = [
    "---",          # 0: NOTHING
    "R",            # 1: RIGHT
    "R+Y",          # 2: RIGHT + Y
    "R+Y+B",        # 3: RIGHT + Y + B
    "R+B",          # 4: RIGHT + B
    "B",            # 5: JUMP
    "L",            # 6: LEFT
    "L+Y",          # 7: LEFT + Y
    "L+Y+B",        # 8: LEFT + Y + B
    "L+B",          # 9: LEFT + B
    "DOWN",         # 10: DOWN
    "A",            # 11: A
    "R+A",          # 12: RIGHT + A
    "UP",           # 13: UP
]


def draw_text(frame: np.ndarray, text: str, x: int, y: int,
              color: tuple = (255, 255, 255), shadow: bool = True) -> None:
    """Draw text on frame using a simple 4x6 bitmap font."""
    FONT = {
        '0': ["0110", "1001", "1001", "1001", "0110"],
        '1': ["0010", "0110", "0010", "0010", "0111"],
        '2': ["0110", "1001", "0010", "0100", "1111"],
        '3': ["1110", "0001", "0110", "0001", "1110"],
        '4': ["1001", "1001", "1111", "0001", "0001"],
        '5': ["1111", "1000", "1110", "0001", "1110"],
        '6': ["0110", "1000", "1110", "1001", "0110"],
        '7': ["1111", "0001", "0010", "0100", "0100"],
        '8': ["0110", "1001", "0110", "1001", "0110"],
        '9': ["0110", "1001", "0111", "0001", "0110"],
        'A': ["0110", "1001", "1111", "1001", "1001"],
        'B': ["1110", "1001", "1110", "1001", "1110"],
        'D': ["1110", "1001", "1001", "1001", "1110"],
        'L': ["1000", "1000", "1000", "1000", "1111"],
        'N': ["1001", "1101", "1011", "1001", "1001"],
        'O': ["0110", "1001", "1001", "1001", "0110"],
        'P': ["1110", "1001", "1110", "1000", "1000"],
        'R': ["1110", "1001", "1110", "1010", "1001"],
        'U': ["1001", "1001", "1001", "1001", "0110"],
        'W': ["1001", "1001", "1011", "1101", "1001"],
        'Y': ["1001", "1001", "0110", "0010", "0010"],
        '+': ["0000", "0100", "1110", "0100", "0000"],
        '-': ["0000", "0000", "1110", "0000", "0000"],
        ' ': ["0000", "0000", "0000", "0000", "0000"],
        ':': ["0000", "0100", "0000", "0100", "0000"],
        '/': ["0001", "0010", "0100", "1000", "0000"],
        '.': ["0000", "0000", "0000", "0000", "0100"],
        'F': ["1111", "1000", "1110", "1000", "1000"],
        'S': ["0111", "1000", "0110", "0001", "1110"],
        'T': ["1110", "0100", "0100", "0100", "0100"],
        'X': ["1001", "0110", "0110", "0110", "1001"],
    }

    h, w = frame.shape[:2]
    cx = x
    for ch in text.upper():
        glyph = FONT.get(ch)
        if glyph is None:
            cx += 5
            continue
        for gy, row in enumerate(glyph):
            for gx, pixel in enumerate(row):
                if pixel == '1':
                    py, px = y + gy, cx + gx
                    if 0 <= py < h and 0 <= px < w:
                        if shadow:
                            # Shadow
                            sy, sx = py + 1, px + 1
                            if 0 <= sy < h and 0 <= sx < w:
                                frame[sy, sx] = (0, 0, 0)
                        frame[py, px] = color
        cx += 5


def record(actions_path: str, level: str, output: str, scale: int = 3) -> None:
    config = get_level_config(level)
    actions = load_actions(Path(actions_path))
    action_table = config.action_table or DEFAULT_PLATFORMER_ACTIONS
    num_actions = len(action_table)

    # Pick labels
    if num_actions == 10:
        labels = DKC_SPEED_LABELS
    elif num_actions == 14:
        labels = DEFAULT_LABELS
    else:
        labels = [str(i) for i in range(num_actions)]

    env = make_env(
        game=config.game_name,
        state=config.start_state,
        game_dir=config.game_dir,
        render_mode="rgb_array",
    )
    obs, _ = env.reset()
    h, w = obs.shape[0], obs.shape[1]
    out_h, out_w = h * scale, w * scale

    schema = config.ram_schema
    ram = env.get_ram()
    initial_values = schema.read(ram)
    initial_cam_x = float(initial_values.get("camera_x", 0))
    initial_lives = initial_values.get("lives")

    # Use ffmpeg pipe for encoding
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "rgb24",
        "-r", "60",
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output,
    ]

    print(f"Recording {len(actions)} frames at {out_w}x{out_h} to {output}")
    print(f"Level: {config.display_name}")
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write_frame(f):
        try:
            proc.stdin.write(f.tobytes())
        except BrokenPipeError:
            pass

    max_progress = 0.0
    for frame_idx, action_idx in enumerate(actions):
        buttons = action_index_to_buttons(action_idx, action_table)
        action_size = env.action_space.shape[0]
        if len(buttons) < action_size:
            buttons = buttons + [0] * (action_size - len(buttons))

        obs, reward, terminated, truncated, info = env.step(
            np.array(buttons, dtype=np.int8)
        )

        ram = env.get_ram()
        values = schema.read(ram)
        cam_x = float(values.get("camera_x", 0))
        lives = values.get("lives", 0)
        level_id = values.get("level_id", 0)

        # Sub-level detection: freeze progress when in bonus room
        # level_id_aliases are part of the same level (not sub-levels)
        _main_ids = {config.target_level_id} | set(config.level_id_aliases)
        in_sub_level = (level_id != 0 and level_id not in _main_ids)

        if not in_sub_level:
            progress = cam_x - initial_cam_x if cam_x > initial_cam_x else 0
            if progress > max_progress:
                max_progress = progress

        # Check completion
        completed = False
        if config.completion_signal == "level_id_change":
            if level_id not in _main_ids and level_id != 0:
                if (max_progress >= config.completion_min_progress
                        and (not config.completion_level_ids
                             or level_id in config.completion_level_ids)
                        and level_id not in config.completion_exclude_ids):
                    completed = True
        elif config.completion_signal == "ram_flag":
            flag_val = values.get(config.completion_ram_key, None)
            if (flag_val is not None
                    and flag_val == config.completion_ram_value
                    and max_progress >= config.completion_min_progress):
                completed = True

        # Check death
        died = False
        if initial_lives is not None and lives < initial_lives:
            died = True

        # Scale frame
        frame = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1).copy()

        # Draw HUD
        btn_label = labels[action_idx] if action_idx < len(labels) else str(action_idx)
        secs = frame_idx / 60.0

        draw_text(frame, f"F:{frame_idx}", 4, 4, (255, 255, 255))
        draw_text(frame, f"T:{secs:.1f}S", 4, 12, (200, 200, 200))
        draw_text(frame, f"BTN:{btn_label}", 4, 20, (100, 255, 100))
        draw_text(frame, f"P:{max_progress:.0f}", 4, 28, (180, 180, 255))

        if in_sub_level:
            draw_text(frame, "BONUS", out_w - 35, 4, (255, 200, 0))
        elif completed:
            draw_text(frame, "DONE", out_w - 30, 4, (0, 255, 0))
        elif died:
            draw_text(frame, "DEAD", out_w - 30, 4, (255, 0, 0))

        write_frame(frame)

        if completed:
            # Write a few more seconds of the completion
            for _ in range(120):
                obs, *_ = env.step(np.zeros(action_size, dtype=np.int8))
                frame = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1).copy()
                draw_text(frame, f"F:{frame_idx}", 4, 4, (255, 255, 255))
                draw_text(frame, "DONE", out_w - 30, 4, (0, 255, 0))
                write_frame(frame)
            break

        if died:
            for _ in range(60):
                obs, *_ = env.step(np.zeros(action_size, dtype=np.int8))
                frame = np.repeat(np.repeat(obs, scale, axis=0), scale, axis=1).copy()
                draw_text(frame, f"F:{frame_idx}", 4, 4, (255, 255, 255))
                draw_text(frame, "DEAD", out_w - 30, 4, (255, 0, 0))
                write_frame(frame)
            break

        if terminated or truncated:
            break

    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    proc.wait()
    if proc.returncode != 0:
        print(f"ffmpeg exited with code {proc.returncode}")
        sys.exit(1)

    env.close()

    # Check file size
    out_path = Path(output)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Done! {output} ({size_mb:.1f} MB)")
    if size_mb > 10:
        print("WARNING: File exceeds 10MB. Re-encoding with higher CRF...")
        temp = output + ".tmp.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", output,
            "-c:v", "libx264", "-preset", "slow", "-crf", "28",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", temp
        ], check=True, capture_output=True)
        Path(temp).rename(output)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"Re-encoded: {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Record replay to MP4 with HUD")
    parser.add_argument("--actions", required=True, help="Path to actions JSON")
    parser.add_argument("--level", "-l", required=True, help="Level ID or alias")
    parser.add_argument("--output", "-o", default="replay.mp4", help="Output MP4 path")
    parser.add_argument("--scale", type=int, default=3, help="Pixel scale (default 3)")
    args = parser.parse_args()
    record(args.actions, args.level, args.output, args.scale)


if __name__ == "__main__":
    main()
