#!/usr/bin/env python3
"""
Record a 3x3 video montage of MK1 matches.

Records 9 individual match videos (all 7 characters represented across
Match 1 and Match 2), then stitches them into a 3x3 grid with ffmpeg
at 2x speed. Adds green/red tint for win/loss and character labels.

Usage:
    python record_montage.py                    # Default: latest model
    python record_montage.py --model path.zip   # Specific model
    python record_montage.py --level 1          # Match 1 only
    python record_montage.py --level 2          # Match 2 only
    python record_montage.py --max-mb 8         # Target max file size
    python record_montage.py --speed 3          # 3x playback speed
"""

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from PIL import Image
from stable_baselines3 import PPO

from fighters_common.fighting_env import (
    DirectRAMReader, DiscreteAction, FightingGameConfig,
    FightingEnv, FrameSkip, FrameStack, GrayscaleResize,
)
from fighters_common.game_configs import get_game_config
import stable_retro as retro

CHARACTERS = ["LiuKang", "Sonya", "JohnnyCage", "Kano", "Raiden", "SubZero", "Scorpion"]
CELL_W, CELL_H = 256, 224  # SNES native resolution
FPS = 15  # Effective FPS after frame skip (60 / 4)

# Opponent mapping for state names (which opponent appears at each level)
MATCH1_OPPONENTS = {
    "LiuKang": "Sonya", "Sonya": "JohnnyCage", "JohnnyCage": "Kano",
    "Kano": "Raiden", "Raiden": "SubZero", "SubZero": "Scorpion",
    "Scorpion": "LiuKang",
}
MATCH2_OPPONENTS = {
    "LiuKang": "Scorpion", "Sonya": "Scorpion", "JohnnyCage": "Sub-Zero",
    "Kano": "Sub-Zero", "Raiden": "Scorpion", "SubZero": "Kano",
    "Scorpion": "Kano",
}
MATCH3_OPPONENTS = {
    "LiuKang": "Sonya", "Sonya": "LiuKang", "JohnnyCage": "Scorpion",
    "Kano": "Scorpion", "Raiden": "Sonya", "SubZero": "Raiden",
    "Scorpion": "Raiden",
}
MATCH4_OPPONENTS = {
    "LiuKang": "Cage", "Sonya": "Raiden", "Raiden": "Cage",
    "SubZero": "LiuKang", "Scorpion": "LiuKang",
}


def build_raw_env(config, game_dir, state_name):
    """Build environment that returns RGB frames for recording."""
    retro.data.Integrations.add_custom_path(str(game_dir / "custom_integrations"))

    base_env = retro.make(
        game=config.game_id,
        state=state_name,
        render_mode="rgb_array",
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        use_restricted_actions=retro.Actions.ALL,
    )

    fight_config = FightingGameConfig(
        max_health=config.max_health,
        health_key=config.health_key,
        enemy_health_key=config.enemy_health_key,
        ram_overrides=config.ram_overrides,
        actions=config.actions,
    )

    env = base_env
    if config.ram_overrides:
        env = DirectRAMReader(env, config.ram_overrides)
    env = FrameSkip(env, n_skip=4)
    env = GrayscaleResize(env, width=84, height=84)
    env = FightingEnv(env, fight_config)
    env = DiscreteAction(env, config.actions)
    env = FrameStack(env, n_frames=4)
    return env, base_env


def record_match(model, config, game_dir, state_name, output_path, max_frames=3000):
    """Record a single match to a video file using ffmpeg pipe."""
    env, base_env = build_raw_env(config, game_dir, state_name)
    obs, info = env.reset()

    # Start ffmpeg process
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{CELL_W}x{CELL_H}",
        "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frames_written = 0
    won = False

    for frame_idx in range(max_frames):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)

        # Get RGB frame from base env
        rgb_frame = base_env.render()
        if rgb_frame is not None:
            # Resize to cell size
            img = Image.fromarray(rgb_frame)
            img = img.resize((CELL_W, CELL_H), Image.NEAREST)
            proc.stdin.write(np.array(img).tobytes())
            frames_written += 1

        if terminated or truncated:
            rw = info.get("rounds_won", 0)
            rl = info.get("rounds_lost", 0)
            won = rw >= 2 and rw > rl
            # Write a few more frames so the ending is visible
            for _ in range(FPS):  # ~1 second
                rgb_frame = base_env.render()
                if rgb_frame is not None:
                    img = Image.fromarray(rgb_frame)
                    img = img.resize((CELL_W, CELL_H), Image.NEAREST)
                    proc.stdin.write(np.array(img).tobytes())
                    frames_written += 1
                base_env.step([0] * 12)
            break

    proc.stdin.close()
    proc.wait()
    env.close()

    return won, frames_written


def pick_matches(level=None, state_dir=None):
    """Pick 9 matches ensuring all 7 characters appear at least once."""
    matches = []

    if level:
        # All 7 chars at one level, skip missing states
        prefix = "Fight" if level == 1 else f"Match{level}"
        for char in CHARACTERS:
            state = f"{prefix}_{char}"
            if state_dir and not (state_dir / f"{state}.state").exists():
                continue
            matches.append((char, state, level))
        # Fill to 9 with random picks
        while len(matches) < 9:
            extra = random.choice(matches)
            matches.append(extra)
    else:
        # Determine available levels (1, 2, and 3 if states exist)
        levels = [1, 2]
        if state_dir:
            m3 = list(state_dir.glob("Match3_*.state"))
            if m3:
                levels.append(3)

        # Guarantee all 7 characters appear at least once
        chars = CHARACTERS[:]
        random.shuffle(chars)
        for char in chars:
            lvl = random.choice(levels)
            prefix = "Fight" if lvl == 1 else f"Match{lvl}"
            state = f"{prefix}_{char}"
            if state_dir and not (state_dir / f"{state}.state").exists():
                state = f"Fight_{char}"
                lvl = 1
            matches.append((char, state, lvl))

        # Fill remaining 2 slots
        for _ in range(2):
            char = random.choice(CHARACTERS)
            lvl = random.choice(levels)
            prefix = "Fight" if lvl == 1 else f"Match{lvl}"
            state = f"{prefix}_{char}"
            if state_dir and not (state_dir / f"{state}.state").exists():
                state = f"Fight_{char}"
                lvl = 1
            matches.append((char, state, lvl))

    random.shuffle(matches)
    return matches[:9]


def get_opponent(char, level):
    """Get the opponent name for a character at a given level."""
    mapping = {1: MATCH1_OPPONENTS, 2: MATCH2_OPPONENTS, 3: MATCH3_OPPONENTS, 4: MATCH4_OPPONENTS}
    return mapping.get(level, {}).get(char, "???")


def stitch_montage(video_files, results, output_path, speed=2.0, max_mb=None):
    """Stitch 9 videos into a 3x3 grid with tint + labels.

    results: list of (char, level, won) for each video
    """
    inputs = []
    filter_parts = []

    for i, vf in enumerate(video_files):
        inputs.extend(["-i", str(vf)])

    for i in range(9):
        char, level, won = results[i]
        opponent = get_opponent(char, level)
        label = f"{char} vs {opponent}"
        match_label = f"M{level}"
        result_text = "WIN" if won else "LOSS"

        # Green tint for wins, red tint for losses
        if won:
            tint = "colorbalance=rs=-0.12:gs=0.12:bs=-0.06"
        else:
            tint = "colorbalance=rs=0.15:gs=-0.12:bs=-0.12"

        # Build per-clip filter: speed, scale, tint, labels
        f = (
            f"[{i}:v]setpts=PTS/{speed},scale={CELL_W}:{CELL_H},"
            f"{tint},"
            f"drawtext=text='{label}':x=5:y=5:fontsize=14:fontcolor=white:borderw=2:bordercolor=black,"
            f"drawtext=text='{match_label}  {result_text}':x=w-tw-5:y=5:fontsize=14:"
            f"fontcolor={'green' if won else 'red'}:borderw=2:bordercolor=black"
            f"[v{i}]"
        )
        filter_parts.append(f)

    # Create rows
    filter_parts.append("[v0][v1][v2]hstack=inputs=3[row0]")
    filter_parts.append("[v3][v4][v5]hstack=inputs=3[row1]")
    filter_parts.append("[v6][v7][v8]hstack=inputs=3[row2]")
    filter_parts.append("[row0][row1][row2]vstack=inputs=3[out]")

    filter_complex = ";".join(filter_parts)

    # Determine encoding params for target file size
    crf = "26"  # Higher CRF = smaller file (default was 20)
    if max_mb:
        # Estimate duration: get longest clip duration
        # Use even higher CRF for smaller targets
        crf = "30" if max_mb <= 8 else "28"

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", crf,
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(output_path),
    ]

    print(f"\nStitching {len(video_files)} videos into 3x3 montage (CRF={crf})...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr[-500:]}")
        return False

    # If over target size, re-encode with higher CRF
    if max_mb and output_path.exists():
        size_mb = output_path.stat().st_size / 1048576
        if size_mb > max_mb:
            print(f"  {size_mb:.1f} MB > {max_mb} MB target, re-encoding with higher CRF...")
            new_crf = str(int(crf) + 4)
            tmp_path = output_path.with_suffix(".tmp.mp4")
            reenc_cmd = [
                "ffmpeg", "-y",
                "-i", str(output_path),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", new_crf,
                "-pix_fmt", "yuv420p",
                str(tmp_path),
            ]
            r2 = subprocess.run(reenc_cmd, capture_output=True, text=True)
            if r2.returncode == 0 and tmp_path.exists():
                tmp_path.rename(output_path)
                print(f"  Re-encoded: {output_path.stat().st_size / 1048576:.1f} MB (CRF={new_crf})")

    return True


def main():
    parser = argparse.ArgumentParser(description="Record MK1 match montage")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--level", type=int, default=None, help="Match level (None=mix)")
    parser.add_argument("--speed", type=float, default=2.0, help="Playback speed multiplier")
    parser.add_argument("--output", type=str, default=None, help="Output filename")
    parser.add_argument("--max-mb", type=float, default=10.0, help="Target max file size in MB")
    args = parser.parse_args()

    config = get_game_config("mk1")
    game_dir = ROOT_DIR / config.game_dir_name
    model_dir = game_dir / "models"
    state_dir = game_dir / "custom_integrations" / config.game_id

    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        candidates = sorted(model_dir.glob("mk1*ppo*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("No model found!")
            return 1
        model_path = candidates[0]

    print(f"Model: {model_path.name}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO.load(str(model_path), device=device)

    # Pick matches (handles missing states internally)
    matches = pick_matches(args.level, state_dir=state_dir)

    # Create temp dir for individual videos
    tmp_dir = game_dir / "_montage_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"\nRecording 9 matches:")
    print("=" * 60)

    video_files = []
    results = []

    for i, (char, state_name, level) in enumerate(matches):
        opponent = get_opponent(char, level)
        print(f"  [{i+1}/9] {char:<12} vs {opponent:<12} (M{level}) ", end="", flush=True)

        video_path = tmp_dir / f"match_{i:02d}.mp4"
        won, frames = record_match(model, config, game_dir, state_name, video_path)
        video_files.append(video_path)

        status = "\033[32mWIN\033[0m" if won else "\033[31mLOSS\033[0m"
        results.append((char, level, won))
        print(f"{status} ({frames} frames)")

    # Stitch montage
    print("=" * 60)
    wins = sum(1 for _, _, w in results if w)
    print(f"Results: {wins}/9 wins")

    montage_dir = game_dir / "montages"
    montage_dir.mkdir(exist_ok=True)
    output_name = args.output or f"mk1_montage_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    output_path = montage_dir / output_name

    if stitch_montage(video_files, results, output_path, speed=args.speed, max_mb=args.max_mb):
        # Get file size
        size_mb = output_path.stat().st_size / 1048576
        print(f"\nMontage saved: {output_path}")
        print(f"Size: {size_mb:.1f} MB {'(under target)' if args.max_mb and size_mb <= args.max_mb else ''}")
        print(f"Speed: {args.speed}x")
        print(f"Grid: 3x3 ({CELL_W*3}x{CELL_H*3})")
    else:
        print("Failed to create montage!")

    # Cleanup temp files
    for vf in video_files:
        if vf.exists():
            vf.unlink()
    if tmp_dir.exists():
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
