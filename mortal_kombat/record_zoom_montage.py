#!/usr/bin/env python3
"""
Record a 5x5 zoom-out montage of MK1 matches.

Cinematic zoom-out animation:
  1. Start zoomed into center fight (full screen)
  2. Smooth ease-out to reveal 3x3 grid
  3. Continue zooming out to full 5x5 grid

Usage:
    python record_zoom_montage.py                    # Default: latest model
    python record_zoom_montage.py --model path.zip   # Specific model
    python record_zoom_montage.py --max-mb 10        # Target max file size
    python record_zoom_montage.py --speed 3          # 3x playback speed
"""

import argparse
import math
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
CELL_W, CELL_H = 256, 224
GRID = 5
FPS = 15

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


def get_opponent(char, level):
    mapping = {1: MATCH1_OPPONENTS, 2: MATCH2_OPPONENTS, 3: MATCH3_OPPONENTS, 4: MATCH4_OPPONENTS}
    return mapping.get(level, {}).get(char, "???")


def build_raw_env(config, game_dir, state_name):
    retro.data.Integrations.add_custom_path(str(game_dir / "custom_integrations"))
    base_env = retro.make(
        game=config.game_id, state=state_name,
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
    env, base_env = build_raw_env(config, game_dir, state_name)
    obs, info = env.reset()
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{CELL_W}x{CELL_H}", "-pix_fmt", "rgb24", "-r", str(FPS),
        "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p", str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames = 0
    won = False
    for _ in range(max_frames):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        rgb = base_env.render()
        if rgb is not None:
            img = Image.fromarray(rgb).resize((CELL_W, CELL_H), Image.NEAREST)
            proc.stdin.write(np.array(img).tobytes())
            frames += 1
        if terminated or truncated:
            rw = info.get("rounds_won", 0)
            rl = info.get("rounds_lost", 0)
            won = rw >= 2 and rw > rl
            for _ in range(FPS):
                rgb = base_env.render()
                if rgb is not None:
                    img = Image.fromarray(rgb).resize((CELL_W, CELL_H), Image.NEAREST)
                    proc.stdin.write(np.array(img).tobytes())
                    frames += 1
                base_env.step([0] * 12)
            break
    proc.stdin.close()
    proc.wait()
    env.close()
    return won, frames


def pick_25_matches(state_dir):
    """Pick 25 matches for 5x5 grid with good distribution across all levels."""
    pool = []

    def add_state(char, level):
        prefix = "Fight" if level == 1 else f"Match{level}"
        s = f"{prefix}_{char}"
        if (state_dir / f"{s}.state").exists():
            pool.append((char, s, level))
        else:
            pool.append((char, f"Fight_{char}", 1))

    # Round 1: all 7 chars M1
    for c in CHARACTERS:
        add_state(c, 1)
    # Round 2: all 7 chars M2
    for c in CHARACTERS:
        add_state(c, 2)
    # Round 3: M3 states (if available) + fill with M1
    m3_chars = [c for c in CHARACTERS if (state_dir / f"Match3_{c}.state").exists()]
    for c in m3_chars:
        add_state(c, 3)
    remaining = 25 - len(pool)
    for c in random.sample(CHARACTERS, min(remaining, len(CHARACTERS))):
        add_state(c, random.choice([1, 2]))
    while len(pool) < 25:
        add_state(random.choice(CHARACTERS), random.choice([1, 2]))

    # Center cell (index 12) = best available fight
    if (state_dir / "Match3_LiuKang.state").exists():
        center = ("LiuKang", "Match3_LiuKang", 3)
    elif (state_dir / "Match2_LiuKang.state").exists():
        center = ("LiuKang", "Match2_LiuKang", 2)
    else:
        center = ("LiuKang", "Fight_LiuKang", 1)

    # Remove one matching entry to avoid duplicate
    for i, m in enumerate(pool):
        if m[1] == center[1]:
            pool.pop(i)
            break

    random.shuffle(pool)
    matches = pool[:12] + [center] + pool[12:24]
    return matches[:25]


def build_grid_video(clip_files, results, grid_path, speed=2.0):
    """Tile 25 clips into a 5x5 grid video with tint + labels."""
    inputs = []
    parts = []

    for vf in clip_files:
        inputs.extend(["-i", str(vf)])

    for i in range(25):
        char, level, won = results[i]
        opp = get_opponent(char, level)
        label = f"{char} vs {opp}"
        tag = f"M{level}"
        res = "WIN" if won else "LOSS"
        tint = ("colorbalance=rs=-0.12:gs=0.12:bs=-0.06" if won
                else "colorbalance=rs=0.15:gs=-0.12:bs=-0.12")
        fc = "green" if won else "red"

        parts.append(
            f"[{i}:v]setpts=PTS/{speed},scale={CELL_W}:{CELL_H},{tint},"
            f"drawtext=text='{label}':x=5:y=5:fontsize=11:fontcolor=white:borderw=1:bordercolor=black,"
            f"drawtext=text='{tag} {res}':x=w-tw-5:y=5:fontsize=11:fontcolor={fc}:borderw=1:bordercolor=black"
            f"[v{i}]"
        )

    for row in range(5):
        s = row * 5
        streams = "".join(f"[v{s+c}]" for c in range(5))
        parts.append(f"{streams}hstack=inputs=5[row{row}]")

    rows = "".join(f"[row{r}]" for r in range(5))
    parts.append(f"{rows}vstack=inputs=5[out]")

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(parts),
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-shortest",
        str(grid_path),
    ]

    print("  Building 5x5 grid video...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FFmpeg grid error: {r.stderr[-500:]}")
        return False
    return True


def apply_zoom_animation(grid_path, output_path, max_mb=None):
    """Read grid video frame-by-frame, apply smooth zoom-out crop animation."""
    gw = CELL_W * GRID  # 1280
    gh = CELL_H * GRID  # 1120
    frame_size = gw * gh * 3

    # Zoom timeline (seconds into the output video)
    HOLD1 = 2.0   # Hold on center cell
    ZOOM1 = 2.0   # Smooth ease to 3x3
    HOLD3 = 3.0   # Hold on 3x3
    ZOOM2 = 2.0   # Smooth ease to 5x5
    # After HOLD1+ZOOM1+HOLD3+ZOOM2 = 9s: full 5x5

    t_z1_start = HOLD1
    t_z1_end = t_z1_start + ZOOM1
    t_z2_start = t_z1_end + HOLD3
    t_z2_end = t_z2_start + ZOOM2

    # Probe grid video duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(grid_path)],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip())
    total_frames = int(duration * FPS)

    print(f"  Grid: {duration:.1f}s, {total_frames} frames ({gw}x{gh})")
    print(f"  Zoom: center {HOLD1}s -> ease {ZOOM1}s -> 3x3 {HOLD3}s -> ease {ZOOM2}s -> 5x5")

    # Read grid video as raw RGB frames
    reader = subprocess.Popen(
        ["ffmpeg", "-i", str(grid_path),
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-r", str(FPS), "-v", "error", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    crf = "26"
    if max_mb and max_mb <= 8:
        crf = "30"
    elif max_mb and max_mb <= 12:
        crf = "28"

    # Write output video
    writer = subprocess.Popen(
        ["ffmpeg", "-y",
         "-f", "rawvideo", "-vcodec", "rawvideo",
         "-s", f"{gw}x{gh}", "-pix_fmt", "rgb24",
         "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", crf,
         "-pix_fmt", "yuv420p",
         str(output_path)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    cw1, ch1 = CELL_W, CELL_H        # 1 cell: 256x224
    cw3, ch3 = CELL_W * 3, CELL_H * 3  # 3x3: 768x672

    def ease(progress):
        """Smooth ease-in-out (cosine)."""
        return (1 - math.cos(math.pi * progress)) / 2

    processed = 0
    while True:
        raw = reader.stdout.read(frame_size)
        if len(raw) < frame_size:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(gh, gw, 3)
        t = processed / FPS

        # Calculate crop dimensions based on zoom phase
        if t < t_z1_start:
            cw, ch = cw1, ch1
        elif t < t_z1_end:
            p = ease((t - t_z1_start) / ZOOM1)
            cw = int(cw1 + (cw3 - cw1) * p)
            ch = int(ch1 + (ch3 - ch1) * p)
        elif t < t_z2_start:
            cw, ch = cw3, ch3
        elif t < t_z2_end:
            p = ease((t - t_z2_start) / ZOOM2)
            cw = int(cw3 + (gw - cw3) * p)
            ch = int(ch3 + (gh - ch3) * p)
        else:
            cw, ch = gw, gh

        # Ensure even dimensions for encoder
        cw = max(2, cw & ~1)
        ch = max(2, ch & ~1)

        # Crop centered on the grid
        x = (gw - cw) // 2
        y = (gh - ch) // 2
        cropped = frame[y:y + ch, x:x + cw]

        # Scale to full output resolution
        img = Image.fromarray(cropped).resize((gw, gh), Image.LANCZOS)
        writer.stdin.write(np.array(img).tobytes())
        processed += 1

        if processed % (FPS * 5) == 0:
            print(f"    {processed}/{total_frames} frames ({t:.1f}s)")

    reader.stdout.close()
    reader.wait()
    writer.stdin.close()
    writer.wait()

    print(f"  Processed {processed} frames")

    # Re-encode if over target size
    if max_mb and output_path.exists():
        size_mb = output_path.stat().st_size / 1048576
        if size_mb > max_mb:
            print(f"  {size_mb:.1f}MB > {max_mb}MB target, re-encoding...")
            new_crf = str(int(crf) + 4)
            tmp = output_path.with_suffix(".tmp.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(output_path),
                 "-c:v", "libx264", "-preset", "medium",
                 "-crf", new_crf, "-pix_fmt", "yuv420p", str(tmp)],
                capture_output=True,
            )
            if tmp.exists():
                tmp.rename(output_path)
                print(f"  Re-encoded: {output_path.stat().st_size / 1048576:.1f}MB (CRF={new_crf})")

    return output_path.exists()


def main():
    parser = argparse.ArgumentParser(description="Record MK1 5x5 zoom-out montage")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--speed", type=float, default=2.0, help="Playback speed")
    parser.add_argument("--output", type=str, default=None, help="Output filename")
    parser.add_argument("--max-mb", type=float, default=10.0, help="Target max MB")
    args = parser.parse_args()

    config = get_game_config("mk1")
    game_dir = ROOT_DIR / config.game_dir_name
    model_dir = game_dir / "models"
    state_dir = game_dir / "custom_integrations" / config.game_id

    # Find model
    if args.model:
        model_path = Path(args.model)
    else:
        candidates = sorted(
            model_dir.glob("mk1*ppo*.zip"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if not candidates:
            print("No model found!")
            return 1
        model_path = candidates[0]

    print(f"Model: {model_path.name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPO.load(str(model_path), device=device)

    matches = pick_25_matches(state_dir)

    tmp_dir = game_dir / "_zoom_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"\nRecording 25 matches for 5x5 grid:")
    print("=" * 70)

    clip_files = []
    results = []

    for i, (char, state_name, level) in enumerate(matches):
        opp = get_opponent(char, level)
        tag = " [CENTER]" if i == 12 else ""
        print(f"  [{i + 1:2d}/25] {char:<12} vs {opp:<12} (M{level}){tag} ", end="", flush=True)

        vpath = tmp_dir / f"clip_{i:02d}.mp4"
        won, frames = record_match(model, config, game_dir, state_name, vpath)
        clip_files.append(vpath)
        results.append((char, level, won))

        color = "\033[32m" if won else "\033[31m"
        print(f"{color}{'WIN' if won else 'LOSS'}\033[0m ({frames}f)")

    wins = sum(1 for _, _, w in results if w)
    print(f"\n{'=' * 70}")
    print(f"Results: {wins}/25 wins ({wins / 25 * 100:.0f}%)")
    print(f"{'=' * 70}")

    # Stage 1: Build 5x5 grid video
    grid_path = tmp_dir / "grid_5x5.mp4"
    if not build_grid_video(clip_files, results, grid_path, speed=args.speed):
        print("Failed to build grid video!")
        return 1

    # Stage 2: Apply zoom-out animation
    montage_dir = game_dir / "montages"
    montage_dir.mkdir(exist_ok=True)
    output_name = args.output or f"mk1_zoom_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    output_path = montage_dir / output_name

    print(f"\nApplying zoom-out animation...")
    if apply_zoom_animation(grid_path, output_path, max_mb=args.max_mb):
        size_mb = output_path.stat().st_size / 1048576
        print(f"\nZoom montage saved: {output_path}")
        print(f"Size: {size_mb:.1f} MB")
        print(f"Grid: {GRID}x{GRID} ({CELL_W * GRID}x{CELL_H * GRID})")
        print(f"Animation: 1 cell -> 3x3 -> 5x5 (smooth cosine ease)")
    else:
        print("Failed to create zoom montage!")
        return 1

    # Cleanup temp files
    for f in clip_files:
        if f.exists():
            f.unlink()
    if grid_path.exists():
        grid_path.unlink()
    if tmp_dir.exists():
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
