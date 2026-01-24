#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

import stable_retro as retro


def bk2_to_mp4(bk2_path: str, out_path: str, fps: int = 60, scale: int = 2) -> None:
    if not os.path.exists(bk2_path):
        raise FileNotFoundError(bk2_path)

    movie = retro.Movie(bk2_path)
    movie.step()

    game = movie.get_game()
    state = movie.get_state()

    env = retro.make(
        game=game,
        state=retro.State.NONE,
        use_restricted_actions=retro.Actions.ALL,
        render_mode="rgb_array",
    )
    env.initial_state = state
    obs, _ = env.reset()

    height, width = obs.shape[0], obs.shape[1]
    scale_filter = f"scale=iw*{scale}:ih*{scale}:flags=neighbor"

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-vf",
        scale_filter,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    try:
        # Write initial frame
        proc.stdin.write(obs.tobytes())

        while movie.step():
            keys = [movie.get_key(i, 0) for i in range(env.num_buttons)]
            obs, _, _, _, _ = env.step(keys)
            proc.stdin.write(obs.tobytes())
    finally:
        env.close()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert .bk2 recordings to .mp4")
    parser.add_argument("bk2", help="Path to .bk2 file")
    parser.add_argument("out", help="Output .mp4 path")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bk2_to_mp4(args.bk2, args.out, fps=args.fps, scale=args.scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
