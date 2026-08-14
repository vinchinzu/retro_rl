"""Headless title → fresh file → Hyrule Castle grounds.

Usage:
    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/boot_to_castle.py
    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/boot_to_castle.py --save
    SDL_VIDEODRIVER=dummy uv run python alttp/scripts/boot_to_castle.py --video
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from alttp.paths import (
    FIRST_ACTION_STATE,
    HYRULE_CASTLE_GROUNDS_STATE,
    INTEGRATION_DIR,
    RECORDINGS_DIR,
)
from alttp.startup import (
    boot_past_title_to_castle,
    build_boot_env,
    create_castle_grounds_state,
    snapshot_env,
)
from retro_harness.env import reset_obs, write_state_bytes
from retro_harness.video import FrameVideoWriter

DEFAULT_VIDEO = RECORDINGS_DIR / "boot_to_castle.mp4"

class _RecordingEnv:
    """Proxy that writes every stepped RGB frame to an MP4 sink."""

    def __init__(
        self,
        env: Any,
        writer: FrameVideoWriter,
        *,
        frame_stride: int = 2,
    ) -> None:
        self._env = env
        self._writer = writer
        self._stride = max(1, frame_stride)
        self.frames_seen = 0
        self.last_obs: np.ndarray | None = None

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        obs, info = reset_obs(self._env)
        self._maybe_write(obs)
        return obs, info

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        obs = result[0]
        self._maybe_write(obs)
        return result

    def render(self, *args: Any, **kwargs: Any) -> Any:
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        self._env.close()

    def get_ram(self) -> Any:
        return self._env.get_ram()

    @property
    def em(self) -> Any:
        return self._env.em

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _maybe_write(self, obs: Any) -> None:
        rgb = np.asarray(obs)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            rendered = self._env.render()
            if rendered is None:
                return
            rgb = np.asarray(rendered)
        self.last_obs = rgb
        if self.frames_seen % self._stride == 0:
            if self._writer.frames_written == 0:
                # FrameVideoWriter is sized at construction; first frame ok.
                pass
            self._writer.write(rgb)
        self.frames_seen += 1

def _configure_headless() -> None:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("SDL_SOFTWARE_RENDERER", "1")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write HyruleCastleGrounds.state and FirstAction.state",
    )
    parser.add_argument(
        "--video",
        nargs="?",
        const=str(DEFAULT_VIDEO),
        default=None,
        help=f"Record MP4 (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video FPS (frame_stride=2 → ~30 from 60Hz)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Nearest-neighbor upscale factor",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=RECORDINGS_DIR / "castle_grounds.png",
        help="PNG path for the final frame",
    )
    args = parser.parse_args()
    _configure_headless()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    writer: FrameVideoWriter | None = None
    env: Any = None
    obs: np.ndarray | None = None
    video_path: Path | None = None
    video_frames = 0

    try:
        if args.video is not None:
            video_path = Path(args.video)
            # Probe resolution from a one-frame reset.
            probe = build_boot_env()
            try:
                probe_obs, _ = probe.reset()
                rgb = np.asarray(probe_obs)
                if rgb.ndim != 3:
                    rgb = np.asarray(probe.render())
                height, width = int(rgb.shape[0]), int(rgb.shape[1])
            finally:
                probe.close()
            writer = FrameVideoWriter(
                video_path,
                width=width,
                height=height,
                fps=args.fps,
                scale=args.scale,
            )
            raw = build_boot_env()
            env = _RecordingEnv(raw, writer, frame_stride=2)
            result = boot_past_title_to_castle(env, close=False)
            obs = env.last_obs
            if args.save:
                state = env.em.get_state()
                write_state_bytes(
                    INTEGRATION_DIR / f"{HYRULE_CASTLE_GROUNDS_STATE}.state",
                    state,
                )
                write_state_bytes(
                    INTEGRATION_DIR / f"{FIRST_ACTION_STATE}.state",
                    state,
                )
        elif args.save:
            result = create_castle_grounds_state(also_first_action=True)
        else:
            env = build_boot_env()
            result = boot_past_title_to_castle(env, close=False)
            obs = env.render()
    finally:
        if writer is not None:
            video_frames = writer.frames_written
            writer.close()
        if env is not None:
            env.close()

    snap = result.snapshot
    report = {
        "phase": result.phase,
        "frames": result.frames,
        "game_mode": snap.game_mode,
        "submodule": snap.submodule,
        "screen_id": snap.screen_id,
        "screen_hex": f"0x{snap.screen_id:02X}",
        "indoors": snap.indoors,
        "dark_world": snap.dark_world,
        "link_x": snap.link_x,
        "link_y": snap.link_y,
        "has_control": snap.has_control,
        "on_castle_grounds": snap.on_castle_grounds,
        "saved": bool(args.save),
        "video": str(video_path) if video_path is not None else None,
        "video_frames": video_frames,
    }
    report_path = RECORDINGS_DIR / "boot_to_castle.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    if obs is None and args.save and args.video is None:
        env = build_boot_env("HyruleCastleGrounds")
        try:
            obs, _ = env.reset()
            rendered = env.render()
            if rendered is not None:
                obs = rendered
            snap = snapshot_env(env)
            report["post_load_screen"] = snap.screen_id
            report_path.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
        finally:
            env.close()

    if obs is not None:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(obs)).save(args.screenshot)
        print(f"Wrote {args.screenshot}")
    if video_path is not None:
        print(f"Wrote {video_path}")

    if not report["on_castle_grounds"]:
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
