"""Record the proven TMNT IV segmented completion path to an MP4.

This is deliberately labeled as a development-checkpoint showcase. It is not
evidence of a continuous title-to-ending attempt: Stage 3--5 transitions and
the Stage 9 form handoff use the canonical development states, and low-health
clips may use the documented development heal.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from retro_harness.env import get_available_states, make_env, reset_obs
from retro_harness.actions import idle_action
from retro_harness.segment_runner import configure_headless
from tmnt_iv.assist import EMERGENCY_HP_RESTORE
from tmnt_iv.paths import GAME, GAME_DIR, RECORDINGS_DIR
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.ram import ADDR_EVENT, parse_game_state, read_u8

@dataclass(frozen=True)
class ShowcaseClip:
    """One replay clip loaded from a documented development checkpoint."""

    label: str
    state: str
    max_frames: int
    stop: str = "fixed"
    note: str = "Development checkpoint cut"

def showcase_clips() -> tuple[ShowcaseClip, ...]:
    """Return the stage-ordered, normal-difficulty completion reel."""
    return (
        ShowcaseClip("Stage 1 - Big Apple", "Stage1", 600),
        ShowcaseClip(
            "Stage 1 - Baxter clear",
            "Stage1_Clear_w3_cam11522",
            5600,
            "stage_advance",
        ),
        ShowcaseClip("Stage 2 - Alleycat Blues", "Stage2", 600),
        ShowcaseClip("Stage 2 - Metalhead clear", "Boss2", 4500, "stage_advance"),
        ShowcaseClip("Stage 3 - Sewer Surfin", "Stage3", 600),
        ShowcaseClip("Stage 3 - Rat King clear", "Boss3_low", 1900, "event_0b"),
        ShowcaseClip("Stage 4 - Technodrome", "Stage4", 600),
        ShowcaseClip("Stage 4 - Tokka and Rahzar", "Boss4_low", 800, "event_0b"),
        ShowcaseClip("Stage 5 - Prehistoric", "Stage5", 600),
        ShowcaseClip("Stage 5 - Slash clear", "Boss5_low", 2400, "event_0b"),
        ShowcaseClip("Stage 6 - Skull and Crossbones", "Stage6", 600),
        ShowcaseClip("Stage 6 - Bebop and Rocksteady", "Boss6_low", 1900, "stage_advance"),
        ShowcaseClip("Stage 7 - Wounded Knee", "Stage7", 600),
        ShowcaseClip("Stage 7 - Leatherhead clear", "Boss7_low", 2300, "stage_advance"),
        ShowcaseClip("Stage 8 - Neon Night Riders", "Stage8", 600),
        ShowcaseClip("Stage 8 - Krang clear", "Boss8_hp5", 2500, "stage_advance"),
        ShowcaseClip("Stage 9 - Starbase", "Stage9", 600),
        ShowcaseClip(
            "Stage 9 - Super Shredder form 1",
            "Boss9_mid",
            900,
            note="Sample, then documented checkpoint handoff",
        ),
        ShowcaseClip(
            "Stage 9 - Form 2 handoff",
            "Stage9_Clear",
            300,
            note="Development checkpoint cut",
        ),
        ShowcaseClip(
            "Finale - Super Shredder and ending",
            "Boss9_phase2_low",
            5000,
            "normal_ending",
            "Normal-difficulty ending sequence",
        ),
    )

class _VideoWriter:
    """Small ffmpeg RGB pipe used only by this game-local artifact script."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: int,
        scale: int,
    ) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to record the showcase")
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.scale = scale
        self.frames_written = 0
        self._proc = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width * scale}x{height * scale}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        """Append one HxWx3 RGB frame."""
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg input is closed")
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(f"unexpected frame shape: {rgb.shape}")
        if self.scale > 1:
            rgb = np.repeat(np.repeat(rgb, self.scale, axis=0), self.scale, axis=1)
        self._proc.stdin.write(rgb.tobytes())
        self.frames_written += 1

    def close(self) -> Path:
        """Finalize the MP4 and raise if ffmpeg failed."""
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        stderr = self._proc.stderr.read() if self._proc.stderr is not None else b""
        code = self._proc.wait()
        if code:
            raise RuntimeError(stderr.decode("utf-8", errors="replace"))
        return self.path

def _title_card(width: int, height: int, lines: list[str]) -> np.ndarray:
    """Render a compact disclosure/stage card at emulator resolution."""
    image = Image.new("RGB", (width, height), (5, 8, 18))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=11)
    y = 34
    for index, line in enumerate(lines):
        current = font if index == 0 else small
        wrapped = textwrap.wrap(line, width=34) or [""]
        for part in wrapped:
            box = draw.textbbox((0, 0), part, font=current)
            text_width = box[2] - box[0]
            draw.text(
                ((width - text_width) // 2, y),
                part,
                font=current,
                fill=(235, 240, 255),
            )
            y += 21 if index == 0 else 15
        y += 8
    return np.asarray(image, dtype=np.uint8)

def _should_stop(
    clip: ShowcaseClip,
    *,
    start_stage: int,
    state: Any,
    event: int,
    saw_ending: bool,
) -> bool:
    if clip.stop == "stage_advance":
        return state.stage > start_stage
    if clip.stop == "event_0b":
        return event == 0x0B
    if clip.stop == "normal_ending":
        return saw_ending and state.mode.name == "TITLE"
    return False

def record_showcase(
    output: Path,
    *,
    frame_stride: int = 2,
    scale: int = 2,
    fps: int = 60,
    card_frames: int = 60,
    max_clips: int | None = None,
) -> dict[str, Any]:
    """Replay the canonical clips and write the video plus JSON manifest."""
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    configure_headless()
    clips = showcase_clips()
    if max_clips is not None:
        clips = clips[:max_clips]
    available = set(get_available_states(GAME, GAME_DIR))
    missing = [clip.state for clip in clips if clip.state not in available]
    if missing:
        raise FileNotFoundError("missing showcase states: " + ", ".join(missing))

    writer = _VideoWriter(output, width=256, height=224, fps=fps, scale=scale)
    clip_reports: list[dict[str, Any]] = []
    intro = _title_card(
        256,
        224,
        [
            "TMNT IV SEGMENTED COMPLETION",
            "Development checkpoints; not a continuous run",
            "Normal difficulty ending; silent emulator capture",
        ],
    )
    for _ in range(card_frames * 2):
        writer.write(intro)

    try:
        for clip in clips:
            card = _title_card(256, 224, [clip.label, clip.note, f"State: {clip.state}"])
            for _ in range(card_frames):
                writer.write(card)

            env = make_env(GAME, clip.state, GAME_DIR, render_mode="rgb_array")
            policy = Stage1Policy()
            heals = 0
            saw_ending = False
            try:
                obs, _ = reset_obs(env)
                start = parse_game_state(env.get_ram())
                start_stage = start.stage
                writer.write(obs)
                end = start
                for frame in range(1, clip.max_frames + 1):
                    state = parse_game_state(env.get_ram(), frame=frame)
                    if 0 < state.health < 28:
                        env.set_value("player_hp", EMERGENCY_HP_RESTORE)
                        env.set_value("player_lives", 2)
                        heals += 1
                        state = parse_game_state(env.get_ram(), frame=frame)
                    tick = policy.tick(state)
                    action = tick.action.action if tick.action is not None else idle_action()
                    obs, *_ = env.step(action)
                    end = parse_game_state(env.get_ram(), frame=frame)
                    event = read_u8(env.get_ram(), ADDR_EVENT)
                    saw_ending = saw_ending or end.stage >= 10
                    if frame % frame_stride == 0:
                        writer.write(obs)
                    if _should_stop(
                        clip,
                        start_stage=start_stage,
                        state=end,
                        event=event,
                        saw_ending=saw_ending,
                    ):
                        break
                clip_reports.append(
                    {
                        **asdict(clip),
                        "frames_replayed": frame,
                        "start_stage": start_stage,
                        "end_stage": end.stage,
                        "end_mode": end.mode.name,
                        "end_event": read_u8(env.get_ram(), ADDR_EVENT),
                        "development_heals": heals,
                    }
                )
                print(
                    f"{clip.label}: state={clip.state} frames={frame} "
                    f"stage={start_stage}->{end.stage} heals={heals}"
                )
            finally:
                env.close()
    finally:
        writer.close()

    manifest: dict[str, Any] = {
        "format": "tmnt-iv-segmented-completion-showcase",
        "continuous_run": False,
        "difficulty": "normal",
        "ending_scope": "normal-difficulty ending sequence; not hard-mode true ending",
        "silent_capture": True,
        "uses_development_checkpoints": True,
        "uses_documented_development_heals": True,
        "frame_stride": frame_stride,
        "video_fps": fps,
        "video_frames": writer.frames_written,
        "video": output.name,
        "clips": clip_reports,
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RECORDINGS_DIR / "tmnt_iv_segmented_completion_showcase.mp4",
    )
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--card-frames", type=int, default=60)
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Record only the first N clips (smoke testing)",
    )
    args = parser.parse_args()
    manifest = record_showcase(
        args.output.resolve(),
        frame_stride=args.frame_stride,
        scale=args.scale,
        fps=args.fps,
        card_frames=args.card_frames,
        max_clips=args.max_clips,
    )
    print(f"video={args.output.resolve()}")
    print(f"manifest={manifest['manifest']}")

if __name__ == "__main__":
    main()
