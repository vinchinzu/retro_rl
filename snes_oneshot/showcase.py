"""Generic segmented showcase recording for oneshot ladder games."""

from __future__ import annotations

import importlib
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from retro_harness.env import get_available_states, make_env
from snes_oneshot.ladder import REPO_ROOT, entry_for
from snes_oneshot.recording import FooterLabels, FrameVideoWriter, RecordingSession
from snes_oneshot.recording_footer import FOOTER_HEIGHT
from snes_oneshot.segment_runner import configure_headless


@dataclass(frozen=True)
class ShowcaseClip:
    """One replay segment loaded from a documented checkpoint."""

    label: str
    state: str
    note: str = "Development checkpoint cut"
    hold_frames: int = 30


class ShowcaseGame(Protocol):
    """Game-specific replay hooks consumed by the generic showcase recorder."""

    slug: str
    game: str
    game_dir: Path
    recordings_dir: Path
    players: int
    manifest_format: str
    ending_scope: str

    def intro_lines(self) -> tuple[str, ...]: ...

    def clips(self) -> tuple[ShowcaseClip, ...]: ...

    def footer_labels(
        self,
        env: object,
        action: list[int],
        frame: int,
        fps: float,
        clip: ShowcaseClip,
    ) -> FooterLabels: ...

    def run_clip(
        self,
        clip: ShowcaseClip,
        session: RecordingSession,
        env: object,
    ) -> dict[str, Any]: ...


def load_showcase_game(slug: str) -> ShowcaseGame:
    """Import a game's showcase builder from ``{slug}.showcase``."""
    entry = entry_for(slug)
    module = importlib.import_module(f"{entry.slug}.showcase")
    builder = getattr(module, "build_showcase", None)
    if builder is None:
        raise AttributeError(f"{entry.slug}.showcase is missing build_showcase()")
    game = builder()
    if game.slug != entry.slug:
        raise ValueError(
            f"showcase slug mismatch: expected {entry.slug}, got {game.slug}"
        )
    return game


def default_output_path(game: ShowcaseGame) -> Path:
    """Return the conventional showcase MP4 path for one ladder game."""
    return game.recordings_dir / f"{game.slug}_segmented_completion_showcase.mp4"


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


def title_card_with_footer(
    lines: list[str],
    *,
    width: int = 256,
    height: int = 224,
) -> np.ndarray:
    """Render a title card with the same dimensions as gameplay frames."""
    card = _title_card(width, height, lines)
    footer = np.full((FOOTER_HEIGHT, width, 3), (5, 10, 18), dtype=np.uint8)
    return np.vstack([card, footer])


def record_showcase(
    game: ShowcaseGame,
    output: Path,
    *,
    frame_stride: int = 2,
    scale: int = 2,
    fps: int = 60,
    card_frames: int = 60,
    ending_hold_frames: int = 240,
    max_clips: int | None = None,
) -> dict[str, Any]:
    """Replay a game's showcase clips and write the video plus JSON manifest."""
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    configure_headless()
    clips = game.clips()
    if max_clips is not None:
        clips = clips[:max_clips]
    available = set(get_available_states(game.game, game.game_dir))
    missing = [clip.state for clip in clips if clip.state not in available]
    if missing:
        raise FileNotFoundError("missing showcase states: " + ", ".join(missing))

    writer = FrameVideoWriter(
        output,
        width=256,
        height=224 + FOOTER_HEIGHT,
        fps=fps,
        scale=scale,
    )
    clip_reports: list[dict[str, Any]] = []

    intro = title_card_with_footer(list(game.intro_lines()))
    for _ in range(card_frames * 2):
        writer.write(intro)

    try:
        for index, clip in enumerate(clips):
            card = title_card_with_footer(
                [clip.label, clip.note, f"State: {clip.state}"],
            )
            for _ in range(card_frames):
                writer.write(card)

            env = make_env(
                game.game,
                clip.state,
                game.game_dir,
                render_mode="rgb_array",
                players=game.players if game.players > 1 else None,
            )
            try:
                result = env.reset()
                obs = result[0] if isinstance(result, tuple) else result

                def footer(
                    active_env: object,
                    action: list[int],
                    frame: int,
                    active_fps: float,
                    *,
                    active_clip: ShowcaseClip = clip,
                ) -> FooterLabels:
                    return game.footer_labels(
                        active_env,
                        action,
                        frame,
                        active_fps,
                        active_clip,
                    )

                from snes_oneshot.actions import idle_action_multi

                idle_builder = (
                    (lambda: idle_action_multi(players=game.players))
                    if game.players > 1
                    else None
                )
                session = RecordingSession(
                    env,
                    sink=writer,
                    footer=footer,
                    fps=float(fps),
                    frame_stride=frame_stride,
                    players=game.players,
                    idle_action=idle_builder,
                )
                session.capture(np.asarray(obs), [])
                report = game.run_clip(clip, session, env)
                report["frames_recorded"] = session.frame
                clip_reports.append({**report, "label": clip.label, "state": clip.state})
                hold = (
                    ending_hold_frames
                    if index == len(clips) - 1
                    else clip.hold_frames
                )
                session.idle(hold)
                print(
                    f"{clip.label}: state={clip.state} frames={session.frame} "
                    f"report={report}",
                )
            finally:
                env.close()
    finally:
        writer.close()

    manifest: dict[str, Any] = {
        "format": game.manifest_format,
        "game": game.slug,
        "continuous_run": False,
        "ending_scope": game.ending_scope,
        "silent_capture": True,
        "uses_development_checkpoints": True,
        "footer_button_tracking": True,
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


def recordings_dir_for(slug: str) -> Path:
    """Return a game's conventional recordings directory."""
    return REPO_ROOT / slug / "recordings"
