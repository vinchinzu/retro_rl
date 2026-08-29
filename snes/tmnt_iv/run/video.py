"""Full-run video config and the post-credits metric overlay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from retro_harness.video import VideoCaptureConfig, VideoRecorder
from tmnt_iv.assist import EMERGENCY_HP_THRESHOLD
from tmnt_iv.run.metrics import RunMetrics, format_duration


def full_run_video_config(
    *,
    native: bool = False,
    scale: int = 3,
    hq: bool = False,
) -> VideoCaptureConfig:
    """Product capture is 1080p60 YouTube; native is the 16px-footer hatch."""
    if native:
        if hq:
            return VideoCaptureConfig.high_quality(scale=scale)
        return VideoCaptureConfig(
            fps=60,
            scale=scale,
            audio=True,
            footer=True,
            layout="native",
        )
    overrides: dict[str, Any] = {}
    if hq:
        overrides["crf"] = 15
        overrides["preset"] = "slow"
    return VideoCaptureConfig.youtube(**overrides)


def open_full_run_capture(
    output: Path,
    *,
    width: int,
    height: int,
    config: VideoCaptureConfig,
    audio_rate: int,
) -> VideoRecorder:
    """Start the product encoder and log the canvas."""
    capture = VideoRecorder(
        output,
        width=width,
        height=height,
        config=config,
        audio_rate=audio_rate,
    )
    canvas = (
        f"{config.canvas_width}x{config.canvas_height}"
        if config.layout == "youtube"
        else f"{width}x{height}*{config.scale}"
    )
    print(
        f"recording {config.layout} {canvas} {config.fps}fps -> {output}",
        flush=True,
    )
    return capture


def render_credits_overlay(
    obs: np.ndarray,
    *,
    frame: int,
    fps: float,
    metrics: RunMetrics,
) -> np.ndarray:
    """Return native RGB, with a metric card after the credits settle."""
    rgb = np.asarray(obs, dtype=np.uint8)
    complete = metrics.credits_complete_frame
    if complete is None or frame < complete:
        return rgb

    height, width = rgb.shape[:2]
    pil_image = Image.fromarray(rgb, mode="RGB")
    overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    card = ImageDraw.Draw(overlay)
    card.rounded_rectangle(
        (13, 35, width - 13, height - 20),
        radius=7,
        fill=(4, 10, 22, 232),
        outline=(82, 224, 168, 255),
        width=1,
    )
    title = ImageFont.load_default(size=13)
    body = ImageFont.load_default(size=9)
    final_seconds = complete / fps
    lines = [
        ("RUN COMPLETE - HARD CREDITS", title, (128, 255, 196, 255)),
        (
            f"POWER-ON TO CREDITS  {format_duration(final_seconds)}",
            body,
            (242, 247, 255, 255),
        ),
        (
            f"DAMAGE TAKEN         {metrics.total_damage_taken}",
            body,
            (242, 247, 255, 255),
        ),
        ("LIFE LOSSES           0", body, (242, 247, 255, 255)),
        (
            "MIN HP SEEN          "
            f"{metrics.min_health_seen if metrics.min_health_seen is not None else '-'}",
            body,
            (194, 207, 222, 255),
        ),
        (
            "EMERGENCY HEALS      "
            f"{metrics.health_guard_interventions} (hp<={EMERGENCY_HP_THRESHOLD})",
            body,
            (194, 207, 222, 255),
        ),
        (
            "F2 I-FRAME GUARD     "
            f"{metrics.final_boss_iframe_guard_frames}f",
            body,
            (194, 207, 222, 255),
        ),
        ("STATE LOADS 0  |  NO FULL-HP SPAM", body, (194, 207, 222, 255)),
        ("ONE EMULATOR SESSION + NATIVE AUDIO", body, (194, 207, 222, 255)),
    ]
    y = 47
    for text, current_font, color in lines:
        card.text((23, y), text, font=current_font, fill=color)
        y += 17 if current_font is title else 13
    return np.asarray(Image.alpha_composite(pil_image.convert("RGBA"), overlay).convert("RGB"))
