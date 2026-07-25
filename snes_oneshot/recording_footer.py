"""Shared bottom banner for oneshot game recordings."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from retro_harness.controls import pressed_snes_buttons

FOOTER_HEIGHT = 16
SNES_PLAYER_STRIDE = 12


def short_clock(frame: int, fps: float) -> str:
    """Return MM:SS for a live footer clock."""
    seconds = int(frame / fps)
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def format_player_buttons(
    action: list[int] | None,
    *,
    players: int = 1,
    stride: int = SNES_PLAYER_STRIDE,
) -> str:
    """Render pressed buttons for one or more players."""
    if action is None:
        return "P1:---"
    parts: list[str] = []
    for player in range(players):
        start = player * stride
        end = start + stride
        if end > len(action):
            break
        names = pressed_snes_buttons(action[start:end])
        label = "+".join(sorted(names)) if names else "---"
        parts.append(f"P{player + 1}:{label}")
    return "  ".join(parts) if parts else "P1:---"


def render_footer_frame(
    obs: np.ndarray,
    *,
    upper_left: str,
    upper_right: str,
    lower_left: str,
    action: list[int] | None = None,
    players: int = 1,
    footer_bg: tuple[int, int, int] = (5, 10, 18),
    upper_left_color: tuple[int, int, int] = (219, 234, 246),
    upper_right_color: tuple[int, int, int] = (103, 232, 164),
    lower_left_color: tuple[int, int, int] = (150, 170, 190),
    button_color: tuple[int, int, int] = (255, 214, 102),
) -> np.ndarray:
    """Extend the frame with a TMNT-style footer and live button labels."""
    rgb = np.asarray(obs, dtype=np.uint8)
    height, width = rgb.shape[:2]
    canvas = np.zeros((height + FOOTER_HEIGHT, width, 3), dtype=np.uint8)
    canvas[:height] = rgb
    canvas[height:] = footer_bg
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=8)

    draw.text((4, height), upper_left, fill=upper_left_color, font=font)
    clock_width = draw.textbbox((0, 0), upper_right, font=font)[2]
    draw.text(
        (width - clock_width - 4, height),
        upper_right,
        fill=upper_right_color,
        font=font,
    )

    draw.text((4, height + 8), lower_left, fill=lower_left_color, font=font)
    buttons = format_player_buttons(action, players=players)
    button_width = draw.textbbox((0, 0), buttons, font=font)[2]
    draw.text(
        (width - button_width - 4, height + 8),
        buttons,
        fill=button_color,
        font=font,
    )
    return np.asarray(image)
