"""Generic YouTube intro slides for continuous full-game recordings.

Reusable across NES/SNES packages. The intro is **pre-roll only** — gameplay
itself remains one continuous emulator session (not a stitch of segment clips).

Typical use::

    from retro_harness.youtube_intro import project_intro_lines, render_intro_card

    lines = project_intro_lines(
        game_title="Super Mario Bros.",
        run_summary="Clean power-on any% warp → 8-4 ending",
    )
    card = render_intro_card(lines, width=256, height=224, with_footer=True)
"""

from __future__ import annotations

import textwrap
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from retro_harness.video import FOOTER_HEIGHT

PROJECT_NAME = "retro_rl"
PROJECT_TAGLINE = "NES + SNES full-game automation"
# ASCII-only: default PIL bitmap fonts lack arrows / middots.
PROJECT_MISSION = "Scripted completion - verified reset-to-ending"
PROJECT_METHOD = "Clean continuous run - one emulator session"
PROJECT_NO_STITCH = "Not a stitch of segment clips"
PROJECT_REPO = "github.com/vinchinzu/retro_rl"

# Default hold at 60 fps (~1.5 s) — snappy pre-roll; slide is still readable.
DEFAULT_INTRO_FRAMES = 90


def _ascii_safe(text: str) -> str:
    """Replace common unicode punctuation the default bitmap font cannot draw."""
    return (
        text.replace("→", "->")
        .replace("←", "<-")
        .replace("·", "-")
        .replace("•", "-")
        .replace("—", "-")
        .replace("–", "-")
        .replace("…", "...")
        .strip()
    )


def project_intro_lines(
    *,
    game_title: str,
    run_summary: str,
    extra_lines: Sequence[str] = (),
    intervention: str | None = "Clean / Bronze runtime observation",
) -> list[str]:
    """Build a generic project intro suitable for any game's YouTube upload.

    Lines 1–2 are always the monorepo brand; game-specific context follows.
    Keep lines short so the card fits NES/SNES playfield height.
    """
    lines = [
        PROJECT_NAME,
        PROJECT_TAGLINE,
        _ascii_safe(game_title),
        _ascii_safe(run_summary),
        PROJECT_MISSION,
        PROJECT_METHOD,
        PROJECT_NO_STITCH,
    ]
    if intervention:
        text = _ascii_safe(intervention)
        if text:
            lines.append(text)
    lines.append(PROJECT_REPO)
    for line in extra_lines:
        text = _ascii_safe(str(line))
        if text:
            lines.append(text)
    return lines


def render_intro_card(
    lines: Sequence[str],
    *,
    width: int = 256,
    height: int = 224,
    with_footer: bool = True,
) -> np.ndarray:
    """Render an intro slide at emulator resolution (optional footer band).

    Matches gameplay frame geometry so recorders can pipe the card through the
    same ffmpeg session as the continuous run.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    image = Image.new("RGB", (width, height), (5, 8, 18))
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=15)
    body_font = ImageFont.load_default(size=10)
    accent = (103, 232, 164)
    body = (235, 240, 255)
    muted = (150, 170, 190)

    y = 12
    for index, line in enumerate(lines):
        if not line:
            y += 6
            continue
        if index == 0:
            font, fill, wrap, step = title_font, accent, 30, 17
        elif index == 1:
            font, fill, wrap, step = body_font, muted, 36, 12
        else:
            font, fill, wrap, step = body_font, body, 36, 12
        for part in textwrap.wrap(line, width=wrap) or [""]:
            if y + step > height - 14:
                break
            box = draw.textbbox((0, 0), part, font=font)
            text_width = box[2] - box[0]
            draw.text(((width - text_width) // 2, y), part, font=font, fill=fill)
            y += step
        y += 3 if index <= 1 else 2
        if y > height - 14:
            break

    # Thin accent rule near the bottom of the playfield.
    rule_y = min(height - 8, max(y + 2, height - 12))
    draw.line((width // 8, rule_y, width * 7 // 8, rule_y), fill=accent, width=1)

    card = np.asarray(image, dtype=np.uint8)
    if not with_footer:
        return card
    footer = np.full((FOOTER_HEIGHT, width, 3), (5, 10, 18), dtype=np.uint8)
    # Label the footer so it is obvious this is pre-roll, not a stalled timer.
    footer_img = Image.fromarray(footer)
    footer_draw = ImageDraw.Draw(footer_img)
    small = ImageFont.load_default(size=8)
    footer_draw.text((4, 2), "INTRO", fill=muted, font=small)
    footer_draw.text((width - 52, 2), "00:00.00", fill=accent, font=small)
    return np.vstack([card, np.asarray(footer_img, dtype=np.uint8)])
