"""16:9 YouTube pad + Twitch-style button sidebars for emulator captures.

YouTube keeps 60 fps on the 720p+ ladder. Native SNES/NES frames are too
small, so product recordings nearest-neighbor scale into a 1920x1080 canvas
and draw the controller on the leftover side bars — not a 16 px footer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from retro_harness.controls import pressed_nes_buttons, pressed_snes_buttons

YOUTUBE_WIDTH = 1920
YOUTUBE_HEIGHT = 1080
CANVAS_BG = (8, 10, 16)
IDLE = (46, 54, 72)
LIT = (103, 232, 164)
LABEL = (235, 240, 255)
MUTED = (150, 170, 190)
PAD_FILL = (18, 22, 32)
GAME_BORDER = (40, 48, 64)


def fit_integer_scale(
    src_width: int,
    src_height: int,
    canvas_width: int = YOUTUBE_WIDTH,
    canvas_height: int = YOUTUBE_HEIGHT,
) -> int:
    """Largest integer NN scale that still fits the canvas."""
    if src_width <= 0 or src_height <= 0:
        raise ValueError("source dimensions must be positive")
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("canvas dimensions must be positive")
    return max(1, min(canvas_width // src_width, canvas_height // src_height))


def nearest_neighbor_scale(rgb: np.ndarray, scale: int) -> np.ndarray:
    """Integer nearest-neighbor upscale (pixel-art safe)."""
    if scale < 1:
        raise ValueError("scale must be >= 1")
    frame = np.asarray(rgb, dtype=np.uint8)
    if scale == 1:
        return frame
    return np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)


def _pressed_names(
    action: np.ndarray | Sequence[int] | None,
    *,
    buttons: str,
) -> set[str]:
    if action is None:
        return set()
    raw = [int(v) for v in action]
    names = (
        pressed_nes_buttons(raw) if buttons == "nes" else pressed_snes_buttons(raw)
    )
    return set(names)


def _fill_circle(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    radius: int,
    fill: tuple[int, int, int],
) -> None:
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)


def _fill_round_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    radius: int = 8,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def _label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = LABEL,
) -> None:
    draw.text(xy, text, fill=fill, font=font)


def _draw_dpad(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    arm: int,
    thick: int,
    pressed: set[str],
    font: ImageFont.ImageFont,
) -> None:
    gap = thick // 2
    # Plus stem.
    _fill_round_rect(
        draw,
        (cx - thick // 2, cy - arm, cx + thick // 2, cy + arm),
        PAD_FILL,
        radius=6,
    )
    _fill_round_rect(
        draw,
        (cx - arm, cy - thick // 2, cx + arm, cy + thick // 2),
        PAD_FILL,
        radius=6,
    )
    keys = {
        "UP": (cx, cy - arm + gap, (cx - 10, cy - arm - 2)),
        "DOWN": (cx, cy + arm - gap, (cx - 18, cy + arm - 6)),
        "LEFT": (cx - arm + gap, cy, (cx - arm - 36, cy - 8)),
        "RIGHT": (cx + arm - gap, cy, (cx + arm + 6, cy - 8)),
    }
    for name, (bx, by, text_xy) in keys.items():
        _fill_circle(draw, bx, by, thick // 2 - 2, LIT if name in pressed else IDLE)
        _label(draw, text_xy, name[:1], font, MUTED)


def _draw_face_cluster(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    radius: int,
    spread: int,
    pressed: set[str],
    font: ImageFont.ImageFont,
    *,
    diamond: bool,
) -> None:
    if diamond:
        spots = {
            "X": (cx, cy - spread),
            "Y": (cx - spread, cy),
            "A": (cx + spread, cy),
            "B": (cx, cy + spread),
        }
    else:
        spots = {
            "B": (cx - spread, cy),
            "A": (cx + spread, cy),
        }
    for name, (x, y) in spots.items():
        _fill_circle(draw, x, y, radius, LIT if name in pressed else IDLE)
        tw = draw.textbbox((0, 0), name, font=font)[2]
        _label(draw, (x - tw // 2, y - 6), name, font)


def _draw_shoulder(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    name: str,
    pressed: set[str],
    font: ImageFont.ImageFont,
) -> None:
    _fill_round_rect(draw, box, LIT if name in pressed else IDLE, radius=10)
    x0, y0, x1, y1 = box
    tw = draw.textbbox((0, 0), name, font=font)[2]
    _label(draw, ((x0 + x1 - tw) // 2, (y0 + y1) // 2 - 6), name, font)


def _draw_snes_bars(
    draw: ImageDraw.ImageDraw,
    *,
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    pressed: set[str],
    font: ImageFont.ImageFont,
) -> None:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    lcx, lcy = (lx0 + lx1) // 2, (ly0 + ly1) // 2
    rcx, rcy = (rx0 + rx1) // 2, (ry0 + ry1) // 2
    bar = max(24, lx1 - lx0)
    arm = min(90, bar // 3)
    thick = max(16, min(52, bar // 4))
    _draw_dpad(draw, lcx, lcy, arm=arm, thick=thick, pressed=pressed, font=font)

    rbar = max(24, rx1 - rx0)
    shoulder_w = min(110, rbar // 2 - 8)
    _draw_shoulder(
        draw,
        (rcx - shoulder_w - 8, ry0 + 80, rcx - 8, ry0 + 124),
        "L",
        pressed,
        font,
    )
    _draw_shoulder(
        draw,
        (rcx + 8, ry0 + 80, rcx + shoulder_w + 8, ry0 + 124),
        "R",
        pressed,
        font,
    )
    face_r = max(10, min(28, rbar // 10))
    face_spread = max(16, min(48, rbar // 6))
    _draw_face_cluster(
        draw,
        rcx,
        rcy + 20,
        radius=face_r,
        spread=face_spread,
        pressed=pressed,
        font=font,
        diamond=True,
    )
    _draw_shoulder(
        draw,
        (rcx - 70, rcy + 130, rcx - 8, rcy + 162),
        "SELECT",
        pressed,
        font,
    )
    _draw_shoulder(
        draw,
        (rcx + 8, rcy + 130, rcx + 70, rcy + 162),
        "START",
        pressed,
        font,
    )


def _draw_nes_bars(
    draw: ImageDraw.ImageDraw,
    *,
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    pressed: set[str],
    font: ImageFont.ImageFont,
) -> None:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    lcx, lcy = (lx0 + lx1) // 2, (ly0 + ly1) // 2
    rcx, rcy = (rx0 + rx1) // 2, (ry0 + ry1) // 2
    bar = max(24, lx1 - lx0)
    arm = min(90, bar // 3)
    thick = max(16, min(52, bar // 4))
    _draw_dpad(draw, lcx, lcy, arm=arm, thick=thick, pressed=pressed, font=font)
    rbar = max(24, rx1 - rx0)
    _draw_face_cluster(
        draw,
        rcx,
        rcy,
        radius=max(10, min(32, rbar // 8)),
        spread=max(16, min(48, rbar // 6)),
        pressed=pressed,
        font=font,
        diamond=False,
    )
    _draw_shoulder(
        draw,
        (rcx - 70, rcy + 90, rcx - 8, rcy + 122),
        "SELECT",
        pressed,
        font,
    )
    _draw_shoulder(
        draw,
        (rcx + 8, rcy + 90, rcx + 70, rcy + 122),
        "START",
        pressed,
        font,
    )


_FONT: ImageFont.ImageFont | None = None
_SMALL_FONT: ImageFont.ImageFont | None = None
_SNES_STAMP_NAMES = (
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "A",
    "B",
    "X",
    "Y",
    "L",
    "R",
    "START",
    "SELECT",
)
_NES_STAMP_NAMES = (
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "A",
    "B",
    "START",
    "SELECT",
)


def _fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    global _FONT, _SMALL_FONT
    if _FONT is None:
        _FONT = ImageFont.load_default(size=12)
        _SMALL_FONT = ImageFont.load_default(size=10)
    assert _SMALL_FONT is not None
    return _FONT, _SMALL_FONT


@dataclass(frozen=True)
class _ButtonStamp:
    y0: int
    x0: int
    patch: np.ndarray


@dataclass(frozen=True)
class _CachedChrome:
    idle: np.ndarray
    stamps: dict[str, _ButtonStamp]
    x0: int
    y0: int
    play_w: int
    play_h: int
    nn: int
    stamp_y: int


_CHROME: dict[tuple[int, int, int, int, int, str], _CachedChrome] = {}


def _paint_chrome(
    *,
    canvas_width: int,
    canvas_height: int,
    x0: int,
    y0: int,
    play_w: int,
    play_h: int,
    buttons: str,
    pressed: set[str],
) -> np.ndarray:
    """Idle canvas: background, game border, controller (no gameplay)."""
    canvas = np.empty((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = CANVAS_BG
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    font, _small = _fonts()
    if x0 > 8:
        draw.rectangle(
            (x0 - 2, y0 - 2, x0 + play_w + 1, y0 + play_h + 1),
            outline=GAME_BORDER,
        )
    left = (0, 0, x0, canvas_height)
    right = (x0 + play_w, 0, canvas_width, canvas_height)
    if buttons == "nes":
        _draw_nes_bars(draw, left=left, right=right, pressed=pressed, font=font)
    else:
        _draw_snes_bars(draw, left=left, right=right, pressed=pressed, font=font)
    return np.asarray(image, dtype=np.uint8).copy()


def _text_patch(text: str) -> np.ndarray:
    """Tiny RGB label so the frame clock is not a full-canvas PIL pass."""
    _, small = _fonts()
    image = Image.new("RGB", (240, 20), CANVAS_BG)
    draw = ImageDraw.Draw(image)
    draw.text((0, 2), text, fill=MUTED, font=small)
    return np.asarray(image, dtype=np.uint8)


def _get_chrome(
    *,
    src_w: int,
    src_h: int,
    canvas_width: int,
    canvas_height: int,
    nn: int,
    buttons: str,
) -> _CachedChrome:
    key = (src_w, src_h, canvas_width, canvas_height, nn, buttons)
    cached = _CHROME.get(key)
    if cached is not None:
        return cached
    play_w, play_h = src_w * nn, src_h * nn
    x0 = (canvas_width - play_w) // 2
    y0 = (canvas_height - play_h) // 2
    idle = _paint_chrome(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        x0=x0,
        y0=y0,
        play_w=play_w,
        play_h=play_h,
        buttons=buttons,
        pressed=set(),
    )
    names = _NES_STAMP_NAMES if buttons == "nes" else _SNES_STAMP_NAMES
    stamps: dict[str, _ButtonStamp] = {}
    for name in names:
        lit = _paint_chrome(
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            x0=x0,
            y0=y0,
            play_w=play_w,
            play_h=play_h,
            buttons=buttons,
            pressed={name},
        )
        changed = np.any(lit != idle, axis=2)
        if not np.any(changed):
            continue
        ys, xs = np.where(changed)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        stamps[name] = _ButtonStamp(y1, x1, lit[y1:y2, x1:x2].copy())
    chrome = _CachedChrome(
        idle=idle,
        stamps=stamps,
        x0=x0,
        y0=y0,
        play_w=play_w,
        play_h=play_h,
        nn=nn,
        stamp_y=max(8, y0 - 22),
    )
    _CHROME[key] = chrome
    return chrome


def compose_youtube_frame(
    obs: np.ndarray,
    *,
    action: np.ndarray | Sequence[int] | None = None,
    frame: int = 0,
    fps: int = 60,
    buttons: str = "snes",
    canvas_width: int = YOUTUBE_WIDTH,
    canvas_height: int = YOUTUBE_HEIGHT,
    scale: int | None = None,
) -> np.ndarray:
    """NN-upscale gameplay into a 16:9 canvas with controller side bars.

    Controller chrome is painted once and cached. Per-frame work is gameplay
    blit plus button stamps — a full 1080p PIL pass every frame is too slow
    for faster-than-realtime dumps.
    """
    rgb = np.asarray(obs, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB frame, got {rgb.shape}")
    src_h, src_w = rgb.shape[:2]
    nn = scale or fit_integer_scale(src_w, src_h, canvas_width, canvas_height)
    play = nearest_neighbor_scale(rgb, nn)
    play_h, play_w = play.shape[:2]
    if play_w > canvas_width or play_h > canvas_height:
        raise ValueError(
            f"scaled gameplay {play_w}x{play_h} exceeds canvas "
            f"{canvas_width}x{canvas_height}"
        )
    chrome = _get_chrome(
        src_w=src_w,
        src_h=src_h,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        nn=nn,
        buttons=buttons,
    )
    out = chrome.idle.copy()
    out[chrome.y0 : chrome.y0 + play_h, chrome.x0 : chrome.x0 + play_w] = play
    for name in _pressed_names(action, buttons=buttons):
        stamp = chrome.stamps.get(name)
        if stamp is None:
            continue
        h, w = stamp.patch.shape[:2]
        out[stamp.y0 : stamp.y0 + h, stamp.x0 : stamp.x0 + w] = stamp.patch
    seconds = int(frame / fps) if fps > 0 else 0
    minutes, secs = divmod(seconds, 60)
    label = _text_patch(f"F{frame:06d}  {minutes:02d}:{secs:02d}")
    th, tw = label.shape[:2]
    tx = (canvas_width - tw) // 2
    out[chrome.stamp_y : chrome.stamp_y + th, tx : tx + tw] = label
    return out
