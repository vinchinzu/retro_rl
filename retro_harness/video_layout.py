"""16:9 YouTube pad + Twitch-style left stack for emulator captures.

YouTube keeps 60 fps on the 720p+ ladder. Native SNES/NES frames are too
small, so product recordings nearest-neighbor scale into a 1920x1080 canvas.

Layout (stream overlay, not a 16 px footer):

- left ~25% (clamped 20–30%): stacked TIMER, INPUT, CAM panels
- right remainder: a square playfield, integer-scaled gameplay centered
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
PANEL_BODY = (22, 26, 38)
GAME_BORDER = (40, 48, 64)
SQUARE_FILL = (12, 14, 22)
CAM_WELL = (14, 16, 24)
SIDEBAR_RATIO = 0.25
SIDEBAR_MIN = 0.20
SIDEBAR_MAX = 0.30

_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}
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


@dataclass(frozen=True)
class _ChromeLayout:
    sidebar_w: int
    timer: tuple[int, int, int, int]
    controller: tuple[int, int, int, int]
    cam: tuple[int, int, int, int]
    square: tuple[int, int, int, int]
    play_inner: tuple[int, int, int, int]
    clock: tuple[int, int, int, int]


def _box_size(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = box
    return x1 - x0, y1 - y0


def _stream_layout(canvas_width: int, canvas_height: int) -> _ChromeLayout:
    """Left stack (20–30%) + square playfield in the remaining width."""
    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("canvas dimensions must be positive")
    pad = max(8, min(canvas_width, canvas_height) // 72)
    sidebar_w = int(round(canvas_width * SIDEBAR_RATIO))
    lo = int(canvas_width * SIDEBAR_MIN)
    hi = int(canvas_width * SIDEBAR_MAX)
    sidebar_w = max(lo, min(hi, sidebar_w))
    remain_w = canvas_width - sidebar_w
    square_size = max(1, min(canvas_height, remain_w))
    sq_x0 = sidebar_w
    sq_y0 = (canvas_height - square_size) // 2
    square = (sq_x0, sq_y0, sq_x0 + square_size, sq_y0 + square_size)
    inset = 2 if square_size > 8 else 0
    play_inner = (
        sq_x0 + inset,
        sq_y0 + inset,
        sq_x0 + square_size - inset,
        sq_y0 + square_size - inset,
    )

    inner_x0, inner_y0 = pad, pad
    inner_x1, inner_y1 = sidebar_w - pad, canvas_height - pad
    if inner_x1 <= inner_x0:
        inner_x0, inner_x1 = 0, sidebar_w
    if inner_y1 <= inner_y0:
        inner_y0, inner_y1 = 0, canvas_height
    inner_h = inner_y1 - inner_y0
    gap = pad
    usable = max(1, inner_h - 2 * gap)
    timer_h = max(64, usable * 20 // 100)
    ctrl_h = max(80, usable * 34 // 100)
    if timer_h + ctrl_h + 16 > usable:
        timer_h = max(48, usable * 22 // 100)
        ctrl_h = max(56, usable * 34 // 100)
    cam_h = max(1, usable - timer_h - ctrl_h)
    timer = (inner_x0, inner_y0, inner_x1, inner_y0 + timer_h)
    controller = (
        inner_x0,
        inner_y0 + timer_h + gap,
        inner_x1,
        inner_y0 + timer_h + gap + ctrl_h,
    )
    cam = (
        inner_x0,
        inner_y0 + timer_h + gap + ctrl_h + gap,
        inner_x1,
        inner_y0 + timer_h + gap + ctrl_h + gap + cam_h,
    )
    title_h = max(18, pad + 8)
    clock = (
        timer[0] + 8,
        timer[1] + title_h,
        timer[2] - 8,
        timer[3] - 8,
    )
    if clock[2] <= clock[0] or clock[3] <= clock[1]:
        clock = timer
    return _ChromeLayout(
        sidebar_w=sidebar_w,
        timer=timer,
        controller=controller,
        cam=cam,
        square=square,
        play_inner=play_inner,
        clock=clock,
    )


def youtube_gameplay_scale(
    src_width: int,
    src_height: int,
    canvas_width: int = YOUTUBE_WIDTH,
    canvas_height: int = YOUTUBE_HEIGHT,
) -> int:
    """Integer NN scale that fits the square playfield, not the full 16:9 canvas."""
    layout = _stream_layout(canvas_width, canvas_height)
    ix0, iy0, ix1, iy1 = layout.play_inner
    return fit_integer_scale(src_width, src_height, ix1 - ix0, iy1 - iy0)


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


def _font(size: int) -> ImageFont.ImageFont:
    cached = _FONT_CACHE.get(size)
    if cached is None:
        cached = ImageFont.load_default(size=size)
        _FONT_CACHE[size] = cached
    return cached


def _fill_circle(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    radius: int,
    fill: tuple[int, int, int],
) -> None:
    radius = max(2, radius)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)


def _fill_round_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    radius: int = 8,
) -> None:
    x0, y0, x1, y1 = box
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    draw.rounded_rectangle(box, radius=max(1, min(radius, (x1 - x0) // 2, (y1 - y0) // 2)), fill=fill)


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
) -> None:
    arm = max(6, arm)
    thick = max(6, min(thick, arm))
    gap = max(2, thick // 2)
    _fill_round_rect(
        draw,
        (cx - thick // 2, cy - arm, cx + thick // 2, cy + arm),
        PAD_FILL,
        radius=4,
    )
    _fill_round_rect(
        draw,
        (cx - arm, cy - thick // 2, cx + arm, cy + thick // 2),
        PAD_FILL,
        radius=4,
    )
    keys = {
        "UP": (cx, cy - arm + gap),
        "DOWN": (cx, cy + arm - gap),
        "LEFT": (cx - arm + gap, cy),
        "RIGHT": (cx + arm - gap, cy),
    }
    radius = max(3, thick // 2 - 1)
    for name, (bx, by) in keys.items():
        _fill_circle(draw, bx, by, radius, LIT if name in pressed else IDLE)


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
    label_inside: bool,
) -> None:
    radius = max(4, radius)
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
        if label_inside:
            _label(draw, (x - tw // 2, y - 6), name, font)
        else:
            _label(draw, (x - tw // 2, y + radius + 1), name, font, MUTED)


def _draw_pill(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    name: str,
    pressed: set[str],
    font: ImageFont.ImageFont,
    label: str | None = None,
) -> None:
    _fill_round_rect(draw, box, LIT if name in pressed else IDLE, radius=6)
    x0, y0, x1, y1 = box
    text = label or name
    tw = draw.textbbox((0, 0), text, font=font)[2]
    th = draw.textbbox((0, 0), text, font=font)[3]
    _label(
        draw,
        ((x0 + x1 - tw) // 2, (y0 + y1 - th) // 2),
        text,
        font,
    )


def _content_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Controller drawing sits below the panel title."""
    x0, y0, x1, y1 = box
    return (x0 + 8, y0 + 24, x1 - 8, y1 - 8)


def _panel_scale(box: tuple[int, int, int, int], ref_w: float, ref_h: float) -> float:
    w, h = _box_size(box)
    if w <= 0 or h <= 0:
        return 0.25
    return max(0.25, min(w / ref_w, h / ref_h, 1.2))


def _draw_nes_controller(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    pressed: set[str],
    font: ImageFont.ImageFont,
) -> None:
    """Compact NES brick — small face buttons, one pad in the INPUT panel."""
    x0, y0, x1, y1 = _content_box(box)
    w, h = x1 - x0, y1 - y0
    s = _panel_scale((x0, y0, x1, y1), 400.0, 240.0)
    brick_w = min(w - 8, int(360 * s))
    brick_h = min(h - 8, int(130 * s))
    brick_w = max(96, brick_w)
    brick_h = max(44, brick_h)
    bx0 = x0 + (w - brick_w) // 2
    by0 = y0 + (h - brick_h) // 2
    _fill_round_rect(
        draw,
        (bx0, by0, bx0 + brick_w, by0 + brick_h),
        PANEL_BODY,
        radius=max(8, int(14 * s)),
    )
    dpad_cx = bx0 + max(22, int(70 * s))
    dpad_cy = by0 + brick_h // 2
    _draw_dpad(
        draw,
        dpad_cx,
        dpad_cy,
        arm=max(10, int(24 * s)),
        thick=max(8, int(15 * s)),
        pressed=pressed,
    )
    pill_h = max(10, int(14 * s))
    pill_w = max(26, int(38 * s))
    mid_x = bx0 + int(brick_w * 0.52)
    pill_y0 = dpad_cy - pill_h // 2
    _draw_pill(
        draw,
        (mid_x - pill_w - 4, pill_y0, mid_x - 4, pill_y0 + pill_h),
        "SELECT",
        pressed,
        font,
        label="SEL",
    )
    _draw_pill(
        draw,
        (mid_x + 4, pill_y0, mid_x + 4 + pill_w, pill_y0 + pill_h),
        "START",
        pressed,
        font,
        label="STR",
    )
    face_cx = bx0 + brick_w - max(28, int(70 * s))
    face_cy = dpad_cy
    _draw_face_cluster(
        draw,
        face_cx,
        face_cy,
        radius=max(6, int(12 * s)),
        spread=max(10, int(20 * s)),
        pressed=pressed,
        font=font,
        diamond=False,
        label_inside=True,
    )


def _draw_snes_controller(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    pressed: set[str],
    font: ImageFont.ImageFont,
) -> None:
    x0, y0, x1, y1 = _content_box(box)
    w, h = x1 - x0, y1 - y0
    s = _panel_scale((x0, y0, x1, y1), 420.0, 300.0)
    body_w = min(w - 16, int(340 * s))
    body_h = min(h - 40, int(150 * s))
    body_w = max(100, body_w)
    body_h = max(48, body_h)
    bx0 = x0 + (w - body_w) // 2
    by0 = y0 + (h - body_h) // 2 + max(8, int(14 * s)) // 2
    _fill_round_rect(
        draw,
        (bx0, by0, bx0 + body_w, by0 + body_h),
        PANEL_BODY,
        radius=max(8, int(18 * s)),
    )
    sh_h = max(10, int(16 * s))
    sh_w = max(28, int(52 * s))
    sh_y0 = by0 - sh_h - 4
    _draw_pill(
        draw,
        (bx0 + 12, sh_y0, bx0 + 12 + sh_w, sh_y0 + sh_h),
        "L",
        pressed,
        font,
    )
    _draw_pill(
        draw,
        (bx0 + body_w - 12 - sh_w, sh_y0, bx0 + body_w - 12, sh_y0 + sh_h),
        "R",
        pressed,
        font,
    )
    dpad_cx = bx0 + max(24, int(70 * s))
    dpad_cy = by0 + body_h // 2 - 4
    _draw_dpad(
        draw,
        dpad_cx,
        dpad_cy,
        arm=max(10, int(28 * s)),
        thick=max(8, int(16 * s)),
        pressed=pressed,
    )
    pill_h = max(10, int(14 * s))
    pill_w = max(36, int(48 * s))
    mid_x = bx0 + body_w // 2
    pill_y0 = by0 + body_h - pill_h - max(8, int(12 * s))
    _draw_pill(
        draw,
        (mid_x - pill_w - 4, pill_y0, mid_x - 4, pill_y0 + pill_h),
        "SELECT",
        pressed,
        font,
    )
    _draw_pill(
        draw,
        (mid_x + 4, pill_y0, mid_x + 4 + pill_w, pill_y0 + pill_h),
        "START",
        pressed,
        font,
    )
    face_cx = bx0 + body_w - max(28, int(78 * s))
    face_cy = dpad_cy
    _draw_face_cluster(
        draw,
        face_cx,
        face_cy,
        radius=max(7, int(14 * s)),
        spread=max(12, int(24 * s)),
        pressed=pressed,
        font=font,
        diamond=True,
        label_inside=True,
    )


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    font: ImageFont.ImageFont,
) -> None:
    _fill_round_rect(draw, box, PAD_FILL, radius=12)
    x0, y0, _, _ = box
    _label(draw, (x0 + 10, y0 + 6), title, font, MUTED)


def _draw_cam_placeholder(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    small: ImageFont.ImageFont,
) -> None:
    _draw_panel(draw, box, "CAM", font)
    x0, y0, x1, y1 = box
    inset = 16
    well = (x0 + inset, y0 + 28, x1 - inset, y1 - inset)
    _fill_round_rect(draw, well, CAM_WELL, radius=10)
    wx0, wy0, wx1, wy1 = well
    cx, cy = (wx0 + wx1) // 2, (wy0 + wy1) // 2 - 8
    bw = max(36, min(120, (wx1 - wx0) // 3))
    bh = max(24, min(72, (wy1 - wy0) // 4))
    _fill_round_rect(
        draw,
        (cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2),
        IDLE,
        radius=8,
    )
    _fill_circle(draw, cx, cy, max(6, min(bw, bh) // 5), PAD_FILL)
    lens_r = max(4, min(bw, bh) // 8)
    _fill_circle(draw, cx, cy, lens_r, GAME_BORDER)
    caption = "video"
    tw = draw.textbbox((0, 0), caption, font=small)[2]
    _label(draw, (cx - tw // 2, cy + bh // 2 + 6), caption, small, MUTED)


def _draw_square_bezel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    _fill_round_rect(draw, box, SQUARE_FILL, radius=6)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=GAME_BORDER)


def _paint_chrome(
    *,
    canvas_width: int,
    canvas_height: int,
    layout: _ChromeLayout,
    buttons: str,
    pressed: set[str],
) -> np.ndarray:
    """Idle canvas: background, panels, square bezel (no gameplay)."""
    canvas = np.empty((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = CANVAS_BG
    image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(image)
    title_font = _font(11)
    small = _font(9)
    _draw_panel(draw, layout.timer, "TIMER", title_font)
    _draw_panel(draw, layout.controller, "INPUT", title_font)
    if buttons == "nes":
        _draw_nes_controller(draw, layout.controller, pressed, small)
    else:
        _draw_snes_controller(draw, layout.controller, pressed, small)
    _draw_cam_placeholder(draw, layout.cam, title_font, small)
    _draw_square_bezel(draw, layout.square)
    return np.asarray(image, dtype=np.uint8).copy()


def _clock_patch(clock: str, sub: str, width: int, height: int) -> np.ndarray:
    """Timer digits on panel fill so we do not rerender the whole 1080p canvas."""
    width = max(1, width)
    height = max(1, height)
    image = Image.new("RGB", (width, height), PAD_FILL)
    draw = ImageDraw.Draw(image)
    clock_size = max(12, min(36, height * 5 // 8))
    sub_size = max(8, min(12, clock_size // 3))
    clock_font = _font(clock_size)
    sub_font = _font(sub_size)
    bbox = draw.textbbox((0, 0), clock, font=clock_font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    sub_bbox = draw.textbbox((0, 0), sub, font=sub_font)
    sw, sh = sub_bbox[2] - sub_bbox[0], sub_bbox[3] - sub_bbox[1]
    gap = 4
    block_h = th + gap + sh
    y = max(0, (height - block_h) // 2)
    _label(draw, ((width - tw) // 2, y), clock, clock_font, LIT)
    _label(draw, ((width - sw) // 2, y + th + gap), sub, sub_font, MUTED)
    return np.asarray(image, dtype=np.uint8)


def _format_clock(frame: int, fps: int) -> tuple[str, str]:
    fps = fps if fps > 0 else 60
    total = max(0, int(frame))
    seconds, frac = divmod(total, fps)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    sub = f"F{total:06d}  +{frac:02d}f"
    return clock, sub


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
    clock: tuple[int, int, int, int]


_CHROME: dict[tuple[int, int, int, int, int, str], _CachedChrome] = {}


def _play_origin(
    layout: _ChromeLayout, play_w: int, play_h: int
) -> tuple[int, int]:
    ix0, iy0, ix1, iy1 = layout.play_inner
    avail_w, avail_h = ix1 - ix0, iy1 - iy0
    return ix0 + (avail_w - play_w) // 2, iy0 + (avail_h - play_h) // 2


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
    layout = _stream_layout(canvas_width, canvas_height)
    play_w, play_h = src_w * nn, src_h * nn
    x0, y0 = _play_origin(layout, play_w, play_h)
    idle = _paint_chrome(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        layout=layout,
        buttons=buttons,
        pressed=set(),
    )
    names = _NES_STAMP_NAMES if buttons == "nes" else _SNES_STAMP_NAMES
    stamps: dict[str, _ButtonStamp] = {}
    for name in names:
        lit = _paint_chrome(
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            layout=layout,
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
        clock=layout.clock,
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
    """NN-upscale gameplay into a 16:9 canvas with a left stream stack.

    Controller chrome is painted once and cached. Per-frame work is gameplay
    blit, button stamps, and a small timer patch — a full 1080p PIL pass every
    frame is too slow for faster-than-realtime dumps.
    """
    rgb = np.asarray(obs, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB frame, got {rgb.shape}")
    src_h, src_w = rgb.shape[:2]
    max_nn = youtube_gameplay_scale(src_w, src_h, canvas_width, canvas_height)
    nn = min(scale, max_nn) if scale else max_nn
    nn = max(1, nn)
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
    y0, x0 = chrome.y0, chrome.x0
    out[y0 : y0 + play_h, x0 : x0 + play_w] = play
    for name in _pressed_names(action, buttons=buttons):
        stamp = chrome.stamps.get(name)
        if stamp is None:
            continue
        h, w = stamp.patch.shape[:2]
        out[stamp.y0 : stamp.y0 + h, stamp.x0 : stamp.x0 + w] = stamp.patch
    clock, sub = _format_clock(frame, fps)
    cx0, cy0, cx1, cy1 = chrome.clock
    patch = _clock_patch(clock, sub, cx1 - cx0, cy1 - cy0)
    ph, pw = patch.shape[:2]
    out[cy0 : cy0 + ph, cx0 : cx0 + pw] = patch
    return out
