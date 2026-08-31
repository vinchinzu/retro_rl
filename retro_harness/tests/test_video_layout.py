"""YouTube 16:9 pad + Twitch-style left stream stack."""

from __future__ import annotations

import numpy as np
from retro_harness.actions import buttons
from retro_harness.video import VideoCaptureConfig
from retro_harness.video_layout import (
    LIT,
    SIDEBAR_MAX,
    SIDEBAR_MIN,
    YOUTUBE_HEIGHT,
    YOUTUBE_WIDTH,
    compose_youtube_frame,
    fit_integer_scale,
    nearest_neighbor_scale,
    youtube_gameplay_scale,
)


def _gameplay_bbox(out: np.ndarray, fill: tuple[int, int, int]) -> tuple[int, int, int, int]:
    mask = np.all(out == np.array(fill, dtype=np.uint8), axis=2)
    ys, xs = np.where(mask)
    assert xs.size > 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _lit_delta(idle: np.ndarray, held: np.ndarray) -> int:
    lit_held = np.all(held == np.array(LIT, dtype=np.uint8), axis=2)
    lit_idle = np.all(idle == np.array(LIT, dtype=np.uint8), axis=2)
    return int(lit_held.sum()) - int(lit_idle.sum())


def test_fit_integer_scale_snes_on_1080p() -> None:
    assert fit_integer_scale(256, 224) == 4
    assert 256 * 4 < YOUTUBE_WIDTH
    assert 224 * 4 < YOUTUBE_HEIGHT


def test_youtube_gameplay_scale_fits_square() -> None:
    nn = youtube_gameplay_scale(256, 240)
    assert nn == 4
    assert 256 * nn <= YOUTUBE_HEIGHT
    assert 240 * nn <= YOUTUBE_HEIGHT


def test_nearest_neighbor_scale_repeats_pixels() -> None:
    src = np.zeros((2, 2, 3), dtype=np.uint8)
    src[0, 0] = (9, 8, 7)
    out = nearest_neighbor_scale(src, 3)
    assert out.shape == (6, 6, 3)
    assert tuple(out[0, 0]) == (9, 8, 7)
    assert tuple(out[2, 2]) == (9, 8, 7)


def test_compose_youtube_frame_is_1080p60_canvas() -> None:
    obs = np.full((224, 256, 3), 40, dtype=np.uint8)
    out = compose_youtube_frame(obs, action=buttons("A"), frame=120, fps=60)
    assert out.shape == (YOUTUBE_HEIGHT, YOUTUBE_WIDTH, 3)


def test_gameplay_sits_in_right_square() -> None:
    fill = (180, 12, 12)
    obs = np.full((240, 256, 3), fill, dtype=np.uint8)
    out = compose_youtube_frame(obs, buttons="nes", frame=60, fps=60)
    x0, y0, x1, y1 = _gameplay_bbox(out, fill)
    nn = youtube_gameplay_scale(256, 240)
    assert x1 - x0 == 256 * nn
    assert y1 - y0 == 240 * nn
    assert x0 >= int(YOUTUBE_WIDTH * SIDEBAR_MIN)
    # Square playfield is on the right of the 20-30% stack.
    assert x1 - x0 <= y1 - y0 + 256  # NES is slightly wide of square pixels
    play_w, play_h = x1 - x0, y1 - y0
    # Centered inside a square whose side is the canvas height.
    assert play_w <= YOUTUBE_HEIGHT
    assert play_h <= YOUTUBE_HEIGHT
    assert y0 >= 0
    # Left 20% is HUD, not gameplay.
    assert not np.any(np.all(out[:, : int(YOUTUBE_WIDTH * SIDEBAR_MIN)] == fill, axis=2))


def test_compose_lights_pressed_face_button() -> None:
    obs = np.zeros((224, 256, 3), dtype=np.uint8)
    idle = compose_youtube_frame(obs, action=buttons(), frame=0, buttons="nes")
    held = compose_youtube_frame(obs, action=buttons("A"), frame=0, buttons="nes")
    left_w = int(YOUTUBE_WIDTH * SIDEBAR_MAX)
    assert _lit_delta(idle[:, :left_w], held[:, :left_w]) > 0
    # Face buttons live in the left stack, not a right-side bar.
    assert _lit_delta(idle[:, left_w:], held[:, left_w:]) == 0


def test_nes_face_buttons_are_smaller_than_snes() -> None:
    obs = np.zeros((224, 256, 3), dtype=np.uint8)
    nes_idle = compose_youtube_frame(obs, action=buttons(), buttons="nes")
    nes_held = compose_youtube_frame(obs, action=buttons("A"), buttons="nes")
    snes_idle = compose_youtube_frame(obs, action=buttons(), buttons="snes")
    snes_held = compose_youtube_frame(obs, action=buttons("A"), buttons="snes")
    assert _lit_delta(nes_idle, nes_held) < _lit_delta(snes_idle, snes_held)


def test_timer_panel_updates_in_left_stack() -> None:
    obs = np.zeros((224, 256, 3), dtype=np.uint8)
    start = compose_youtube_frame(obs, frame=0, fps=60, buttons="nes")
    later = compose_youtube_frame(obs, frame=3600 * 60, fps=60, buttons="nes")
    left = slice(0, int(YOUTUBE_WIDTH * SIDEBAR_MAX))
    top = slice(0, YOUTUBE_HEIGHT // 3)
    assert not np.array_equal(start[top, left], later[top, left])


def test_cam_placeholder_occupies_bottom_left() -> None:
    obs = np.zeros((224, 256, 3), dtype=np.uint8)
    out = compose_youtube_frame(obs, buttons="nes")
    cam = out[int(YOUTUBE_HEIGHT * 0.55) :, : int(YOUTUBE_WIDTH * SIDEBAR_MAX)]
    # Panel fill is not the canvas background.
    bg = np.array((8, 10, 16), dtype=np.uint8)
    assert np.any(~np.all(cam == bg, axis=2))


def test_youtube_capture_preset() -> None:
    cfg = VideoCaptureConfig.youtube()
    assert cfg.layout == "youtube"
    assert cfg.fps == 60
    assert cfg.footer is False
    assert cfg.canvas_width == YOUTUBE_WIDTH
    assert cfg.canvas_height == YOUTUBE_HEIGHT
    assert cfg.scale == 0
    assert cfg.preset == "veryfast"


def test_compose_returned_frames_are_independent() -> None:
    obs = np.zeros((224, 256, 3), dtype=np.uint8)
    idle = compose_youtube_frame(obs, action=buttons(), frame=0)
    snapshot = idle.copy()
    compose_youtube_frame(obs, action=buttons("A"), frame=1)
    assert np.array_equal(idle, snapshot)
