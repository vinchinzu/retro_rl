"""YouTube 16:9 pad + Twitch-style button sidebars."""

from __future__ import annotations

import numpy as np
from retro_harness.actions import buttons
from retro_harness.video import VideoCaptureConfig
from retro_harness.video_layout import (
    LIT,
    YOUTUBE_HEIGHT,
    YOUTUBE_WIDTH,
    compose_youtube_frame,
    fit_integer_scale,
    nearest_neighbor_scale,
)


def test_fit_integer_scale_snes_on_1080p() -> None:
    assert fit_integer_scale(256, 224) == 4
    assert 256 * 4 < YOUTUBE_WIDTH
    assert 224 * 4 < YOUTUBE_HEIGHT


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


def test_compose_lights_pressed_face_button() -> None:
    obs = np.zeros((224, 256, 3), dtype=np.uint8)
    idle = compose_youtube_frame(obs, action=buttons(), frame=0)
    held = compose_youtube_frame(obs, action=buttons("A"), frame=0)
    # Right-side cluster sits in the right pad; A should add lit pixels.
    right_idle = idle[:, YOUTUBE_WIDTH // 2 :]
    right_held = held[:, YOUTUBE_WIDTH // 2 :]
    lit = np.all(right_held == np.array(LIT, dtype=np.uint8), axis=2)
    lit_idle = np.all(right_idle == np.array(LIT, dtype=np.uint8), axis=2)
    assert int(lit.sum()) > int(lit_idle.sum())


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
