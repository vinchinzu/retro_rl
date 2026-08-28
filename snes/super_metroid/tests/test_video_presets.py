"""Metroid continuous-video presets on the shared recorder."""

from __future__ import annotations

from super_metroid.video import (
    DEFAULT_OPENING_CREDITS_CUTOFF,
    ZEBES_LANDING_ROOM_ID,
    continuous_video_config,
    opening_credits_cutoff,
)


def test_opening_credits_cutoff_default() -> None:
    assert opening_credits_cutoff() == DEFAULT_OPENING_CREDITS_CUTOFF
    assert opening_credits_cutoff(1200) == 1200


def test_continuous_video_config_defaults_drop_credits_and_pad_youtube() -> None:
    cfg = continuous_video_config()
    assert cfg.start_frame == DEFAULT_OPENING_CREDITS_CUTOFF
    assert cfg.start_room_id is None
    assert cfg.layout == "youtube"
    assert cfg.footer is False
    assert cfg.fps == 60
    assert cfg.canvas_width == 1920
    assert cfg.canvas_height == 1080


def test_continuous_video_config_zebes() -> None:
    cfg = continuous_video_config(start="zebes")
    assert cfg.start_room_id == ZEBES_LANDING_ROOM_ID
    assert cfg.start_frame is None
    assert cfg.audio is True
    assert cfg.layout == "youtube"
    assert cfg.fps == 60


def test_continuous_video_config_native_keeps_footer() -> None:
    cfg = continuous_video_config(start="zebes", layout="native")
    assert cfg.layout == "native"
    assert cfg.footer is True
    assert cfg.scale == 2


def test_continuous_video_config_after_credits() -> None:
    cfg = continuous_video_config(start="after_credits")
    assert cfg.start_frame == DEFAULT_OPENING_CREDITS_CUTOFF
    cfg2 = continuous_video_config(start="after_credits", start_frame=500)
    assert cfg2.start_frame == 500


def test_continuous_video_config_hq() -> None:
    cfg = continuous_video_config(start="power_on", hq=True)
    assert cfg.layout == "youtube"
    assert cfg.crf == 15
    assert cfg.preset == "slow"
    native = continuous_video_config(start="power_on", hq=True, layout="native")
    assert native.scale == 3
    assert native.crf == 15
    assert native.preset == "slow"
