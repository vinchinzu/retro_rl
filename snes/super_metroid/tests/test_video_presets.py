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


def test_continuous_video_config_zebes() -> None:
    cfg = continuous_video_config(start="zebes")
    assert cfg.start_room_id == ZEBES_LANDING_ROOM_ID
    assert cfg.start_frame is None
    assert cfg.audio is True
    assert cfg.footer is True
    assert cfg.fps == 60


def test_continuous_video_config_after_credits() -> None:
    cfg = continuous_video_config(start="after_credits")
    assert cfg.start_frame == DEFAULT_OPENING_CREDITS_CUTOFF
    cfg2 = continuous_video_config(start="after_credits", start_frame=500)
    assert cfg2.start_frame == 500


def test_continuous_video_config_hq() -> None:
    cfg = continuous_video_config(start="power_on", hq=True)
    assert cfg.scale == 3
    assert cfg.crf == 15
    assert cfg.preset == "slow"
