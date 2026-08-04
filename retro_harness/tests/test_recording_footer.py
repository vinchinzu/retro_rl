"""Tests for shared recording footer helpers."""

from __future__ import annotations

from retro_harness.actions import buttons, buttons_multi
from retro_harness.video import (
    FOOTER_HEIGHT,
    format_player_buttons,
    frame_timestamp,
    render_footer_frame,
)


def test_footer_extends_frame_height() -> None:
    import numpy as np

    obs = np.zeros((224, 256, 3), dtype=np.uint8)
    frame = render_footer_frame(
        obs,
        upper_left="SCENE 01/05 TEST",
        upper_right="00:12",
        lower_left="SCORE 01234  CUR 032,100",
        action=buttons_multi(p1=("A",), p2=("A",)),
        players=2,
    )
    assert frame.shape == (224 + FOOTER_HEIGHT, 256, 3)


def test_format_player_buttons_multi() -> None:
    action = buttons_multi(p1=("RIGHT", "Y"), p2=("A",))
    label = format_player_buttons(action, players=2)
    assert label == "P1:RIGHT+Y  P2:A"


def test_format_player_buttons_idle() -> None:
    label = format_player_buttons(buttons(), players=1)
    assert label == "P1:---"


def test_format_player_buttons_nes_layout() -> None:
    # NES: [B, null, Select, Start, Up, Down, Left, Right, A]
    action = [1, 0, 0, 0, 0, 0, 0, 1, 1]  # B+RIGHT+A
    label = format_player_buttons(action, players=1, layout="nes")
    assert label == "P1:A+B+RIGHT"


def test_frame_timestamp_format() -> None:
    assert frame_timestamp(0, 60.0) == "F00000  00:00.00"
    assert frame_timestamp(60, 60.0) == "F00060  00:01.00"
    assert frame_timestamp(3661, 60.0).startswith("F03661  ")
