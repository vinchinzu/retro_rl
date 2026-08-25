"""Repo-wide --headed flag. No pygame window, no emulator."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from retro_harness.headed import (
    HEADED_FLAG_HELP,
    add_headed_flag,
    bot_speed_timing,
    configure_headed,
    headed_emu_repeat,
)


def test_add_headed_flag_is_store_true() -> None:
    parser = argparse.ArgumentParser()
    add_headed_flag(parser)
    assert parser.parse_args([]).headed is False
    assert parser.parse_args(["--headed"]).headed is True
    help_text = parser.format_help()
    assert "--headed" in help_text
    assert "bot on" in help_text.lower() or "pygame" in help_text.lower()


def test_add_headed_flag_help_override() -> None:
    parser = argparse.ArgumentParser()
    add_headed_flag(parser, help="Start BOT immediately (window already open).")
    help_text = parser.format_help()
    assert "Start BOT immediately" in help_text
    assert HEADED_FLAG_HELP not in help_text


def test_configure_headed_drops_dummy_and_picks_a_driver(
    monkeypatch: object,
) -> None:
    monkeypatch.setenv("HEADLESS", "1")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
    monkeypatch.delenv("DISPLAY", raising=False)
    configure_headed()
    assert "HEADLESS" not in os.environ
    assert os.environ["SDL_VIDEODRIVER"] == "wayland"
    assert os.environ.get("SDL_AUDIODRIVER") == "dummy"


def test_configure_headed_falls_back_to_x11_without_wayland(
    monkeypatch: object,
) -> None:
    monkeypatch.delenv("HEADLESS", raising=False)
    monkeypatch.delenv("SDL_VIDEODRIVER", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    configure_headed()
    assert os.environ["SDL_VIDEODRIVER"] == "x11"


def test_headed_module_does_not_import_harvest_or_play_session() -> None:
    src = (Path(__file__).resolve().parents[1] / "headed.py").read_text(
        encoding="utf-8"
    )
    assert "harvest" not in src
    assert "play_session" not in src
    assert "add_headed_flag" in src
    assert "attach_headed" in src
    assert "headed_emu_repeat" in src
    assert "[ ] speed" in src


def test_bot_4x_repeats_four_emu_steps_at_60hz() -> None:
    repeat, tick, skip = bot_speed_timing(4.0, bot=True)
    assert repeat == 4
    assert tick == 60
    assert skip is False


def test_tab_turbo_unthrottles() -> None:
    repeat, tick, skip = bot_speed_timing(4.0, turbo=True, bot=True)
    assert repeat == 1
    assert tick == 0
    assert skip is True


def test_headed_emu_repeat_is_one_without_window() -> None:
    assert headed_emu_repeat(object()) == 1
