"""Repo-wide --headed flag. No pygame window, no emulator."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from retro_harness.headed import HEADED_FLAG_HELP, add_headed_flag, configure_headed


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
