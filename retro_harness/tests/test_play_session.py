"""Tests for retro_harness.play_session module (non-pygame, import-only)."""
import ast
from pathlib import Path

import pytest


def test_syntax_valid():
    """Verify play_session.py is syntactically valid Python."""
    src = Path(__file__).parent.parent / "play_session.py"
    ast.parse(src.read_text())


def test_class_structure():
    """Verify PlaySession class has expected methods."""
    src = Path(__file__).parent.parent / "play_session.py"
    tree = ast.parse(src.read_text())

    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert len(classes) == 1
    assert classes[0].name == "PlaySession"

    methods = [n.name for n in ast.walk(classes[0]) if isinstance(n, ast.FunctionDef)]
    expected = {"__init__", "run", "save_state", "load_state", "set_bot",
                "_main_loop", "_render_frame", "_draw_hud", "_gather_action",
                "_handle_keydown"}
    assert expected.issubset(set(methods)), f"Missing methods: {expected - set(methods)}"


def test_sdl_videodriver_set():
    """Verify SDL_VIDEODRIVER is set to x11 for Hyprland compatibility."""
    src = Path(__file__).parent.parent / "play_session.py"
    text = src.read_text()
    assert 'os.environ.setdefault("SDL_VIDEODRIVER", "x11")' in text


def test_line_count():
    """Play session should be under 400 lines."""
    src = Path(__file__).parent.parent / "play_session.py"
    lines = src.read_text().count("\n")
    assert lines <= 400, f"play_session.py is {lines} lines, expected <= 400"
