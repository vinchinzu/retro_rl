"""Tests for retro_harness.play_session module (non-pygame, import-only)."""
import ast
from pathlib import Path

from retro_harness.play_session import PlaySession, SPEED_LEVELS


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
                "_limit_frame_rate", "_handle_keydown"}
    assert expected.issubset(set(methods)), f"Missing methods: {expected - set(methods)}"


def test_sdl_videodriver_set():
    """Verify SDL_VIDEODRIVER is selected before pygame import."""
    src = Path(__file__).parent.parent / "play_session.py"
    text = src.read_text()
    assert 'if "SDL_VIDEODRIVER" not in os.environ:' in text
    assert 'os.environ["SDL_VIDEODRIVER"] = "wayland"' in text
    assert 'os.environ["SDL_VIDEODRIVER"] = "x11"' in text


def test_speed_ladder_includes_practice_slow_motion_steps():
    assert SPEED_LEVELS[SPEED_LEVELS.index(0.5):SPEED_LEVELS.index(1.0) + 1] == [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]


def test_initial_speed_uses_nearest_supported_level():
    session = PlaySession(object(), initial_speed=0.79)

    assert session.speed == 0.8


def test_last_action_post_sanitize_returns_copy():
    session = PlaySession(object(), action_size=4)
    session._last_action_post_sanitize = [1, 0, 1, 0]

    action = session.last_action_post_sanitize
    action[0] = 0

    assert session.last_action_post_sanitize == [1, 0, 1, 0]


def test_keyboard_checkpoints_use_public_hooks():
    class Keys:
        mods = 0

        @classmethod
        def get_mods(cls):
            return cls.mods

    class Pygame:
        K_ESCAPE = 0
        K_F1, K_F2, K_F3, K_F4 = range(1, 5)
        KMOD_SHIFT = 1
        key = Keys

    calls = []
    session = PlaySession(object())
    session.on_trigger_save = lambda slot: calls.append(("save", slot))
    session.on_trigger_load = lambda slot: calls.append(("load", slot))

    session._handle_keydown(Pygame, Pygame.K_F2)
    Keys.mods = Pygame.KMOD_SHIFT
    session._handle_keydown(Pygame, Pygame.K_F2)

    assert calls == [("save", 2), ("load", 2)]


def test_line_count():
    """Play session should stay reasonably compact."""
    src = Path(__file__).parent.parent / "play_session.py"
    lines = src.read_text().count("\n")
    assert lines <= 570, f"play_session.py is {lines} lines, expected <= 570"
