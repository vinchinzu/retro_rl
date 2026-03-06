"""No-ROM tests for the shared live-play launcher."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retro_harness.live_play import play_game


def test_play_game_uses_supplied_factories():
    ran = []
    seen = {}

    class _FakeEnv:
        pass

    class _FakeSession:
        def __init__(
            self,
            env,
            *,
            game_dir,
            game,
            scale,
            title,
            bot,
            headless,
            action_size,
            base_fps,
        ):
            seen["env"] = env
            seen["game_dir"] = game_dir
            seen["game"] = game
            seen["scale"] = scale
            seen["title"] = title
            seen["bot"] = bot
            seen["headless"] = headless
            seen["action_size"] = action_size
            seen["base_fps"] = base_fps
            self.on_hud = None

        def run(self):
            ran.append(self.on_hud({}))

    def _env_factory():
        return _FakeEnv()

    def _hud(_info):
        return ["ok"]

    play_game(
        game="TestGame-Snes",
        state="StartState",
        game_dir=ROOT,
        title="Test Title",
        scale=4,
        action_size=9,
        base_fps=30,
        headless=True,
        on_hud=_hud,
        env_factory=_env_factory,
        session_factory=_FakeSession,
    )

    assert seen["game"] == "TestGame-Snes"
    assert seen["title"] == "Test Title"
    assert seen["scale"] == 4
    assert seen["action_size"] == 9
    assert seen["base_fps"] == 30
    assert seen["headless"] is True
    assert ran == [["ok"]]
