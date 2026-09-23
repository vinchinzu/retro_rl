"""The process's one live emulator, for controllers built mid-stage.

stable-retro allows one emulator per process, so "the env" is well defined.
Stage runners bind it (``route.chain.bind_controller_env``); controllers that
a stage builds lazily (a room clear inside a door hop) read it here instead
of walking the ROM tile map blind.
"""

from __future__ import annotations

from typing import Any

_ENV: Any = None


def bind(env: Any) -> None:
    global _ENV
    _ENV = env


def current() -> Any:
    return _ENV
