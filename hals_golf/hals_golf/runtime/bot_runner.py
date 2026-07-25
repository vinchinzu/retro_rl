"""Golf-aware BotRunner that injects emulator RAM and forwards handoff hooks."""

from __future__ import annotations

from typing import Any

import numpy as np

from retro_harness.bot_runner import BotRunner
from retro_harness.mission_control import MissionSnapshot


class GolfBotRunner(BotRunner):
    """BotRunner that reads ``env.get_ram()`` and forwards mission handoff."""

    def __init__(self, task: Any, *, env: Any = None, action_size: int = 12) -> None:
        super().__init__(task, action_size=action_size)
        self.env = env

    def bind_env(self, env: Any) -> None:
        """Attach the live emulator env after construction."""
        self.env = env

    def __call__(self, obs: Any, info: dict[str, Any]) -> np.ndarray | None:
        enriched = dict(info)
        if self.env is not None and hasattr(self.env, "get_ram"):
            enriched["ram"] = np.asarray(self.env.get_ram(), dtype=np.uint8)
        return super().__call__(obs, enriched)

    def mission_status(self) -> MissionSnapshot:
        status_fn = getattr(self.task, "mission_status", None)
        if callable(status_fn):
            return status_fn()
        return super().mission_status()

    def on_human_takeover(self) -> None:
        hook = getattr(self.task, "on_human_takeover", None)
        if callable(hook):
            hook()

    def on_autopilot_resume(self) -> None:
        hook = getattr(self.task, "on_autopilot_resume", None)
        if callable(hook):
            hook()
