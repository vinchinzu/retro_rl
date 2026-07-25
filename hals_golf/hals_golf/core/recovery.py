"""Recovery helpers after human takeover or sticky menus."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hals_golf.core.actions import CANCEL, CONFIRM, idle, press, tap_sequence
from hals_golf.core.scene import SceneDecision


@dataclass
class RecoveryController:
    """Dismiss accidental menus opened by the human↔bot chord.

    PlaySession toggles with ``~`` or L+R+SELECT. L/R can nudge aim; SELECT
    may open tournament stats. Pulse cancel/confirm until the scene settles.
    """

    warmup_frames: int = 90
    _frames_left: int = 0
    _queue: list[np.ndarray] = field(default_factory=list)

    def start(self, *, reason: str = "hotswap") -> None:
        """Begin a recovery warmup sequence."""
        del reason  # reserved for logging hooks
        self._frames_left = self.warmup_frames
        self._queue = []
        # Cancel first (close stats / submenus), then confirm leftovers.
        self._queue.extend(tap_sequence(CANCEL, hold=2, gap=6, times=4))
        self._queue.extend(tap_sequence(CONFIRM, hold=2, gap=10, times=2))
        self._queue.extend(idle() for _ in range(20))

    @property
    def active(self) -> bool:
        return self._frames_left > 0 or bool(self._queue)

    def step(self, decision: SceneDecision) -> np.ndarray | None:
        """Return a recovery action, or None when finished."""
        if self._queue:
            self._frames_left = max(0, self._frames_left - 1)
            return self._queue.pop(0)
        if self._frames_left > 0:
            self._frames_left -= 1
            if decision.needs_dismiss:
                # Alternate cancel/confirm while waiting.
                return press(CANCEL if self._frames_left % 20 < 4 else CONFIRM)
            return idle()
        return None
