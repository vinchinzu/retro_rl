"""Stage 1 segment policy for TMNT II (NES).

First-wave clear (M3): from ``Level1``, score ≥ 5 without death.

Recipe discovered by probe:

1. **Open (score < 3):** walk RIGHT, jump (A), mash B / RIGHT+B.
2. **Lock face (score < 5):** feet pin the right edge — face LEFT and B
   (pure RIGHT+B stalls at score 4).
3. **Push (score ≥ 5):** RIGHT pressure + B for later packs.

NES buttons: B=attack, A=jump (fceumm 9-button layout).
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.nes import nes_action, nes_idle_action
from snes_oneshot.primitives import FrameAction


@dataclass
class Stage1Policy:
    """Score-gated Stage 1 combat policy."""

    target_score: int = 5

    def tick(self, *, frame: int, score: int, health: int) -> FrameAction:
        """Choose one frame of controller input."""
        if health <= 0:
            return FrameAction(nes_idle_action(), "dead")
        if score < 3:
            return self._open(frame)
        if score < self.target_score:
            return self._face_left_lock(frame)
        return self._push(frame)

    def _open(self, frame: int) -> FrameAction:
        phase = frame % 25
        if phase < 5:
            return FrameAction(nes_action("RIGHT"), "open_walk")
        if phase < 8:
            return FrameAction(nes_action("A"), "open_jump")
        if phase < 16:
            return FrameAction(nes_action("B"), "open_attack")
        if phase < 20:
            return FrameAction(nes_action("RIGHT", "B"), "open_rb")
        return FrameAction(nes_action("B"), "open_attack")

    def _face_left_lock(self, frame: int) -> FrameAction:
        phase = frame % 20
        if phase < 4:
            return FrameAction(nes_action("LEFT"), "lock_face")
        if phase < 12:
            return FrameAction(nes_action("B"), "lock_attack")
        if phase < 15:
            return FrameAction(nes_action("LEFT", "B"), "lock_lb")
        if phase < 17:
            return FrameAction(nes_action("A"), "lock_jump")
        return FrameAction(nes_action("B"), "lock_attack")

    def _push(self, frame: int) -> FrameAction:
        phase = frame % 30
        if phase < 8:
            return FrameAction(nes_action("RIGHT"), "push_walk")
        if phase < 18:
            return FrameAction(nes_action("B"), "push_attack")
        if phase < 22:
            return FrameAction(nes_action("LEFT"), "push_face")
        return FrameAction(nes_action("B"), "push_attack")
