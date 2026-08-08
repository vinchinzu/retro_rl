"""Air Man early-stage policies for Mega Man 2 (NES).

M3 isolated segment: from ``Level1``, reach camera X screen ≥ 1
(first horizontal screen transition) without pit death.

Recipe (probe 2026-08-08, 3/3 deterministic @ ~248 frames):

- Hold RIGHT every frame
- Jump (A) for ``jump_hold`` frames every ``jump_period`` (50 / 12)
- Pulse B (shoot) 2 frames every 40

NES buttons: B=shoot, A=jump (fceumm 9-button layout).
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action


@dataclass
class AirScreen1Policy:
    """Periodic jump-run to clear Air Man screen 0 → screen 1."""

    jump_period: int = 50
    jump_hold: int = 12
    shoot_period: int = 40
    shoot_hold: int = 2
    target_camera_screen: int = 1

    def tick(
        self,
        *,
        frame: int,
        health: int,
        camera_x_screen: int,
        fallen: bool = False,
    ) -> FrameAction:
        """Choose one frame of controller input."""
        if health <= 0 or fallen:
            return FrameAction(nes_idle_action(), "dead")
        if camera_x_screen >= self.target_camera_screen:
            return FrameAction(nes_idle_action(), "clear_hold")

        # Policy frame index is 0-based for the modulo recipe.
        i = max(0, frame - 1)
        buttons: list[str] = ["RIGHT"]
        if self.jump_period > 0 and (i % self.jump_period) < self.jump_hold:
            buttons.append("A")
        if self.shoot_period > 0 and (i % self.shoot_period) < self.shoot_hold:
            buttons.append("B")
        reason = "run"
        if "A" in buttons and "B" in buttons:
            reason = "run_jump_shoot"
        elif "A" in buttons:
            reason = "run_jump"
        elif "B" in buttons:
            reason = "run_shoot"
        return FrameAction(nes_action(*buttons), reason)
