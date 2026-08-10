"""Air Man stage policies for Mega Man 2 (NES).

M3 isolated segments (Clean Bronze):

- **Screen ≥ 1** from ``Level1``: periodic jump-run (legacy ``AirScreen1Policy``).
- **Screen ≥ 2** from ``Level1`` or ``AirLanded``: multi-phase ``AirManPolicy``
  (~521f from Level1, ~225f from AirLanded; 3/3; verified 2026-08-08).
- **Screen ≥ 3 / ≥ 4** from ``AirScreen2``: late-stage ``AirManPolicy(start=screen2)``
  (~241f → s3 HP20; ~502f → s4 HP16; 3/3; verified 2026-08-09).

Level1 recipe (0-based frame index ``i``):

1. ``i < 180``: RIGHT + jump period 50 / hold 12 (+ shoot pulse)
2. ``216 ≤ i < 230``: long land jump onto screen-1 platforms
3. After land (``i ≥ 301`` at grounded AirLanded pose):
   - relative ``r = i - 301``: period 50/12 until ``r = 142``
   - gap jump ``142 ≤ r < 156``
4. Hold clear when ``camera_x_screen ≥ target``

From ``AirLanded``, only phase 3 applies with ``r = i``.

AirScreen2 late recipe (0-based ``i``):

1. ``48 ≤ i < 145``: approach period 45 / hold 16
2. ``145 ≤ i < 180``: continuous jump (fan / tall-gap boost)
3. ``i ≥ 180``: late period 40 / hold 16 through screens 3–4
4. Hold clear when ``camera_x_screen ≥ target``

NES buttons: B=shoot, A=jump (fceumm 9-button layout).
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_harness.input_script import FrameAction
from retro_harness.nes import nes_action, nes_idle_action


def _run_buttons(*, jump: bool, shoot: bool) -> tuple[list[str], str]:
    buttons = ["RIGHT"]
    if jump:
        buttons.append("A")
    if shoot:
        buttons.append("B")
    if jump and shoot:
        reason = "run_jump_shoot"
    elif jump:
        reason = "run_jump"
    elif shoot:
        reason = "run_shoot"
    else:
        reason = "run"
    return buttons, reason


@dataclass
class AirScreen1Policy:
    """Periodic jump-run to clear Air Man screen 0 → screen 1 (legacy)."""

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

        i = max(0, frame - 1)
        jump = self.jump_period > 0 and (i % self.jump_period) < self.jump_hold
        shoot = self.shoot_period > 0 and (i % self.shoot_period) < self.shoot_hold
        buttons, reason = _run_buttons(jump=jump, shoot=shoot)
        return FrameAction(nes_action(*buttons), reason)


@dataclass
class AirManPolicy:
    """Multi-phase Air Man route: Level1 / AirLanded / AirScreen2 starts."""

    # Early screen-0 hop
    early_until: int = 180
    early_period: int = 50
    early_hold: int = 12
    # Land onto screen-1 platforms
    land_jump_start: int = 216
    land_jump_hold: int = 14
    # Absolute 0-based frame of grounded AirLanded (Level1 path)
    land_frame: int = 301
    # From AirLanded / post-land: approach + gap into screen 2
    mid_period: int = 50
    mid_hold: int = 12
    gap_rel: int = 142
    gap_hold: int = 14
    # AirScreen2 late-stage (fans / gaps → screens 3–4)
    s2_approach_start: int = 48
    s2_approach_period: int = 45
    s2_approach_hold: int = 16
    s2_fan_start: int = 145
    s2_fan_end: int = 180
    s2_late_period: int = 40
    s2_late_hold: int = 16
    shoot_period: int = 40
    shoot_hold: int = 2
    target_camera_screen: int = 2
    # "level1" full recipe; "landed" post-land only; "screen2" late-stage
    start: str = "level1"

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

        i = max(0, frame - 1)
        shoot = self.shoot_period > 0 and (i % self.shoot_period) < self.shoot_hold
        jump = self._want_jump(i)
        buttons, reason = _run_buttons(jump=jump, shoot=shoot)
        if jump and self.start == "level1" and self.land_jump_start <= i < (
            self.land_jump_start + self.land_jump_hold
        ):
            reason = "land_jump" if not shoot else "land_jump_shoot"
        elif jump and self.start == "screen2" and self._in_s2_fan(i):
            reason = "fan_hold" if not shoot else "fan_hold_shoot"
        elif jump and self.start == "screen2" and self._in_s2_late(i):
            reason = "late_jump" if not shoot else "late_jump_shoot"
        elif jump and self._in_gap(i):
            reason = "gap_jump" if not shoot else "gap_jump_shoot"
        return FrameAction(nes_action(*buttons), reason)

    def _in_gap(self, i: int) -> bool:
        r = self._rel_mid(i)
        if r is None:
            return False
        return self.gap_rel <= r < self.gap_rel + self.gap_hold

    def _rel_mid(self, i: int) -> int | None:
        if self.start == "landed":
            return i
        if self.start == "level1" and i >= self.land_frame:
            return i - self.land_frame
        return None

    def _in_s2_fan(self, i: int) -> bool:
        return self.s2_fan_start <= i < self.s2_fan_end

    def _in_s2_late(self, i: int) -> bool:
        return i >= self.s2_fan_end

    def _want_jump(self, i: int) -> bool:
        if self.start == "screen2":
            return self._screen2_jump(i)
        if self.start == "landed":
            return self._mid_jump(i)
        if i < self.early_until:
            return (i % self.early_period) < self.early_hold
        if self.land_jump_start <= i < self.land_jump_start + self.land_jump_hold:
            return True
        r = self._rel_mid(i)
        if r is not None:
            return self._mid_jump(r)
        return False

    def _screen2_jump(self, i: int) -> bool:
        """Late-stage from AirScreen2: approach → fan hold → period 40/16."""
        if i < self.s2_approach_start:
            return False
        if i < self.s2_fan_start:
            r = i - self.s2_approach_start
            return (r % self.s2_approach_period) < self.s2_approach_hold
        if i < self.s2_fan_end:
            return True
        r = i - self.s2_fan_end
        return (r % self.s2_late_period) < self.s2_late_hold

    def _mid_jump(self, r: int) -> bool:
        if r < self.gap_rel:
            return (r % self.mid_period) < self.mid_hold
        return self.gap_rel <= r < self.gap_rel + self.gap_hold
