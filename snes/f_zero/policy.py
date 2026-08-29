"""Mute City centerline follow: match heading to checkpoint, recover off walls."""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from retro_harness.bot_runner import BehaviorNode, NodeStatus, TickResult
from retro_harness.input_script import FrameAction
from retro_harness.ram_state import GameState

from f_zero.ram import SCREEN_REVERSE

DEADZONE = 3
SHARP_ERROR = 16


class CenterlinePolicy(BehaviorNode):
    """Hold B and steer Angle8 heading onto the current checkpoint facing."""

    name = "MuteCityCenterline"

    def tick(self, state: GameState) -> TickResult:
        extras = state.extras
        if state.player_dead or state.level_complete:
            reason = "lap" if state.level_complete else "crash"
            status = (
                NodeStatus.SUCCESS if state.level_complete else NodeStatus.FAILURE
            )
            return TickResult(
                status=status,
                action=FrameAction(idle_action(), reason),
                reason=reason,
            )
        if not extras.get("racing"):
            return TickResult(
                status=NodeStatus.RUNNING,
                action=FrameAction(buttons("B"), "countdown"),
                reason="countdown",
            )
        err = int(extras.get("heading_error", 0))
        damaged = bool(extras.get("damaged"))
        reverse = bool(int(extras.get("screen_text", 0)) & SCREEN_REVERSE)
        keys = ["B"]
        if damaged or reverse:
            if err > 0:
                keys.append("LEFT")
                reason = "recover_left"
            elif err < 0:
                keys.append("RIGHT")
                reason = "recover_right"
            else:
                reason = "recover_hold"
        elif err > SHARP_ERROR:
            keys.extend(("LEFT", "L"))
            reason = "sharp_left"
        elif err > DEADZONE:
            keys.append("LEFT")
            reason = "left"
        elif err < -SHARP_ERROR:
            keys.extend(("RIGHT", "R"))
            reason = "sharp_right"
        elif err < -DEADZONE:
            keys.append("RIGHT")
            reason = "right"
        else:
            reason = "center"
        return TickResult(
            status=NodeStatus.RUNNING,
            action=FrameAction(buttons(*keys), reason),
            reason=reason,
        )
