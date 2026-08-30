"""Red Tower Ice edge: frozen lower_ripper_3 → lower_ripper_4.

136px gap: standing Hi-Jump apex is only ~3px above r4 and falls through.
Crouch-jump from the ice clears it. Freeze r4 on the facing (right) side
with UP+X (no d-pad), crouch, Hi-Jump crouch-jump, drift onto the ice
from above.

Do not RIGHT+A from aim-up: that becomes pose 81 and falls through ice.
Do not treat jump-apex vy=0 as a landing.
"""

from __future__ import annotations

from typing import Any

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.kpdr.red_tower.red_ice_climb import (
    LOWER_RIPPER_2,
    LOWER_RIPPER_3,
    LOWER_RIPPER_4,
    LOW_RIPPER_3_Y,
    LOW_RIPPER_4_LAND_Y,
    LOW_RIPPER_4_Y,
    VARIANT_ID,
    can_attach_ripper3_edge,
    checkpoint_supported,
    ripper_at_height,
)
from super_metroid.routes.kpdr.rooms import ROOM_RED_TOWER
from super_metroid.routes.runtime import ControllerSession

POLICY_ID = "red_tower_ice_ripper3_to_ripper4"

_STAND = frozenset({1, 2})
_CROUCH = frozenset({39, 40})
_TRUE_MORPH = frozenset({29, 30, 31, 32})
_JUMP_UNTIL_Y = 2008
_DRIFT_HIGH_Y = LOW_RIPPER_4_LAND_Y - 8  # 2015
_HOVER_Y = (LOWER_RIPPER_4.y_range[0], LOWER_RIPPER_4.y_range[1] + 5)


def _action(*names: str):
    return buttons(*names) if names else idle_action()


class RedIceRipper34EdgeRunner:
    """One-action-per-call runner: freeze r4 at offset, crouch-jump on top."""

    policy_id = POLICY_ID
    variant_id = VARIANT_ID
    from_checkpoint = LOWER_RIPPER_3.checkpoint_id
    to_checkpoint = LOWER_RIPPER_4.checkpoint_id

    def __init__(self, env: Any, *, max_frames: int = 360, max_attempts: int = 2) -> None:
        self.env = env
        self.max_frames = max(1, int(max_frames))
        self.max_attempts = max(1, int(max_attempts))
        self.phase = "stand"
        self.detail = "stand"
        self.frames = 0
        self.attempts = 0
        self.complete = False
        self.failed = False
        self.failure = ""
        self.last_reason = "red_ice_r34_init"
        self._phase_frames = 0
        self._settle_frames = 0
        self._target_x = 0

    def _fail(self, reason: str) -> None:
        self.failed = True
        self.failure = reason
        self.phase = "failed"
        self.detail = reason

    def _emit(self, action, reason: str):
        self.frames += 1
        self._phase_frames += 1
        self.last_reason = reason
        if self.frames > self.max_frames:
            self._fail(f"budget>{self.max_frames}f")
            return idle_action()
        return action

    def _set_phase(self, phase: str, detail: str = "") -> None:
        self.phase = phase
        self.detail = detail or phase
        self._phase_frames = 0

    def _retry_or_fail(self, state: SuperMetroidState, reason: str) -> None:
        self.attempts += 1
        if self.attempts >= self.max_attempts or not LOWER_RIPPER_3.matches(state):
            self._fail(reason)
            return
        self._set_phase("acquire", f"retry {self.attempts}")

    def action(self, state: SuperMetroidState):
        if self.complete or self.failed:
            return None
        if int(state.room_id) != ROOM_RED_TOWER:
            self._fail(f"left room 0x{int(state.room_id):04X}")
            return None

        while not self.complete and not self.failed:
            if self.phase == "stand":
                if int(state.pose) in _STAND or int(state.pose) in (3, 4):
                    self._set_phase("acquire", "track r4")
                    continue
                return self._emit(_action("UP"), "red_ice_r34_stand")

            if self.phase == "acquire":
                support = ripper_at_height(self.env, LOW_RIPPER_3_Y)
                enemy = ripper_at_height(self.env, LOW_RIPPER_4_Y)
                if support is None or support.freeze_timer < 22:
                    self._fail("r3 thawed before r4 freeze")
                    break
                if enemy is None:
                    return self._emit(idle_action(), "red_ice_r34_wait_r4")
                signed = int(enemy.x) - int(state.samus_x)
                if enemy.freeze_timer > 40 and signed >= 8:
                    self._target_x = int(enemy.x)
                    self._set_phase("drop_aim", f"frozen x={enemy.x}")
                    continue
                if 8 <= signed <= 36:
                    return self._emit(_action("UP", "X"), "red_ice_r34_freeze_shot")
                return self._emit(_action("UP"), "red_ice_r34_wait_dx")

            if self.phase == "drop_aim":
                if int(state.pose) in _STAND or self._phase_frames >= 10:
                    self._set_phase("crouch", "crouch-jump setup")
                    continue
                return self._emit(idle_action(), "red_ice_r34_drop_aim")

            if self.phase == "crouch":
                if int(state.pose) in _CROUCH or self._phase_frames >= 8:
                    self._set_phase("jump", "Hi-Jump crouch-jump")
                    continue
                return self._emit(_action("DOWN"), "red_ice_r34_crouch")

            if self.phase == "jump":
                if int(state.samus_y) <= _JUMP_UNTIL_Y or self._phase_frames >= 36:
                    self._set_phase("land", "drift onto ice top")
                    continue
                return self._emit(_action("A"), "red_ice_r34_jump")

            if self.phase == "land":
                if checkpoint_supported(self.env, state, LOWER_RIPPER_4):
                    self._set_phase("settle", "verify frozen support")
                    continue
                grounded = (
                    int(state.velocity_y) == 0
                    and int(state.vertical_direction) == 0
                )
                if grounded and LOWER_RIPPER_2.matches(state):
                    self._fail(
                        f"fell past r3 xy=({state.samus_x},{state.samus_y})"
                    )
                    break
                if grounded and LOWER_RIPPER_3.matches(state):
                    self._retry_or_fail(state, "landed back on r3")
                    continue
                if int(state.pose) in _TRUE_MORPH:
                    return self._emit(_action("UP"), "red_ice_r34_unmorph")
                enemy = ripper_at_height(self.env, LOW_RIPPER_4_Y)
                ex = int(enemy.x) if enemy is not None else self._target_x
                y = int(state.samus_y)
                x = int(state.samus_x)
                if y <= _DRIFT_HIGH_Y and abs(x - ex) > 3:
                    direction = "RIGHT" if x < ex else "LEFT"
                    return self._emit(_action(direction), "red_ice_r34_drift_high")
                if _HOVER_Y[0] <= y <= _HOVER_Y[1]:
                    if abs(x - ex) > 3:
                        direction = "RIGHT" if x < ex else "LEFT"
                        return self._emit(_action(direction), "red_ice_r34_hover_track")
                    return self._emit(idle_action(), "red_ice_r34_hover")
                if abs(x - ex) > 3:
                    direction = "RIGHT" if x < ex else "LEFT"
                    return self._emit(_action(direction), "red_ice_r34_track")
                return self._emit(idle_action(), "red_ice_r34_fall")

            if self.phase == "settle":
                if not checkpoint_supported(self.env, state, LOWER_RIPPER_4):
                    self._retry_or_fail(state, "unstable r4 support")
                    continue
                if self._settle_frames >= 8:
                    self.complete = True
                    self._set_phase("complete", LOWER_RIPPER_4.checkpoint_id)
                    break
                self._settle_frames += 1
                return self._emit(idle_action(), "red_ice_r34_checkpoint_settle")

            self._fail(f"unknown phase {self.phase}")

        return None

    def status(self) -> dict[str, Any]:
        return {
            "policy": self.policy_id,
            "variant": self.variant_id,
            "phase": self.phase,
            "detail": self.detail,
            "frames": self.frames,
            "attempts": self.attempts,
            "complete": self.complete,
            "failed": self.failed,
            "failure": self.failure,
            "from_checkpoint": self.from_checkpoint,
            "to_checkpoint": self.to_checkpoint,
        }


def play_ripper3_to_ripper4(
    session: ControllerSession,
    *,
    max_frames: int = 360,
) -> SuperMetroidState:
    """Synchronous facade: frozen r3 → grounded frozen r4."""
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError("Red Ice r3→r4 needs session.env")
    if not can_attach_ripper3_edge(session.state):
        raise TimeoutError(
            f"{POLICY_ID}: not on lower_ripper_3 "
            f"xy=({session.state.samus_x},{session.state.samus_y}) "
            f"p={session.state.pose}"
        )
    runner = RedIceRipper34EdgeRunner(env, max_frames=max_frames)
    while not runner.complete and not runner.failed:
        action = runner.action(session.state)
        if action is None:
            break
        session.step(action, runner.last_reason)
    if runner.failed or not runner.complete:
        raise TimeoutError(
            f"{POLICY_ID}: {runner.failure or 'did not complete'}; "
            f"phase={runner.phase} frames={runner.frames} "
            f"xy=({session.state.samus_x},{session.state.samus_y})"
        )
    return session.state


__all__ = [
    "POLICY_ID",
    "RedIceRipper34EdgeRunner",
    "play_ripper3_to_ripper4",
]
