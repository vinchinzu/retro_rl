"""Red Tower Ice edge: thin seat → frozen upper_ripper_1.

The thin seat is solid, so a short RIGHT turn is legal. Freeze the y≈520
Ripper on the facing (right) side with UP+X (no d-pad while aiming),
standing Hi-Jump, then drift onto the ice from above.

Do not claim the isolated `_ice_ripper_ladder` hop as natural-predecessor
proof. Do not RIGHT+A from aim-up. Do not treat jump-apex vy=0 as a landing.
"""

from __future__ import annotations

from typing import Any

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import FACING_RIGHT, SuperMetroidState
from super_metroid.routes.kpdr.red_tower.red_ice_climb import (
    THIN_SEAT,
    UPPER_RIPPER_1,
    UPPER_RIPPER_1_LAND_Y,
    UPPER_RIPPER_1_Y,
    VARIANT_ID,
    can_attach_thin_seat_edge,
    checkpoint_supported,
    ripper_at_height,
)
from super_metroid.routes.kpdr.rooms import ROOM_RED_TOWER
from super_metroid.routes.runtime import ControllerSession

POLICY_ID = "red_tower_ice_thin_seat_to_upper_ripper1"

_STAND = frozenset({1, 2})
_TRUE_MORPH = frozenset({29, 30, 31, 32})
_JUMP_UNTIL_Y = UPPER_RIPPER_1_LAND_Y - 27  # 468
_DRIFT_HIGH_Y = UPPER_RIPPER_1_LAND_Y - 10  # 485
_HOVER_Y = (UPPER_RIPPER_1.y_range[0], UPPER_RIPPER_1.y_range[1] + 5)


def _action(*names: str):
    return buttons(*names) if names else idle_action()


class RedIceThinToUr1EdgeRunner:
    """One-action-per-call runner: freeze ur1 at offset, standing hop on top."""

    policy_id = POLICY_ID
    variant_id = VARIANT_ID
    from_checkpoint = THIN_SEAT.checkpoint_id
    to_checkpoint = UPPER_RIPPER_1.checkpoint_id

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
        self.last_reason = "red_ice_thin_ur1_init"
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
        if self.attempts >= self.max_attempts or not THIN_SEAT.matches(state):
            self._fail(reason)
            return
        self._set_phase("face", f"retry {self.attempts}")

    def action(self, state: SuperMetroidState):
        if self.complete or self.failed:
            return None
        if int(state.room_id) != ROOM_RED_TOWER:
            self._fail(f"left room 0x{int(state.room_id):04X}")
            return None

        while not self.complete and not self.failed:
            if self.phase == "stand":
                if int(state.pose) in _STAND or int(state.pose) in (3, 4):
                    self._set_phase("face", "face right")
                    continue
                return self._emit(_action("UP"), "red_ice_thin_ur1_stand")

            if self.phase == "face":
                if int(state.facing) == FACING_RIGHT or self._phase_frames >= 8:
                    self._set_phase("acquire", "track ur1")
                    continue
                if int(state.samus_x) >= 100:
                    return self._emit(_action("UP"), "red_ice_thin_ur1_hold_seat")
                return self._emit(_action("RIGHT"), "red_ice_thin_ur1_face")

            if self.phase == "acquire":
                enemy = ripper_at_height(self.env, UPPER_RIPPER_1_Y)
                if enemy is None:
                    return self._emit(idle_action(), "red_ice_thin_ur1_wait_r")
                signed = int(enemy.x) - int(state.samus_x)
                if enemy.freeze_timer > 40 and signed >= 8:
                    self._target_x = int(enemy.x)
                    self._set_phase("drop_aim", f"frozen x={enemy.x}")
                    continue
                if 8 <= signed <= 36:
                    return self._emit(_action("UP", "X"), "red_ice_thin_ur1_freeze_shot")
                return self._emit(_action("UP"), "red_ice_thin_ur1_wait_dx")

            if self.phase == "drop_aim":
                if int(state.pose) in _STAND or self._phase_frames >= 10:
                    self._set_phase("jump", "standing Hi-Jump")
                    continue
                return self._emit(idle_action(), "red_ice_thin_ur1_drop_aim")

            if self.phase == "jump":
                if int(state.samus_y) <= _JUMP_UNTIL_Y or self._phase_frames >= 32:
                    self._set_phase("land", "drift onto ice top")
                    continue
                return self._emit(_action("A"), "red_ice_thin_ur1_jump")

            if self.phase == "land":
                if checkpoint_supported(self.env, state, UPPER_RIPPER_1):
                    self._set_phase("settle", "verify frozen support")
                    continue
                grounded = (
                    int(state.velocity_y) == 0
                    and int(state.vertical_direction) == 0
                )
                if grounded and int(state.samus_y) >= 620:
                    self._fail(
                        f"fell off seat xy=({state.samus_x},{state.samus_y})"
                    )
                    break
                if grounded and THIN_SEAT.matches(state):
                    self._retry_or_fail(state, "landed back on thin_seat")
                    continue
                if int(state.pose) in _TRUE_MORPH:
                    return self._emit(_action("UP"), "red_ice_thin_ur1_unmorph")
                enemy = ripper_at_height(self.env, UPPER_RIPPER_1_Y)
                ex = int(enemy.x) if enemy is not None else self._target_x
                y = int(state.samus_y)
                x = int(state.samus_x)
                if y <= _DRIFT_HIGH_Y and abs(x - ex) > 3:
                    direction = "RIGHT" if x < ex else "LEFT"
                    return self._emit(_action(direction), "red_ice_thin_ur1_drift_high")
                if _HOVER_Y[0] <= y <= _HOVER_Y[1]:
                    if abs(x - ex) > 3:
                        direction = "RIGHT" if x < ex else "LEFT"
                        return self._emit(_action(direction), "red_ice_thin_ur1_hover_track")
                    return self._emit(idle_action(), "red_ice_thin_ur1_hover")
                if abs(x - ex) > 3:
                    direction = "RIGHT" if x < ex else "LEFT"
                    return self._emit(_action(direction), "red_ice_thin_ur1_track")
                return self._emit(idle_action(), "red_ice_thin_ur1_fall")

            if self.phase == "settle":
                if not checkpoint_supported(self.env, state, UPPER_RIPPER_1):
                    self._retry_or_fail(state, "unstable ur1 support")
                    continue
                if self._settle_frames >= 8:
                    self.complete = True
                    self._set_phase("complete", UPPER_RIPPER_1.checkpoint_id)
                    break
                self._settle_frames += 1
                return self._emit(idle_action(), "red_ice_thin_ur1_checkpoint_settle")

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


def play_thin_seat_to_upper_ripper1(
    session: ControllerSession,
    *,
    max_frames: int = 360,
) -> SuperMetroidState:
    """Synchronous facade: grounded thin seat → grounded frozen ur1."""
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError("Red Ice thin→ur1 needs session.env")
    if not can_attach_thin_seat_edge(session.state):
        raise TimeoutError(
            f"{POLICY_ID}: not on thin_seat "
            f"xy=({session.state.samus_x},{session.state.samus_y}) "
            f"p={session.state.pose}"
        )
    runner = RedIceThinToUr1EdgeRunner(env, max_frames=max_frames)
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
    "RedIceThinToUr1EdgeRunner",
    "play_thin_seat_to_upper_ripper1",
]
