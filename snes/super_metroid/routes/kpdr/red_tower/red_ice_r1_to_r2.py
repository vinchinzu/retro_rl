"""Red Tower Ice edge: frozen lower_ripper_1 → lower_ripper_2.

Starts on the first frozen Ripper (floor freeze + standing hop). Freezes the next
Ripper with a vertical Ice shot (no d-pad — walking falls off the ice),
drops aim-up, Hi-Jump standing, drifts onto the ice top from above.

Do not RIGHT+A from aim-up: that becomes pose 81 and falls through ice.
Do not treat jump-apex vy=0 as a landing.
"""

from __future__ import annotations

from typing import Any

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.kpdr.red_tower.red_ice_climb import (
    BOTTOM_FLOOR,
    LOWER_RIPPER_1,
    LOWER_RIPPER_2,
    MID_RIPPER_Y,
    VARIANT_ID,
    can_attach_ripper1_edge,
    checkpoint_supported,
    ripper_at_height,
)
from super_metroid.routes.kpdr.rooms import ROOM_RED_TOWER
from super_metroid.routes.runtime import ControllerSession

POLICY_ID = "red_tower_ice_ripper1_to_ripper2"

_STAND = frozenset({1, 2})
_TRUE_MORPH = frozenset({29, 30, 31, 32})


def _action(*names: str):
    return buttons(*names) if names else idle_action()


class RedIceRipper12EdgeRunner:
    """One-action-per-call runner: freeze r2 at offset, standing hop on top."""

    policy_id = POLICY_ID
    variant_id = VARIANT_ID
    from_checkpoint = LOWER_RIPPER_1.checkpoint_id
    to_checkpoint = LOWER_RIPPER_2.checkpoint_id

    def __init__(self, env: Any, *, max_frames: int = 280, max_attempts: int = 2) -> None:
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
        self.last_reason = "red_ice_r12_init"
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
        if self.attempts >= self.max_attempts or not LOWER_RIPPER_1.matches(state):
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
                    self._set_phase("acquire", "track r2")
                    continue
                return self._emit(_action("UP"), "red_ice_r12_stand")

            if self.phase == "acquire":
                support = ripper_at_height(self.env, LOWER_RIPPER_1.support_enemy_y or 0)
                enemy = ripper_at_height(self.env, MID_RIPPER_Y)
                if support is None or support.freeze_timer < 22:
                    self._fail("r1 thawed before r2 freeze")
                    break
                if enemy is None:
                    return self._emit(idle_action(), "red_ice_r12_wait_r2")
                dx = abs(int(enemy.x) - int(state.samus_x))
                if enemy.freeze_timer > 40 and dx >= 8:
                    self._target_x = int(enemy.x)
                    self._set_phase("drop_aim", f"frozen x={enemy.x}")
                    continue
                if 8 <= dx <= 36:
                    return self._emit(_action("UP", "X"), "red_ice_r12_freeze_shot")
                return self._emit(_action("UP"), "red_ice_r12_wait_dx")

            if self.phase == "drop_aim":
                if int(state.pose) in _STAND or self._phase_frames >= 10:
                    self._set_phase("jump", "standing Hi-Jump")
                    continue
                return self._emit(idle_action(), "red_ice_r12_drop_aim")

            if self.phase == "jump":
                if int(state.samus_y) <= 2228 or self._phase_frames >= 32:
                    self._set_phase("land", "drift onto ice top")
                    continue
                return self._emit(_action("A"), "red_ice_r12_jump")

            if self.phase == "land":
                if checkpoint_supported(self.env, state, LOWER_RIPPER_2):
                    self._set_phase("settle", "verify frozen support")
                    continue
                grounded = (
                    int(state.velocity_y) == 0
                    and int(state.vertical_direction) == 0
                )
                if grounded and BOTTOM_FLOOR.matches(state):
                    self._retry_or_fail(
                        state,
                        f"fell to floor xy=({state.samus_x},{state.samus_y})",
                    )
                    continue
                if grounded and LOWER_RIPPER_1.matches(state):
                    self._retry_or_fail(state, "landed back on r1")
                    continue
                if int(state.pose) in _TRUE_MORPH:
                    return self._emit(_action("UP"), "red_ice_r12_unmorph")
                enemy = ripper_at_height(self.env, MID_RIPPER_Y)
                ex = int(enemy.x) if enemy is not None else self._target_x
                y = int(state.samus_y)
                x = int(state.samus_x)
                if y <= 2245 and abs(x - ex) > 3:
                    direction = "RIGHT" if x < ex else "LEFT"
                    return self._emit(_action(direction), "red_ice_r12_drift_high")
                if 2238 <= y <= 2275:
                    return self._emit(idle_action(), "red_ice_r12_hover")
                if abs(x - ex) > 3:
                    direction = "RIGHT" if x < ex else "LEFT"
                    return self._emit(_action(direction), "red_ice_r12_track")
                return self._emit(idle_action(), "red_ice_r12_fall")

            if self.phase == "settle":
                if not checkpoint_supported(self.env, state, LOWER_RIPPER_2):
                    self._retry_or_fail(state, "unstable r2 support")
                    continue
                if self._settle_frames >= 8:
                    self.complete = True
                    self._set_phase("complete", LOWER_RIPPER_2.checkpoint_id)
                    break
                self._settle_frames += 1
                return self._emit(idle_action(), "red_ice_r12_checkpoint_settle")

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


def play_ripper1_to_ripper2(
    session: ControllerSession,
    *,
    max_frames: int = 280,
) -> SuperMetroidState:
    """Synchronous facade: frozen r1 → grounded frozen r2."""
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError("Red Ice r1→r2 needs session.env")
    if not can_attach_ripper1_edge(session.state):
        raise TimeoutError(
            f"{POLICY_ID}: not on lower_ripper_1 "
            f"xy=({session.state.samus_x},{session.state.samus_y}) "
            f"p={session.state.pose}"
        )
    runner = RedIceRipper12EdgeRunner(env, max_frames=max_frames)
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
    "RedIceRipper12EdgeRunner",
    "play_ripper1_to_ripper2",
]
