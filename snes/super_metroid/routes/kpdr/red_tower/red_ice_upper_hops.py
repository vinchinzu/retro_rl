"""Red Tower Ice edges: frozen upper_ripper_1 → 2 → 3 → 4.

Same family as the lower standing hops: freeze the next Ripper with a
vertical Ice shot (no d-pad — walking falls off the ice), drop aim-up,
Hi-Jump standing, drift onto the ice top from above.

Do not RIGHT+A from aim-up. Do not treat jump-apex vy=0 as a landing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.kpdr.red_tower.red_ice_climb import (
    THIN_SEAT,
    UPPER_RIPPER_1,
    UPPER_RIPPER_1_Y,
    UPPER_RIPPER_2,
    UPPER_RIPPER_2_LAND_Y,
    UPPER_RIPPER_2_Y,
    UPPER_RIPPER_3,
    UPPER_RIPPER_3_LAND_Y,
    UPPER_RIPPER_3_Y,
    UPPER_RIPPER_4,
    UPPER_RIPPER_4_LAND_Y,
    UPPER_RIPPER_4_Y,
    VARIANT_ID,
    RedIceCheckpoint,
    can_attach_upper_ripper1_edge,
    can_attach_upper_ripper2_edge,
    can_attach_upper_ripper3_edge,
    checkpoint_supported,
    ripper_at_height,
)
from super_metroid.routes.kpdr.rooms import ROOM_RED_TOWER
from super_metroid.routes.runtime import ControllerSession

_STAND = frozenset({1, 2})
_TRUE_MORPH = frozenset({29, 30, 31, 32})


def _action(*names: str):
    return buttons(*names) if names else idle_action()


@dataclass(frozen=True)
class UpperRipperHopSpec:
    policy_id: str
    from_checkpoint: RedIceCheckpoint
    to_checkpoint: RedIceCheckpoint
    support_y: int
    target_y: int
    land_y: int
    past_checkpoint: RedIceCheckpoint
    freeze_dx: tuple[int, int] = (8, 36)
    aim_before_shot: bool = False

    @property
    def jump_until_y(self) -> int:
        return self.land_y - 27

    @property
    def drift_high_y(self) -> int:
        return self.land_y - 10

    @property
    def hover_y(self) -> tuple[int, int]:
        return (
            self.to_checkpoint.y_range[0],
            self.to_checkpoint.y_range[1] + 5,
        )


UR12 = UpperRipperHopSpec(
    "red_tower_ice_upper_ripper1_to_2",
    UPPER_RIPPER_1,
    UPPER_RIPPER_2,
    UPPER_RIPPER_1_Y,
    UPPER_RIPPER_2_Y,
    UPPER_RIPPER_2_LAND_Y,
    THIN_SEAT,
)
UR23 = UpperRipperHopSpec(
    "red_tower_ice_upper_ripper2_to_3",
    UPPER_RIPPER_2,
    UPPER_RIPPER_3,
    UPPER_RIPPER_2_Y,
    UPPER_RIPPER_3_Y,
    UPPER_RIPPER_3_LAND_Y,
    UPPER_RIPPER_1,
)
UR34 = UpperRipperHopSpec(
    "red_tower_ice_upper_ripper3_to_4",
    UPPER_RIPPER_3,
    UPPER_RIPPER_4,
    UPPER_RIPPER_3_Y,
    UPPER_RIPPER_4_Y,
    UPPER_RIPPER_4_LAND_Y,
    UPPER_RIPPER_2,
    freeze_dx=(10, 28),
    aim_before_shot=True,
)


class RedIceUpperRipperHopRunner:
    """One-action-per-call runner: freeze next upper Ripper, standing hop."""

    variant_id = VARIANT_ID

    def __init__(
        self,
        env: Any,
        spec: UpperRipperHopSpec,
        *,
        max_frames: int = 360,
        max_attempts: int = 2,
    ) -> None:
        self.env = env
        self.spec = spec
        self.policy_id = spec.policy_id
        self.from_checkpoint = spec.from_checkpoint.checkpoint_id
        self.to_checkpoint = spec.to_checkpoint.checkpoint_id
        self.max_frames = max(1, int(max_frames))
        self.max_attempts = max(1, int(max_attempts))
        self.phase = "stand"
        self.detail = "stand"
        self.frames = 0
        self.attempts = 0
        self.complete = False
        self.failed = False
        self.failure = ""
        self.last_reason = f"{spec.policy_id}_init"
        self._phase_frames = 0
        self._settle_frames = 0
        self._target_x = 0
        self._tag = spec.policy_id.replace("red_tower_ice_", "red_ice_")

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
        if self.attempts >= self.max_attempts or not self.spec.from_checkpoint.matches(state):
            self._fail(reason)
            return
        self._set_phase("acquire", f"retry {self.attempts}")

    def action(self, state: SuperMetroidState):
        if self.complete or self.failed:
            return None
        if int(state.room_id) != ROOM_RED_TOWER:
            self._fail(f"left room 0x{int(state.room_id):04X}")
            return None
        spec = self.spec
        tag = self._tag

        while not self.complete and not self.failed:
            if self.phase == "stand":
                if int(state.pose) in _STAND or int(state.pose) in (3, 4):
                    self._set_phase("acquire", f"track {spec.to_checkpoint.checkpoint_id}")
                    continue
                return self._emit(_action("UP"), f"{tag}_stand")

            if self.phase == "acquire":
                support = ripper_at_height(self.env, spec.support_y)
                enemy = ripper_at_height(self.env, spec.target_y)
                if support is None or support.freeze_timer < 22:
                    self._fail("support thawed before next freeze")
                    break
                if enemy is None:
                    return self._emit(idle_action(), f"{tag}_wait_r")
                signed = int(enemy.x) - int(state.samus_x)
                if enemy.freeze_timer > 40 and signed >= spec.freeze_dx[0]:
                    self._target_x = int(enemy.x)
                    self._set_phase("drop_aim", f"frozen x={enemy.x}")
                    continue
                lo, hi = spec.freeze_dx
                if lo <= signed <= hi:
                    if spec.aim_before_shot and int(state.pose) not in (3, 4):
                        return self._emit(_action("UP"), f"{tag}_aim")
                    return self._emit(_action("UP", "X"), f"{tag}_freeze_shot")
                return self._emit(_action("UP"), f"{tag}_wait_dx")

            if self.phase == "drop_aim":
                if int(state.pose) in _STAND or self._phase_frames >= 10:
                    self._set_phase("jump", "standing Hi-Jump")
                    continue
                return self._emit(idle_action(), f"{tag}_drop_aim")

            if self.phase == "jump":
                if int(state.samus_y) <= spec.jump_until_y or self._phase_frames >= 32:
                    self._set_phase("land", "drift onto ice top")
                    continue
                return self._emit(_action("A"), f"{tag}_jump")

            if self.phase == "land":
                if checkpoint_supported(self.env, state, spec.to_checkpoint):
                    self._set_phase("settle", "verify frozen support")
                    continue
                grounded = (
                    int(state.velocity_y) == 0
                    and int(state.vertical_direction) == 0
                )
                if grounded and spec.past_checkpoint.matches(state):
                    self._fail(
                        f"fell past {spec.from_checkpoint.checkpoint_id} "
                        f"xy=({state.samus_x},{state.samus_y})"
                    )
                    break
                if grounded and spec.from_checkpoint.matches(state):
                    self._retry_or_fail(
                        state,
                        f"landed back on {spec.from_checkpoint.checkpoint_id}",
                    )
                    continue
                if int(state.pose) in _TRUE_MORPH:
                    return self._emit(_action("UP"), f"{tag}_unmorph")
                enemy = ripper_at_height(self.env, spec.target_y)
                ex = int(enemy.x) if enemy is not None else self._target_x
                y = int(state.samus_y)
                x = int(state.samus_x)
                if y <= spec.drift_high_y and abs(x - ex) > 3:
                    direction = "RIGHT" if x < ex else "LEFT"
                    return self._emit(_action(direction), f"{tag}_drift_high")
                hover_lo, hover_hi = spec.hover_y
                if hover_lo <= y <= hover_hi:
                    if abs(x - ex) > 3:
                        direction = "RIGHT" if x < ex else "LEFT"
                        return self._emit(_action(direction), f"{tag}_hover_track")
                    return self._emit(idle_action(), f"{tag}_hover")
                if abs(x - ex) > 3:
                    direction = "RIGHT" if x < ex else "LEFT"
                    return self._emit(_action(direction), f"{tag}_track")
                return self._emit(idle_action(), f"{tag}_fall")

            if self.phase == "settle":
                if not checkpoint_supported(self.env, state, spec.to_checkpoint):
                    self._retry_or_fail(state, "unstable frozen support")
                    continue
                if self._settle_frames >= 8:
                    self.complete = True
                    self._set_phase("complete", spec.to_checkpoint.checkpoint_id)
                    break
                self._settle_frames += 1
                return self._emit(idle_action(), f"{tag}_checkpoint_settle")

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


def _play_hop(
    session: ControllerSession,
    spec: UpperRipperHopSpec,
    attach,
    *,
    max_frames: int,
) -> SuperMetroidState:
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError(f"{spec.policy_id}: needs session.env")
    if not attach(session.state):
        raise TimeoutError(
            f"{spec.policy_id}: not on {spec.from_checkpoint.checkpoint_id} "
            f"xy=({session.state.samus_x},{session.state.samus_y}) "
            f"p={session.state.pose}"
        )
    runner = RedIceUpperRipperHopRunner(env, spec, max_frames=max_frames)
    while not runner.complete and not runner.failed:
        action = runner.action(session.state)
        if action is None:
            break
        session.step(action, runner.last_reason)
    if runner.failed or not runner.complete:
        raise TimeoutError(
            f"{spec.policy_id}: {runner.failure or 'did not complete'}; "
            f"phase={runner.phase} frames={runner.frames} "
            f"xy=({session.state.samus_x},{session.state.samus_y})"
        )
    return session.state


def play_upper_ripper1_to_2(
    session: ControllerSession,
    *,
    max_frames: int = 360,
) -> SuperMetroidState:
    """Frozen ur1 → grounded frozen ur2."""
    return _play_hop(session, UR12, can_attach_upper_ripper1_edge, max_frames=max_frames)


def play_upper_ripper2_to_3(
    session: ControllerSession,
    *,
    max_frames: int = 360,
) -> SuperMetroidState:
    """Frozen ur2 → grounded frozen ur3."""
    return _play_hop(session, UR23, can_attach_upper_ripper2_edge, max_frames=max_frames)


def play_upper_ripper3_to_4(
    session: ControllerSession,
    *,
    max_frames: int = 360,
) -> SuperMetroidState:
    """Frozen ur3 → grounded frozen ur4."""
    return _play_hop(session, UR34, can_attach_upper_ripper3_edge, max_frames=max_frames)


POLICY_ID = UR12.policy_id
POLICY_ID_UR12 = UR12.policy_id
POLICY_ID_UR23 = UR23.policy_id
POLICY_ID_UR34 = UR34.policy_id

__all__ = [
    "POLICY_ID",
    "POLICY_ID_UR12",
    "POLICY_ID_UR23",
    "POLICY_ID_UR34",
    "RedIceUpperRipperHopRunner",
    "UR12",
    "UR23",
    "UR34",
    "UpperRipperHopSpec",
    "play_upper_ripper1_to_2",
    "play_upper_ripper2_to_3",
    "play_upper_ripper3_to_4",
]
