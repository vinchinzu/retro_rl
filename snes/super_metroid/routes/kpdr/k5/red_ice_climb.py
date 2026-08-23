"""Enemy-aware checkpoint policy for the Red Tower Ice climb.

Red Tower is too tall and phase-sensitive for one blind room tape.  This
module owns the first small edge in a checkpoint graph:

``bottom_floor -> lower_ripper_1``

The edge observes the live Ripper patrol, freezes it only in a repeatable
launch band, then uses the shared consecutive-wall-jump timing family and
steers back onto the frozen enemy.  Later edges belong beside this one, not in
the already-large :mod:`red_to_hellway` product controller.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import HI_JUMP_MASK, SuperMetroidState
from super_metroid.routes.kpdr.ice.geometry import ICE_BEAM_MASK
from super_metroid.routes.kpdr.rooms import ROOM_RED_TOWER
from super_metroid.routes.runtime import ControllerSession

ENEMY_BASE = 0x0F78
ENEMY_STRIDE = 0x40
RIPPER_ID = 0xD47F

BOTTOM_RIPPER_Y = 2376
BOTTOM_FLOOR_Y = 2443
BOTTOM_RIPPER_LAND_Y = 2351
MID_RIPPER_Y = 2280
MID_RIPPER_LAND_Y = 2255
LOW_RIPPER_3_Y = 2184
LOW_RIPPER_3_LAND_Y = 2159
LOW_RIPPER_4_Y = 2048
LOW_RIPPER_4_LAND_Y = 2023

POLICY_ID = "red_tower_ice_bottom_to_ripper1"
VARIANT_ID = "ice_hi_jump"


def _u16(ram: np.ndarray, address: int) -> int:
    return int(ram[address]) | (int(ram[address + 1]) << 8)


@dataclass(frozen=True)
class RipperObservation:
    """One live Ripper slot needed by the checkpoint policy."""

    slot: int
    x: int
    y: int
    freeze_timer: int


@dataclass(frozen=True)
class RedIceCheckpoint:
    checkpoint_id: str
    x_range: tuple[int, int]
    y_range: tuple[int, int]
    grounded: bool = True
    support_enemy_y: int | None = None
    min_freeze_timer: int = 0

    def matches(self, state: SuperMetroidState) -> bool:
        if int(state.room_id) != ROOM_RED_TOWER:
            return False
        if not self.x_range[0] <= int(state.samus_x) <= self.x_range[1]:
            return False
        if not self.y_range[0] <= int(state.samus_y) <= self.y_range[1]:
            return False
        if self.grounded and not (
            int(state.velocity_y) == 0 and int(state.vertical_direction) == 0
        ):
            return False
        return True


BOTTOM_FLOOR = RedIceCheckpoint(
    "bottom_floor",
    (48, 220),
    (2435, 2450),
)
LOWER_RIPPER_1 = RedIceCheckpoint(
    "lower_ripper_1",
    (55, 175),
    (2335, 2360),
    support_enemy_y=BOTTOM_RIPPER_Y,
    min_freeze_timer=30,
)
LOWER_RIPPER_2 = RedIceCheckpoint(
    "lower_ripper_2",
    (55, 205),
    (2238, 2270),
    support_enemy_y=MID_RIPPER_Y,
    min_freeze_timer=30,
)
LOWER_RIPPER_3 = RedIceCheckpoint(
    "lower_ripper_3",
    (55, 205),
    (2142, 2174),
    support_enemy_y=LOW_RIPPER_3_Y,
    min_freeze_timer=30,
)
LOWER_RIPPER_4 = RedIceCheckpoint(
    "lower_ripper_4",
    (55, 205),
    (2006, 2038),
    support_enemy_y=LOW_RIPPER_4_Y,
    min_freeze_timer=30,
)


def read_rippers(env: Any) -> tuple[RipperObservation, ...]:
    """Read every live Red Tower Ripper, including off-screen lower slots."""
    ram = env.get_ram()
    out: list[RipperObservation] = []
    for slot in range(12):
        base = ENEMY_BASE + slot * ENEMY_STRIDE
        if _u16(ram, base) != RIPPER_ID:
            continue
        x = _u16(ram, base + 0x02)
        y = _u16(ram, base + 0x06)
        if x >= 0xFE00 or y >= 0xFE00 or (x == 0 and y == 0):
            continue
        out.append(
            RipperObservation(
                slot=slot,
                x=x,
                y=y,
                freeze_timer=_u16(ram, base + 0x26),
            )
        )
    return tuple(out)


def ripper_at_height(env: Any, target_y: int, *, tolerance: int = 12) -> RipperObservation | None:
    candidates = [
        enemy
        for enemy in read_rippers(env)
        if abs(int(enemy.y) - int(target_y)) <= int(tolerance)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda enemy: (abs(enemy.y - target_y), enemy.slot))


def checkpoint_supported(
    env: Any,
    state: SuperMetroidState,
    checkpoint: RedIceCheckpoint,
) -> bool:
    """Require both a stable Samus state and the expected frozen support."""
    if not checkpoint.matches(state):
        return False
    if checkpoint.support_enemy_y is None:
        return True
    enemy = ripper_at_height(env, checkpoint.support_enemy_y)
    if enemy is None or enemy.freeze_timer < checkpoint.min_freeze_timer:
        return False
    return abs(int(state.samus_x) - enemy.x) <= 24


def _has_ice_hi_jump(state: SuperMetroidState) -> bool:
    return (
        int(state.equipped_beams) & ICE_BEAM_MASK == ICE_BEAM_MASK
        and int(state.equipped_items) & HI_JUMP_MASK == HI_JUMP_MASK
    )


def can_attach_bottom_edge(state: SuperMetroidState) -> bool:
    """Equipment and geometry gate for the built-in interactive runner."""
    return BOTTOM_FLOOR.matches(state) and _has_ice_hi_jump(state)


def can_attach_ripper1_edge(state: SuperMetroidState) -> bool:
    """Gate for the r1 → r2 hop. Support freeze is checked live by the runner."""
    return LOWER_RIPPER_1.matches(state) and _has_ice_hi_jump(state)


def can_attach_ripper2_edge(state: SuperMetroidState) -> bool:
    """Gate for the r2 → r3 hop. Support freeze is checked live by the runner."""
    return LOWER_RIPPER_2.matches(state) and _has_ice_hi_jump(state)


def can_attach_ripper3_edge(state: SuperMetroidState) -> bool:
    """Gate for the r3 → r4 hop. Support freeze is checked live by the runner."""
    return LOWER_RIPPER_3.matches(state) and _has_ice_hi_jump(state)


def _action(*names: str) -> np.ndarray:
    return buttons(*names) if names else idle_action()


class RedIceBottomEdgeRunner:
    """One-action-per-call runner for human-hot-swappable autopilot.

    Enemy alignment and landing are evaluated every frame.  The WJ technique
    itself is a short, named span sequence; it is never mutated frame by frame.
    """

    policy_id = POLICY_ID
    variant_id = VARIANT_ID
    from_checkpoint = BOTTOM_FLOOR.checkpoint_id
    to_checkpoint = LOWER_RIPPER_1.checkpoint_id

    def __init__(self, env: Any, *, max_frames: int = 720, max_attempts: int = 2) -> None:
        self.env = env
        self.max_frames = max(1, int(max_frames))
        self.max_attempts = max(1, int(max_attempts))
        self.phase = "select_beam"
        self.detail = "beam"
        self.frames = 0
        self.attempts = 0
        self.complete = False
        self.failed = False
        self.failure = ""
        self.last_reason = "red_ice_init"
        self._phase_frames = 0
        self._settle_frames = 0
        self._spans: deque[tuple[np.ndarray, str]] = deque()

    def _fail(self, reason: str) -> None:
        self.failed = True
        self.failure = reason
        self.phase = "failed"
        self.detail = reason

    def _emit(self, action: np.ndarray, reason: str) -> np.ndarray:
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

    def _queue_wj(self) -> None:
        # Shared Bubble/Parlor consecutive-WJ family, tuned by the first
        # natural Red pin sweep: WJ1 20/4/8, WJ2 14/2/6.
        spans = (
            (20, ("LEFT", "A"), "dwj_wj1_into"),
            (4, ("A",), "dwj_wj1_amid"),
            (8, ("RIGHT", "A"), "dwj_wj1_flip"),
            (14, ("LEFT", "A"), "dwj_wj2_into"),
            (2, ("A",), "dwj_wj2_amid"),
            (6, ("RIGHT", "A"), "dwj_wj2_flip"),
        )
        for frames, names, label in spans:
            action = _action(*names)
            self._spans.extend((action, f"red_ice_{label}") for _ in range(frames))
        self._set_phase("double_wj", "consecutive WJ")

    def _retry_or_fail(self, state: SuperMetroidState, reason: str) -> None:
        self.attempts += 1
        if self.attempts >= self.max_attempts or not BOTTOM_FLOOR.matches(state):
            self._fail(reason)
            return
        self._set_phase("acquire", f"retry {self.attempts}")

    def action(self, state: SuperMetroidState) -> np.ndarray | None:
        """Return the next SNES-12 action, or ``None`` after completion/fail."""
        if self.complete or self.failed:
            return None
        if int(state.room_id) != ROOM_RED_TOWER:
            self._fail(f"left room 0x{int(state.room_id):04X}")
            return None

        if self._spans:
            action, reason = self._spans.popleft()
            if not self._spans:
                self._set_phase("land", "track frozen Ripper")
            return self._emit(action, reason)

        while not self.complete and not self.failed:
            if self.phase == "select_beam":
                if int(state.equipped_beams) & ICE_BEAM_MASK != ICE_BEAM_MASK:
                    self._fail("Ice Beam is not equipped")
                    break
                if int(state.equipped_items) & HI_JUMP_MASK != HI_JUMP_MASK:
                    self._fail("Hi-Jump is not equipped")
                    break
                if int(state.selected_item) == 0:
                    self._set_phase("acquire", "track lower Ripper")
                    continue
                return self._emit(_action("SELECT"), "red_ice_select_beam")

            if self.phase == "acquire":
                enemy = ripper_at_height(self.env, BOTTOM_RIPPER_Y)
                if enemy is None:
                    self._fail("lower Ripper missing")
                    break
                if enemy.freeze_timer > 40:
                    self._set_phase("runup", f"frozen x={enemy.x}")
                    continue
                samus_x = int(state.samus_x)
                # Beam travel shifts a left-moving target ~6 px.  Fire only
                # inside this band so every WJ starts from usable geometry.
                if 92 <= enemy.x <= 145 and abs(enemy.x - samus_x) <= 6:
                    return self._emit(_action("UP", "X"), "red_ice_freeze_shot")
                target_x = max(90, min(148, enemy.x))
                dx = target_x - samus_x
                if abs(dx) <= 8:
                    return self._emit(_action("UP"), "red_ice_wait_phase")
                direction = "RIGHT" if dx > 0 else "LEFT"
                return self._emit(_action(direction), "red_ice_track_phase")

            if self.phase == "runup":
                enemy = ripper_at_height(self.env, BOTTOM_RIPPER_Y)
                if enemy is None:
                    self._fail("lower Ripper missing during runup")
                    break
                # Start the first wall arc only after Samus has cleared the
                # frozen Ripper horizontally.  A fixed 16f runup could put
                # the rising spin within its collision box before the actual
                # double wall jump began.
                clear_of_ripper = int(state.samus_x) >= int(enemy.x) + 36
                if self._phase_frames >= 16 and clear_of_ripper:
                    self._set_phase("spin", "right-wall approach")
                    continue
                if self._phase_frames >= 60:
                    self._retry_or_fail(state, "could not clear Ripper for launch")
                    continue
                return self._emit(_action("RIGHT", "B"), "red_ice_runup")

            if self.phase == "spin":
                if int(state.samus_x) >= 215 and self._phase_frames >= 30:
                    self._set_phase("coast", "arm WJ")
                    continue
                if self._phase_frames >= 90:
                    self._retry_or_fail(state, "right wall not reached")
                    continue
                return self._emit(
                    _action("RIGHT", "B", "A"),
                    "red_ice_spin_to_wall",
                )

            if self.phase == "coast":
                if self._phase_frames >= 4:
                    self._set_phase("release", "release jump")
                    continue
                return self._emit(_action("B", "A"), "red_ice_wj_coast")

            if self.phase == "release":
                if self._phase_frames >= 1:
                    self._set_phase("idle_turn", "turn window")
                    continue
                return self._emit(_action("B"), "red_ice_wj_release")

            if self.phase == "idle_turn":
                if self._phase_frames < 2:
                    return self._emit(idle_action(), "red_ice_wj_idle")
                if self._phase_frames < 4:
                    return self._emit(_action("LEFT"), "red_ice_wj_turn")
                self._queue_wj()
                action, reason = self._spans.popleft()
                return self._emit(action, reason)

            if self.phase == "land":
                if checkpoint_supported(self.env, state, LOWER_RIPPER_1):
                    self._set_phase("settle", "verify frozen support")
                    continue
                if int(state.velocity_y) == 0 and int(state.vertical_direction) == 0:
                    self._retry_or_fail(
                        state,
                        f"missed Ripper xy=({state.samus_x},{state.samus_y})",
                    )
                    continue
                enemy = ripper_at_height(self.env, BOTTOM_RIPPER_Y)
                if enemy is None or enemy.freeze_timer <= LOWER_RIPPER_1.min_freeze_timer:
                    self._fail("Ripper thawed before landing")
                    break
                dx = enemy.x - int(state.samus_x)
                if dx > 2:
                    action = _action("RIGHT")
                elif dx < -2:
                    action = _action("LEFT")
                else:
                    action = idle_action()
                return self._emit(action, "red_ice_land_track")

            if self.phase == "settle":
                if not checkpoint_supported(self.env, state, LOWER_RIPPER_1):
                    self._retry_or_fail(state, "unstable frozen support")
                    continue
                if self._settle_frames >= 8:
                    self.complete = True
                    self._set_phase("complete", LOWER_RIPPER_1.checkpoint_id)
                    break
                self._settle_frames += 1
                return self._emit(idle_action(), "red_ice_checkpoint_settle")

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


def play_bottom_to_ripper1(
    session: ControllerSession,
    *,
    max_frames: int = 720,
) -> SuperMetroidState:
    """Synchronous route/probe facade over the interactive tick runner."""
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError("Red Ice checkpoint policy needs session.env")
    runner = RedIceBottomEdgeRunner(env, max_frames=max_frames)
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
    "BOTTOM_FLOOR",
    "BOTTOM_RIPPER_LAND_Y",
    "BOTTOM_RIPPER_Y",
    "LOWER_RIPPER_1",
    "LOWER_RIPPER_2",
    "LOWER_RIPPER_3",
    "LOWER_RIPPER_4",
    "LOW_RIPPER_3_LAND_Y",
    "LOW_RIPPER_3_Y",
    "LOW_RIPPER_4_LAND_Y",
    "LOW_RIPPER_4_Y",
    "MID_RIPPER_LAND_Y",
    "MID_RIPPER_Y",
    "POLICY_ID",
    "RIPPER_ID",
    "RedIceBottomEdgeRunner",
    "RedIceCheckpoint",
    "RipperObservation",
    "VARIANT_ID",
    "can_attach_bottom_edge",
    "can_attach_ripper1_edge",
    "can_attach_ripper2_edge",
    "can_attach_ripper3_edge",
    "checkpoint_supported",
    "play_bottom_to_ripper1",
    "read_rippers",
    "ripper_at_height",
]
