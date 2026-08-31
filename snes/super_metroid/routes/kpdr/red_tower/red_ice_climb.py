"""Enemy-aware checkpoint policy for the Red Tower Ice climb.

Red Tower is too tall and phase-sensitive for one blind room tape. This
module owns checkpoint geometry and the Ice-pin hop chain:

``bottom_floor -> ordinary Hellway left-door``

Freeze Rippers, hop the platforms, keep RIGHT until ordinary Hellway
left-door (gs=8, x≤80). No wall-jump on ice tops. Do not RIGHT+A from
aim-up. Do not treat jump-apex vy=0 as a landing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import FACING_RIGHT, HI_JUMP_MASK, SuperMetroidState
from super_metroid.routes.controller_common import ensure_morph, hold, is_morph, settle_hold
from super_metroid.routes.kpdr.ice.geometry import ICE_BEAM_MASK
from super_metroid.routes.kpdr.red_tower.red_to_hellway import (
    _HUMAN_FLOOR_RLE,
    _MORPH,
    _ibj_double,
    _period_wj,
    _play_upper_rle,
    _seat_left_after_handoff,
    _tunnel_to_midplat,
)
from super_metroid.routes.kpdr.rooms import ROOM_HELLWAY, ROOM_RED_TOWER
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
TUNNEL_FLOOR_Y = 1883
MID_FLOOR_Y = 1625
THIN_SEAT_Y = 587
UPPER_RIPPER_1_Y = 520
UPPER_RIPPER_1_LAND_Y = 495
UPPER_RIPPER_2_Y = 416
UPPER_RIPPER_2_LAND_Y = 391
UPPER_RIPPER_3_Y = 320
UPPER_RIPPER_3_LAND_Y = 295
UPPER_RIPPER_4_Y = 232
UPPER_RIPPER_4_LAND_Y = 207

POLICY_ID = "red_tower_ice_bottom_to_ripper1"
POLICY_ID_R12 = "red_tower_ice_ripper1_to_ripper2"
POLICY_ID_R23 = "red_tower_ice_ripper2_to_ripper3"
POLICY_ID_R34 = "red_tower_ice_ripper3_to_ripper4"
POLICY_ID_R4TUN = "red_tower_ice_ripper4_to_tunnel"
POLICY_ID_TUNMID = "red_tower_ice_tunnel_to_mid_floor"
POLICY_ID_MIDTHIN = "red_tower_ice_mid_floor_to_thin_seat"
POLICY_ID_THINUR1 = "red_tower_ice_thin_seat_to_upper_ripper1"
POLICY_ID_UR12 = "red_tower_ice_upper_ripper1_to_2"
POLICY_ID_UR23 = "red_tower_ice_upper_ripper2_to_3"
POLICY_ID_UR34 = "red_tower_ice_upper_ripper3_to_4"
POLICY_ID_UR3HW = "red_tower_ice_upper_ripper3_to_hellway"
POLICY_ID_ICEHW = "red_tower_ice_bottom_to_hellway"
VARIANT_ID = "ice_hi_jump"

_STAND = frozenset({1, 2})
_CROUCH = frozenset({39, 40})
_TRUE_MORPH = frozenset({29, 30, 31, 32})
_JUMP_UNTIL_Y = 2320
_STEP_OFF_PX = 28
_BRAKE_FRAMES = 4
_R4TUN_AIR_Y = 2015
_R4TUN_RISE_UNTIL_Y = 1860
_R4TUN_SEAT_X = 104
_R4TUN_SHAFT_X_MAX = 125
_BREAK_SHOT_FRAMES = 12
_DOOR_SHOT_X = 200


def _u16(ram: np.ndarray, address: int) -> int:
    return int(ram[address]) | (int(ram[address + 1]) << 8)


def _action(*names: str) -> np.ndarray:
    return buttons(*names) if names else idle_action()


def _grounded(state: SuperMetroidState) -> bool:
    return int(state.velocity_y) == 0 and int(state.vertical_direction) == 0


def _toward(x: int, target: int) -> str:
    return "RIGHT" if x < target else "LEFT"


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
    room_id: int = ROOM_RED_TOWER

    def matches(self, state: SuperMetroidState) -> bool:
        if int(state.room_id) != int(self.room_id):
            return False
        if not self.x_range[0] <= int(state.samus_x) <= self.x_range[1]:
            return False
        if not self.y_range[0] <= int(state.samus_y) <= self.y_range[1]:
            return False
        if self.grounded and not _grounded(state):
            return False
        return True


BOTTOM_FLOOR = RedIceCheckpoint("bottom_floor", (48, 220), (2435, 2450))
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
TUNNEL_FLOOR = RedIceCheckpoint("tunnel_floor", (80, 125), (1870, 1895))
MID_FLOOR = RedIceCheckpoint("mid_floor", (130, 195), (1618, 1632))
THIN_SEAT = RedIceCheckpoint("thin_seat", (70, 110), (575, 600))
UPPER_RIPPER_1 = RedIceCheckpoint(
    "upper_ripper_1",
    (70, 180),
    (478, 512),
    support_enemy_y=UPPER_RIPPER_1_Y,
    min_freeze_timer=30,
)
UPPER_RIPPER_2 = RedIceCheckpoint(
    "upper_ripper_2",
    (65, 180),
    (374, 408),
    support_enemy_y=UPPER_RIPPER_2_Y,
    min_freeze_timer=30,
)
UPPER_RIPPER_3 = RedIceCheckpoint(
    "upper_ripper_3",
    (65, 180),
    (278, 312),
    support_enemy_y=UPPER_RIPPER_3_Y,
    min_freeze_timer=30,
)
UPPER_RIPPER_4 = RedIceCheckpoint(
    "upper_ripper_4",
    (65, 180),
    (190, 224),
    support_enemy_y=UPPER_RIPPER_4_Y,
    min_freeze_timer=30,
)
HELLWAY_SILL = RedIceCheckpoint(
    "hellway_sill",
    (16, 80),
    (120, 175),
    grounded=False,
    room_id=ROOM_HELLWAY,
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


def can_attach_ripper4_edge(state: SuperMetroidState) -> bool:
    """Gate for the r4 → tunnel alcove hop."""
    return LOWER_RIPPER_4.matches(state) and _has_ice_hi_jump(state)


def can_attach_tunnel_edge(state: SuperMetroidState) -> bool:
    """Gate for the tunnel alcove → temporary mid-floor climb."""
    return TUNNEL_FLOOR.matches(state) and _has_ice_hi_jump(state)


def can_attach_mid_floor_edge(state: SuperMetroidState) -> bool:
    """Gate for the temporary mid floor → thin upper seat climb."""
    return MID_FLOOR.matches(state) and _has_ice_hi_jump(state)


def can_attach_thin_seat_edge(state: SuperMetroidState) -> bool:
    """Gate for the thin seat → frozen upper_ripper_1 hop."""
    return THIN_SEAT.matches(state) and _has_ice_hi_jump(state)


def can_attach_upper_ripper1_edge(state: SuperMetroidState) -> bool:
    """Gate for the frozen ur1 → ur2 hop."""
    return UPPER_RIPPER_1.matches(state) and _has_ice_hi_jump(state)


def can_attach_upper_ripper2_edge(state: SuperMetroidState) -> bool:
    """Gate for the frozen ur2 → ur3 hop."""
    return UPPER_RIPPER_2.matches(state) and _has_ice_hi_jump(state)


def can_attach_upper_ripper3_edge(state: SuperMetroidState) -> bool:
    """Gate for the frozen ur3 → ur4 hop."""
    return UPPER_RIPPER_3.matches(state) and _has_ice_hi_jump(state)


@dataclass(frozen=True)
class IceRipperHopSpec:
    """Freeze-next + standing/crouch hop from one ice seat onto the next."""

    policy_id: str
    from_checkpoint: RedIceCheckpoint
    to_checkpoint: RedIceCheckpoint
    support_y: int | None
    target_y: int
    land_y: int
    past_checkpoint: RedIceCheckpoint | None = None
    freeze_dx: tuple[int, int] = (8, 36)
    freeze_signed: bool = True
    aim_before_shot: bool = False
    crouch_jump: bool = False
    hover_track: bool = True
    past_is_retry: bool = False
    face_right: bool = False
    fail_y: int | None = None
    jump_until_override: int | None = None
    drift_high_delta: int = 10
    jump_max_frames: int = 32
    default_max_frames: int = 360

    @property
    def jump_until_y(self) -> int:
        if self.jump_until_override is not None:
            return self.jump_until_override
        return self.land_y - 27

    @property
    def drift_high_y(self) -> int:
        return self.land_y - self.drift_high_delta

    @property
    def hover_y(self) -> tuple[int, int]:
        return (
            self.to_checkpoint.y_range[0],
            self.to_checkpoint.y_range[1] + 5,
        )


R12 = IceRipperHopSpec(
    POLICY_ID_R12,
    LOWER_RIPPER_1,
    LOWER_RIPPER_2,
    BOTTOM_RIPPER_Y,
    MID_RIPPER_Y,
    MID_RIPPER_LAND_Y,
    BOTTOM_FLOOR,
    freeze_signed=False,
    hover_track=False,
    past_is_retry=True,
    default_max_frames=280,
)
R23 = IceRipperHopSpec(
    POLICY_ID_R23,
    LOWER_RIPPER_2,
    LOWER_RIPPER_3,
    MID_RIPPER_Y,
    LOW_RIPPER_3_Y,
    LOW_RIPPER_3_LAND_Y,
    LOWER_RIPPER_1,
    hover_track=False,
    default_max_frames=280,
)
R34 = IceRipperHopSpec(
    POLICY_ID_R34,
    LOWER_RIPPER_3,
    LOWER_RIPPER_4,
    LOW_RIPPER_3_Y,
    LOW_RIPPER_4_Y,
    LOW_RIPPER_4_LAND_Y,
    LOWER_RIPPER_2,
    crouch_jump=True,
    jump_until_override=2008,
    drift_high_delta=8,
    jump_max_frames=36,
)
THINUR1 = IceRipperHopSpec(
    POLICY_ID_THINUR1,
    THIN_SEAT,
    UPPER_RIPPER_1,
    None,
    UPPER_RIPPER_1_Y,
    UPPER_RIPPER_1_LAND_Y,
    face_right=True,
    fail_y=620,
)
UR12 = IceRipperHopSpec(
    POLICY_ID_UR12,
    UPPER_RIPPER_1,
    UPPER_RIPPER_2,
    UPPER_RIPPER_1_Y,
    UPPER_RIPPER_2_Y,
    UPPER_RIPPER_2_LAND_Y,
    THIN_SEAT,
)
UR23 = IceRipperHopSpec(
    POLICY_ID_UR23,
    UPPER_RIPPER_2,
    UPPER_RIPPER_3,
    UPPER_RIPPER_2_Y,
    UPPER_RIPPER_3_Y,
    UPPER_RIPPER_3_LAND_Y,
    UPPER_RIPPER_1,
)
UR34 = IceRipperHopSpec(
    POLICY_ID_UR34,
    UPPER_RIPPER_3,
    UPPER_RIPPER_4,
    UPPER_RIPPER_3_Y,
    UPPER_RIPPER_4_Y,
    UPPER_RIPPER_4_LAND_Y,
    UPPER_RIPPER_2,
    freeze_dx=(10, 28),
    aim_before_shot=True,
)
UpperRipperHopSpec = IceRipperHopSpec


class _IceTick:
    variant_id = VARIANT_ID

    def __init__(
        self,
        env: Any,
        *,
        policy_id: str,
        from_checkpoint: str,
        to_checkpoint: str,
        max_frames: int,
        max_attempts: int = 2,
        phase: str = "stand",
        init_reason: str = "",
    ) -> None:
        self.env = env
        self.policy_id = policy_id
        self.from_checkpoint = from_checkpoint
        self.to_checkpoint = to_checkpoint
        self.max_frames = max(1, int(max_frames))
        self.max_attempts = max(1, int(max_attempts))
        self.phase = phase
        self.detail = phase
        self.frames = 0
        self.attempts = 0
        self.complete = False
        self.failed = False
        self.failure = ""
        self.last_reason = init_reason or f"{policy_id}_init"
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


class RedIceBottomEdgeRunner(_IceTick):
    """Floor freeze, step off the ice column, standing hop onto r1.

    Same-column Hi-Jump bonks the frozen Ripper from below.  The floor is
    solid, so walk ~24px away after the freeze, then drift onto the ice
    from above.  No wall-jump.
    """

    def __init__(self, env: Any, *, max_frames: int = 720, max_attempts: int = 2) -> None:
        super().__init__(
            env,
            policy_id=POLICY_ID,
            from_checkpoint=BOTTOM_FLOOR.checkpoint_id,
            to_checkpoint=LOWER_RIPPER_1.checkpoint_id,
            max_frames=max_frames,
            max_attempts=max_attempts,
            phase="select_beam",
            init_reason="red_ice_init",
        )
        self.detail = "beam"

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
                    self._target_x = int(enemy.x)
                    self._set_phase("drop_aim", f"frozen x={enemy.x}")
                    continue
                samus_x = int(state.samus_x)
                if 92 <= enemy.x <= 145 and abs(enemy.x - samus_x) <= 6:
                    return self._emit(_action("UP", "X"), "red_ice_freeze_shot")
                target_x = max(90, min(148, enemy.x))
                dx = target_x - samus_x
                if abs(dx) <= 8:
                    return self._emit(_action("UP"), "red_ice_wait_phase")
                return self._emit(_action(_toward(samus_x, target_x)), "red_ice_track_phase")

            if self.phase == "drop_aim":
                if int(state.pose) in _STAND or self._phase_frames >= 10:
                    self._set_phase("step_off", "walk off ice column")
                    continue
                return self._emit(idle_action(), "red_ice_drop_aim")

            if self.phase == "step_off":
                enemy = ripper_at_height(self.env, BOTTOM_RIPPER_Y)
                ex = int(enemy.x) if enemy is not None else self._target_x
                x = int(state.samus_x)
                if abs(x - ex) >= _STEP_OFF_PX or x <= 68 or x >= 180 or self._phase_frames >= 28:
                    self._set_phase("brake", "kill leftover run")
                    continue
                if x >= ex:
                    direction = "RIGHT" if x < 190 else "LEFT"
                else:
                    direction = "LEFT" if x > 60 else "RIGHT"
                return self._emit(_action(direction), "red_ice_step_off")

            if self.phase == "brake":
                if self._phase_frames >= _BRAKE_FRAMES:
                    self._set_phase("jump", "standing Hi-Jump")
                    continue
                enemy = ripper_at_height(self.env, BOTTOM_RIPPER_Y)
                ex = int(enemy.x) if enemy is not None else self._target_x
                return self._emit(_action(_toward(int(state.samus_x), ex)), "red_ice_brake")

            if self.phase == "jump":
                if int(state.samus_y) <= _JUMP_UNTIL_Y or self._phase_frames >= 36:
                    self._set_phase("land", "drift onto ice top")
                    continue
                return self._emit(_action("A"), "red_ice_jump")

            if self.phase == "land":
                if checkpoint_supported(self.env, state, LOWER_RIPPER_1):
                    self._set_phase("settle", "verify frozen support")
                    continue
                if _grounded(state) and (
                    BOTTOM_FLOOR.matches(state) or not LOWER_RIPPER_1.matches(state)
                ):
                    self._retry_or_fail(
                        state,
                        f"missed Ripper xy=({state.samus_x},{state.samus_y})",
                    )
                    continue
                if int(state.pose) in _TRUE_MORPH:
                    return self._emit(_action("UP"), "red_ice_unmorph")
                enemy = ripper_at_height(self.env, BOTTOM_RIPPER_Y)
                if enemy is None or enemy.freeze_timer <= LOWER_RIPPER_1.min_freeze_timer:
                    self._fail("Ripper thawed before landing")
                    break
                x = int(state.samus_x)
                if abs(x - enemy.x) > 3:
                    return self._emit(_action(_toward(x, enemy.x)), "red_ice_land_track")
                return self._emit(idle_action(), "red_ice_fall")

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


class RedIceRipperHopRunner(_IceTick):
    """One-action-per-call runner: freeze next Ripper, standing or crouch hop."""

    def __init__(
        self,
        env: Any,
        spec: IceRipperHopSpec,
        *,
        max_frames: int | None = None,
        max_attempts: int = 2,
    ) -> None:
        super().__init__(
            env,
            policy_id=spec.policy_id,
            from_checkpoint=spec.from_checkpoint.checkpoint_id,
            to_checkpoint=spec.to_checkpoint.checkpoint_id,
            max_frames=spec.default_max_frames if max_frames is None else max_frames,
            max_attempts=max_attempts,
        )
        self.spec = spec
        self._tag = spec.policy_id.replace("red_tower_ice_", "red_ice_")

    def _retry_or_fail(self, state: SuperMetroidState, reason: str) -> None:
        self.attempts += 1
        if self.attempts >= self.max_attempts or not self.spec.from_checkpoint.matches(state):
            self._fail(reason)
            return
        retry = "face" if self.spec.face_right else "acquire"
        self._set_phase(retry, f"retry {self.attempts}")

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
                    if spec.face_right:
                        self._set_phase("face", "face right")
                    else:
                        self._set_phase("acquire", f"track {spec.to_checkpoint.checkpoint_id}")
                    continue
                return self._emit(_action("UP"), f"{tag}_stand")

            if self.phase == "face":
                if int(state.facing) == FACING_RIGHT or self._phase_frames >= 8:
                    self._set_phase("acquire", f"track {spec.to_checkpoint.checkpoint_id}")
                    continue
                if int(state.samus_x) >= 100:
                    return self._emit(_action("UP"), f"{tag}_hold_seat")
                return self._emit(_action("RIGHT"), f"{tag}_face")

            if self.phase == "acquire":
                if spec.support_y is not None:
                    support = ripper_at_height(self.env, spec.support_y)
                    if support is None or support.freeze_timer < 22:
                        self._fail("support thawed before next freeze")
                        break
                enemy = ripper_at_height(self.env, spec.target_y)
                if enemy is None:
                    return self._emit(idle_action(), f"{tag}_wait_r")
                signed = int(enemy.x) - int(state.samus_x)
                offset = signed if spec.freeze_signed else abs(signed)
                lo, hi = spec.freeze_dx
                if enemy.freeze_timer > 40 and offset >= lo:
                    self._target_x = int(enemy.x)
                    self._set_phase("drop_aim", f"frozen x={enemy.x}")
                    continue
                if lo <= offset <= hi:
                    if spec.aim_before_shot and int(state.pose) not in (3, 4):
                        return self._emit(_action("UP"), f"{tag}_aim")
                    return self._emit(_action("UP", "X"), f"{tag}_freeze_shot")
                return self._emit(_action("UP"), f"{tag}_wait_dx")

            if self.phase == "drop_aim":
                if int(state.pose) in _STAND or self._phase_frames >= 10:
                    if spec.crouch_jump:
                        self._set_phase("crouch", "crouch-jump setup")
                    else:
                        self._set_phase("jump", "standing Hi-Jump")
                    continue
                return self._emit(idle_action(), f"{tag}_drop_aim")

            if self.phase == "crouch":
                if int(state.pose) in _CROUCH or self._phase_frames >= 8:
                    self._set_phase("jump", "Hi-Jump crouch-jump")
                    continue
                return self._emit(_action("DOWN"), f"{tag}_crouch")

            if self.phase == "jump":
                if int(state.samus_y) <= spec.jump_until_y or self._phase_frames >= spec.jump_max_frames:
                    self._set_phase("land", "drift onto ice top")
                    continue
                return self._emit(_action("A"), f"{tag}_jump")

            if self.phase == "land":
                if checkpoint_supported(self.env, state, spec.to_checkpoint):
                    self._set_phase("settle", "verify frozen support")
                    continue
                if _grounded(state) and spec.fail_y is not None and int(state.samus_y) >= spec.fail_y:
                    self._fail(f"fell off seat xy=({state.samus_x},{state.samus_y})")
                    break
                if _grounded(state) and spec.past_checkpoint is not None and spec.past_checkpoint.matches(state):
                    reason = (
                        f"fell past {spec.from_checkpoint.checkpoint_id} "
                        f"xy=({state.samus_x},{state.samus_y})"
                    )
                    if spec.past_is_retry:
                        self._retry_or_fail(state, reason)
                        continue
                    self._fail(reason)
                    break
                if _grounded(state) and spec.from_checkpoint.matches(state):
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
                    return self._emit(_action(_toward(x, ex)), f"{tag}_drift_high")
                hover_lo, hover_hi = spec.hover_y
                if hover_lo <= y <= hover_hi:
                    if spec.hover_track and abs(x - ex) > 3:
                        return self._emit(_action(_toward(x, ex)), f"{tag}_hover_track")
                    return self._emit(idle_action(), f"{tag}_hover")
                if abs(x - ex) > 3:
                    return self._emit(_action(_toward(x, ex)), f"{tag}_track")
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


RedIceUpperRipperHopRunner = RedIceRipperHopRunner


class RedIceRipper12EdgeRunner(RedIceRipperHopRunner):
    def __init__(self, env: Any, *, max_frames: int = 280, max_attempts: int = 2) -> None:
        super().__init__(env, R12, max_frames=max_frames, max_attempts=max_attempts)


class RedIceRipper23EdgeRunner(RedIceRipperHopRunner):
    def __init__(self, env: Any, *, max_frames: int = 280, max_attempts: int = 2) -> None:
        super().__init__(env, R23, max_frames=max_frames, max_attempts=max_attempts)


class RedIceRipper34EdgeRunner(RedIceRipperHopRunner):
    def __init__(self, env: Any, *, max_frames: int = 360, max_attempts: int = 2) -> None:
        super().__init__(env, R34, max_frames=max_frames, max_attempts=max_attempts)


class RedIceThinToUr1EdgeRunner(RedIceRipperHopRunner):
    def __init__(self, env: Any, *, max_frames: int = 360, max_attempts: int = 2) -> None:
        super().__init__(env, THINUR1, max_frames=max_frames, max_attempts=max_attempts)


class RedIceRipper4TunnelEdgeRunner(_IceTick):
    """Crouch-jump left onto the tunnel alcove. A-only until airborne, then LEFT+A."""

    def __init__(self, env: Any, *, max_frames: int = 240, max_attempts: int = 2) -> None:
        super().__init__(
            env,
            policy_id=POLICY_ID_R4TUN,
            from_checkpoint=LOWER_RIPPER_4.checkpoint_id,
            to_checkpoint=TUNNEL_FLOOR.checkpoint_id,
            max_frames=max_frames,
            max_attempts=max_attempts,
            init_reason="red_ice_r4tun_init",
        )

    def _retry_or_fail(self, state: SuperMetroidState, reason: str) -> None:
        self.attempts += 1
        if self.attempts >= self.max_attempts or not LOWER_RIPPER_4.matches(state):
            self._fail(reason)
            return
        self._set_phase("crouch", f"retry {self.attempts}")

    def action(self, state: SuperMetroidState):
        if self.complete or self.failed:
            return None
        if int(state.room_id) != ROOM_RED_TOWER:
            self._fail(f"left room 0x{int(state.room_id):04X}")
            return None

        while not self.complete and not self.failed:
            if self.phase == "stand":
                if int(state.pose) in _STAND or int(state.pose) in _CROUCH:
                    self._set_phase("crouch", "crouch-jump setup")
                    continue
                return self._emit(_action("UP"), "red_ice_r4tun_stand")

            if self.phase == "crouch":
                if int(state.pose) in _CROUCH or self._phase_frames >= 8:
                    self._set_phase("jump", "Hi-Jump crouch-jump")
                    continue
                return self._emit(_action("DOWN"), "red_ice_r4tun_crouch")

            if self.phase == "jump":
                y = int(state.samus_y)
                x = int(state.samus_x)
                airborne = (not _grounded(state)) or y <= _R4TUN_AIR_Y
                if airborne and (y <= _R4TUN_RISE_UNTIL_Y or x <= _R4TUN_SHAFT_X_MAX):
                    self._set_phase("land", "drift onto alcove")
                    continue
                if airborne:
                    return self._emit(_action("LEFT", "A"), "red_ice_r4tun_rise_left")
                return self._emit(_action("A"), "red_ice_r4tun_jump")

            if self.phase == "land":
                if TUNNEL_FLOOR.matches(state):
                    self._set_phase("settle", "verify alcove seat")
                    continue
                if _grounded(state) and LOWER_RIPPER_4.matches(state):
                    self._retry_or_fail(state, "landed back on r4")
                    continue
                if _grounded(state) and LOWER_RIPPER_3.matches(state):
                    self._fail(f"fell past r4 xy=({state.samus_x},{state.samus_y})")
                    break
                if int(state.samus_y) >= 2300:
                    self._fail(f"fell to shaft xy=({state.samus_x},{state.samus_y})")
                    break
                if int(state.pose) in _TRUE_MORPH:
                    return self._emit(_action("UP"), "red_ice_r4tun_unmorph")
                x = int(state.samus_x)
                y = int(state.samus_y)
                if x > _R4TUN_SEAT_X + 3:
                    return self._emit(_action("LEFT"), "red_ice_r4tun_drift_left")
                if x < _R4TUN_SEAT_X - 12 and y <= TUNNEL_FLOOR_Y + 20:
                    return self._emit(_action("RIGHT"), "red_ice_r4tun_nudge")
                return self._emit(idle_action(), "red_ice_r4tun_fall")

            if self.phase == "settle":
                if not TUNNEL_FLOOR.matches(state):
                    self._retry_or_fail(state, "unstable tunnel seat")
                    continue
                if self._settle_frames >= 8:
                    self.complete = True
                    self._set_phase("complete", TUNNEL_FLOOR.checkpoint_id)
                    break
                self._settle_frames += 1
                return self._emit(idle_action(), "red_ice_r4tun_checkpoint_settle")

            self._fail(f"unknown phase {self.phase}")

        return None


class RedIceUr3ToHellwayRunner(_IceTick):
    """Freeze ur4 in the UR34 band, gap-jump the x=134 hole, walk the sill."""

    def __init__(
        self,
        env: Any,
        *,
        max_frames: int = 480,
        max_attempts: int = 2,
    ) -> None:
        super().__init__(
            env,
            policy_id=POLICY_ID_UR3HW,
            from_checkpoint=UPPER_RIPPER_3.checkpoint_id,
            to_checkpoint=HELLWAY_SILL.checkpoint_id,
            max_frames=max_frames,
            max_attempts=max_attempts,
            init_reason="red_ice_ur3_hw_init",
        )

    def _on_hellway(self, state: SuperMetroidState) -> bool:
        """True Hellway left-door seat, not the Red Tower door-slot fire."""
        if int(state.room_id) != ROOM_HELLWAY:
            return False
        gs = int(getattr(state, "game_state", 8))
        door = int(getattr(state, "door_transition", 0))
        x = int(state.samus_x)
        y = int(state.samus_y)
        return gs == 8 and door == 0 and 16 <= x <= 80 and 100 <= y <= 180

    def action(self, state: SuperMetroidState):
        if self.complete or self.failed:
            return None
        if self._on_hellway(state):
            self.complete = True
            self._set_phase("complete", "hellway")
            return None
        if int(state.room_id) == ROOM_HELLWAY:
            return self._emit(_action("RIGHT"), "red_ice_ur3_hw_door")
        if int(state.room_id) != ROOM_RED_TOWER:
            self._fail(f"left room 0x{int(state.room_id):04X}")
            return None

        freeze_lo, freeze_hi = UR34.freeze_dx
        while not self.complete and not self.failed:
            if self.phase == "stand":
                if int(state.pose) in _STAND or int(state.pose) in (3, 4):
                    self._set_phase("acquire", "track upper_ripper_4")
                    continue
                return self._emit(_action("UP"), "red_ice_ur3_hw_stand")

            if self.phase == "acquire":
                support = ripper_at_height(self.env, UPPER_RIPPER_3_Y)
                enemy = ripper_at_height(self.env, UPPER_RIPPER_4_Y)
                if support is None or support.freeze_timer < 22:
                    self._fail("support thawed before next freeze")
                    break
                if enemy is None:
                    return self._emit(idle_action(), "red_ice_ur3_hw_wait_r")
                signed = int(enemy.x) - int(state.samus_x)
                if enemy.freeze_timer > 40 and signed >= freeze_lo:
                    self._set_phase("drop_aim", f"frozen x={enemy.x}")
                    continue
                if freeze_lo <= signed <= freeze_hi:
                    if int(state.pose) not in (3, 4):
                        return self._emit(_action("UP"), "red_ice_ur3_hw_aim")
                    return self._emit(_action("UP", "X"), "red_ice_ur3_hw_freeze_shot")
                return self._emit(_action("UP"), "red_ice_ur3_hw_wait_dx")

            if self.phase == "drop_aim":
                if int(state.pose) in _STAND or self._phase_frames >= 10:
                    self._set_phase("break", "UP+X+A through hole")
                    continue
                return self._emit(idle_action(), "red_ice_ur3_hw_drop_aim")

            if self.phase == "break":
                if self._phase_frames >= _BREAK_SHOT_FRAMES:
                    self._set_phase("rise", "A-only through hole")
                    continue
                return self._emit(_action("UP", "X", "A"), "red_ice_ur3_hw_break_shot")

            if self.phase == "rise":
                y = int(state.samus_y)
                if y <= 140:
                    self._set_phase("sill", "walk door floor")
                    continue
                if y >= 360:
                    self._fail(f"fell xy=({state.samus_x},{state.samus_y})")
                    break
                if UPPER_RIPPER_3.matches(state):
                    self._fail("landed back on upper_ripper_3")
                    break
                return self._emit(_action("A"), "red_ice_ur3_hw_rise")

            if self.phase == "sill":
                if self._on_hellway(state):
                    self.complete = True
                    self._set_phase("complete", "hellway")
                    break
                if int(state.samus_y) >= 360:
                    self._fail(f"fell xy=({state.samus_x},{state.samus_y})")
                    break
                y = int(state.samus_y)
                x = int(state.samus_x)
                if y <= 155 and (_grounded(state) or y <= 142):
                    names = ("RIGHT", "X") if x >= _DOOR_SHOT_X else ("RIGHT",)
                    return self._emit(_action(*names), "red_ice_ur3_hw_sill_right")
                return self._emit(_action("A"), "red_ice_ur3_hw_sill_keep_up")

            self._fail(f"unknown phase {self.phase}")

        return None


def _play_runner(
    session: ControllerSession,
    runner: _IceTick,
    attach: Callable[[SuperMetroidState], bool],
    *,
    from_name: str,
) -> SuperMetroidState:
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError(f"{runner.policy_id}: needs session.env")
    if not attach(session.state):
        raise TimeoutError(
            f"{runner.policy_id}: not on {from_name} "
            f"xy=({session.state.samus_x},{session.state.samus_y}) "
            f"p={session.state.pose}"
        )
    while not runner.complete and not runner.failed:
        action = runner.action(session.state)
        if action is None:
            break
        session.step(action, runner.last_reason)
    if runner.failed or not runner.complete:
        raise TimeoutError(
            f"{runner.policy_id}: {runner.failure or 'did not complete'}; "
            f"phase={runner.phase} frames={runner.frames} "
            f"xy=({session.state.samus_x},{session.state.samus_y})"
        )
    return session.state


def _play_hop(
    session: ControllerSession,
    spec: IceRipperHopSpec,
    attach: Callable[[SuperMetroidState], bool],
    *,
    max_frames: int,
) -> SuperMetroidState:
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError(f"{spec.policy_id}: needs session.env")
    return _play_runner(
        session,
        RedIceRipperHopRunner(env, spec, max_frames=max_frames),
        attach,
        from_name=spec.from_checkpoint.checkpoint_id,
    )


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


def play_ripper1_to_ripper2(
    session: ControllerSession,
    *,
    max_frames: int = 280,
) -> SuperMetroidState:
    """Frozen r1 → grounded frozen r2."""
    return _play_hop(session, R12, can_attach_ripper1_edge, max_frames=max_frames)


def play_ripper2_to_ripper3(
    session: ControllerSession,
    *,
    max_frames: int = 280,
) -> SuperMetroidState:
    """Frozen r2 → grounded frozen r3."""
    return _play_hop(session, R23, can_attach_ripper2_edge, max_frames=max_frames)


def play_ripper3_to_ripper4(
    session: ControllerSession,
    *,
    max_frames: int = 360,
) -> SuperMetroidState:
    """Frozen r3 → grounded frozen r4."""
    return _play_hop(session, R34, can_attach_ripper3_edge, max_frames=max_frames)


def play_ripper4_to_tunnel(
    session: ControllerSession,
    *,
    max_frames: int = 240,
) -> SuperMetroidState:
    """Frozen r4 → grounded tunnel alcove."""
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError(f"{POLICY_ID_R4TUN}: needs session.env")
    return _play_runner(
        session,
        RedIceRipper4TunnelEdgeRunner(env, max_frames=max_frames),
        can_attach_ripper4_edge,
        from_name="lower_ripper_4",
    )


def play_tunnel_to_mid_floor(
    session: ControllerSession,
    *,
    max_cycles: int = 50,
) -> SuperMetroidState:
    """Climb from the natural tunnel checkpoint to the grounded mid floor."""
    if not can_attach_tunnel_edge(session.state):
        raise TimeoutError(
            f"{POLICY_ID_TUNMID}: not on tunnel_floor "
            f"xy=({session.state.samus_x},{session.state.samus_y}) "
            f"p={session.state.pose}"
        )

    _tunnel_to_midplat(session, f"{POLICY_ID_TUNMID}_ledge")
    for _ in range(50):
        state = session.state
        if (
            _grounded(state)
            and 1740 <= int(state.samus_y) <= 1770
            and 115 <= int(state.samus_x) <= 180
        ):
            break
        hold(session, 1, reason=f"{POLICY_ID_TUNMID}_ledge_land")
    else:
        raise TimeoutError(
            f"{POLICY_ID_TUNMID}: missed bomb ledge "
            f"xy=({session.state.samus_x},{session.state.samus_y})"
        )

    hold(session, 1, "UP", reason=f"{POLICY_ID_TUNMID}_stand")
    for _ in range(12):
        hold(session, 1, reason=f"{POLICY_ID_TUNMID}_drop_aim")
    for _ in range(50):
        if int(session.state.samus_x) >= 168:
            break
        hold(session, 1, "RIGHT", reason=f"{POLICY_ID_TUNMID}_center")
    settle_hold(session, 4, reason=f"{POLICY_ID_TUNMID}_center_settle")
    if not is_morph(session.state.pose) and int(session.state.pose) not in _MORPH:
        ensure_morph(session)

    for cycle in range(max(1, int(max_cycles))):
        _ibj_double(
            session,
            f"{POLICY_ID_TUNMID}_ibj_{cycle}",
            center_x=171,
            stop_y=1595,
        )
        if int(session.state.room_id) != ROOM_RED_TOWER:
            break
        if int(session.state.samus_y) <= 1610:
            for _ in range(80):
                state = hold(session, 1, "LEFT", reason=f"{POLICY_ID_TUNMID}_catch")
                if MID_FLOOR.matches(state):
                    settle_hold(session, 8, reason=f"{POLICY_ID_TUNMID}_settle")
                    if MID_FLOOR.matches(session.state):
                        return session.state
                    break

    raise TimeoutError(
        f"{POLICY_ID_TUNMID}: mid floor not reached "
        f"xy=({session.state.samus_x},{session.state.samus_y}) "
        f"p={session.state.pose}"
    )


def play_mid_floor_to_thin_seat(session: ControllerSession) -> SuperMetroidState:
    """Climb the alternating solid ledges and land the thin seat at y=587."""
    if not can_attach_mid_floor_edge(session.state):
        raise TimeoutError(
            f"{POLICY_ID_MIDTHIN}: not on mid_floor "
            f"xy=({session.state.samus_x},{session.state.samus_y}) "
            f"p={session.state.pose}"
        )

    _play_upper_rle(session, _HUMAN_FLOOR_RLE, f"{POLICY_ID_MIDTHIN}_handoff")
    _seat_left_after_handoff(session, f"{POLICY_ID_MIDTHIN}_left_seat")
    hold(session, 3, "LEFT", "B", reason=f"{POLICY_ID_MIDTHIN}_launch_run")
    hold(session, 12, "LEFT", "B", "A", reason=f"{POLICY_ID_MIDTHIN}_launch")
    _period_wj(session, f"{POLICY_ID_MIDTHIN}_p0", side="LEFT", frames=600, stop_y=1200)
    _period_wj(session, f"{POLICY_ID_MIDTHIN}_p1", side="RIGHT", frames=800, stop_y=1050)
    _period_wj(
        session,
        f"{POLICY_ID_MIDTHIN}_p2",
        side="LEFT",
        frames=800,
        stop_y=900,
        period=16,
        into=2,
        flip=2,
    )
    _period_wj(
        session,
        f"{POLICY_ID_MIDTHIN}_p3",
        side="RIGHT",
        frames=800,
        stop_y=720,
        period=16,
        into=2,
        flip=2,
    )
    _period_wj(
        session,
        f"{POLICY_ID_MIDTHIN}_p4",
        side="LEFT",
        frames=500,
        period=22,
        into=2,
        flip=2,
    )
    _period_wj(
        session,
        f"{POLICY_ID_MIDTHIN}_p5",
        side="RIGHT",
        frames=500,
        period=22,
        into=8,
        flip=2,
    )
    settle_hold(session, 8, reason=f"{POLICY_ID_MIDTHIN}_settle")
    if THIN_SEAT.matches(session.state):
        return session.state
    raise TimeoutError(
        f"{POLICY_ID_MIDTHIN}: thin seat not reached "
        f"xy=({session.state.samus_x},{session.state.samus_y}) "
        f"p={session.state.pose}"
    )


def play_thin_seat_to_upper_ripper1(
    session: ControllerSession,
    *,
    max_frames: int = 360,
) -> SuperMetroidState:
    """Grounded thin seat → grounded frozen ur1."""
    return _play_hop(session, THINUR1, can_attach_thin_seat_edge, max_frames=max_frames)


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


def play_upper_ripper3_to_hellway(
    session: ControllerSession,
    *,
    max_frames: int = 480,
) -> SuperMetroidState:
    """Frozen ur3 → ordinary Hellway left-door (gap-jump, no ur4 settle)."""
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError(f"{POLICY_ID_UR3HW}: needs session.env")
    return _play_runner(
        session,
        RedIceUr3ToHellwayRunner(env, max_frames=max_frames),
        can_attach_upper_ripper3_edge,
        from_name="upper_ripper_3",
    )


def play_ice_climb_to_hellway(session: ControllerSession) -> SuperMetroidState:
    """Bottom floor → ordinary Hellway left-door. No door-slot fire, no settle."""
    if not can_attach_bottom_edge(session.state):
        raise TimeoutError(
            f"{POLICY_ID_ICEHW}: not on Ice+HJ bottom floor "
            f"xy=({session.state.samus_x},{session.state.samus_y}) "
            f"p={session.state.pose}"
        )
    play_bottom_to_ripper1(session)
    play_ripper1_to_ripper2(session)
    play_ripper2_to_ripper3(session)
    play_ripper3_to_ripper4(session)
    play_ripper4_to_tunnel(session)
    play_tunnel_to_mid_floor(session)
    play_mid_floor_to_thin_seat(session)
    play_thin_seat_to_upper_ripper1(session)
    play_upper_ripper1_to_2(session)
    play_upper_ripper2_to_3(session)
    play_upper_ripper3_to_hellway(session)
    state = session.state
    if not HELLWAY_SILL.matches(state):
        raise TimeoutError(
            f"{POLICY_ID_ICEHW}: not ordinary Hellway left-door "
            f"room=0x{int(state.room_id):04X} "
            f"xy=({state.samus_x},{state.samus_y}) p={state.pose}"
        )
    return state


__all__ = [
    "BOTTOM_FLOOR",
    "BOTTOM_RIPPER_LAND_Y",
    "BOTTOM_RIPPER_Y",
    "HELLWAY_SILL",
    "IceRipperHopSpec",
    "LOWER_RIPPER_1",
    "LOWER_RIPPER_2",
    "LOWER_RIPPER_3",
    "LOWER_RIPPER_4",
    "LOW_RIPPER_3_LAND_Y",
    "LOW_RIPPER_3_Y",
    "LOW_RIPPER_4_LAND_Y",
    "LOW_RIPPER_4_Y",
    "MID_FLOOR",
    "MID_FLOOR_Y",
    "MID_RIPPER_LAND_Y",
    "MID_RIPPER_Y",
    "POLICY_ID",
    "POLICY_ID_ICEHW",
    "POLICY_ID_MIDTHIN",
    "POLICY_ID_R12",
    "POLICY_ID_R23",
    "POLICY_ID_R34",
    "POLICY_ID_R4TUN",
    "POLICY_ID_THINUR1",
    "POLICY_ID_TUNMID",
    "POLICY_ID_UR12",
    "POLICY_ID_UR23",
    "POLICY_ID_UR34",
    "POLICY_ID_UR3HW",
    "R12",
    "R23",
    "R34",
    "RIPPER_ID",
    "RedIceBottomEdgeRunner",
    "RedIceCheckpoint",
    "RedIceRipper12EdgeRunner",
    "RedIceRipper23EdgeRunner",
    "RedIceRipper34EdgeRunner",
    "RedIceRipper4TunnelEdgeRunner",
    "RedIceRipperHopRunner",
    "RedIceThinToUr1EdgeRunner",
    "RedIceUpperRipperHopRunner",
    "RedIceUr3ToHellwayRunner",
    "RipperObservation",
    "THINUR1",
    "THIN_SEAT",
    "THIN_SEAT_Y",
    "TUNNEL_FLOOR",
    "TUNNEL_FLOOR_Y",
    "UR12",
    "UR23",
    "UR34",
    "UPPER_RIPPER_1",
    "UPPER_RIPPER_1_LAND_Y",
    "UPPER_RIPPER_1_Y",
    "UPPER_RIPPER_2",
    "UPPER_RIPPER_2_LAND_Y",
    "UPPER_RIPPER_2_Y",
    "UPPER_RIPPER_3",
    "UPPER_RIPPER_3_LAND_Y",
    "UPPER_RIPPER_3_Y",
    "UPPER_RIPPER_4",
    "UPPER_RIPPER_4_LAND_Y",
    "UPPER_RIPPER_4_Y",
    "UpperRipperHopSpec",
    "VARIANT_ID",
    "can_attach_bottom_edge",
    "can_attach_mid_floor_edge",
    "can_attach_ripper1_edge",
    "can_attach_ripper2_edge",
    "can_attach_ripper3_edge",
    "can_attach_ripper4_edge",
    "can_attach_thin_seat_edge",
    "can_attach_tunnel_edge",
    "can_attach_upper_ripper1_edge",
    "can_attach_upper_ripper2_edge",
    "can_attach_upper_ripper3_edge",
    "checkpoint_supported",
    "play_bottom_to_ripper1",
    "play_ice_climb_to_hellway",
    "play_mid_floor_to_thin_seat",
    "play_ripper1_to_ripper2",
    "play_ripper2_to_ripper3",
    "play_ripper3_to_ripper4",
    "play_ripper4_to_tunnel",
    "play_thin_seat_to_upper_ripper1",
    "play_tunnel_to_mid_floor",
    "play_upper_ripper1_to_2",
    "play_upper_ripper2_to_3",
    "play_upper_ripper3_to_4",
    "play_upper_ripper3_to_hellway",
    "read_rippers",
    "ripper_at_height",
]
