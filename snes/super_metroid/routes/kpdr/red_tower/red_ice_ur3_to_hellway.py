"""Red Tower Ice edge: frozen upper_ripper_3 → Hellway sill.

Freeze ur4 with the verified UR34 (10, 28) aim-then-shot band, then do
**not** land-and-settle on ur4. A 12f UP+X+A burst from standing clears
the 1-tile hole at x≈134; A-only through the hole; RIGHT only once Samus
is above the door floor (y≤140). Early RIGHT hits the door-floor lip.

Do not complete on the first Hellway ``room_id``. That fire is still the
Red Tower door slot ~(237,139) p11; x underflows to 65522 if RIGHT stops.
Keep RIGHT until ordinary Hellway left-door (gs=8, x≤80).

Product ``play_upper_ripper3_to_4`` is unchanged. This hop attaches from
the same ur3 pin and supersedes ur4 as a stop.
"""

from __future__ import annotations

from typing import Any

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.kpdr.red_tower.red_ice_climb import (
    HELLWAY_SILL,
    UPPER_RIPPER_3,
    UPPER_RIPPER_3_Y,
    UPPER_RIPPER_4_Y,
    VARIANT_ID,
    can_attach_upper_ripper3_edge,
    ripper_at_height,
)
from super_metroid.routes.kpdr.red_tower.red_ice_upper_hops import UR34
from super_metroid.routes.kpdr.rooms import ROOM_HELLWAY, ROOM_RED_TOWER
from super_metroid.routes.runtime import ControllerSession

POLICY_ID = "red_tower_ice_upper_ripper3_to_hellway"

_STAND = frozenset({1, 2})
_BREAK_SHOT_FRAMES = 12
_DOOR_SHOT_X = 200


def _action(*names: str):
    return buttons(*names) if names else idle_action()


class RedIceUr3ToHellwayRunner:
    """Freeze ur4 in the UR34 band, gap-jump the x=134 hole, walk the sill."""

    variant_id = VARIANT_ID
    policy_id = POLICY_ID
    from_checkpoint = UPPER_RIPPER_3.checkpoint_id
    to_checkpoint = HELLWAY_SILL.checkpoint_id

    def __init__(
        self,
        env: Any,
        *,
        max_frames: int = 480,
        max_attempts: int = 2,
    ) -> None:
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
        self.last_reason = "red_ice_ur3_hw_init"
        self._phase_frames = 0

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
                grounded = (
                    int(state.velocity_y) == 0 and int(state.vertical_direction) == 0
                )
                if y <= 155 and (grounded or y <= 142):
                    names = ("RIGHT", "X") if x >= _DOOR_SHOT_X else ("RIGHT",)
                    return self._emit(_action(*names), "red_ice_ur3_hw_sill_right")
                return self._emit(_action("A"), "red_ice_ur3_hw_sill_keep_up")

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


def play_upper_ripper3_to_hellway(
    session: ControllerSession,
    *,
    max_frames: int = 480,
) -> SuperMetroidState:
    """Frozen ur3 → ordinary Hellway left-door (gap-jump, no ur4 settle)."""
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError(f"{POLICY_ID}: needs session.env")
    if not can_attach_upper_ripper3_edge(session.state):
        raise TimeoutError(
            f"{POLICY_ID}: not on upper_ripper_3 "
            f"xy=({session.state.samus_x},{session.state.samus_y}) "
            f"p={session.state.pose}"
        )
    runner = RedIceUr3ToHellwayRunner(env, max_frames=max_frames)
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
    "RedIceUr3ToHellwayRunner",
    "play_upper_ripper3_to_hellway",
]
