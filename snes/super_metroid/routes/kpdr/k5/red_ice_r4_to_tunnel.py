"""Red Tower Ice edge: frozen lower_ripper_4 → solid tunnel_floor.

Crouch-jump from r4 clears the alcove height (apex ~1847 vs seat y1883)
but the open shaft at x≈155 has no floor there. Do not walk left on the
ice (falls off). A-only until airborne, then LEFT+A so the hop travels
onto the left seat (~x104) from above.

Do not RIGHT+A from aim-up: that becomes pose 81 and falls through ice.
"""

from __future__ import annotations

from typing import Any

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.kpdr.k5.red_ice_climb import (
    LOWER_RIPPER_3,
    LOWER_RIPPER_4,
    TUNNEL_FLOOR,
    TUNNEL_FLOOR_Y,
    VARIANT_ID,
    can_attach_ripper4_edge,
)
from super_metroid.routes.kpdr.rooms import ROOM_RED_TOWER
from super_metroid.routes.runtime import ControllerSession

POLICY_ID = "red_tower_ice_ripper4_to_tunnel"

_STAND = frozenset({1, 2})
_CROUCH = frozenset({39, 40})
_TRUE_MORPH = frozenset({29, 30, 31, 32})
# Off the r4 ice top (land 2023 / crouch 2028) before any LEFT.
_AIR_Y = 2015
# Keep LEFT+A until at/above the alcove, then drift onto ~x104.
_RISE_UNTIL_Y = 1860
_SEAT_X = 104
_SHAFT_X_MAX = 125


def _action(*names: str):
    return buttons(*names) if names else idle_action()


def _grounded(state: SuperMetroidState) -> bool:
    return int(state.velocity_y) == 0 and int(state.vertical_direction) == 0


class RedIceRipper4TunnelEdgeRunner:
    """One-action-per-call runner: crouch-jump left onto the tunnel alcove."""

    policy_id = POLICY_ID
    variant_id = VARIANT_ID
    from_checkpoint = LOWER_RIPPER_4.checkpoint_id
    to_checkpoint = TUNNEL_FLOOR.checkpoint_id

    def __init__(self, env: Any, *, max_frames: int = 240, max_attempts: int = 2) -> None:
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
        self.last_reason = "red_ice_r4tun_init"
        self._phase_frames = 0
        self._settle_frames = 0

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
                airborne = (not _grounded(state)) or y <= _AIR_Y
                if airborne and (y <= _RISE_UNTIL_Y or x <= _SHAFT_X_MAX):
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
                    self._fail(
                        f"fell past r4 xy=({state.samus_x},{state.samus_y})"
                    )
                    break
                if int(state.samus_y) >= 2300:
                    self._fail(
                        f"fell to shaft xy=({state.samus_x},{state.samus_y})"
                    )
                    break
                if int(state.pose) in _TRUE_MORPH:
                    return self._emit(_action("UP"), "red_ice_r4tun_unmorph")
                x = int(state.samus_x)
                y = int(state.samus_y)
                if x > _SEAT_X + 3:
                    return self._emit(_action("LEFT"), "red_ice_r4tun_drift_left")
                if x < _SEAT_X - 12 and y <= TUNNEL_FLOOR_Y + 20:
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


def play_ripper4_to_tunnel(
    session: ControllerSession,
    *,
    max_frames: int = 240,
) -> SuperMetroidState:
    """Synchronous facade: frozen r4 → grounded tunnel alcove."""
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError("Red Ice r4→tunnel needs session.env")
    if not can_attach_ripper4_edge(session.state):
        raise TimeoutError(
            f"{POLICY_ID}: not on lower_ripper_4 "
            f"xy=({session.state.samus_x},{session.state.samus_y}) "
            f"p={session.state.pose}"
        )
    runner = RedIceRipper4TunnelEdgeRunner(env, max_frames=max_frames)
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
    "RedIceRipper4TunnelEdgeRunner",
    "play_ripper4_to_tunnel",
]
