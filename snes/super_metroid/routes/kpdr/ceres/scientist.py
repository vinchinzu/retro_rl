"""Dead Scientist Room 0xE021: arm-pump run, never jump.

https://wiki.supermetroid.run/Dead_Scientist_Room — two raised door ledges,
stairs into a pit, stairs out. Sniq 100% lsnes never presses A either way:
outbound gs=8 f9577→9680 RIGHT+B+L/R off (39,139) p9; reverse f12198→12299
LEFT+B+L/R off (472,139) p18. A on the alcove bonks. The floor hop window
was the +20f / pose-137 stall versus TAS.

Fade already matches (161 vs 161 outbound, 162 vs 162 reverse). Leftover
dwell is the door-lip stall (outbound x=467 pose 207; reverse x=45 pose 208).
"""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import GS_ORDINARY, SuperMetroidState
from super_metroid.routes.kpdr.ceres.arm_pump import _ceres_clear_knockback
from super_metroid.routes.kpdr.ceres.geometry import (
    _CERES_SCI_DOOR_Y,
    _CERES_SCI_ENTRY_LEDGE_X,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_FALLING,
    ROOM_CERES_FLAT,
    ROOM_CERES_MAGNET,
    ROOM_CERES_RIDLEY,
    ROOM_CERES_SCIENTIST,
)
from super_metroid.routes.runtime import RouteSession
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.takeoff import shoulder_pump_button


def scientist_on_entry_ledge(state: SuperMetroidState) -> bool:
    """True on the left door alcove."""
    return (
        int(state.room_id) == ROOM_CERES_SCIENTIST
        and abs(int(state.samus_y) - _CERES_SCI_DOOR_Y) <= 16
        and int(state.samus_x) <= _CERES_SCI_ENTRY_LEDGE_X
    )


class CeresScientistCross:
    """One-frame Dead Scientist Room policy. Never jump.

    Outbound (RIGHT): Sniq 100% lsnes gs=8 f9577→door f9680.
    Reverse (LEFT): gs=8 f12198→door f12299. Stairs + y=187 pit, no A.
    """

    def __init__(self, direction: str = "RIGHT") -> None:
        if direction not in ("LEFT", "RIGHT"):
            raise ValueError(f"scientist direction must be LEFT or RIGHT, got {direction!r}")
        self.direction = direction
        self.pump_i = 0

    def action(self, state: SuperMetroidState) -> tuple[str, ...]:
        if int(state.game_state) != GS_ORDINARY:
            return (self.direction,)
        names = (self.direction, "B", shoulder_pump_button(self.pump_i))
        self.pump_i += 1
        return names


def _scientist_past(state: SuperMetroidState) -> bool:
    """True in Flat/Ridley ordinary — not the scientist→flat door (gs 9/11)."""
    if int(state.game_state) != GS_ORDINARY:
        return False
    return int(state.room_id) in (ROOM_CERES_FLAT, ROOM_CERES_RIDLEY)


def play_ceres_scientist_to_flat(session: RouteSession) -> None:
    """Scientist ordinary → Flat (or Ridley if the door overshoots).

    No-op when already past the room. Waits out the magnet→scientist door
    before treating x-stagnation as a ledge.
    """
    if _scientist_past(session.state):
        return
    for _ in range(160):
        st = session.state
        if _scientist_past(st):
            return
        if int(st.room_id) == ROOM_CERES_SCIENTIST and int(st.game_state) == GS_ORDINARY:
            break
        session.step(buttons("RIGHT"), "ceres_sci_door")
    else:
        st = session.state
        if int(st.room_id) != ROOM_CERES_SCIENTIST:
            raise TimeoutError(f"ceres scientist ordinary missed: {st}")

    cross = CeresScientistCross()
    for _ in range(400):
        st = session.state
        if _scientist_past(st):
            return
        if is_knockback(st):
            _ceres_clear_knockback(session, "RIGHT", reason="ceres_sci")
            continue
        names = cross.action(st)
        if scientist_on_entry_ledge(st):
            reason = "ceres_sci_ledge"
        elif int(st.game_state) != GS_ORDINARY:
            reason = "ceres_sci_fade"
        else:
            reason = "ceres_sci"
        session.step(buttons(*names) if names else idle_action(), reason)
    raise TimeoutError(f"ceres scientist missed Flat: {session.state}")


def _scientist_escape_past(state: SuperMetroidState) -> bool:
    """True in Magnet/Falling ordinary — not the scientist→magnet door."""
    if int(state.game_state) != GS_ORDINARY:
        return False
    return int(state.room_id) in (ROOM_CERES_MAGNET, ROOM_CERES_FALLING)


def play_ceres_scientist_to_magnet(session: RouteSession) -> None:
    """Scientist ordinary → Magnet (or Falling if the door overshoots).

    Reverse Ceres 4. TAS dwell (f12198–12299) never presses A: LEFT+B+L/R
    off (472,139) p18, down the east stairs, across y=187, up the west
    stairs into the left door. stuck-jump on the stairs is leftover.
    """
    if _scientist_escape_past(session.state):
        return
    for _ in range(180):
        st = session.state
        if _scientist_escape_past(st):
            return
        if int(st.room_id) == ROOM_CERES_SCIENTIST and int(st.game_state) == GS_ORDINARY:
            break
        session.step(buttons("LEFT"), "ceres_sci_rev_door")
    else:
        st = session.state
        if int(st.room_id) != ROOM_CERES_SCIENTIST:
            raise TimeoutError(f"ceres scientist reverse ordinary missed: {st}")

    cross = CeresScientistCross("LEFT")
    for _ in range(400):
        st = session.state
        if _scientist_escape_past(st):
            return
        if is_knockback(st):
            _ceres_clear_knockback(session, "LEFT", reason="ceres_sci_rev")
            continue
        names = cross.action(st)
        if int(st.game_state) != GS_ORDINARY:
            reason = "ceres_sci_rev_fade"
        else:
            reason = "ceres_sci_rev"
        session.step(buttons(*names) if names else idle_action(), reason)
    raise TimeoutError(f"ceres scientist reverse missed Magnet: {session.state}")


__all__ = [
    "CeresScientistCross",
    "play_ceres_scientist_to_flat",
    "play_ceres_scientist_to_magnet",
    "scientist_on_entry_ledge",
]
