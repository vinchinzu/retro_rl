"""Dead Scientist Room 0xE021 outbound: walk off the left alcove, hop the pit.

https://wiki.supermetroid.run/Ceres_Station — two raised door ledges, stairs
into a pit, stairs out. Holding RIGHT on the left lip (y≈139, x≲80) is the
stall; a panic A there bonks the alcove ceiling. Walk down. Jump from the
pit floor onto the right stairs, not from the door.
"""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import GS_ORDINARY, SuperMetroidState
from super_metroid.routes.kpdr.ceres.arm_pump import _ceres_clear_knockback
from super_metroid.routes.kpdr.ceres.geometry import (
    CERES_SCIENTIST_FLOOR_HOP,
    _CERES_SCI_DOOR_Y,
    _CERES_SCI_ENTRY_LEDGE_X,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_FLAT,
    ROOM_CERES_RIDLEY,
    ROOM_CERES_SCIENTIST,
)
from super_metroid.routes.runtime import RouteSession
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.takeoff import shoulder_pump_button, spin_jump


def scientist_on_entry_ledge(state: SuperMetroidState) -> bool:
    """True on the left door alcove — walk off, do not jump."""
    return (
        int(state.room_id) == ROOM_CERES_SCIENTIST
        and abs(int(state.samus_y) - _CERES_SCI_DOOR_Y) <= 16
        and int(state.samus_x) <= _CERES_SCI_ENTRY_LEDGE_X
    )


class CeresScientistCross:
    """One-frame outbound policy for Dead Scientist Room."""

    def __init__(self) -> None:
        self.pump_i = 0

    def action(self, state: SuperMetroidState) -> tuple[str, ...]:
        if int(state.game_state) != GS_ORDINARY:
            return ("RIGHT",)
        if scientist_on_entry_ledge(state):
            return ("RIGHT", "B")
        hop = CERES_SCIENTIST_FLOOR_HOP
        y = int(state.samus_y)
        x = int(state.samus_x)
        if hop.covers_y(y) and (hop.ready(state) or hop.at_ledge_end(x)):
            return spin_jump("RIGHT")
        running = int(state.speed_flag) != 0 or abs(int(state.momentum_x)) >= 1
        if running:
            names = ("RIGHT", "B", shoulder_pump_button(self.pump_i))
            self.pump_i += 1
            return names
        return ("RIGHT", "B")


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
        elif "A" in names:
            reason = "ceres_sci_stair"
        else:
            reason = "ceres_sci"
        session.step(buttons(*names) if names else idle_action(), reason)
    raise TimeoutError(f"ceres scientist missed Flat: {session.state}")


__all__ = [
    "CeresScientistCross",
    "play_ceres_scientist_to_flat",
    "scientist_on_entry_ledge",
]
