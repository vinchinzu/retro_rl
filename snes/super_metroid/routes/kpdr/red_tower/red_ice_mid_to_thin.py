"""Red Tower Ice edge: temporary mid floor → thin upper seat."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, settle_hold
from super_metroid.routes.kpdr.red_tower.red_ice_climb import (
    THIN_SEAT,
    VARIANT_ID,
    can_attach_mid_floor_edge,
)
from super_metroid.routes.kpdr.red_tower.red_to_hellway_common import _HUMAN_FLOOR_RLE
from super_metroid.routes.kpdr.red_tower.red_to_hellway_upper import (
    _period_wj,
    _play_upper_rle,
    _seat_left_after_handoff,
)
from super_metroid.routes.runtime import ControllerSession

POLICY_ID = "red_tower_ice_mid_floor_to_thin_seat"


def play_mid_floor_to_thin_seat(session: ControllerSession) -> SuperMetroidState:
    """Climb the alternating solid ledges and land the thin seat at y=587."""
    if not can_attach_mid_floor_edge(session.state):
        raise TimeoutError(
            f"{POLICY_ID}: not on mid_floor "
            f"xy=({session.state.samus_x},{session.state.samus_y}) "
            f"p={session.state.pose}"
        )

    _play_upper_rle(session, _HUMAN_FLOOR_RLE, f"{POLICY_ID}_handoff")
    _seat_left_after_handoff(session, f"{POLICY_ID}_left_seat")
    hold(session, 3, "LEFT", "B", reason=f"{POLICY_ID}_launch_run")
    hold(session, 12, "LEFT", "B", "A", reason=f"{POLICY_ID}_launch")

    # The inherited 16f 6/8 cadence gains the first two ledges.  Above y=1050
    # it loses height, so the remaining ledges use short 2f wall contacts.
    _period_wj(session, f"{POLICY_ID}_p0", side="LEFT", frames=600, stop_y=1200)
    _period_wj(session, f"{POLICY_ID}_p1", side="RIGHT", frames=800, stop_y=1050)
    _period_wj(
        session,
        f"{POLICY_ID}_p2",
        side="LEFT",
        frames=800,
        stop_y=900,
        period=16,
        into=2,
        flip=2,
    )
    _period_wj(
        session,
        f"{POLICY_ID}_p3",
        side="RIGHT",
        frames=800,
        stop_y=720,
        period=16,
        into=2,
        flip=2,
    )
    _period_wj(
        session,
        f"{POLICY_ID}_p4",
        side="LEFT",
        frames=500,
        period=22,
        into=2,
        flip=2,
    )
    _period_wj(
        session,
        f"{POLICY_ID}_p5",
        side="RIGHT",
        frames=500,
        period=22,
        into=8,
        flip=2,
    )
    settle_hold(session, 8, reason=f"{POLICY_ID}_settle")
    if THIN_SEAT.matches(session.state):
        return session.state
    raise TimeoutError(
        f"{POLICY_ID}: thin seat not reached "
        f"xy=({session.state.samus_x},{session.state.samus_y}) "
        f"p={session.state.pose}"
    )


__all__ = ["POLICY_ID", "VARIANT_ID", "play_mid_floor_to_thin_seat"]
