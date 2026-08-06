"""Below Spazer solid top → Super green door → Spazer Room."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import ROOM_BELOW_SPAZER, ROOM_SPAZER
from super_metroid.routes.kpdr.spazer.climb import play_below_spazer_climb
from super_metroid.routes.kpdr.spazer.geometry import (
    DOOR_X_MIN,
    SOLID_TOP_X_MIN,
    SOLID_TOP_Y,
    TOP_Y_MAX,
    in_below_spazer,
    is_lag_pose,
    on_solid_top,
    on_super_door_approach,
    solid_ish_top,
)
from super_metroid.routes.kpdr.spazer.helpers import break_lag, play_script
from super_metroid.routes.kpdr.spazer.scripts import TOP_DOOR_APPROACH_RLE
from super_metroid.routes.runtime import ControllerSession


def approach_super_door_from_top(session: ControllerSession) -> SuperMetroidState:
    """From solid top / node4, morph-tunnel crawl to Super-door approach band.

    Mainline door approach: guide-shaped morph tunnel + RIGHT hops to green
    Super lip. Open-loop must not break lag mid-script (pose 137 @ x≈411 is
    part of hop timing).
    """
    if on_super_door_approach(session.state):
        return session.state
    # Clear pre-RLE lag only; mid-script lag is intentional.
    if is_lag_pose(session.state):
        break_lag(session)

    if int(session.state.samus_x) < DOOR_X_MIN:
        play_script(
            session,
            TOP_DOOR_APPROACH_RLE,
            reason="spazer_top_door_tunnel",
            room_id=ROOM_BELOW_SPAZER,
            stop_when=on_super_door_approach,
            on_lag="ignore",
        )

    if not on_super_door_approach(session.state):
        unmorph(session)
        for _ in range(80):
            if not in_below_spazer(session.state):
                return session.state
            if on_super_door_approach(session.state):
                return session.state
            hold(session, 1, "RIGHT", "B", reason="spazer_door_approach")
    return session.state


def play_below_spazer_to_spazer(
    session: ControllerSession,
) -> SuperMetroidState:
    """Below Spazer → Spazer Room via Super green door (climbs if needed).

    Floor entry: climb to solid top, morph-tunnel right to the green Super door
    (bombs=X on shelf — mainline approach), Super, enter Spazer. Pre-door pure
    pin (~460,139) skips climb/tunnel.

    Expected exit: Spazer Room ordinary, just inside left door.
    """
    require_room(session, ROOM_BELOW_SPAZER, "below_spazer_to_spazer")
    unmorph(session)
    break_lag(session)

    if not (on_solid_top(session.state) or on_super_door_approach(session.state)):
        play_below_spazer_climb(session)
        break_lag(session)
        if not solid_ish_top(session.state):
            raise TimeoutError(
                "below_spazer_to_spazer: climb failed solid top "
                f"(need x≥{SOLID_TOP_X_MIN}, y∈{SOLID_TOP_Y} or y≤{TOP_Y_MAX} "
                f"grounded): {session.state}"
            )

    approach_super_door_from_top(session)
    if not on_super_door_approach(session.state):
        raise TimeoutError(
            f"below_spazer_to_spazer: missed green-door lip: {session.state}"
        )

    hold(session, 10, reason="spazer_door_settle")
    select_weapon(session, 2)
    hold(session, 6, reason="spazer_super_ready")
    hold(session, 3, "RIGHT", reason="spazer_face_door")
    hold(session, 3, reason="spazer_face_door_release")
    hold(session, 8, "X", reason="spazer_green_door_super")
    hold(session, 50, reason="spazer_green_door_fuse")
    for _ in range(250):
        state = hold(session, 1, "RIGHT", "B", reason="spazer_enter")
        if state.room_id == ROOM_SPAZER:
            break
    else:
        raise TimeoutError(
            f"below_spazer_to_spazer: green door did not open: {session.state}"
        )
    return wait_ordinary_room(
        session, ROOM_SPAZER, settle_frames=120, label="below_spazer_to_spazer"
    )


__all__ = [
    "approach_super_door_from_top",
    "play_below_spazer_to_spazer",
]
