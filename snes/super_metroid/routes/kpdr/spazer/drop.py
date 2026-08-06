"""Top return handoff → mid/floor, then West Tunnel exit."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, require_room, unmorph
from super_metroid.routes.kpdr.below_spazer_west import play_below_spazer_floor_to_west
from super_metroid.routes.kpdr.rooms import ROOM_BELOW_SPAZER
from super_metroid.routes.kpdr.spazer.geometry import (
    FLOOR_UNMORPH_POSES,
    FLOOR_Y_MIN,
    MID_PLATFORM_Y,
    in_below_spazer,
    is_lag_pose,
    is_true_ground_pose,
    on_mid_or_floor,
)
from super_metroid.routes.kpdr.spazer.helpers import break_lag, play_script
from super_metroid.routes.kpdr.spazer.scripts import TOP_MID_RLE
from super_metroid.routes.runtime import ControllerSession


def _settle_floor_land(session: ControllerSession) -> SuperMetroidState:
    """After y≥360 fall, wait for standing pose before West runner."""
    for _ in range(80):
        if not in_below_spazer(session.state):
            return session.state
        if is_true_ground_pose(session.state):
            hold(session, 8, reason="spazer_top_mid_floor_stand")
            return session.state
        if is_lag_pose(session.state):
            break_lag(session)
            continue
        if int(session.state.pose) in FLOOR_UNMORPH_POSES:
            hold(session, 1, "UP", reason="spazer_top_mid_unmorph")
            continue
        hold(session, 1, reason="spazer_top_mid_land")
    return session.state


def play_spazer_top_to_mid(session: ControllerSession) -> SuperMetroidState:
    """Top return handoff ~(380,155) → mid band / floor via guide-shaped RLE.

    Spin-left across top, morph crawl left shelf (bombs = X), fall left shaft
    to floor ~(43,395). Lag recovery during script. **Do not** RIGHT into Super.
    """
    require_room(session, ROOM_BELOW_SPAZER, "spazer_top_to_mid")
    unmorph(session)
    if on_mid_or_floor(session.state):
        return session.state
    break_lag(session)

    def _stop(state: SuperMetroidState) -> bool:
        if int(state.samus_y) >= FLOOR_Y_MIN:
            return True
        return on_mid_or_floor(state) and is_true_ground_pose(state)

    play_script(
        session,
        TOP_MID_RLE,
        reason="spazer_top_mid",
        room_id=ROOM_BELOW_SPAZER,
        stop_when=_stop,
        on_lag="break",
    )

    if not in_below_spazer(session.state):
        raise TimeoutError(f"spazer_top_to_mid: left Below Spazer: {session.state}")
    if int(session.state.samus_y) >= FLOOR_Y_MIN:
        return _settle_floor_land(session)
    if not on_mid_or_floor(session.state):
        raise TimeoutError(
            "spazer_top_to_mid: missed mid/floor "
            f"(need y≥{MID_PLATFORM_Y}): {session.state}"
        )
    return session.state


def _walk_mid_platforms_to_floor(session: ControllerSession) -> None:
    """Walk LEFT off mid platforms into water/floor when still elevated."""
    for _ in range(250):
        if not in_below_spazer(session.state):
            return
        if int(session.state.samus_y) >= FLOOR_Y_MIN:
            return
        if is_lag_pose(session.state):
            hold(session, 6, "A", reason="spazer_mid_lag")
            hold(session, 4, reason="spazer_mid_lag")
            continue
        hold(session, 1, "LEFT", "B", reason="spazer_mid_left")


def mid_or_floor_to_west(session: ControllerSession) -> SuperMetroidState:
    """From mid platforms or floor in Below Spazer → West Tunnel."""
    _walk_mid_platforms_to_floor(session)
    return play_below_spazer_floor_to_west(session)


def play_spazer_top_to_west(session: ControllerSession) -> SuperMetroidState:
    """Below Spazer after Spazer return → West (mainline fuse).

    From top return handoff ~(380,155): guide-shaped top→mid/floor drop, then
    stock mid/floor → West. From mid-platform band (y≥220) or floor: stock West
    only. **Do not** RIGHT into open Super door from the handoff pin.
    """
    require_room(session, ROOM_BELOW_SPAZER, "spazer_top_to_west")
    unmorph(session)

    if not on_mid_or_floor(session.state):
        play_spazer_top_to_mid(session)
    if not on_mid_or_floor(session.state):
        raise TimeoutError(
            "spazer_top_to_west: top→mid failed "
            f"(need y≥{MID_PLATFORM_Y}; got y={session.state.samus_y}): "
            f"{session.state}"
        )
    return mid_or_floor_to_west(session)


__all__ = [
    "mid_or_floor_to_west",
    "play_spazer_top_to_mid",
    "play_spazer_top_to_west",
]
