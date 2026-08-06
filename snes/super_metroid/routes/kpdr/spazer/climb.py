"""Below Spazer climb: floor → standing mid → solid top (node 4)."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    consecutive_walljumps,
    hold,
    require_room,
    unmorph,
)
from super_metroid.routes.kpdr.rooms import ROOM_BELOW_SPAZER
from super_metroid.routes.kpdr.spazer.geometry import (
    CACATAC_OFF_DOOR_X,
    CREST_FAIL_Y,
    CREST_LAND_X,
    CREST_PERIOD,
    CREST_PERIOD_FLIP,
    CREST_PERIOD_INTO,
    DOOR_SAFE_X,
    FLOOR_LIP_X,
    HIGH_AIR_Y_MAX,
    MID_Y,
    OVER_LIP_Y_MAX,
    SOLID_TOP_X_MIN,
    SOLID_TOP_Y,
    WJ_PAIR,
    in_below_spazer,
    is_lag_pose,
    is_true_ground_pose,
    mid_band,
    on_solid_top,
    standing_mid_seat,
)
from super_metroid.routes.kpdr.spazer.helpers import (
    break_lag,
    play_script,
    try_select_weapon,
)
from super_metroid.routes.kpdr.spazer.scripts import FLOOR_MID_RLE
from super_metroid.routes.runtime import ControllerSession


def _off_bat_door(
    session: ControllerSession, *, target_x: int = DOOR_SAFE_X
) -> None:
    """Nudge RIGHT off the Bat door face before left-shaft wall-jumps.

    Grounded / low: full run nudge. High airborne (RLE peak ~y284): only a
    short RIGHT tap — long RIGHT+B dumps height and floors the climb.
    """
    high_air = int(session.state.samus_y) <= HIGH_AIR_Y_MAX and not is_true_ground_pose(
        session.state
    )
    limit = 10 if high_air else 40
    for _ in range(limit):
        if not in_below_spazer(session.state):
            return
        if int(session.state.samus_x) >= target_x and int(session.state.door_transition) == 0:
            return
        if high_air:
            hold(session, 1, "RIGHT", reason="spazer_off_bat_door_air")
        else:
            hold(session, 1, "RIGHT", "B", reason="spazer_off_bat_door")


def _settle_over_lip(session: ControllerSession) -> SuperMetroidState:
    """After clearing the shaft wall, keep x on ledge and wait for solid top."""
    for _ in range(45):
        if not in_below_spazer(session.state):
            return session.state
        if on_solid_top(session.state):
            return session.state
        sx = int(session.state.samus_x)
        if sx < CREST_LAND_X[0]:
            hold(session, 1, "RIGHT", reason="spazer_crest_land")
        elif sx > CREST_LAND_X[1]:
            hold(session, 1, "LEFT", reason="spazer_crest_land")
        else:
            hold(session, 1, reason="spazer_crest_land")
        if on_solid_top(session.state):
            return session.state
    return session.state


def _crest_period_over_lip(
    session: ControllerSession,
    *,
    frames: int = 100,
) -> SuperMetroidState:
    """Period WJ crest: clear shaft lip then settle on node 4.

    Mid WJ alone peaks air ~y124 @ x≈59 (blocked by shaft wall). Open-loop
    period carries over the lip onto solid left top (natural stand ~y91).
    """
    for i in range(frames):
        if not in_below_spazer(session.state):
            return session.state
        if on_solid_top(session.state):
            return session.state
        x = int(session.state.samus_x)
        y = int(session.state.samus_y)
        if x >= SOLID_TOP_X_MIN and y <= OVER_LIP_Y_MAX:
            return _settle_over_lip(session)
        if y > CREST_FAIL_Y:
            return session.state
        if x < 40:
            hold(session, 2, "RIGHT", "B", reason="spazer_crest_door")
            continue
        ph = i % CREST_PERIOD
        if ph < CREST_PERIOD_INTO:
            hold(session, 1, "LEFT", "A", reason="spazer_crest_period")
        elif ph < CREST_PERIOD_INTO + CREST_PERIOD_FLIP:
            hold(session, 1, "RIGHT", "A", reason="spazer_crest_period")
        else:
            hold(session, 1, "RIGHT", "B", "A", reason="spazer_crest_period")
    return session.state


def play_below_spazer_mid_to_top(
    session: ControllerSession,
    *,
    max_attempts: int = 5,
) -> SuperMetroidState:
    """Mid shaft → solid top via WJ height + period over-lip crest land.

    Place-green from standing mid y≈230–280, x≈48–55. Success is **only**
    :func:`~super_metroid.routes.kpdr.spazer.geometry.on_solid_top`.
    """
    require_room(session, ROOM_BELOW_SPAZER, "below_spazer_mid_to_top")
    # Standing / low: unmorph + off-door. High air (RLE peak): keep height.
    if int(session.state.samus_y) > HIGH_AIR_Y_MAX or is_true_ground_pose(session.state):
        unmorph(session)
        _off_bat_door(session)
    elif int(session.state.samus_x) < 40:
        _off_bat_door(session)

    def _stop_high(state: SuperMetroidState) -> bool:
        if not in_below_spazer(state):
            return True
        if on_solid_top(state):
            return True
        if int(state.door_transition) != 0 and int(state.samus_x) < DOOR_SAFE_X:
            return True
        return int(state.samus_y) <= 160

    for attempt in range(max_attempts):
        if not in_below_spazer(session.state):
            break
        if on_solid_top(session.state):
            break
        if int(session.state.samus_x) < DOOR_SAFE_X and int(session.state.samus_y) > HIGH_AIR_Y_MAX:
            _off_bat_door(session)
        # Two pairs to gain height, then period crest (three pairs overshoot).
        consecutive_walljumps(
            session,
            WJ_PAIR + WJ_PAIR,
            reason=f"spazer_wj{attempt}",
            gap_frames=0,
            stop_when=_stop_high,
        )
        if on_solid_top(session.state):
            break
        if int(session.state.samus_y) <= 200:
            _crest_period_over_lip(session)
        if on_solid_top(session.state):
            break
        hold(session, 3, reason="spazer_wj_gap")

    if not in_below_spazer(session.state):
        raise TimeoutError(
            f"below_spazer_mid_to_top: left Below Spazer: {session.state}"
        )
    if not on_solid_top(session.state):
        raise TimeoutError(
            "below_spazer_mid_to_top: missed solid top "
            f"(need x≥{SOLID_TOP_X_MIN}, y∈{SOLID_TOP_Y}, grounded pose): "
            f"{session.state}"
        )
    hold(session, 8, reason="spazer_top_settle")
    return session.state


def _clear_floor_cacatac(session: ControllerSession) -> None:
    """Shoot the floor-lip Cacatac before spin (spike = knockoff into gap)."""
    try_select_weapon(session, 0)
    hold(session, 4, reason="spazer_cacatac_weapon")
    if int(session.state.samus_x) < CACATAC_OFF_DOOR_X:
        hold(session, 6, "RIGHT", reason="spazer_cacatac_off_door")
        hold(session, 4, reason="spazer_cacatac_off_door")
    for _ in range(6):
        if not in_below_spazer(session.state):
            return
        if mid_band(session.state) or on_solid_top(session.state):
            return
        hold(session, 3, "UP", reason="spazer_cacatac_aim")
        hold(session, 2, "UP", "X", reason="spazer_cacatac_shot")
        hold(session, 16, "UP", reason="spazer_cacatac_wait")
    hold(session, 8, reason="spazer_cacatac_settle")
    # Re-seat on lip for open-loop spin (x≈48–55, grounded).
    for _ in range(20):
        if not in_below_spazer(session.state):
            return
        if is_lag_pose(session.state):
            break_lag(session, reason="spazer_cacatac_lag")
            continue
        x = int(session.state.samus_x)
        if x < FLOOR_LIP_X[0]:
            hold(session, 1, "RIGHT", reason="spazer_cacatac_reseal")
        elif x > FLOOR_LIP_X[1]:
            hold(session, 1, "LEFT", reason="spazer_cacatac_reseal")
        elif not is_true_ground_pose(session.state):
            hold(session, 1, reason="spazer_cacatac_reseal")
        else:
            break
    hold(session, 6, reason="spazer_cacatac_ready")


def play_below_spazer_floor_to_mid(
    session: ControllerSession,
) -> SuperMetroidState:
    """Floor lip → mid band for WJ via guide-shaped open-loop phases.

    1. Clear Cacatac (Charge-cadence UP+X).
    2. Re-seat lip, then open-loop spin-crest (guide ``floor_to_mid`` shape).
    Early-exits on mid band or solid top. **Does not** off-door (dumps height).
    """
    require_room(session, ROOM_BELOW_SPAZER, "below_spazer_floor_to_mid")
    unmorph(session)
    if mid_band(session.state) or on_solid_top(session.state):
        return session.state
    _clear_floor_cacatac(session)
    if mid_band(session.state) or on_solid_top(session.state):
        return session.state

    def _stop(state: SuperMetroidState) -> bool:
        return standing_mid_seat(state) or on_solid_top(state)

    play_script(
        session,
        FLOOR_MID_RLE,
        reason="spazer_floor_mid",
        room_id=ROOM_BELOW_SPAZER,
        stop_when=_stop,
        on_lag="ignore",
    )
    # Settle spin/fall (continuous often ends RLE mid-air ~y218–236).
    for _ in range(60):
        if not in_below_spazer(session.state):
            return session.state
        if mid_band(session.state) or on_solid_top(session.state):
            return session.state
        if is_lag_pose(session.state):
            break_lag(session, reason="spazer_floor_mid_lag")
            continue
        hold(session, 1, reason="spazer_floor_mid_settle")
    if not in_below_spazer(session.state):
        raise TimeoutError(
            f"below_spazer_floor_to_mid: left Below Spazer: {session.state}"
        )
    if not mid_band(session.state) and not on_solid_top(session.state):
        raise TimeoutError(
            "below_spazer_floor_to_mid: missed mid band "
            f"(need y∈[210,300], x≈40–80): {session.state}"
        )
    return session.state


def play_below_spazer_climb(
    session: ControllerSession,
) -> SuperMetroidState:
    """Floor (or mid) → solid top wall-jump climb in Below Spazer.

    Floor→standing mid via guide-shaped phases; mid→top via WJ + period
    over-lip crest. Success is **only** solid top.
    """
    require_room(session, ROOM_BELOW_SPAZER, "below_spazer_climb")
    unmorph(session)
    if on_solid_top(session.state):
        return session.state
    if not standing_mid_seat(session.state) and int(session.state.samus_y) > 280:
        play_below_spazer_floor_to_mid(session)
    if on_solid_top(session.state):
        return session.state
    if int(session.state.samus_y) <= MID_Y:
        play_below_spazer_mid_to_top(session)
    if not on_solid_top(session.state):
        raise TimeoutError(
            "below_spazer_climb: missed solid top "
            f"(need x≥{SOLID_TOP_X_MIN}, y∈{SOLID_TOP_Y}, grounded pose): "
            f"{session.state}"
        )
    return session.state


__all__ = [
    "play_below_spazer_climb",
    "play_below_spazer_floor_to_mid",
    "play_below_spazer_mid_to_top",
]
