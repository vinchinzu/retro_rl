"""Powered Wrecked Ship Main Shaft → Attic (rr-kw8t hop 2).

Pin is hop-1 leave ``0xCAF6`` ~(1173,1979) p1 gs=8. Take02 two-hop onto
the fire slope ~(1223,1860), then climb (do not go DOWN to Basement;
save ``0xCE8A`` is x≳1240). Ice Atomics if they block. Blue ceiling
door UP into Attic ``0xCA52``. West Super ``0xCDA8`` is an optional
in-hop side trip.

Phases live in ``ws_main_geometry``. Climb loop: ``ws_main_shaft``.
Actions: ``ws_main_actions``. Overlay Ice: ``ws_main_ice``.

https://wiki.supermetroid.run/Wrecked_Ship_Main_Shaft
"""

from __future__ import annotations

from super_metroid.hop_glance import LeaveMiss, raise_leave_miss
from super_metroid.leave_specs import WS_MAIN_TO_ATTIC
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import hold, require_room, select_weapon
from super_metroid.routes.kpdr.wrecked_ship.phantoon_fight import phantoon_boss_bit_set
from super_metroid.routes.kpdr.wrecked_ship.ws_ceiling_door import (
    play_ceiling_door,
    settle_ceiling_dest,
)
from super_metroid.routes.kpdr.wrecked_ship.ws_main_actions import (
    attic_door_action,
    climb_action,
)
from super_metroid.routes.kpdr.wrecked_ship.ws_main_geometry import (
    at_ws_main_mid_climb,
    at_ws_main_usable_grate_seat,
    at_ws_main_west_super_band,
    classify_region,
    ws_main_attic_settled,
    ws_main_phase_index,
)
from super_metroid.routes.kpdr.wrecked_ship.ws_main_shaft import (
    WEAPON_BEAM,
    at_attic_climb_done,
    climb_until,
    exit_side_room,
    guard_main_shaft,
    knockback_main_shaft,
    three_shot_tunnel,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_WS_ATTIC,
    ROOM_WS_MAIN,
    ROOM_WS_SPONGE,
    ROOM_WS_WEST_SUPER,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.geometry import PhaseStop

_GRATE_SEAT_SETTLE_FRAMES = 5


def _settle_grate_seat_momentum(session: ControllerSession, label: str) -> None:
    """Release the natural-entry slide before the tape-locked lip shot."""
    if abs(int(session.state.momentum_x)) == 0:
        return
    hold(
        session,
        _GRATE_SEAT_SETTLE_FRAMES,
        reason=f"{label}_grate_seat_settle",
    )


def _jump_up_attic(session: ControllerSession, label: str) -> None:
    """Shoot then jump UP through the blue ceiling door."""
    if int(session.state.room_id) == ROOM_WS_ATTIC:
        return
    select_weapon(session, WEAPON_BEAM)

    def _remount(st: SuperMetroidState) -> tuple[str, ...]:
        return climb_action(
            int(st.samus_x),
            int(st.samus_y),
            int(st.pose),
            int(st.facing),
            int(st.velocity_y),
            int(st.movement_type),
            int(session.frame),
            region=classify_region(st),
        )

    def _door(st: SuperMetroidState, shoot_i: int) -> tuple[str, ...]:
        return attic_door_action(
            int(st.samus_x), int(st.samus_y), int(st.pose), shoot_i
        )

    play_ceiling_door(
        session,
        label=label,
        dest_room=ROOM_WS_ATTIC,
        lip_y=160,
        remount=_remount,
        door_action=_door,
        guard=guard_main_shaft,
        on_knockback=knockback_main_shaft,
        side_rooms=(ROOM_WS_WEST_SUPER, ROOM_WS_SPONGE),
        on_side_room=exit_side_room,
    )


def play_ws_main_to_attic(
    session: ControllerSession,
    *,
    start: str = "pit_shot",
    stop: str = "attic_door",
) -> SuperMetroidState:
    """Powered Main Shaft climb. AFS Wave 3-shot, jump UP into Attic.

    Pin: ``scratch/post_ws_basement_to_main.state`` ``0xCAF6`` ~(1173,1979)
    p1 gs=8. Take02 two-hop onto ~(1223,1860), climb, tap-shot the blue
    ceiling door ~x=1135. Lands ordinary ``gs=8`` in ``0xCA52``. Six
    in-room phases: see ``ws_main_geometry``.

    Diagnostic ``stop`` raises ``PhaseStop`` unless it is ``attic_door``.
    """
    label = "ws_main_to_attic"
    start_i = ws_main_phase_index(start)
    stop_i = ws_main_phase_index(stop)
    if start_i > stop_i:
        raise ValueError(f"start phase {start!r} is after stop {stop!r}")

    def _maybe_stop(phase: str) -> None:
        if ws_main_phase_index(phase) >= stop_i and phase != "attic_door":
            raise PhaseStop(phase, session.state, label="ws_main_phase_stop")

    try:
        if ws_main_attic_settled(session.state):
            return session.state
        require_room(session, ROOM_WS_MAIN, label)
        if not phantoon_boss_bit_set(session):
            raise RuntimeError(f"{label}: Phantoon not defeated: {session.state}")
        if start_i <= 0:
            three_shot_tunnel(session, f"{label}_pit_shot")
            _maybe_stop("pit_shot")
        if start_i <= 1:
            climb_until(session, f"{label}_grate_seat", at_ws_main_usable_grate_seat)
            _maybe_stop("grate_seat")
            _settle_grate_seat_momentum(session, label)
        if start_i <= 2:
            climb_until(session, f"{label}_west_super", at_ws_main_west_super_band)
            _maybe_stop("west_super")
        if start_i <= 3:
            climb_until(session, f"{label}_mid_climb", at_ws_main_mid_climb)
            _maybe_stop("mid_climb")
        if start_i <= 4:
            climb_until(session, f"{label}_attic_seat", at_attic_climb_done)
            _maybe_stop("attic_seat")
        _jump_up_attic(session, label)
        return settle_ceiling_dest(session, ROOM_WS_ATTIC, label=label)
    except (LeaveMiss, PhaseStop):
        raise
    except Exception as exc:
        raise_leave_miss(
            session.state,
            "ws_main_to_attic",
            WS_MAIN_TO_ATTIC,
            room_label="Attic",
            to_room=ROOM_WS_ATTIC,
            exc=exc,
        )
        raise


__all__ = [
    "play_ws_main_to_attic",
    "ws_main_attic_settled",
]
