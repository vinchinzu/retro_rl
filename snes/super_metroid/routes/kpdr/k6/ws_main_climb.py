"""Powered Wrecked Ship Main Shaft → Attic (rr-kw8t hop 2).

Pin is hop-1 leave ``0xCAF6`` ~(1173,1979) p1 gs=8. First jump is a
hatch-column gun-jump onto the right lip ~(1184,1883) (do not go DOWN to
Basement; save ``0xCE8A`` is x≳1240). Ice Atomics if they block. Blue
ceiling door UP into Attic ``0xCA52``. West Super ``0xCDA8`` is an
optional in-hop side trip.

https://wiki.supermetroid.run/Wrecked_Ship_Main_Shaft
"""

from __future__ import annotations

from super_metroid.hop_glance import (
    LeaveMiss,
    WS_MAIN_TO_ATTIC,
    final_from_state,
    grade_final,
)
from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    is_morph,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.k6.phantoon_fight import phantoon_boss_bit_set
from super_metroid.routes.kpdr.k6.ws_main_actions import (
    SHAFT_HOPS,
    THREE_SHOT_FRAMES,
    THREE_SHOT_X_MAX,
    THREE_SHOT_X_MIN,
    TUNNEL_CLEAR_X,
    WS_MAIN_ATTIC_DOOR_X,
    WS_MAIN_FLOOR_Y,
    WS_MAIN_PIT_Y,
    WS_MAIN_STAIR_Y,
    WS_MAIN_SAVE_X,
    WS_MAIN_SHAFT_CENTER,
    at_ws_main_attic_door_seat,
    at_ws_main_first_jump_land,
    at_ws_main_pit,
    attic_door_action,
    climb_action,
    grate_clear_action,
    pit_exit_action,
    three_shot_action,
    ws_main_attic_settled,
)
from super_metroid.routes.kpdr.k6.ws_main_ice import (
    ATOMIC_ID,
    ShaftEnemy,
    ice_keepaway_action,
    list_shaft_enemies,
)
from super_metroid.routes.kpdr.k6.ws_main_phases import (
    WS_MAIN_PHASES,
    at_ws_main_grate_seat,
    at_ws_main_mid_climb,
    at_ws_main_west_super_band,
    ws_main_phase_index,
)
from super_metroid.routes.skills.geometry import PhaseStop
from super_metroid.routes.kpdr.room_ids import ROOM_WS_ATTIC, ROOM_WS_BASEMENT, ROOM_WS_MAIN
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.charge_shot import session_beam_charge
from super_metroid.routes.skills.knockback import escape_knockback_spin, is_knockback

ROOM_WS_WEST_SUPER = 0xCDA8
ROOM_WS_SAVE = 0xCE8A
ROOM_WS_SPONGE = 0xCD5C
WEAPON_BEAM = 0
_SETTLE = 200
_THREE_SHOT_FRAMES = THREE_SHOT_FRAMES + 40
_CLIMB_BUDGET = 3600
_DOOR_BUDGET = 800
_SIDE_TRIP_BUDGET = 400


def _guard(session: ControllerSession, label: str) -> None:
    room = int(session.state.room_id)
    if room in (ROOM_WS_MAIN, ROOM_WS_ATTIC, ROOM_WS_WEST_SUPER, ROOM_WS_SPONGE):
        return
    if room == ROOM_WS_SAVE:
        raise TimeoutError(f"{label}: entered save 0xCE8A: {session.state}")
    if room == ROOM_WS_BASEMENT:
        raise TimeoutError(f"{label}: dropped back to Basement 0xCC6F: {session.state}")
    raise TimeoutError(f"{label}: left Main Shaft into 0x{room:04X}: {session.state}")


def _kb(session: ControllerSession, label: str) -> None:
    x = int(session.state.samus_x)
    prefer = "LEFT" if x > WS_MAIN_SHAFT_CENTER else "RIGHT"
    escape_knockback_spin(
        session,
        prefer_dir=prefer,
        run_frames=6,
        spin_frames=24,
        label=label,
        stop_room_id=ROOM_WS_ATTIC,
    )


def _exit_side_room(session: ControllerSession, label: str) -> None:
    """West Super RIGHT, Sponge Bath LEFT — both rooms one change."""
    for _ in range(_SIDE_TRIP_BUDGET):
        st = session.state
        room = int(st.room_id)
        if room in (ROOM_WS_MAIN, ROOM_WS_ATTIC):
            return
        _guard(session, label)
        if is_knockback(st):
            _kb(session, f"{label}_side_kb")
            continue
        if is_morph(int(st.pose)):
            unmorph(session)
            continue
        if room == ROOM_WS_WEST_SUPER:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_west_super")
        else:
            hold(session, 1, "LEFT", "B", reason=f"{label}_sponge")
    if int(session.state.room_id) not in (ROOM_WS_MAIN, ROOM_WS_ATTIC):
        raise TimeoutError(f"{label}: side room did not return: {session.state}")


def _three_shot_tunnel(session: ControllerSession, label: str) -> None:
    """Walk to the hatch column and gun-jump onto the right lip ~(1184, 1883)."""
    if int(session.state.room_id) in (ROOM_WS_ATTIC, ROOM_WS_WEST_SUPER):
        return
    if at_ws_main_grate_seat(session.state) or not at_ws_main_pit(session.state):
        return
    select_weapon(session, WEAPON_BEAM)
    shot_i = 0
    for _ in range(_THREE_SHOT_FRAMES):
        st = session.state
        _guard(session, label)
        if int(st.room_id) in (ROOM_WS_ATTIC, ROOM_WS_WEST_SUPER):
            return
        if at_ws_main_grate_seat(st) or not at_ws_main_pit(st):
            return
        if is_knockback(st):
            _kb(session, f"{label}_shot_kb")
            continue
        if is_morph(int(st.pose)):
            unmorph(session)
            continue
        names = three_shot_action(
            int(st.samus_x),
            int(st.samus_y),
            int(st.pose),
            int(st.facing),
            shot_i,
            session_beam_charge(session),
            int(getattr(st, "movement_type", 0) or 0),
            int(st.velocity_y),
        )
        shot_i += 1
        if names:
            hold(session, 1, *names, reason=f"{label}_3shot")
        else:
            hold(session, 1, reason=f"{label}_3shot_land")


def _at_attic_climb_done(state: SuperMetroidState) -> bool:
    return (
        int(state.room_id) == ROOM_WS_ATTIC
        or at_ws_main_attic_door_seat(state)
        or int(state.samus_y) <= 160
    )


def _climb_until(session: ControllerSession, label: str, done) -> None:
    """Spin-hop the shaft until ``done(state)``. Ice nearby Atomics."""
    if int(session.state.room_id) == ROOM_WS_ATTIC or done(session.state):
        return
    if is_morph(int(session.state.pose)):
        unmorph(session)
    select_weapon(session, WEAPON_BEAM)
    for _ in range(_CLIMB_BUDGET):
        st = session.state
        _guard(session, label)
        if int(st.room_id) == ROOM_WS_ATTIC or done(st):
            return
        if int(st.room_id) in (ROOM_WS_WEST_SUPER, ROOM_WS_SPONGE):
            _exit_side_room(session, label)
            continue
        if is_knockback(st):
            _kb(session, f"{label}_climb_kb")
            continue
        if is_morph(int(st.pose)):
            if int(st.samus_x) > TUNNEL_CLEAR_X and int(st.samus_y) < WS_MAIN_STAIR_Y:
                hold(session, 1, "LEFT", reason=f"{label}_roll")
            elif int(st.samus_y) < WS_MAIN_STAIR_Y:
                # UP only — generic unmorph A-settle idles over the gap.
                hold(session, 1, "UP", reason=f"{label}_unmorph")
            else:
                unmorph(session)
            continue
        if at_ws_main_grate_seat(st):
            ensure_morph(session)
            continue
        if (
            at_ws_main_pit(st)
            and not at_ws_main_grate_seat(st)
            and int(st.samus_y) >= WS_MAIN_STAIR_Y
        ):
            names = three_shot_action(
                int(st.samus_x),
                int(st.samus_y),
                int(st.pose),
                int(st.facing),
                int(session.frame),
                session_beam_charge(session),
                int(getattr(st, "movement_type", 0) or 0),
                int(st.velocity_y),
            )
            if names:
                hold(session, 1, *names, reason=f"{label}_pit")
            else:
                hold(session, 1, reason=f"{label}_pit_fire")
            continue
        grate = grate_clear_action(
            int(st.samus_x),
            int(st.samus_y),
            int(st.pose),
            int(st.facing),
            int(session.frame),
            session_beam_charge(session),
            int(st.velocity_y),
            int(getattr(st, "movement_type", 0) or 0),
        )
        if grate is not None:
            if grate:
                hold(session, 1, *grate, reason=f"{label}_grate")
            else:
                hold(session, 1, reason=f"{label}_grate_wait")
            continue
        # Grate band is the first-jump landing / west_super takeoff. Ice
        # jump-shot faces LEFT at the Covern and pulls off the lip.
        if int(st.samus_y) >= 1760:
            names = climb_action(
                int(st.samus_x),
                int(st.samus_y),
                int(st.pose),
                int(st.facing),
                int(st.velocity_y),
                int(getattr(st, "movement_type", 0) or 0),
                int(session.frame),
            )
            if names:
                hold(session, 1, *names, reason=f"{label}_climb")
            else:
                hold(session, 1, reason=f"{label}_wait")
            continue
        enemies = list_shaft_enemies(session)
        keepaway = ice_keepaway_action(
            int(st.samus_x),
            int(st.samus_y),
            int(st.facing),
            enemies,
            movement_type=int(getattr(st, "movement_type", 0) or 0),
            charge=session_beam_charge(session),
            velocity_y=int(st.velocity_y),
        )
        if keepaway is not None:
            if keepaway:
                hold(session, 1, *keepaway, reason=f"{label}_ice")
            else:
                hold(session, 1, reason=f"{label}_ice_wait")
            continue
        names = climb_action(
            int(st.samus_x),
            int(st.samus_y),
            int(st.pose),
            int(st.facing),
            int(st.velocity_y),
            int(getattr(st, "movement_type", 0) or 0),
            int(session.frame),
        )
        if names:
            hold(session, 1, *names, reason=f"{label}_climb")
        else:
            hold(session, 1, reason=f"{label}_wait")
    if int(session.state.room_id) != ROOM_WS_ATTIC and not done(session.state):
        raise TimeoutError(f"{label}: did not reach phase seat: {session.state}")


def _climb_to_attic_door(session: ControllerSession, label: str) -> None:
    """Spin-hop the shaft. Ice nearby Atomics. Stop on attic-door seat."""
    _climb_until(session, label, _at_attic_climb_done)


def _jump_up_attic(session: ControllerSession, label: str) -> None:
    """Shoot then jump UP through the blue ceiling door."""
    if int(session.state.room_id) == ROOM_WS_ATTIC:
        return
    select_weapon(session, WEAPON_BEAM)
    shoot_i = 0
    for _ in range(_DOOR_BUDGET):
        st = session.state
        _guard(session, label)
        if int(st.room_id) == ROOM_WS_ATTIC:
            return
        if int(st.room_id) in (ROOM_WS_WEST_SUPER, ROOM_WS_SPONGE):
            _exit_side_room(session, label)
            continue
        if is_knockback(st):
            _kb(session, f"{label}_door_kb")
            continue
        if is_morph(int(st.pose)):
            unmorph(session)
            continue
        if int(st.samus_y) > 160:
            shoot_i = 0
            names = climb_action(
                int(st.samus_x),
                int(st.samus_y),
                int(st.pose),
                int(st.facing),
                int(st.velocity_y),
                int(getattr(st, "movement_type", 0) or 0),
                int(session.frame),
            )
            reason = f"{label}_remount"
        else:
            names = attic_door_action(
                int(st.samus_x), int(st.samus_y), int(st.pose), shoot_i
            )
            shoot_i += 1
            reason = f"{label}_door"
        if names:
            hold(session, 1, *names, reason=reason)
        else:
            hold(session, 1, reason=f"{label}_hurt")
    if int(session.state.room_id) != ROOM_WS_ATTIC:
        raise TimeoutError(f"{label}: attic door missed: {session.state}")


def _raise_leave_miss(session: ControllerSession, exc: BaseException | None = None) -> None:
    leftover = final_from_state(session.state)
    misses = list(grade_final(leftover, WS_MAIN_TO_ATTIC))
    if exc is not None:
        misses.append(f"{type(exc).__name__}: {exc}")
    raise LeaveMiss(
        "ws_main_to_attic",
        leftover,
        misses or ["leave failed"],
        room_label="Attic",
        to_room=ROOM_WS_ATTIC,
    ) from exc


def _settle_attic(session: ControllerSession, label: str) -> SuperMetroidState:
    wait_ordinary_room(session, ROOM_WS_ATTIC, settle_frames=_SETTLE, label=label)
    for _ in range(90):
        st = session.state
        if int(st.pose) in (1, 2, 9, 10) and abs(int(st.velocity_y)) <= 1:
            break
        hold(session, 1, reason=f"{label}_land")
    return session.state


def play_ws_main_to_attic_phased(
    session: ControllerSession,
    *,
    start: str = "pit_shot",
    stop: str = "attic_door",
) -> SuperMetroidState:
    """Run a slice of the Main Shaft climb. ``PhaseStop`` at ``stop`` unless Attic.

    Diagnostic only. Hop GREEN is still ``play_ws_main_to_attic`` to Attic gs=8.
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
            _three_shot_tunnel(session, f"{label}_pit_shot")
            _maybe_stop("pit_shot")
        if start_i <= 1:
            _climb_until(session, f"{label}_grate_seat", at_ws_main_grate_seat)
            _maybe_stop("grate_seat")
        if start_i <= 2:
            _climb_until(session, f"{label}_west_super", at_ws_main_west_super_band)
            _maybe_stop("west_super")
        if start_i <= 3:
            _climb_until(session, f"{label}_mid_climb", at_ws_main_mid_climb)
            _maybe_stop("mid_climb")
        if start_i <= 4:
            _climb_until(session, f"{label}_attic_seat", _at_attic_climb_done)
            _maybe_stop("attic_seat")
        _jump_up_attic(session, label)
        return _settle_attic(session, label)
    except (LeaveMiss, PhaseStop):
        raise
    except Exception as exc:
        _raise_leave_miss(session, exc)
        raise


def play_ws_main_to_attic(session: ControllerSession) -> SuperMetroidState:
    """Powered Main Shaft climb. AFS Wave 3-shot, jump UP into Attic.

    Pin: ``scratch/post_ws_basement_to_main.state`` ``0xCAF6`` ~(1173,1979)
    p1 gs=8. Hatch-column gun-jump onto ~(1184,1883), climb, tap-shot the
    blue ceiling door ~x=1135. Lands ordinary ``gs=8`` in ``0xCA52``. Six
    in-room phases: see ``ws_main_phases``.
    """
    label = "ws_main_to_attic"
    try:
        if ws_main_attic_settled(session.state):
            return session.state
        require_room(session, ROOM_WS_MAIN, label)
        if not phantoon_boss_bit_set(session):
            raise RuntimeError(f"{label}: Phantoon not defeated: {session.state}")
        _three_shot_tunnel(session, label)
        _climb_to_attic_door(session, label)
        _jump_up_attic(session, label)
        return _settle_attic(session, label)
    except LeaveMiss:
        raise
    except PhaseStop:
        raise
    except Exception as exc:
        _raise_leave_miss(session, exc)
        raise  # unreachable; keeps type checkers happy


__all__ = [
    "ATOMIC_ID",
    "ROOM_WS_SAVE",
    "ROOM_WS_SPONGE",
    "ROOM_WS_WEST_SUPER",
    "SHAFT_HOPS",
    "THREE_SHOT_X_MAX",
    "THREE_SHOT_X_MIN",
    "TUNNEL_CLEAR_X",
    "WEAPON_BEAM",
    "WS_MAIN_ATTIC_DOOR_X",
    "WS_MAIN_PIT_Y",
    "WS_MAIN_SAVE_X",
    "ShaftEnemy",
    "at_ws_main_attic_door_seat",
    "at_ws_main_first_jump_land",
    "at_ws_main_pit",
    "attic_door_action",
    "climb_action",
    "grate_clear_action",
    "ice_keepaway_action",
    "list_shaft_enemies",
    "pit_exit_action",
    "play_ws_main_to_attic",
    "play_ws_main_to_attic_phased",
    "three_shot_action",
    "ws_main_attic_settled",
    "WS_MAIN_PHASES",
]
