"""Powered Main Shaft climb loop (rr-kw8t hop 2).

Pit 3-shot, Ice keepaway, save-column wall-jump. Play wrappers and the
attic door live in ``ws_main_climb``. Product ``west_super`` is y~1675
in the shaft — not West Super ``0xCDA8``.
"""

from __future__ import annotations

from collections.abc import Callable

from super_metroid.combat.enemies import list_enemies
from super_metroid.plm import session_ram, shot_block_spawns, snapshot_plms
from super_metroid.ram import FACING_LEFT, FACING_RIGHT, SuperMetroidState
from super_metroid.routes.controller_common import (
    WallJumpTiming,
    hold,
    is_morph,
    select_weapon,
    unmorph,
    walljump_once,
)
from super_metroid.takeoff import spin_jump
from super_metroid.routes.kpdr.k6.ws_main_actions import (
    SAVE_LEDGE_Y,
    THREE_SHOT_FRAMES,
    TUNNEL_CLEAR_X,
    WS_MAIN_SAVE_X,
    WS_MAIN_SHAFT_CENTER,
    WS_MAIN_STAIR_Y,
    at_ws_main_attic_door_seat,
    at_ws_main_left_platform,
    at_ws_main_pit,
    climb_action,
    grate_clear_action,
    three_shot_action,
)
from super_metroid.routes.kpdr.k6.ws_main_grate import (
    MORPH_DROP_BOMB_FRAMES,
    at_ws_main_morph_drop,
)
from super_metroid.routes.kpdr.k6.ws_main_ice import (
    SHELF_HOLE_FRAMES,
    ice_keepaway_action,
    shelf_covern_ice_action,
)
from super_metroid.routes.kpdr.k6.ws_main_phases import (
    at_ws_main_grate_seat,
    classify_ws_main_phase,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_WS_ATTIC,
    ROOM_WS_BASEMENT,
    ROOM_WS_MAIN,
    ROOM_WS_SAVE,
    ROOM_WS_SPONGE,
    ROOM_WS_WEST_SUPER,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.basic_moves import shoot_up_action
from super_metroid.routes.skills.charge_shot import session_beam_charge
from super_metroid.routes.skills.knockback import escape_knockback_spin, is_knockback

WEAPON_BEAM = 0
_THREE_SHOT_FRAMES = THREE_SHOT_FRAMES + 40
_CLIMB_BUDGET = 3600
_SIDE_TRIP_BUDGET = 400
# Past the right-lip land (x≤1210) and left of save (x≥1240). Human WJ
# (1216, 1852) p19; leftover jam (1220, 1843) p77. Peak 1827 is in-band.
SAVE_COLUMN_WJ_X = (1212, 1232)
SAVE_COLUMN_WJ_Y = (1701, 1888)
SAVE_COLUMN_LATCH_X = 1216
# Same-wall climb on the save-column LEFT face. Short into (long RIGHT
# walked onto the save ledge at x=1232 and aborted before flip). Kick LEFT.
SAVE_COLUMN_WJ = WallJumpTiming(
    into="RIGHT",
    flip="LEFT",
    into_frames=3,
    amid_frames=2,
    flip_frames=14,
    delay_into_frames=0,
)
_WJ_POSES = frozenset({19, 20, 132})
_GROUNDED = frozenset({1, 2, 3, 4, 9, 10})
_TURN = frozenset({37, 38})
_CROUCH = frozenset({39, 40})


def guard_main_shaft(session: ControllerSession, label: str) -> None:
    room = int(session.state.room_id)
    if room in (ROOM_WS_MAIN, ROOM_WS_ATTIC, ROOM_WS_WEST_SUPER, ROOM_WS_SPONGE):
        return
    if room == ROOM_WS_SAVE:
        raise TimeoutError(f"{label}: entered save 0xCE8A: {session.state}")
    if room == ROOM_WS_BASEMENT:
        raise TimeoutError(f"{label}: dropped back to Basement 0xCC6F: {session.state}")
    raise TimeoutError(f"{label}: left Main Shaft into 0x{room:04X}: {session.state}")


def knockback_main_shaft(session: ControllerSession, label: str) -> None:
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


def exit_side_room(session: ControllerSession, label: str) -> None:
    """West Super RIGHT, Sponge Bath LEFT — both rooms one change."""
    for _ in range(_SIDE_TRIP_BUDGET):
        st = session.state
        room = int(st.room_id)
        if room in (ROOM_WS_MAIN, ROOM_WS_ATTIC):
            return
        guard_main_shaft(session, label)
        if is_knockback(st):
            knockback_main_shaft(session, f"{label}_side_kb")
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


def three_shot_tunnel(session: ControllerSession, label: str) -> None:
    """Walk to the hatch column and gun-jump onto the right lip ~(1184, 1883)."""
    if int(session.state.room_id) in (ROOM_WS_ATTIC, ROOM_WS_WEST_SUPER):
        return
    if at_ws_main_grate_seat(session.state) or not at_ws_main_pit(session.state):
        return
    select_weapon(session, WEAPON_BEAM)
    shot_i = 0
    for _ in range(_THREE_SHOT_FRAMES):
        st = session.state
        guard_main_shaft(session, label)
        if int(st.room_id) in (ROOM_WS_ATTIC, ROOM_WS_WEST_SUPER):
            return
        if at_ws_main_grate_seat(st) or not at_ws_main_pit(st):
            return
        if is_knockback(st):
            knockback_main_shaft(session, f"{label}_shot_kb")
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
            int(st.movement_type),
            int(st.velocity_y),
        )
        shot_i += 1
        if names:
            hold(session, 1, *names, reason=f"{label}_3shot")
        else:
            hold(session, 1, reason=f"{label}_3shot_land")


def at_attic_climb_done(state: SuperMetroidState) -> bool:
    return (
        int(state.room_id) == ROOM_WS_ATTIC
        or at_ws_main_attic_door_seat(state)
        or int(state.samus_y) <= 160
    )


def at_ws_main_save_column_wj(state: SuperMetroidState) -> bool:
    """Air against the save-column LEFT face. Not the save door, not the lip."""
    pose = int(state.pose)
    x, y = int(state.samus_x), int(state.samus_y)
    airborne = (
        (pose not in _GROUNDED and pose not in _CROUCH)
        or abs(int(state.velocity_y)) > 1
    )
    return (
        int(state.room_id) == ROOM_WS_MAIN
        and SAVE_COLUMN_WJ_X[0] <= x < SAVE_COLUMN_WJ_X[1]
        and SAVE_COLUMN_WJ_Y[0] <= y <= SAVE_COLUMN_WJ_Y[1]
        and (airborne or pose in _WJ_POSES)
        and not is_morph(pose)
        and pose not in _CROUCH
        # Facing RIGHT = approaching the LEFT face from the shaft. Alcove
        # leftover (1235, 1851) p10 faces LEFT — do not steal that jump.
        and int(state.facing) == FACING_RIGHT
    )


def at_ws_main_save_alcove(state: SuperMetroidState) -> bool:
    """Planted on the save-door alcove ~(1235, 1851). Jump LEFT into the shaft."""
    pose = int(state.pose)
    x, y = int(state.samus_x), int(state.samus_y)
    return (
        int(state.room_id) == ROOM_WS_MAIN
        and WS_MAIN_SAVE_X - 16 <= x < WS_MAIN_SAVE_X
        and SAVE_LEDGE_Y[0] <= y <= SAVE_LEDGE_Y[1]
        and pose in _GROUNDED | _TURN
        and abs(int(state.velocity_y)) <= 1
        and not is_morph(pose)
    )


def save_alcove_jump(session: ControllerSession, label: str) -> None:
    """Cubby ceiling blocks A; still try LEFT so we do not LEFT+B into the wall."""
    st = session.state
    turning = int(st.movement_type) == 14
    if int(st.facing) != FACING_LEFT or turning:
        hold(session, 1, "LEFT", reason=f"{label}_alcove_face")
        return
    hold(session, 1, *spin_jump("LEFT"), reason=f"{label}_alcove_jump")


def _save_column_stop(
    state: SuperMetroidState, done: Callable[[SuperMetroidState], bool]
) -> bool:
    return (
        done(state)
        or int(state.room_id) != ROOM_WS_MAIN
        or int(state.samus_x) >= WS_MAIN_SAVE_X
        or int(state.samus_y) > WS_MAIN_STAIR_Y
    )


def save_column_walljump(
    session: ControllerSession,
    label: str,
    done: Callable[[SuperMetroidState], bool],
) -> None:
    """Seek pose 19 at x~1216, then one walljump_once. Loop repeats for 2–3."""

    def _stop(state: SuperMetroidState) -> bool:
        return _save_column_stop(state, done)

    pose = int(session.state.pose)
    x = int(session.state.samus_x)
    if pose not in _WJ_POSES and x < SAVE_COLUMN_LATCH_X:
        for _ in range(8):
            st = session.state
            if _stop(st) or int(st.pose) in _WJ_POSES:
                break
            if int(st.samus_x) >= SAVE_COLUMN_LATCH_X:
                break
            hold(session, 1, "RIGHT", reason=f"{label}_wj_seek")
    if _stop(session.state):
        return
    walljump_once(
        session,
        SAVE_COLUMN_WJ,
        reason=f"{label}_save_wj",
        stop_when=_stop,
    )


def _hold_names(
    session: ControllerSession,
    names: tuple[str, ...],
    reason: str,
    wait_reason: str,
) -> None:
    if names:
        hold(session, 1, *names, reason=reason)
    else:
        hold(session, 1, reason=wait_reason)


def _shelf_hole_buttons(st: SuperMetroidState, shelf_open: int) -> tuple[str, ...]:
    if int(st.pose) in (39, 40):
        return ("UP",)
    if int(st.facing) != FACING_RIGHT:
        return ("RIGHT",)
    if shelf_open % 10 < 4:
        return ("X",)
    return ()


def _dispatch_pit_shot(session: ControllerSession, st: SuperMetroidState, label: str) -> None:
    names = three_shot_action(
        int(st.samus_x),
        int(st.samus_y),
        int(st.pose),
        int(st.facing),
        int(session.frame),
        session_beam_charge(session),
        int(st.movement_type),
        int(st.velocity_y),
    )
    _hold_names(session, names, f"{label}_pit", f"{label}_pit_fire")


def _update_lip_hit(
    session: ControllerSession,
    prev_plms: tuple[dict[str, int], ...],
    lip_hit: bool,
) -> tuple[bool, tuple[dict[str, int], ...]]:
    """Latch True on a new 0xD080-family PLM. Empty prev is a seed, not a hit."""
    ram = session_ram(session)
    if ram is None:
        return lip_hit, prev_plms
    cur = snapshot_plms(ram)
    if shot_block_spawns(prev_plms, cur):
        return True, cur
    return lip_hit, cur


def _dispatch_west_super_band(
    session: ControllerSession,
    st: SuperMetroidState,
    label: str,
    shelf_open: int,
    lip_hit: bool,
) -> tuple[int, bool]:
    """Shelf-hole + Ice, then grate-band takeoff. Returns shelf / lip-hit."""
    x = int(st.samus_x)
    y = int(st.samus_y)
    pose = int(st.pose)
    if at_ws_main_left_platform(x, y, pose, int(st.velocity_y)):
        if shelf_open < SHELF_HOLE_FRAMES:
            shelf_open += 1
            names = _shelf_hole_buttons(st, shelf_open)
            reason = f"{label}_shelf_stand" if names == ("UP",) else (
                f"{label}_shelf_face" if names == ("RIGHT",) else (
                    f"{label}_shelf_hole" if names else f"{label}_shelf_hole_tap"
                )
            )
            _hold_names(session, names, reason, reason)
            return shelf_open, lip_hit
        keepaway = shelf_covern_ice_action(
            x,
            y,
            int(st.facing),
            list_enemies(session),
            movement_type=int(st.movement_type),
            charge=session_beam_charge(session),
            velocity_y=int(st.velocity_y),
        )
        if keepaway:
            hold(session, 1, *keepaway, reason=f"{label}_shelf_ice")
            return shelf_open, lip_hit
    grate = grate_clear_action(
        x,
        y,
        pose,
        int(st.facing),
        int(session.frame),
        session_beam_charge(session),
        int(st.velocity_y),
        int(st.movement_type),
        lip_hit,
    )
    if grate is not None:
        reason = f"{label}_lip_up" if grate == shoot_up_action() else (
            f"{label}_lip_jump" if "A" in grate else (
                f"{label}_drop_morph" if "DOWN" in grate else f"{label}_grate"
            )
        )
        _hold_names(session, grate, reason, f"{label}_grate_wait")
        return shelf_open, lip_hit
    names = climb_action(
        x,
        y,
        pose,
        int(st.facing),
        int(st.velocity_y),
        int(st.movement_type),
        int(session.frame),
        lip_hit,
        session_beam_charge(session),
    )
    _hold_names(session, names, f"{label}_climb", f"{label}_wait")
    return shelf_open, lip_hit


def _dispatch_mid_shaft(session: ControllerSession, st: SuperMetroidState, label: str) -> None:
    keepaway = ice_keepaway_action(
        int(st.samus_x),
        int(st.samus_y),
        int(st.facing),
        list_enemies(session),
        movement_type=int(st.movement_type),
        charge=session_beam_charge(session),
        velocity_y=int(st.velocity_y),
    )
    if keepaway is not None:
        _hold_names(session, keepaway, f"{label}_ice", f"{label}_ice_wait")
        return
    names = climb_action(
        int(st.samus_x),
        int(st.samus_y),
        int(st.pose),
        int(st.facing),
        int(st.velocity_y),
        int(st.movement_type),
        int(session.frame),
    )
    _hold_names(session, names, f"{label}_climb", f"{label}_wait")


def climb_until(
    session: ControllerSession,
    label: str,
    done: Callable[[SuperMetroidState], bool],
) -> None:
    """Spin-hop the shaft until ``done(state)``. Ice nearby Atomics."""
    if int(session.state.room_id) == ROOM_WS_ATTIC or done(session.state):
        return
    if is_morph(int(session.state.pose)):
        unmorph(session)
    select_weapon(session, WEAPON_BEAM)
    shelf_open = 0
    lip_hit = False
    morph_bombs = 0
    prev_plms: tuple[dict[str, int], ...] = ()
    for _ in range(_CLIMB_BUDGET):
        st = session.state
        guard_main_shaft(session, label)
        if int(st.room_id) == ROOM_WS_ATTIC or done(st):
            return
        if int(st.room_id) in (ROOM_WS_WEST_SUPER, ROOM_WS_SPONGE):
            exit_side_room(session, label)
            continue
        if is_knockback(st):
            knockback_main_shaft(session, f"{label}_climb_kb")
            continue
        if is_morph(int(st.pose)):
            mx, my = int(st.samus_x), int(st.samus_y)
            if (
                lip_hit
                and at_ws_main_morph_drop(mx, my)
                and morph_bombs < MORPH_DROP_BOMB_FRAMES
            ):
                morph_bombs += 1
                hold(session, 1, "X", reason=f"{label}_drop_bomb")
                continue
            if mx > TUNNEL_CLEAR_X and my < WS_MAIN_STAIR_Y:
                hold(session, 1, "LEFT", reason=f"{label}_roll")
            elif my < WS_MAIN_STAIR_Y:
                # UP only — generic unmorph A-settle idles over the gap.
                hold(session, 1, "UP", reason=f"{label}_unmorph")
            else:
                unmorph(session)
            continue
        if at_ws_main_save_alcove(st):
            save_alcove_jump(session, label)
            continue
        if at_ws_main_save_column_wj(st):
            save_column_walljump(session, label, done)
            continue
        phase = classify_ws_main_phase(st)
        y = int(st.samus_y)
        # Floor fall-recovery is pit_shot. y>=1760 leftover (shelf / lip /
        # grate) is the west_super takeoff band — not the y~1675 hop.
        if (
            phase == "pit_shot"
            and at_ws_main_pit(st)
            and not at_ws_main_grate_seat(st)
            and y >= WS_MAIN_STAIR_Y
        ):
            _dispatch_pit_shot(session, st, label)
            continue
        if phase == "grate_seat" or y >= 1760:
            lip_hit, prev_plms = _update_lip_hit(session, prev_plms, lip_hit)
            shelf_open, lip_hit = _dispatch_west_super_band(
                session, st, label, shelf_open, lip_hit
            )
            continue
        _dispatch_mid_shaft(session, st, label)
    if int(session.state.room_id) != ROOM_WS_ATTIC and not done(session.state):
        raise TimeoutError(f"{label}: did not reach phase seat: {session.state}")


__all__ = [
    "SAVE_COLUMN_LATCH_X",
    "SAVE_COLUMN_WJ",
    "SAVE_COLUMN_WJ_X",
    "SAVE_COLUMN_WJ_Y",
    "WEAPON_BEAM",
    "at_attic_climb_done",
    "at_ws_main_save_alcove",
    "at_ws_main_save_column_wj",
    "climb_until",
    "exit_side_room",
    "guard_main_shaft",
    "knockback_main_shaft",
    "save_alcove_jump",
    "save_column_walljump",
    "three_shot_tunnel",
]
