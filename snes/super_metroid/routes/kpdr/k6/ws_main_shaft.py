"""Powered Main Shaft climb loop (rr-kw8t hop 2).

Pit two-hop, Ice keepaway, save-column wall-jump. Dispatch is
``classify_region`` — stairs leftover is PIT, not the Wave-hole shelf.
Product ``west_super`` is y~1675 in the shaft — not West Super ``0xCDA8``.
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
from super_metroid.routes.kpdr.k6.ws_main_actions import (
    at_take02_departure,
    climb_action,
    three_shot_action,
)
from super_metroid.routes.kpdr.k6.ws_main_geometry import (
    MORPH_DROP_BOMB_FRAMES,
    SAVE_COLUMN_LATCH_X,
    THREE_SHOT_FRAMES,
    TUNNEL_CLEAR_X,
    WJ_POSES,
    WS_MAIN_SAVE_X,
    WS_MAIN_SHAFT_CENTER,
    WS_MAIN_STAIR_Y,
    ShaftRegion,
    at_ws_main_attic_door_seat,
    at_ws_main_grate_seat,
    at_ws_main_morph_drop,
    at_ws_main_pit,
    at_ws_main_save_alcove,
    at_ws_main_save_column_wj,
    classify_region,
)
from super_metroid.routes.kpdr.k6.ws_main_ice import (
    SHELF_HOLE_FRAMES,
    ice_keepaway_action,
    shelf_covern_ice_action,
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
from super_metroid.takeoff import spin_jump

WEAPON_BEAM = 0
_THREE_SHOT_FRAMES = THREE_SHOT_FRAMES + 40
_CLIMB_BUDGET = 3600
_SIDE_TRIP_BUDGET = 400
_TAKE02_DROP_HANDOFF = (
    (5, ("LEFT",)),
    (9, ("LEFT", "A")),
    (6, ("LEFT",)),
    (3, ("LEFT", "X")),
    (4, ("X",)),
    (5, ()),
    (6, ("X",)),
    (9, ()),
    (7, ("A",)),
    (1, ("DOWN", "A")),
    (9, ("DOWN", "A", "X")),
    (1, ("DOWN", "A")),
    (4, ("DOWN",)),
    (2, ("X",)),
    (4, ("LEFT", "X")),
    (3, ("LEFT",)),
    (3, ("UP", "LEFT")),
    (1, ("LEFT",)),
    (11, ()),
    (1, ("DOWN",)),
)
_TAKE02_TUNNEL_HANDOFF = (
    (11, ("UP",)),
    (6, ("UP", "X")),
    (7, ("UP",)),
    (5, ("UP", "X")),
    (15, ("UP",)),
    (4, ("X",)),
    (4, ()),
    (11, ("UP",)),
    (6, ("UP", "X")),
    (5, ("UP",)),
    (6, ("UP", "X")),
    (1, ("UP",)),
    (7, ()),
    (4, ("A",)),
    (20, ("RIGHT", "A")),
)
SAVE_COLUMN_WJ = WallJumpTiming(
    into="RIGHT",
    flip="LEFT",
    into_frames=3,
    amid_frames=2,
    flip_frames=14,
    delay_into_frames=0,
)


def _rle_action(
    recipe: tuple[tuple[int, tuple[str, ...]], ...], frame: int
) -> tuple[str, ...] | None:
    cursor = int(frame)
    for count, names in recipe:
        if cursor < count:
            return names
        cursor -= count
    return None


def _take02_drop_handoff_action(frame: int) -> tuple[str, ...] | None:
    """Tape-locked x=1189 contact clear through the first DOWN-morph."""
    return _rle_action(_TAKE02_DROP_HANDOFF, frame)


def _take02_tunnel_handoff_action(frame: int) -> tuple[str, ...] | None:
    """Tape-locked x=1093 unmorph, ceiling shots, and west-super jump."""
    return _rle_action(_TAKE02_TUNNEL_HANDOFF, frame)


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
    """Take02 two-hop onto the fire slope: short A at 1166, committed at 1156."""
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
    if pose not in WJ_POSES and x < SAVE_COLUMN_LATCH_X:
        for _ in range(8):
            st = session.state
            if _stop(st) or int(st.pose) in WJ_POSES:
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


def _climb_reason(
    label: str, region: ShaftRegion, names: tuple[str, ...]
) -> str:
    if names == shoot_up_action():
        return f"{label}_lip_up"
    if "DOWN" in names:
        return f"{label}_drop_morph"
    if region is ShaftRegion.GRATE_SEAT and "A" in names:
        return f"{label}_lip_jump"
    if region is ShaftRegion.PIT:
        return f"{label}_pit" if names else f"{label}_pit_fire"
    if names:
        return f"{label}_climb"
    return f"{label}_wait"


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
    drop_handoff_frame: int | None = None
    tunnel_handoff_frame: int | None = None
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
        if drop_handoff_frame is not None:
            drop_names = _take02_drop_handoff_action(drop_handoff_frame)
            if drop_names is not None:
                drop_handoff_frame += 1
                _hold_names(
                    session,
                    drop_names,
                    f"{label}_drop_handoff",
                    f"{label}_drop_handoff_wait",
                )
                continue
        if tunnel_handoff_frame is not None:
            tunnel_names = _take02_tunnel_handoff_action(tunnel_handoff_frame)
            if tunnel_names is not None:
                tunnel_handoff_frame += 1
                _hold_names(
                    session,
                    tunnel_names,
                    f"{label}_tunnel_handoff",
                    f"{label}_tunnel_handoff_wait",
                )
                continue
        if is_knockback(st):
            if lip_hit and at_ws_main_morph_drop(
                int(st.samus_x), int(st.samus_y)
            ):
                drop_handoff_frame = 1
                _hold_names(
                    session,
                    _TAKE02_DROP_HANDOFF[0][1],
                    f"{label}_drop_handoff",
                    f"{label}_drop_handoff_wait",
                )
                continue
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
            # The take02 tape plants against the left tunnel wall at x=1093;
            # x=1088 is geometry clearance, not a reachable morph-ball center.
            if mx > TUNNEL_CLEAR_X + 5 and my < WS_MAIN_STAIR_Y:
                hold(session, 1, "LEFT", reason=f"{label}_roll")
            elif my < WS_MAIN_STAIR_Y:
                tunnel_handoff_frame = 1
                _hold_names(
                    session,
                    _TAKE02_TUNNEL_HANDOFF[0][1],
                    f"{label}_tunnel_handoff",
                    f"{label}_tunnel_handoff_wait",
                )
            else:
                unmorph(session)
            continue
        lip_hit, prev_plms = _update_lip_hit(session, prev_plms, lip_hit)
        x, y = int(st.samus_x), int(st.samus_y)
        pose = int(st.pose)
        region = classify_region(st, lip_hit=lip_hit)
        if at_take02_departure(x, y, int(st.velocity_y)):
            region = ShaftRegion.GRATE_SEAT
        take02_active = lip_hit
        if (
            take02_active
            and drop_handoff_frame is None
            and at_ws_main_morph_drop(x, y, pose, int(st.velocity_y))
        ):
            hold(session, 1, "B", "LEFT", reason=f"{label}_drop_plant")
            continue
        if region is not ShaftRegion.GRATE_SEAT and not take02_active:
            if at_ws_main_save_alcove(st):
                save_alcove_jump(session, label)
                continue
            if at_ws_main_save_column_wj(st):
                save_column_walljump(session, label, done)
                continue
        if region is ShaftRegion.SHELF:
            if shelf_open < SHELF_HOLE_FRAMES:
                shelf_open += 1
                names = _shelf_hole_buttons(st, shelf_open)
                reason = (
                    f"{label}_shelf_stand"
                    if names == ("UP",)
                    else (
                        f"{label}_shelf_face"
                        if names == ("RIGHT",)
                        else (
                            f"{label}_shelf_hole"
                            if names
                            else f"{label}_shelf_hole_tap"
                        )
                    )
                )
                _hold_names(session, names, reason, reason)
                continue
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
                continue
        elif region is ShaftRegion.SHAFT and not take02_active:
            keepaway = ice_keepaway_action(
                x,
                y,
                int(st.facing),
                list_enemies(session),
                movement_type=int(st.movement_type),
                charge=session_beam_charge(session),
                velocity_y=int(st.velocity_y),
            )
            if keepaway is not None:
                _hold_names(session, keepaway, f"{label}_ice", f"{label}_ice_wait")
                continue
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
            region=region,
            take02_active=take02_active,
        )
        _hold_names(
            session,
            names,
            _climb_reason(label, region, names),
            f"{label}_wait",
        )
    if int(session.state.room_id) != ROOM_WS_ATTIC and not done(session.state):
        raise TimeoutError(f"{label}: did not reach phase seat: {session.state}")


__all__ = [
    "SAVE_COLUMN_LATCH_X",
    "SAVE_COLUMN_WJ",
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
