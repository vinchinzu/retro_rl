"""Red Tower → Hellway pure return (K5 hop 12).

Source: ``post_ice_bat_to_red_pure`` ~(206–216, 2443) after Bat→Red dual
**718f**. Climb reverse of ``play_red_tower_to_bat`` descent bands, then RIGHT
into top-right Hellway door ``0xA2F7``.

Hybrid pure (Hi-Jump held on K5 stack)::

  1. Accept Red bottom residual; clear Bat door lip (never RIGHT into 0xA3DD)
  2. Right-wall WJ chain lower shaft → pocket / mid-right wall y≲2090
  3. Right-wall re-catch cycles past pocket ceiling → y≲1960 open shaft
  4. Double-bomb IBJ mid climb through tunnel band y≲1880 → temporary floor
  5. Bomb-floor reverse + upper WJ zigzag → top-right door band
  6. RIGHT into Hellway ordinary settle

Tape: ``tasks/speed_to_wave_ice_moat_human.json`` Phase C hop 29→30.
Human thrash RLE desyncs from pure pin (enemy state); prefer clean climb.
Right pocket ~(225,2091) is a **dead-end under ceiling** (A peaks ~y1964) —
do not treat y≤2090 alone as lower-done.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    is_morph,
    play_run_shoot_exit,
    require_room,
    select_weapon,
    settle_hold,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.k5.geometry import (
    RED_BOTTOM_Y,
    RED_CLIMB_FRAMES,
    RED_FLOOR_Y,
    RED_LOWER_LIP_Y,
    RED_TO_HELLWAY_EXIT_HOLD,
    RED_TO_HELLWAY_EXIT_RUN,
    RED_TO_HELLWAY_EXIT_SETTLE,
    RED_TO_HELLWAY_EXIT_SHOOT,
    RED_TO_HELLWAY_EXIT_SPIN,
    RED_TOP_DOOR_X,
    RED_TOP_DOOR_Y,
    RED_TUNNEL_Y,
    RED_ZIG_X_MAX,
    RED_ZIG_X_MIN,
)
from super_metroid.routes.kpdr.rooms import ROOM_HELLWAY, ROOM_RED_TOWER
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import escape_knockback_spin, is_knockback

_MORPH = frozenset({0x1D, 0x1E, 0x1F, 0x20, 29, 30, 39, 40, 41, 42, 81, 82})
_STAND = frozenset({1, 2, 9, 10, 12, 27, 28, 137, 138})

# Double-bomb IBJ cadence (probe-validated mid-shaft; peaks through tunnel).
_IBJ_WAIT1 = 18
_IBJ_WAIT2 = 30
# Right-wall re-catch past pocket ceiling (stable ~y1942 from pocket pin).
_RWJ_INTO = 2
_RWJ_WJ = 10
_RWJ_OUT = 18
_RWJ_BACK = 36


def _in_red(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_RED_TOWER


def _in_hellway(state: SuperMetroidState) -> bool:
    return int(state.room_id) == ROOM_HELLWAY


def _unmorph(session: ControllerSession, label: str) -> None:
    for _ in range(12):
        st = session.state
        if not (is_morph(st.pose) or int(st.pose) in _MORPH):
            return
        hold(session, 1, "UP", reason=f"{label}_unmorph")
    unmorph(session)


def _bat_safe(session: ControllerSession, label: str) -> None:
    """Stay left of Bat door on the bottom floor / low spin."""
    st = session.state
    if not _in_red(st):
        return
    # Bat right door is lethal below y~2360 at x≳215.
    if int(st.samus_y) >= RED_BOTTOM_Y - 100 and int(st.samus_x) >= 210:
        hold(session, 1, "LEFT", reason=f"{label}_bat")


def _bat_abort(session: ControllerSession, label: str) -> bool:
    """True if we just corrected a Bat-door drift (caller should break RIGHT hold)."""
    st = session.state
    if not _in_red(st):
        return True
    if int(st.samus_y) >= RED_BOTTOM_Y - 90 and int(st.samus_x) >= 220:
        hold(session, 4, "LEFT", reason=f"{label}_bat_abort")
        return True
    return False


def _kb(session: ControllerSession, label: str, prefer: str = "LEFT") -> None:
    if is_knockback(session.state):
        escape_knockback_spin(
            session,
            prefer_dir=prefer,
            run_frames=2,
            spin_frames=8,
            label=f"{label}_kb",
            ensure_beam=True,
            break_on_motion_clear=True,
        )


def _right_wall_wj_cycle(
    session: ControllerSession, label: str, *, from_floor: bool
) -> None:
    """One right-wall wall-jump cycle (human-validated lower-shaft primitive)."""
    if not _in_red(session.state):
        return
    _unmorph(session, label)
    _kb(session, label)

    if from_floor or int(session.state.samus_y) >= RED_BOTTOM_Y - 40:
        # Runway mid-left then jump into right wall — avoid Bat door x≥230 low.
        for _ in range(70):
            st = session.state
            if not _in_red(st):
                return
            _bat_safe(session, label)
            if int(st.pose) in _MORPH:
                hold(session, 1, "UP", reason=f"{label}_u")
                continue
            if int(st.samus_y) < RED_BOTTOM_Y - 40:
                break
            if 145 <= int(st.samus_x) <= 155 and int(st.velocity_y) == 0:
                break
            hold(
                session,
                1,
                "LEFT" if int(st.samus_x) > 155 else "RIGHT",
                "B",
                reason=f"{label}_runway",
            )
        settle_hold(session, 3, reason=f"{label}_runway_s")
        hold(session, 3, "RIGHT", "B", reason=f"{label}_run")
        for f in range(75):
            st = hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_jwall")
            if not _in_red(st):
                return
            if _bat_abort(session, label):
                return
            if int(st.samus_x) >= 218 and int(st.velocity_y) == 0 and f > 18:
                break
    else:
        for _ in range(35):
            st = session.state
            if not _in_red(st) or int(st.samus_y) >= RED_BOTTOM_Y - 20:
                return
            if int(st.pose) in _MORPH:
                hold(session, 1, "UP", reason=f"{label}_u")
                continue
            if _bat_abort(session, label):
                return
            if int(st.samus_x) >= 216:
                break
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_re")

    hold(session, 2, reason=f"{label}_contact")
    hold(session, 2, "LEFT", reason=f"{label}_face")
    for _ in range(18):
        st = hold(session, 1, "LEFT", "A", reason=f"{label}_wj")
        if not _in_red(st):
            return
    for f in range(36):
        st = hold(session, 1, "RIGHT", "A", reason=f"{label}_back")
        if not _in_red(st):
            return
        if int(st.samus_x) >= 217 and int(st.velocity_y) <= 1 and f > 8:
            hold(session, 1, "LEFT", reason=f"{label}_face2")
            for _ in range(15):
                hold(session, 1, "LEFT", "A", reason=f"{label}_wj2")
            for _ in range(22):
                hold(session, 1, "RIGHT", "A", reason=f"{label}_back2")
            break


def _right_wall_recatch(
    session: ControllerSession, label: str
) -> SuperMetroidState:
    """WJ off right wall, short outward LEFT, re-catch right (past pocket ceiling).

    Probe-stable from pocket ~(225,2091) to open shaft y≈1942.
    """
    if not _in_red(session.state):
        return session.state
    _unmorph(session, label)
    _kb(session, label)
    # Approach right wall if grounded mid-shaft.
    st = session.state
    if int(st.velocity_y) == 0 and int(st.samus_x) < 200 and int(st.samus_y) > 2000:
        hold(session, 8, "RIGHT", "B", reason=f"{label}_ap")
        hold(session, 16, "RIGHT", "B", "A", reason=f"{label}_jap")
    hold(session, _RWJ_INTO, "RIGHT", reason=f"{label}_into")
    for _ in range(_RWJ_WJ):
        st = hold(session, 1, "RIGHT", "A", reason=f"{label}_wj")
        if not _in_red(st):
            return st
    for _ in range(_RWJ_OUT):
        st = hold(session, 1, "LEFT", "A", reason=f"{label}_out")
        if not _in_red(st):
            return st
        if int(st.samus_x) <= 175:
            break
    for f in range(_RWJ_BACK):
        st = hold(session, 1, "RIGHT", "A", reason=f"{label}_back")
        if not _in_red(st):
            return st
        if int(st.samus_x) >= 218 and int(st.velocity_y) <= 1 and f > 6:
            break
    return session.state


def _ibj_double(session: ControllerSession, label: str) -> SuperMetroidState:
    """One double-bomb IBJ cycle (morph X / wait / X / wait)."""
    if not _in_red(session.state):
        return session.state
    _kb(session, label)
    if not is_morph(session.state.pose) and int(session.state.pose) not in _MORPH:
        ensure_morph(session)
    st = session.state
    x = int(st.samus_x)
    y = int(st.samus_y)
    # Low fall: bias LEFT hard so we never drift into Bat door.
    if y >= RED_BOTTOM_Y - 120:
        hold(session, 4, "LEFT", reason=f"{label}_bat_bias")
    elif x > 185:
        hold(session, 2, "LEFT", reason=f"{label}_cL")
    elif x < 100:
        hold(session, 2, "RIGHT", reason=f"{label}_cR")
    hold(session, 2, "X", reason=f"{label}_b1")
    for _ in range(_IBJ_WAIT1):
        st = hold(session, 1, reason=f"{label}_w1")
        if not _in_red(st) or int(st.samus_y) <= RED_TUNNEL_Y:
            return st
        if int(st.samus_y) >= RED_BOTTOM_Y - 100 and int(st.samus_x) >= 200:
            hold(session, 1, "LEFT", reason=f"{label}_bat_w1")
    hold(session, 2, "X", reason=f"{label}_b2")
    for _ in range(_IBJ_WAIT2):
        st = hold(session, 1, reason=f"{label}_w2")
        if not _in_red(st) or int(st.samus_y) <= RED_TUNNEL_Y:
            return st
        if int(st.samus_y) >= RED_BOTTOM_Y - 100 and int(st.samus_x) >= 200:
            hold(session, 1, "LEFT", reason=f"{label}_bat_w2")
    return session.state


def _climb_lower(session: ControllerSession, label: str) -> SuperMetroidState:
    """Bottom → past pocket ceiling into open shaft y≲1960.

    Do **not** accept right-pocket ~(225,2091) as done. Chain right-wall WJ
    to natural pocket land, then re-catch until y<=1960 (or tunnel).
    """
    target_y = 1960
    # Re-entry from mid fall: clear Bat door before any RIGHT runway.
    st0 = session.state
    if _in_red(st0) and int(st0.samus_y) >= RED_BOTTOM_Y - 80:
        for _ in range(60):
            st0 = session.state
            if not _in_red(st0):
                break
            if int(st0.samus_x) <= 165 and int(st0.velocity_y) == 0:
                break
            hold(session, 1, "LEFT", "B", reason=f"{label}_reentry_bat")
        settle_hold(session, 4, reason=f"{label}_reentry_s")

    for attempt in range(48):
        st = session.state
        if _in_hellway(st):
            return st
        if not _in_red(st):
            raise TimeoutError(
                f"{label}: left Red lower room=0x{int(st.room_id):04X} "
                f"xy=({st.samus_x},{st.samus_y})"
            )
        y = int(st.samus_y)
        x = int(st.samus_x)
        if y <= RED_TUNNEL_Y:
            return st
        if y <= target_y and x <= 230:
            return st

        _kb(session, f"{label}_a{attempt}")
        _bat_safe(session, f"{label}_a{attempt}")

        # Floor / deep lower: classic right-wall WJ toward pocket.
        if y >= RED_BOTTOM_Y - 100:
            _right_wall_wj_cycle(
                session,
                f"{label}_a{attempt}",
                from_floor=True,
            )
            continue

        # Between floor and pocket lip: keep right-wall WJ (not recatch yet).
        if y > RED_LOWER_LIP_Y + 30:
            _right_wall_wj_cycle(
                session,
                f"{label}_midwj{attempt}",
                from_floor=False,
            )
            continue

        # Pocket band y~2050–2120: wait for grounded right seat, then recatch.
        if RED_LOWER_LIP_Y - 50 <= y <= RED_LOWER_LIP_Y + 50:
            # Land / settle on pocket before recatch (mid-air recatch free-falls).
            if int(st.velocity_y) != 0 or int(st.pose) not in _STAND:
                for _ in range(40):
                    st2 = hold(session, 1, reason=f"{label}_pocket_land")
                    if not _in_red(st2):
                        return st2
                    if int(st2.velocity_y) == 0 and int(st2.pose) in _STAND:
                        break
                    if int(st2.samus_y) >= RED_BOTTOM_Y - 40:
                        break
                st = session.state
                y = int(st.samus_y)
                x = int(st.samus_x)
            # If we fell, floor WJ next loop.
            if y >= RED_BOTTOM_Y - 80:
                continue
            # Grounded on/near right pocket → recatch past ceiling.
            if x >= 200 and y <= RED_LOWER_LIP_Y + 40:
                _right_wall_recatch(session, f"{label}_rc{attempt}")
                continue
            # On pocket height but left of wall: approach right then recatch.
            if y <= RED_LOWER_LIP_Y + 40:
                hold(session, 10, "RIGHT", "B", reason=f"{label}_pocket_ap")
                _right_wall_recatch(session, f"{label}_rcap{attempt}")
                continue

        # Above pocket / open mid: recatch or re-approach right wall.
        if x >= 200 or y <= target_y + 40:
            _right_wall_recatch(session, f"{label}_hi{attempt}")
            continue

        hold(session, 12, "RIGHT", "B", "A", reason=f"{label}_mid_ap")
        _right_wall_recatch(session, f"{label}_rcm{attempt}")

    st = session.state
    if int(st.samus_y) > target_y + 80:
        raise TimeoutError(
            f"{label}: lower climb stalled y={st.samus_y} x={st.samus_x} p={st.pose}"
        )
    return st


def _climb_mid(session: ControllerSession, label: str) -> SuperMetroidState:
    """Open shaft y~1960 → tunnel + temporary floor to y<=RED_FLOOR_Y.

    Double-bomb IBJ through tunnel band, then reverse bomb-floor of
    ``play_red_tower_to_bat``.
    """
    for attempt in range(80):
        st = session.state
        if _in_hellway(st):
            return st
        if not _in_red(st):
            raise TimeoutError(
                f"{label}: left Red mid room=0x{int(st.room_id):04X} "
                f"xy=({st.samus_x},{st.samus_y})"
            )
        y = int(st.samus_y)
        x = int(st.samus_x)
        if y <= RED_FLOOR_Y - 20:
            return st

        _kb(session, f"{label}_a{attempt}")

        if y >= RED_BOTTOM_Y - 40:
            _climb_lower(session, f"{label}_relower")
            continue

        # Temporary floor / just below: bomb reverse + hop up.
        if RED_FLOOR_Y - 30 <= y <= RED_FLOOR_Y + 220:
            if int(st.pose) in _MORPH or is_morph(st.pose):
                hold(session, 2, "X", reason=f"{label}_floor_bomb")
                hold(session, 10, "LEFT", reason=f"{label}_floor_ret")
                hold(session, 30, reason=f"{label}_floor_wait")
                _unmorph(session, label)
                hold(session, 36, "A", reason=f"{label}_floor_up")
                hold(session, 16, "LEFT", "A", reason=f"{label}_floor_left")
                continue
            ensure_morph(session)
            for _ in range(40):
                st2 = session.state
                if not _in_red(st2):
                    return st2
                if 145 <= int(st2.samus_x) <= 160:
                    break
                hold(
                    session,
                    1,
                    "LEFT" if int(st2.samus_x) > 155 else "RIGHT",
                    reason=f"{label}_floor_pos",
                )
            continue

        # Tunnel / high mid: prefer standing hops if grounded, else IBJ.
        if y <= RED_TUNNEL_Y + 40:
            if int(st.velocity_y) == 0 and int(st.pose) in _STAND:
                _unmorph(session, label)
                hold(session, 32, "A", reason=f"{label}_tun_v")
                d = "LEFT" if x > 150 else "RIGHT"
                hold(session, 14, d, "A", reason=f"{label}_tun_steer")
                settle_hold(session, 12, reason=f"{label}_tun_land")
                continue
            # Light IBJ to gain the last tiles.
            _ibj_double(session, f"{label}_tun_ibj{attempt}")
            continue

        # Below tunnel: prefer right-wall recatch if on/near right wall high,
        # else double-bomb IBJ in open shaft (x≲200).
        if x >= 210 and y <= RED_LOWER_LIP_Y + 80:
            _right_wall_recatch(session, f"{label}_wall_rc{attempt}")
            # Nudge left into open shaft for IBJ (keep y if possible).
            hold(session, 4, "LEFT", reason=f"{label}_off")
            if int(session.state.samus_x) >= 200:
                hold(session, 6, "LEFT", "A", reason=f"{label}_nudge")

        _ibj_double(session, f"{label}_ibj{attempt}")

        # If peaking into tunnel band, unmorph to try a seat/hop.
        st = session.state
        if _in_red(st) and int(st.samus_y) <= RED_TUNNEL_Y + 50:
            _unmorph(session, label)
            hold(session, 6, "A", reason=f"{label}_peak_catch")

        # Bail mid early with a clean pin if we keep falling to floor.
        if attempt > 20 and int(session.state.samus_y) >= RED_BOTTOM_Y - 40:
            # One re-lower attempt then report stall at best effort.
            try:
                _climb_lower(session, f"{label}_relower2")
            except TimeoutError:
                pass
            st = session.state
            if int(st.samus_y) > RED_FLOOR_Y + 80:
                raise TimeoutError(
                    f"{label}: mid climb stalled after re-lower "
                    f"y={st.samus_y} x={st.samus_x} p={st.pose}"
                )

    st = session.state
    if int(st.samus_y) > RED_FLOOR_Y + 120:
        raise TimeoutError(
            f"{label}: mid climb stalled y={st.samus_y} x={st.samus_x} p={st.pose}"
        )
    return st


def _climb_upper(session: ControllerSession, label: str) -> SuperMetroidState:
    """Above temporary floor → top door band (reverse upper zigzag)."""
    direction = "RIGHT"
    for frame in range(RED_CLIMB_FRAMES):
        st = session.state
        if _in_hellway(st):
            return st
        if not _in_red(st):
            raise TimeoutError(
                f"{label}: left Red upper room=0x{int(st.room_id):04X} "
                f"xy=({st.samus_x},{st.samus_y})"
            )
        y = int(st.samus_y)
        x = int(st.samus_x)
        if y <= RED_TOP_DOOR_Y + 30 and int(st.velocity_y) == 0:
            return st
        if y <= RED_TOP_DOOR_Y:
            return st

        _kb(session, label, prefer=direction)

        if int(st.pose) in _MORPH or is_morph(st.pose):
            hold(session, 1, "UP", reason=f"{label}_u")
            continue

        if y >= RED_BOTTOM_Y - 80:
            _climb_lower(session, f"{label}_relower")
            _climb_mid(session, f"{label}_remid")
            continue

        if y >= RED_FLOOR_Y + 40:
            _climb_mid(session, f"{label}_remid")
            continue

        if x >= RED_ZIG_X_MAX:
            direction = "LEFT"
        elif x <= RED_ZIG_X_MIN:
            direction = "RIGHT"

        if x >= 214:
            hold(session, 1, "LEFT", reason=f"{label}_face")
            hold(session, 12, "LEFT", "A", reason=f"{label}_wj")
            hold(session, 10, "RIGHT", "A", reason=f"{label}_back")
            continue
        if x <= 48:
            hold(session, 1, "RIGHT", reason=f"{label}_face")
            hold(session, 12, "RIGHT", "A", reason=f"{label}_wj")
            hold(session, 10, "LEFT", "A", reason=f"{label}_back")
            continue

        phase = frame % 28
        if phase < 14:
            hold(session, 1, direction, "B", "A", reason=f"{label}_spin")
        elif phase < 22:
            hold(session, 1, direction, "B", reason=f"{label}_run")
        else:
            hold(session, 1, direction, reason=f"{label}_walk")

    return session.state


def play_red_to_hellway(session: ControllerSession) -> SuperMetroidState:
    """Red Tower bottom → ordinary Hellway left (K5 hop 12)."""
    label = "red_to_hellway"
    require_room(session, ROOM_RED_TOWER, label)
    unmorph(session)
    select_weapon(session, 0)
    hold(session, 6, reason=f"{label}_entry_glide")

    # Clear Bat door lip before any RIGHT spin.
    for _ in range(100):
        st = session.state
        if not _in_red(st):
            break
        if int(st.samus_x) <= 165 and int(st.velocity_y) == 0:
            break
        hold(session, 1, "LEFT", "B", reason=f"{label}_clear_bat")
    settle_hold(session, 6, reason=f"{label}_bottom_settle")

    _climb_lower(session, f"{label}_lower")
    if _in_hellway(session.state):
        return wait_ordinary_room(
            session, ROOM_HELLWAY, settle_frames=RED_TO_HELLWAY_EXIT_SETTLE, label=label
        )

    _climb_mid(session, f"{label}_mid")
    if _in_hellway(session.state):
        return wait_ordinary_room(
            session, ROOM_HELLWAY, settle_frames=RED_TO_HELLWAY_EXIT_SETTLE, label=label
        )

    _climb_upper(session, f"{label}_upper")
    if _in_hellway(session.state):
        return wait_ordinary_room(
            session, ROOM_HELLWAY, settle_frames=RED_TO_HELLWAY_EXIT_SETTLE, label=label
        )
    if not _in_red(session.state):
        raise TimeoutError(f"{label}: left Red unexpectedly: {session.state}")

    _unmorph(session, label)
    if int(session.state.samus_y) > RED_TOP_DOOR_Y + 100:
        _climb_upper(session, f"{label}_reclimb")
        if _in_hellway(session.state):
            return wait_ordinary_room(
                session,
                ROOM_HELLWAY,
                settle_frames=RED_TO_HELLWAY_EXIT_SETTLE,
                label=label,
            )

    for _ in range(100):
        st = session.state
        if _in_hellway(st) or not _in_red(st):
            break
        if int(st.samus_y) > RED_TOP_DOOR_Y + 50:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_top_hop")
            continue
        if int(st.samus_x) < RED_TOP_DOOR_X - 30:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_top_run")
            continue
        break

    if _in_hellway(session.state):
        return wait_ordinary_room(
            session, ROOM_HELLWAY, settle_frames=RED_TO_HELLWAY_EXIT_SETTLE, label=label
        )

    return play_run_shoot_exit(
        session,
        from_room=ROOM_RED_TOWER,
        to_room=ROOM_HELLWAY,
        direction="RIGHT",
        label=label,
        run_frames=RED_TO_HELLWAY_EXIT_RUN,
        shoot_frames=RED_TO_HELLWAY_EXIT_SHOOT,
        spin_frames=RED_TO_HELLWAY_EXIT_SPIN,
        hold_frames=RED_TO_HELLWAY_EXIT_HOLD,
        settle_frames=RED_TO_HELLWAY_EXIT_SETTLE,
    )


__all__ = ["play_red_to_hellway"]
