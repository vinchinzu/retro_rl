"""Red Tower mid-shaft climb: IBJ, tunnel, pocket, lower wall-jump."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    is_morph,
    settle_hold,
)
from super_metroid.routes.kpdr.k5.geometry import (
    RED_BOTTOM_Y,
    RED_FLOOR_Y,
    RED_LOWER_LIP_Y,
    RED_TUNNEL_Y,
)
from super_metroid.routes.kpdr.k5.red_to_hellway_common import (
    _IBJ_WAIT1,
    _IBJ_WAIT2,
    _MORPH,
    _RWJ_BACK,
    _RWJ_INTO,
    _RWJ_OUT,
    _RWJ_WJ,
    _STAND,
    _in_hellway,
    _in_red,
    _kb,
    _unmorph,
)
from super_metroid.routes.runtime import ControllerSession


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


def _ibj_double(
    session: ControllerSession,
    label: str,
    *,
    center_x: int = 150,
    stop_y: int = RED_FLOOR_Y - 40,
) -> SuperMetroidState:
    """One double-bomb IBJ cycle (morph X / wait / X / wait).

    Dual-stable lower climb: center_x≈150, waits 18/30 — peaks tunnel ~y1820
    from past-pocket morph fall; midplat center_x≈171 peaks temp floor ~y1600.
    """
    if not _in_red(session.state):
        return session.state
    _kb(session, label)
    if not is_morph(session.state.pose) and int(session.state.pose) not in _MORPH:
        ensure_morph(session)
    st = session.state
    x = int(st.samus_x)
    y = int(st.samus_y)
    # Low fall: bias LEFT hard so we never drift into Bat door.
    if y >= RED_BOTTOM_Y - 120 and x >= 200:
        hold(session, 4, "LEFT", reason=f"{label}_bat_bias")
    elif x > center_x + 18:
        hold(session, 2, "LEFT", reason=f"{label}_cL")
    elif x < center_x - 18:
        hold(session, 2, "RIGHT", reason=f"{label}_cR")
    hold(session, 2, "X", reason=f"{label}_b1")
    for _ in range(_IBJ_WAIT1):
        st = hold(session, 1, reason=f"{label}_w1")
        if not _in_red(st) or int(st.samus_y) <= stop_y:
            return st
        if int(st.samus_y) >= RED_BOTTOM_Y - 100 and int(st.samus_x) >= 200:
            hold(session, 1, "LEFT", reason=f"{label}_bat_w1")
    hold(session, 2, "X", reason=f"{label}_b2")
    for _ in range(_IBJ_WAIT2):
        st = hold(session, 1, reason=f"{label}_w2")
        if not _in_red(st) or int(st.samus_y) <= stop_y:
            return st
        if int(st.samus_y) >= RED_BOTTOM_Y - 100 and int(st.samus_x) >= 200:
            hold(session, 1, "LEFT", reason=f"{label}_bat_w2")
    return session.state


def _tunnel_to_midplat(session: ControllerSession, label: str) -> SuperMetroidState:
    """From tunnel peak ~y1820: seat y1883 x≈104 → hop to midplat ~y1720 x≈116.

    Human-matched UP+X then A / A+X / RIGHT+A+X into bomb-block mid shelf.
    Dual-stable from past-pocket IBJ peak pin.
    """
    if not _in_red(session.state):
        return session.state
    _unmorph(session, label)
    for _ in range(16):
        hold(session, 1, "UP", reason=f"{label}_stand")
    settle_hold(session, 5, reason=f"{label}_tun_s")
    # Drop / walk left onto solid tunnel floor y≈1883.
    for _ in range(35):
        st = session.state
        if not _in_red(st):
            return st
        if int(st.samus_y) >= RED_TUNNEL_Y:
            break
        hold(session, 1, "LEFT", reason=f"{label}_to_tun")
    for _ in range(40):
        st = session.state
        if not _in_red(st) or int(st.velocity_y) == 0:
            break
        hold(session, 1, reason=f"{label}_tun_land")
    for _ in range(50):
        st = session.state
        if not _in_red(st):
            return st
        if abs(int(st.samus_x) - 104) < 6 and int(st.velocity_y) == 0:
            break
        hold(
            session,
            1,
            "LEFT" if int(st.samus_x) > 104 else "RIGHT",
            reason=f"{label}_tun_x",
        )
    settle_hold(session, 5, reason=f"{label}_tun_seat")
    for _ in range(8):
        hold(session, 2, "UP", "X", reason=f"{label}_tun_shot")
    for i in range(50):
        st = session.state
        if not _in_red(st) or int(st.samus_y) <= RED_FLOOR_Y + 80:
            return st
        if i < 15:
            hold(session, 1, "A", reason=f"{label}_tun_j")
        elif i < 25:
            hold(session, 1, "A", "X", reason=f"{label}_tun_jx")
        else:
            hold(session, 1, "RIGHT", "A", "X", reason=f"{label}_tun_jrx")
    for _ in range(35):
        st = session.state
        if not _in_red(st) or int(st.velocity_y) == 0:
            break
        hold(session, 1, reason=f"{label}_mid_land")
    return session.state


def _pocket_spin_mid(session: ControllerSession, label: str) -> SuperMetroidState:
    """From right pocket ~(225,2091) or wall seat: B+LEFT+A spin into mid shaft.

    Human + probe: dual-stable peak/seat ~y1932–1942 at x≈170–185 (crouch under
    tunnel lip). Right-wall pure-A at y1942 gains 0 (hard ceiling) — must leave
    via spin, not recatch thrash. Still ~50px short of RED_TUNNEL_Y=1880.
    """
    if not _in_red(session.state):
        return session.state
    _unmorph(session, label)
    _kb(session, label)
    # Stand from crouch if needed (pocket often pose 1/9/137).
    for _ in range(14):
        st = session.state
        if int(st.pose) in _STAND:
            break
        hold(session, 1, "UP", reason=f"{label}_stand")
    settle_hold(session, 3, reason=f"{label}_spin_s")
    hold(session, 4, "LEFT", reason=f"{label}_face")
    for f in range(56):
        st = hold(session, 1, "B", "LEFT", "A", reason=f"{label}_spin")
        if not _in_red(st):
            return st
        y = int(st.samus_y)
        # Plateau on mid crouch seat / peak.
        if f > 32 and int(st.velocity_y) <= 1 and y <= 1955:
            break
        if y <= RED_TUNNEL_Y:
            break
    for _ in range(16):
        st = session.state
        if not _in_red(st) or int(st.velocity_y) == 0:
            break
        hold(session, 1, reason=f"{label}_spin_land")
    return session.state


def _on_pocket_seat(state: SuperMetroidState) -> bool:
    """Right-pocket dead-end ledge used as spin launch (not climb-done)."""
    return (
        _in_red(state)
        and int(state.samus_x) >= 210
        and RED_LOWER_LIP_Y - 40 <= int(state.samus_y) <= RED_LOWER_LIP_Y + 40
        and int(state.velocity_y) == 0
    )


def _on_right_wall_ceiling(state: SuperMetroidState) -> bool:
    """Hard right-wall ceiling seat ~y1942 (pure-A gain 0)."""
    return (
        _in_red(state)
        and int(state.samus_x) >= 210
        and 1925 <= int(state.samus_y) <= 1985
        and int(state.velocity_y) == 0
    )


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
    """Bottom → tunnel peak → midplat → temporary floor (phased dual path).

    Probe path (rr-av5s night2)::

      1. Morph + IBJ 18/30 c150 from pure bottom — dual peak ~y1820
         (do **not** climb_lower first — desyncs IBJ)
      2. ``_tunnel_to_midplat``: tunnel seat → bomb-hop midplat ~y1720 x171
      3. Midplat IBJ 18/30 c171 dual peaks temp floor ~y1600 (hard lip)
      4. Bomb-floor reverse residual when y locks ≤RED_FLOOR_Y

    Human mid ledges y2255/2159/2023 are frozen rippers — not pure solid tiles.
    """
    best_y = int(session.state.samus_y)

    def _ensure_m() -> None:
        st = session.state
        if is_morph(st.pose) or int(st.pose) in _MORPH:
            return
        try:
            ensure_morph(session)
        except Exception:
            hold(session, 1, "DOWN", reason=f"{label}_morph_d")
            hold(session, 1, reason=f"{label}_morph_r")
            hold(session, 1, "DOWN", reason=f"{label}_morph_d2")

    # --- Phase 1: bottom IBJ → tunnel peak ---
    _ensure_m()
    for c in range(90):
        st = session.state
        if _in_hellway(st):
            return st
        if not _in_red(st):
            raise TimeoutError(
                f"{label}: left Red p1 room=0x{int(st.room_id):04X} "
                f"xy=({st.samus_x},{st.samus_y})"
            )
        y = int(st.samus_y)
        if y < best_y:
            best_y = y
        if y <= 1830 and int(st.velocity_y) == 0 and c > 20:
            break
        if y <= 1820:
            break
        _kb(session, f"{label}_p1_{c}")
        _ensure_m()
        _ibj_double(session, f"{label}_p1_{c}", center_x=150, stop_y=1820)
    st = session.state
    if int(st.samus_y) > 1900:
        raise TimeoutError(
            f"{label}: tunnel peak failed best_y={best_y} "
            f"y={st.samus_y} x={st.samus_x} p={st.pose}"
        )

    # --- Phase 2: tunnel → midplat ---
    if int(session.state.samus_y) > 1680:
        _tunnel_to_midplat(session, f"{label}_tun")
    st = session.state
    if not _in_red(st):
        if _in_hellway(st):
            return st
        raise TimeoutError(
            f"{label}: left Red tun room=0x{int(st.room_id):04X}"
        )
    if int(st.samus_y) > 1850:
        # Retry peak IBJ then tunnel once.
        _ensure_m()
        for c in range(40):
            _ibj_double(session, f"{label}_p1r_{c}", center_x=150, stop_y=1820)
            if int(session.state.samus_y) <= 1830:
                break
        _tunnel_to_midplat(session, f"{label}_tun2")
    st = session.state
    if int(st.samus_y) > 1850:
        raise TimeoutError(
            f"{label}: midplat failed best_y={best_y} "
            f"y={st.samus_y} x={st.samus_x} p={st.pose}"
        )

    # --- Phase 3: stand → x171 → morph IBJ 18/30 (dual peaks ~y1606) ---
    _unmorph(session, label)
    for _ in range(15):
        hold(session, 1, "UP", reason=f"{label}_mid_stand")
    settle_hold(session, 6, reason=f"{label}_mid_s")
    for _ in range(50):
        st = session.state
        if not _in_red(st) or int(st.samus_y) > 1820:
            break
        if int(st.samus_x) >= 168:
            break
        hold(session, 1, "RIGHT", reason=f"{label}_mid_r")
    settle_hold(session, 4, reason=f"{label}_mid_s2")
    _ensure_m()
    for c in range(45):
        st = session.state
        if _in_hellway(st):
            return st
        if not _in_red(st):
            raise TimeoutError(
                f"{label}: left Red p3 room=0x{int(st.room_id):04X} "
                f"xy=({st.samus_x},{st.samus_y})"
            )
        y = int(st.samus_y)
        if y < best_y:
            best_y = y
        if y <= RED_FLOOR_Y + 10:
            return st
        if y >= 1950:
            break
        _ensure_m()
        _ibj_double(
            session, f"{label}_p3_{c}", center_x=171, stop_y=RED_FLOOR_Y - 40
        )
        # At dual peak ~y1606, spray bombs against temp floor lip.
        st = session.state
        if _in_red(st) and int(st.samus_y) <= RED_FLOOR_Y + 30:
            for _ in range(6):
                hold(session, 2, "X", reason=f"{label}_lip")
                hold(session, 6, reason=f"{label}_lip_w")
            if int(session.state.samus_y) <= RED_FLOOR_Y + 10:
                return session.state

    st = session.state
    if int(st.samus_y) <= RED_FLOOR_Y + 20:
        return st
    # Progress residual: tunnel+midplat dual-stable; temp floor hard lip ~y1600.
    raise TimeoutError(
        f"{label}: floor peak failed best_y={best_y} "
        f"y={st.samus_y} x={st.samus_x} p={st.pose}"
    )
