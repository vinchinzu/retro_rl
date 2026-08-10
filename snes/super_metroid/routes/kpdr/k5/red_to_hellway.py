"""Red Tower → Hellway pure return (K5 hop 12).

Source: ``post_ice_bat_to_red_pure`` ~(216, 2443) pose 10 Red bottom after
Bat→Red dual **718f**. Climb reverse of ``play_red_tower_to_bat`` descent
bands (lower zigzag → tunnel → temporary floor → upper zigzag), then RIGHT
into top-right Hellway door ``0xA2F7``.

Hybrid pure (Hi-Jump held on K5 stack)::

  1. Accept Red bottom residual; clear Bat door lip
  2. Right-wall WJ chain lower shaft → y≲2090 (reverse lower zigzag)
  3. Mid tunnel / bomb-floor reverse of red_tower_to_bat mid bands
  4. Upper WJ zigzag → top-right door band
  5. RIGHT into Hellway ordinary settle

Tape: ``tasks/speed_to_wave_ice_moat_human.json`` Phase C hop 29→30.
Prefer clean reverse of descent, not full 7k freeze thrash.
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
_STAND = frozenset({1, 2, 9, 10, 12, 25, 26, 27, 28, 137, 138})


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
    """Stay left of Bat door on the bottom floor."""
    st = session.state
    if int(st.samus_y) >= RED_BOTTOM_Y - 50 and int(st.samus_x) >= 218:
        hold(session, 1, "LEFT", reason=f"{label}_bat")


def _right_wall_wj_cycle(session: ControllerSession, label: str, *, from_floor: bool) -> None:
    """One right-wall wall-jump cycle (human-validated lower-shaft primitive)."""
    if not _in_red(session.state):
        return
    _unmorph(session, label)
    if is_knockback(session.state):
        escape_knockback_spin(
            session,
            prefer_dir="LEFT",
            run_frames=3,
            spin_frames=10,
            label=f"{label}_kb",
            ensure_beam=True,
            break_on_motion_clear=True,
        )

    if from_floor or int(session.state.samus_y) >= RED_BOTTOM_Y - 40:
        # Runway mid-left then jump into right wall.
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
            if int(st.samus_x) >= 218 and int(st.velocity_y) == 0 and f > 18:
                break
    else:
        # Re-approach right wall from mid-height.
        for _ in range(35):
            st = session.state
            if not _in_red(st) or int(st.samus_y) >= RED_BOTTOM_Y - 20:
                return
            if int(st.pose) in _MORPH:
                hold(session, 1, "UP", reason=f"{label}_u")
                continue
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
            # Chain second WJ off right wall.
            hold(session, 1, "LEFT", reason=f"{label}_face2")
            for _ in range(15):
                hold(session, 1, "LEFT", "A", reason=f"{label}_wj2")
            for _ in range(22):
                hold(session, 1, "RIGHT", "A", reason=f"{label}_back2")
            break


def _climb_lower(session: ControllerSession, label: str) -> SuperMetroidState:
    """Bottom → open mid-shaft y≲2000 via right-wall WJ (multi-attempt).

    Do **not** accept the right-wall pocket ledge ~(225,2091) as done — that
    shelf is a dead-end under a ceiling. Keep chaining until y<=2000 with
    x in the open shaft (or any y<=1880).
    """
    target_y = RED_TUNNEL_Y + 40  # ~1920 — past the right-pocket ceiling
    for attempt in range(36):
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
        # Success: clear of right-pocket ceiling into open shaft.
        if y <= target_y and x <= 210:
            return st
        if y <= RED_TUNNEL_Y:
            return st
        # Stuck on right pocket ledge: apex-WJ left into shaft.
        if RED_LOWER_LIP_Y - 20 <= y <= RED_LOWER_LIP_Y + 40 and x >= 210:
            hold(session, 2, "RIGHT", reason=f"{label}_pocket_face")
            for f in range(36):
                st2 = hold(session, 1, "A", reason=f"{label}_pocket_up")
                if not _in_red(st2):
                    return st2
                if int(st2.samus_y) <= 1985 and int(st2.velocity_y) <= 1:
                    hold(session, 1, "LEFT", reason=f"{label}_pocket_faceL")
                    for _ in range(16):
                        hold(session, 1, "LEFT", "A", reason=f"{label}_pocket_wj")
                    # Aim for mid platforms, not free fall left.
                    for _ in range(12):
                        hold(session, 1, "LEFT", "A", reason=f"{label}_pocket_fly")
                    for _ in range(20):
                        hold(session, 1, "RIGHT", "A", reason=f"{label}_pocket_mid")
                    break
            continue
        _right_wall_wj_cycle(
            session,
            f"{label}_a{attempt}",
            from_floor=int(session.state.samus_y) >= RED_BOTTOM_Y - 60,
        )
    st = session.state
    if int(st.samus_y) > target_y + 100:
        raise TimeoutError(
            f"{label}: lower climb stalled y={st.samus_y} x={st.samus_x} p={st.pose}"
        )
    return st


def _vertical_ledge_hops(
    session: ControllerSession,
    label: str,
    *,
    count: int = 4,
) -> None:
    """Human mid-shaft primitive: grounded A hops (pose 78) then LEFT/RIGHT steer.

    Tape f28000–28130: stand ~y2159/2091, hold A to rise, mix LEFT+A / RIGHT+A.
    """
    _unmorph(session, label)
    settle_hold(session, 6, reason=f"{label}_ledge_s")
    for i in range(count):
        st = session.state
        if not _in_red(st) or _in_hellway(st):
            return
        if int(st.samus_y) <= RED_FLOOR_Y:
            return
        # Vertical A burst (human crouch-jump / spin up).
        hold(session, 28, "A", reason=f"{label}_v{i}")
        # Steer while still rising / floating.
        d = "LEFT" if int(session.state.samus_x) > 180 else "RIGHT"
        hold(session, 16, d, "A", reason=f"{label}_steer{i}")
        settle_hold(session, 10, reason=f"{label}_land{i}")
        _unmorph(session, label)


def _climb_mid(session: ControllerSession, label: str) -> SuperMetroidState:
    """y~2090 → through tunnel + temporary floor to y<=RED_FLOOR_Y.

    Reverse of red_tower_to_bat mid bands. Human uses vertical A hops on the
    right/mid ledges then morph through ~y1760 bomb floor.
    """
    for attempt in range(36):
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

        if is_knockback(st):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=2,
                spin_frames=8,
                label=f"{label}_kb",
                ensure_beam=True,
                break_on_motion_clear=True,
            )
            continue

        if int(st.pose) in _MORPH or is_morph(st.pose):
            hold(session, 2, "X", reason=f"{label}_bomb")
            d = "RIGHT" if x < 150 else "LEFT"
            hold(session, 14, d, reason=f"{label}_roll")
            hold(session, 8, "UP", reason=f"{label}_u")
            continue

        if y >= RED_BOTTOM_Y - 40:
            _climb_lower(session, f"{label}_relower")
            continue

        # Right-lip / mid ledge band (~y2000–2160): vertical A hops.
        if RED_TUNNEL_Y - 20 <= y <= RED_LOWER_LIP_Y + 80:
            # Prefer x~170–200 like human final ascent, not Bat lip x≥230.
            if x >= 220:
                hold(session, 12, "LEFT", "B", reason=f"{label}_off_wall")
                settle_hold(session, 8, reason=f"{label}_off_s")
            _vertical_ledge_hops(session, f"{label}_v{attempt}", count=3)
            # LEFT spin toward tunnel after hops.
            hold(session, 20, "LEFT", "B", "A", reason=f"{label}_tun_in")
            settle_hold(session, 8, reason=f"{label}_tun_s")
            continue

        # Temporary floor band: reverse of floor bomb-cross.
        if RED_FLOOR_Y - 30 <= y <= RED_FLOOR_Y + 220:
            ensure_morph(session)
            for _ in range(50):
                st2 = session.state
                if not _in_red(st2):
                    return st2
                if 145 <= int(st2.samus_x) <= 175:
                    break
                hold(
                    session,
                    1,
                    "LEFT" if int(st2.samus_x) > 165 else "RIGHT",
                    reason=f"{label}_floor_pos",
                )
            hold(session, 2, "X", reason=f"{label}_floor_bomb")
            hold(session, 10, "LEFT", reason=f"{label}_floor_ret")
            hold(session, 36, reason=f"{label}_floor_wait")
            for _ in range(100):
                st2 = hold(session, 1, "A", reason=f"{label}_floor_up")
                if not _in_red(st2):
                    return st2
                if int(st2.samus_y) < RED_FLOOR_Y - 30:
                    break
            # Through hole left into upper.
            hold(session, 24, "LEFT", "B", "A", reason=f"{label}_floor_left")
            _unmorph(session, label)
            continue

        # Between tunnel and floor: keep hopping up.
        d = "RIGHT" if x < 140 else "LEFT"
        hold(session, 18, "A", reason=f"{label}_mid_v")
        hold(session, 12, d, "A", reason=f"{label}_mid_steer")
        settle_hold(session, 8, reason=f"{label}_mid_land")

    st = session.state
    if int(st.samus_y) > RED_FLOOR_Y + 100:
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

        if is_knockback(st):
            escape_knockback_spin(
                session,
                prefer_dir=direction,
                run_frames=2,
                spin_frames=8,
                label=f"{label}_kb",
                ensure_beam=True,
                break_on_motion_clear=True,
            )
            continue

        if int(st.pose) in _MORPH or is_morph(st.pose):
            hold(session, 1, "UP", reason=f"{label}_u")
            continue

        if y >= RED_BOTTOM_Y - 80:
            _climb_lower(session, f"{label}_relower")
            _climb_mid(session, f"{label}_remid")
            continue

        if x >= RED_ZIG_X_MAX:
            direction = "LEFT"
        elif x <= RED_ZIG_X_MIN:
            direction = "RIGHT"

        # Right-wall WJ preference near wall.
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
    for _ in range(90):
        st = session.state
        if not _in_red(st):
            break
        if int(st.samus_x) <= 170 and int(st.velocity_y) == 0:
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

    # Top door push.
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
