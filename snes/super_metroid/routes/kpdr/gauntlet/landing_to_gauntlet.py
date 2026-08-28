"""Landing Site floor (parlor door) → Gauntlet Entrance with Morph + Bombs.

Live sequence (do not IBJ the V-gap or hug the cliff):

1. Hop right out of the parlor cave onto the ship floor (x≳650).
2. Open-air IBJ at x≈840–900: first bomb wait 52, then 18/30 to y≲520.
3. More IBJ + 4f-left drift to the cliff face, climb, 12f-left above A.
4. Morph-bomb Obstacle A, hop the cave ledge, shoot the blue door.
"""

from __future__ import annotations

from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    require_room,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.gauntlet.geometry import (
    BOMB_WALL_CYCLES,
    BOMB_WALL_X,
    CAVE_HOP_MAX,
    IBJ_CENTER_X,
    IBJ_DOOR_Y,
    IBJ_FIRST_WAIT,
    IBJ_MAX_CYCLES,
    IBJ_STOP_Y,
    IBJ_WAIT1,
    IBJ_WAIT2,
    IBJ_X,
    LANDING_FLOOR_Y,
    LEDGE_Y,
    SHIP_FLOOR_MIN_X,
    at_cliff_lip,
    at_gauntlet_ledge,
    at_ship_floor,
    is_grounded,
    is_morph_pose,
)
from super_metroid.routes.kpdr.room_ids import ROOM_GAUNTLET_ENTRANCE, ROOM_LANDING_SITE
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.geometry import PhaseStop
from super_metroid.routes.skills.knockback import escape_knockback_spin, is_knockback


def _in_landing(session: ControllerSession) -> bool:
    return int(session.state.room_id) == ROOM_LANDING_SITE


def _near_bomb_wall(session: ControllerSession) -> bool:
    st = session.state
    return (
        _in_landing(session)
        and int(st.samus_x) <= BOMB_WALL_X + 40
        and int(st.samus_y) <= LEDGE_Y[1]
    )


def _settle_grounded(session: ControllerSession, *, timeout: int = 50) -> None:
    """Idle through spin/fall until a ground pose (ship-floor pin is pose 25)."""
    for _ in range(timeout):
        st = session.state
        if not _in_landing(session):
            return
        if is_knockback(st):
            escape_knockback_spin(
                session,
                prefer_dir="RIGHT",
                run_frames=4,
                spin_frames=12,
                label="landing_kb",
            )
            continue
        if is_grounded(st) or (
            is_morph_pose(int(st.pose)) and abs(int(st.velocity_y)) <= 1
        ):
            return
        hold(session, 1, reason="landing_settle")


def _hop_to_ship_floor(session: ControllerSession) -> None:
    """Parlor door dumps us in the left floor cave; hop the wall at x≈495."""
    if at_ship_floor(session.state) or int(session.state.samus_x) >= SHIP_FLOOR_MIN_X:
        _settle_grounded(session)
        return
    unmorph(session)
    _settle_grounded(session)
    for _ in range(CAVE_HOP_MAX):
        if not _in_landing(session):
            return
        if int(session.state.samus_x) >= SHIP_FLOOR_MIN_X:
            break
        if is_knockback(session.state):
            escape_knockback_spin(
                session,
                prefer_dir="RIGHT",
                run_frames=4,
                spin_frames=10,
                label="landing_cave_kb",
            )
        hold(session, 10, "RIGHT", "B", reason="landing_cave_hop_run")
        hold(session, 14, "RIGHT", "A", "B", reason="landing_cave_hop_jump")
        hold(session, 20, "RIGHT", "B", reason="landing_cave_hop_land")
    _settle_grounded(session)
    hold(session, 6, reason="landing_ship_settle")


def _align_ibj(session: ControllerSession, *, center_x: int = IBJ_CENTER_X) -> None:
    del center_x
    lo, hi = IBJ_X
    unmorph(session)
    _settle_grounded(session)
    for _ in range(120):
        if not _in_landing(session):
            return
        x = int(session.state.samus_x)
        if lo <= x <= hi:
            break
        hold(
            session,
            1,
            "RIGHT" if x < lo else "LEFT",
            "B",
            reason="landing_ibj_align",
        )
    hold(session, 4, reason="landing_ibj_align_settle")


def ibj_first_bomb(
    session: ControllerSession,
    *,
    label: str = "landing_ibj",
    stop_y: int = IBJ_STOP_Y,
) -> bool:
    """Rest starter: X then wait 55 so the first explosion lifts off the floor."""
    if not _in_landing(session):
        return True
    if not is_morph_pose(int(session.state.pose)):
        ensure_morph(session)
    hold(session, 2, "X", reason=f"{label}_b0")
    airborne = int(session.state.samus_y) < LANDING_FLOOR_Y[0] - 20
    wait = IBJ_WAIT1 if airborne else IBJ_FIRST_WAIT
    for _ in range(wait):
        st = hold(session, 1, reason=f"{label}_w0")
        if int(st.room_id) != ROOM_LANDING_SITE or int(st.samus_y) <= stop_y:
            return True
    return int(session.state.samus_y) <= stop_y


def ibj_cycle(
    session: ControllerSession,
    *,
    label: str = "landing_ibj",
    center_x: int = IBJ_CENTER_X,
    stop_y: int = IBJ_STOP_Y,
    wait1: int = IBJ_WAIT1,
    wait2: int = IBJ_WAIT2,
    left_frames: int = 0,
) -> bool:
    """One double-bomb IBJ cycle. Returns True if ``samus_y <= stop_y``.

    Do not hold LEFT for the whole wait — that falls. A short LEFT tap during
    wait2 is the live drift onto the cliff / Obstacle A.
    """
    if int(session.state.room_id) != ROOM_LANDING_SITE:
        return True
    if not is_morph_pose(int(session.state.pose)):
        ensure_morph(session)
    x = int(session.state.samus_x)
    # Wide deadzone: a 2f tap during the first bounce kills the column.
    if left_frames == 0:
        if x > center_x + 80:
            hold(session, 1, "LEFT", reason=f"{label}_cL")
        elif x < center_x - 80:
            hold(session, 1, "RIGHT", reason=f"{label}_cR")
    hold(session, 2, "X", reason=f"{label}_b1")
    for _ in range(wait1):
        st = hold(session, 1, reason=f"{label}_w1")
        if int(st.room_id) != ROOM_LANDING_SITE:
            return True
    hold(session, 2, "X", reason=f"{label}_b2")
    left_n = max(0, min(int(left_frames), wait2))
    for i in range(wait2):
        if i < left_n:
            st = hold(session, 1, "LEFT", reason=f"{label}_w2L")
        else:
            st = hold(session, 1, reason=f"{label}_w2")
        if int(st.room_id) != ROOM_LANDING_SITE:
            return True
    return int(session.state.samus_y) <= stop_y


# Morph rest on the ship grass is y≈1161–1195. The first-bomb peak is ~1119
# with vy≈0 — that is NOT the floor. Restarting the 55-wait there drops us.
_IBJ_FLOOR_Y = 1155


def climb_open_air_ibj(
    session: ControllerSession,
    *,
    stop_y: int = IBJ_STOP_Y,
    center_x: int = IBJ_CENTER_X,
    label: str = "landing_ibj",
) -> None:
    """Long IBJ in open air (x≈870) past Gauntlet-door height."""
    ensure_morph(session)
    hold(session, 8, reason=f"{label}_morph_rest")
    reached = ibj_first_bomb(session, label=label, stop_y=stop_y)
    best_y = int(session.state.samus_y)
    floor_retries = 0
    for _ in range(IBJ_MAX_CYCLES):
        if not _in_landing(session) or reached:
            return
        y = int(session.state.samus_y)
        best_y = min(best_y, y)
        if y <= stop_y:
            return
        on_floor = y >= _IBJ_FLOOR_Y and abs(int(session.state.velocity_y)) <= 1
        if on_floor:
            floor_retries += 1
            if floor_retries > 3:
                break
            if ibj_first_bomb(session, label=label, stop_y=stop_y):
                return
            continue
        if ibj_cycle(session, label=label, center_x=center_x, stop_y=stop_y):
            return
    y = int(session.state.samus_y)
    if y > IBJ_DOOR_Y:
        raise TimeoutError(
            f"landing_to_gauntlet: open-air IBJ timed out y={y} best={best_y} "
            f"{session.state}"
        )


def drift_to_bomb_wall(session: ControllerSession) -> None:
    """From a rested open-air peak ~y520, IBJ + drift onto Obstacle A.

    The idle seats are part of the bomb cadence, not cosmetic.  From
    ``gauntlet_ibj_peak2.state`` this exact chain reaches (629,573):

    1 up + 10×4f-left → 50 idle → 12 up → 30 idle → 8 up + 6×12f-left.
    """
    if _near_bomb_wall(session):
        return

    def _y() -> int:
        return int(session.state.samus_y)

    def _raw(n: int, left: int = 0, tag: str = "landing_ibj") -> None:
        # Byte-match of the peak-pin search: no recenter, no mid-cycle abort.
        for _ in range(n):
            if not _in_landing(session) or _y() >= 1100:
                return
            hold(session, 2, "X", reason=f"{tag}_b1")
            hold(session, 18, reason=f"{tag}_w1")
            hold(session, 2, "X", reason=f"{tag}_b2")
            if left:
                hold(session, left, "LEFT", reason=f"{tag}_L")
                hold(session, max(0, 30 - left), reason=f"{tag}_w2")
            else:
                hold(session, 30, reason=f"{tag}_w2")

    # From rest peak (867,518): 1 extra 18/30 then 10×4f-left → (645,738).
    _raw(1, 0, "landing_ibj_up")
    _raw(10, 4, "landing_ibj_dL4")
    if _y() >= 900:
        return
    hold(session, 50, reason="landing_ibj_cliff_seat")
    _raw(12, 0, "landing_ibj_face")
    if _y() >= 900:
        return
    hold(session, 30, reason="landing_ibj_face_seat")
    _raw(8, 0, "landing_ibj_above_a")
    _raw(6, 12, "landing_ibj_dL12")
    hold(session, 40, reason="landing_ibj_a_seat")


def _jump_off_lip_right(session: ControllerSession) -> None:
    """Step back into open air from the node-7 lip so IBJ is unobstructed."""
    unmorph(session)
    _settle_grounded(session)
    hold(session, 6, "RIGHT", "B", reason="landing_lip_away")
    hold(session, 18, "RIGHT", "A", "B", reason="landing_lip_jump")
    hold(session, 8, "RIGHT", "B", reason="landing_lip_air")
    ensure_morph(session)


def climb_from_lip(session: ControllerSession) -> None:
    """Remaining height from the node-7 lip: jump-into IBJ, then drift to A."""
    if _near_bomb_wall(session) and int(session.state.samus_y) <= IBJ_DOOR_Y:
        return
    _jump_off_lip_right(session)
    climb_open_air_ibj(session, stop_y=IBJ_STOP_Y, label="landing_lip_ibj")
    drift_to_bomb_wall(session)


def climb_to_ledge(session: ControllerSession) -> None:
    """Ship-floor open-air IBJ, then drift onto the Obstacle A ledge."""
    climb_open_air_ibj(session)
    drift_to_bomb_wall(session)


def bomb_gauntlet_wall(session: ControllerSession) -> None:
    """Morph-bomb Obstacle A. Live from (629,573): 4f LEFT + X + 32 idle."""
    ensure_morph(session)
    for _ in range(BOMB_WALL_CYCLES):
        st = session.state
        if not _in_landing(session):
            return
        if int(st.samus_x) <= 500 and 600 <= int(st.samus_y) <= 720:
            return
        hold(session, 4, "LEFT", reason="landing_bomb_wall_roll")
        hold(session, 2, "X", reason="landing_bomb_wall")
        hold(session, 32, reason="landing_bomb_wall_wait")


def _enter_gauntlet_door(session: ControllerSession) -> None:
    """Unmorph, shoot the blue door, hop the last cave ledge, walk in.

    Live from cave mouth (489,649): run ~150, hop 18, shot, run → 0x92B3.
    """
    if int(session.state.samus_y) >= 900:
        raise TimeoutError(
            f"landing_to_gauntlet: on the floor before the door {session.state}"
        )
    unmorph(session)
    if is_knockback(session.state):
        escape_knockback_spin(
            session,
            prefer_dir="LEFT",
            run_frames=4,
            spin_frames=12,
            label="landing_door_kb",
        )
    hold(session, 4, "X", reason="landing_door_shot")
    for i in range(150):
        if not _in_landing(session):
            break
        if is_knockback(session.state):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=4,
                spin_frames=10,
                label="landing_door_kb",
            )
        if i % 20 < 4:
            hold(session, 1, "LEFT", "B", "X", reason="landing_door_run")
        else:
            hold(session, 1, "LEFT", "B", reason="landing_door_run")
    if _in_landing(session):
        hold(session, 18, "LEFT", "A", "B", reason="landing_door_hop")
        hold(session, 6, "LEFT", "X", reason="landing_door_shot")
    for _ in range(220):
        if not _in_landing(session):
            break
        if is_knockback(session.state):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=4,
                spin_frames=10,
                label="landing_door_kb",
            )
            continue
        hold(session, 1, "LEFT", "B", reason="landing_gauntlet_door")
    if _in_landing(session):
        raise TimeoutError(
            f"landing_to_gauntlet: Gauntlet door did not open {session.state}"
        )
    wait_ordinary_room(
        session,
        ROOM_GAUNTLET_ENTRANCE,
        settle_frames=280,
        label="landing_to_gauntlet",
    )


def play_landing_to_gauntlet(
    session: ControllerSession,
    *,
    stop_at: str | None = None,
    start_at: str | None = None,
) -> None:
    """Landing floor cave → Gauntlet Entrance (Morph + Bombs)."""
    require_room(session, ROOM_LANDING_SITE, "landing_to_gauntlet")
    if start_at == "cave":
        _enter_gauntlet_door(session)
        return
    if start_at == "obstacle_a":
        bomb_gauntlet_wall(session)
        if stop_at == "wall":
            raise PhaseStop("wall", session.state, label="landing_to_gauntlet")
        _enter_gauntlet_door(session)
        return
    if start_at == "ibj_high":
        if stop_at == "ibj_high":
            raise PhaseStop("ibj_high", session.state, label="landing_to_gauntlet")
        drift_to_bomb_wall(session)
        if stop_at in ("ledge", "lip"):
            raise PhaseStop(stop_at, session.state, label="landing_to_gauntlet")
        bomb_gauntlet_wall(session)
        if stop_at == "wall":
            raise PhaseStop("wall", session.state, label="landing_to_gauntlet")
        _enter_gauntlet_door(session)
        return
    from_lip = start_at == "lip" or (
        start_at is None and (at_cliff_lip(session.state) or at_gauntlet_ledge(session.state))
        and int(session.state.samus_y) < LANDING_FLOOR_Y[0]
    )
    if from_lip:
        if stop_at == "lip":
            raise PhaseStop("lip", session.state, label="landing_to_gauntlet")
        climb_from_lip(session)
        if stop_at == "ledge":
            raise PhaseStop("ledge", session.state, label="landing_to_gauntlet")
        bomb_gauntlet_wall(session)
        if stop_at == "wall":
            raise PhaseStop("wall", session.state, label="landing_to_gauntlet")
        _enter_gauntlet_door(session)
        return

    _hop_to_ship_floor(session)
    if stop_at in ("cave_exit", "ship"):
        raise PhaseStop(stop_at, session.state, label="landing_to_gauntlet")
    _align_ibj(session)
    if stop_at == "ibj_high":
        climb_open_air_ibj(session)
        raise PhaseStop("ibj_high", session.state, label="landing_to_gauntlet")
    climb_to_ledge(session)
    if stop_at in ("ledge", "lip"):
        raise PhaseStop(stop_at, session.state, label="landing_to_gauntlet")
    bomb_gauntlet_wall(session)
    if stop_at == "wall":
        raise PhaseStop("wall", session.state, label="landing_to_gauntlet")
    _enter_gauntlet_door(session)


__all__ = [
    "bomb_gauntlet_wall",
    "climb_from_lip",
    "climb_open_air_ibj",
    "climb_to_ledge",
    "drift_to_bomb_wall",
    "ibj_cycle",
    "ibj_first_bomb",
    "play_landing_to_gauntlet",
]
