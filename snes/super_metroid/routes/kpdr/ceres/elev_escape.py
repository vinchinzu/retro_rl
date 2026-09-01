"""Ceres elevator shaft climb after Falling → ship leave (WRAM-reactive).

Inbound settle waits for ordinary control (gs==8). A spin jump through the
previous door preserves a y≈628 fast phase. TAS air-turns LEFT and latches
pose 132 at x≈204 y≈597, left of the Ceres-door overlay ($E23F at 224/232).
One TAS wall-jump trajectory; missed plants raise. Ship handoff remains
right-wall KB → LEFT+A.
"""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from super_metroid.combat.enemies import CERES_DOOR_ID, list_enemies
from super_metroid.combat.enemies.species import enemy_overlaps
from super_metroid.ram import GS_CERES_LEAVE, GS_ORDINARY
from super_metroid.routes.controller_common import POSE_WALL_LATCH
from super_metroid.routes.kpdr.ceres.geometry import (
    _CERES_ELEV_BOTTOM_Y,
    _CERES_ELEV_SHIP_X,
    _CERES_ELEV_SHIP_Y,
    _CERES_ELEV_TOP_X,
    _CERES_ELEV_TOP_Y,
)
from super_metroid.routes.kpdr.room_ids import ROOM_CERES_ELEVATOR
from super_metroid.routes.runtime import ActionSpan, RouteSession
from super_metroid.routes.skills.geometry import (
    CROUCH_POSES,
    LAND_POSES,
    POSE_KNOCKBACK,
    SPIN_POSES,
    STAND_LOCOMOTION_POSES,
)
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.takeoff import walk_toward_x

_POSE_WALL_LATCH_LEFT = 131
# 475→363: LEFT into the wall, RIGHT away; latch pose 131.
_CERES_475_TO_363_INTO = "LEFT"
_CERES_475_TO_363_AWAY = "RIGHT"
# Last measured product elev_to_landing. Overwrite only after a live TAS bench.
CERES_ELEV_BENCH_FRAMES = 3239


def _ceres_elev_ship_band(state) -> bool:
    """Grounded on ship pad (product leave ~x145 y75 pose 2/10 → gs 32)."""
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and int(state.game_state) == GS_ORDINARY
        and int(state.samus_y) <= _CERES_ELEV_SHIP_Y
        and abs(int(state.velocity_y)) <= 1
    )


def ship_pad_action(state) -> tuple[str, ...]:
    """Walk through the Ceres pad x that starts gs 32."""
    return walk_toward_x(int(state.samus_x), _CERES_ELEV_SHIP_X)


def _ceres_elev_leaving(state) -> bool:
    """True ship leave: left the elevator, or Ceres success / Zebes load.

    Inbound Falling→elev door (gs 9/11, often fake y≈139) is not leave.
    """
    if int(state.room_id) != ROOM_CERES_ELEVATOR:
        return True
    return int(state.game_state) in GS_CERES_LEAVE


def _ceres_elev_top_seat(state) -> bool:
    """s10 land / right-wall KB — the only shaft→ship handoff."""
    if int(state.room_id) != ROOM_CERES_ELEVATOR:
        return False
    if int(state.game_state) != GS_ORDINARY:
        return False
    if abs(int(state.samus_y) - _CERES_ELEV_TOP_Y) > 16:
        return False
    x = int(state.samus_x)
    if int(state.pose) in POSE_KNOCKBACK and x >= _CERES_ELEV_TOP_X - 30:
        return True
    return x >= _CERES_ELEV_TOP_X - 20


def _ceres_elev_entry_action(state) -> tuple[str, ...] | None:
    """Inputs until the shaft may start. ``None`` means the window is ready.

    Hold LEFT only while the Falling door is still settling. Ordinary y≈628
    spin is the precise-WJ phase — walking LEFT there leaves x=216 and dumps
    the well. A floor remap (y≈651) is a failed product entry.
    """
    if _ceres_elev_leaving(state):
        return None
    if int(state.room_id) != ROOM_CERES_ELEVATOR:
        return ()
    if int(state.game_state) != GS_ORDINARY:
        # Door-transition inputs are not movement frames. Preserve the
        # predecessor's pose-25 rise; the first ordinary frame owns the WJ.
        return ()
    if _ceres_fast_entry_window(state):
        return None
    if int(state.samus_y) >= _CERES_ELEV_BOTTOM_Y - 20:
        return None
    return ()


def _ceres_reactive_elev_climb(session: RouteSession) -> None:
    """Elev after Falling → ship leave through one TAS wall-jump climb.

    The predecessor's late door jump remaps to y≈628 with its spin phase intact.
    Missed fast-entry, 475, or 363 plants raise; there is no checkpoint recover.
    """
    session.wait_until(
        lambda s: s.room_id == ROOM_CERES_ELEVATOR,
        timeout=300,
        reason="ceres_elev_door",
    )
    for _ in range(160):
        names = _ceres_elev_entry_action(session.state)
        if names is None:
            break
        session.step(
            buttons(*names) if names else idle_action(),
            "ceres_elev_entry",
        )
    if not _ceres_fast_entry_window(session.state):
        raise TimeoutError(f"ceres elev fast entry missed: {session.state}")
    overlay = _ceres_door_blocks_wj(session)
    if not _ceres_entry_to_475(session):
        raise TimeoutError(
            f"ceres 475 plant missed overlay={overlay}: {session.state}"
        )
    if not _ceres_475_to_363(session):
        raise TimeoutError(f"ceres 363 plant missed: {session.state}")
    _ceres_elev_top_to_ship(session)


def _ceres_planted_at(state, target_y: int, *, slack: int = 18) -> bool:
    return (
        int(state.game_state) == GS_ORDINARY
        and abs(int(state.samus_y) - target_y) <= slack
        and abs(int(state.velocity_y)) <= 1
        and (
            int(state.pose) in STAND_LOCOMOTION_POSES
            or int(state.pose) in LAND_POSES
        )
    )


def _ceres_fast_entry_window(state) -> bool:
    """Natural door-jump phase from which the y=475 wall jump is repeatable."""
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and int(state.game_state) == GS_ORDINARY
        and 210 <= int(state.samus_x) <= 220
        and 624 <= int(state.samus_y) <= 641
        and int(state.pose) in SPIN_POSES
    )


def _ceres_any_wall_latch(state) -> bool:
    """Right-wall latch 132 or left-wall latch 131 (TAS 475→363 contact)."""
    return int(state.pose) in (POSE_WALL_LATCH, _POSE_WALL_LATCH_LEFT)


def _ceres_entry_to_475(session: RouteSession) -> bool:
    """Two measured wall jumps from the natural door phase to y=475."""
    if not _ceres_fast_entry_window(session.state):
        return False

    # Reach the lower edge of $E23F, release A for one frame, then wall-jump
    # at x≈208/y620. This predecessor is lower than Sniq's native entry, so
    # the first wall jump needs a 40-frame hold before the reverse.
    for names, frames in (
        (("LEFT", "A"), 1),
        (("A",), 1),
        (("LEFT", "A"), 1),
        (("A",), 3),
        (("LEFT",), 1),
    ):
        session.span(ActionSpan(names, frames, "ceres_elev_direct_entry"))

    first_latch = False
    for _ in range(40):
        session.step(buttons("LEFT", "A"), "ceres_elev_direct_first_wj")
        first_latch = first_latch or int(session.state.pose) == POSE_WALL_LATCH
    if not first_latch:
        return False

    # Unlatch at x≈165/y524, reverse for one frame, and catch the left wall.
    # The second jump clears the underside and plants x≈164/y475 naturally.
    tail = (
        (("RIGHT", "A"), 1),
        (("RIGHT",), 1),
        (("RIGHT", "A"), 1),
        (("LEFT", "A"), 7),
        (("DOWN", "RIGHT", "A"), 1),
        (("LEFT",), 1),
        ((), 1),
        (("LEFT",), 1),
        (("LEFT", "A"), 3),
    )
    second_latch = False
    for names, frames in tail:
        for _ in range(frames):
            session.step(
                buttons(*names) if names else idle_action(),
                "ceres_elev_direct_second_wj",
            )
            second_latch = second_latch or int(session.state.pose) == _POSE_WALL_LATCH_LEFT
            if _ceres_planted_at(session.state, 475, slack=4):
                return second_latch
    return False


def _ceres_475_to_363(session: RouteSession) -> bool:
    """Left-wall jump from the planted 475 seat onto y=363.

    TAS (lsnes sniq_100): plant x=163 pose 165/10, LEFT+A, pose 131 at
    y≈404, land x=156 y=363. Do not RIGHT-run from this seat.
    """
    if not _ceres_planted_at(session.state, 475, slack=4):
        return False
    spans = (
        ((_CERES_475_TO_363_INTO, "A"), 3),
        (("A",), 2),
        ((_CERES_475_TO_363_AWAY, "A"), 1),
        ((_CERES_475_TO_363_INTO, "A"), 1),
        ((_CERES_475_TO_363_AWAY, "A"), 1),
        (("A",), 9),
        ((_CERES_475_TO_363_AWAY, "A"), 1),
        (("A",), 1),
        ((_CERES_475_TO_363_INTO, _CERES_475_TO_363_AWAY), 1),
        ((_CERES_475_TO_363_INTO, _CERES_475_TO_363_AWAY, "A"), 1),
        (("A",), 5),
        (("A", "X"), 1),
        (("DOWN", "A"), 1),
        ((_CERES_475_TO_363_AWAY,), 1),
        ((), 1),
        ((_CERES_475_TO_363_AWAY,), 1),
    )
    latched = False
    for names, frames in spans:
        for _ in range(frames):
            session.step(
                buttons(*names) if names else idle_action(),
                "ceres_elev_475_363",
            )
            latched = latched or int(session.state.pose) == _POSE_WALL_LATCH_LEFT
            if _ceres_planted_at(session.state, 363, slack=4):
                return latched
    return False


def _ceres_door_blocks_wj(session: RouteSession) -> bool:
    """True when a Ceres-door overlay occupies the right-wall WJ contact."""
    st = session.state
    sx, sy = int(st.samus_x), int(st.samus_y)
    for enemy in list_enemies(session):
        if int(enemy.enemy_id) != CERES_DOOR_ID:
            continue
        if enemy_overlaps(enemy, sx, sy, samus_r=16):
            return True
    return False


def _ceres_elev_top_to_ship(session: RouteSession) -> None:
    """From s10 land (~y171) force right-wall pose 137 then LEFT+A to ship pad.

    Product (open-loop s10 tail): land x211 y171 pose 9 → idle → pose 137 →
    LEFT+A 38 peaks ~y65 → LEFT 25 walks to x≈145 y75 → Ceres success (gs 32).
    Already on the pad still walks through ``_CERES_ELEV_SHIP_X`` until gs 32.
    """
    if session.state.room_id != ROOM_CERES_ELEVATOR:
        return
    if _ceres_elev_leaving(session.state):
        return

    if int(session.state.pose) in CROUCH_POSES:
        for _ in range(10):
            if int(session.state.pose) not in CROUCH_POSES:
                break
            session.step(buttons("UP"), "ceres_elev_uncrouch")

    # Pose 166 needs ~12f LEFT to ordinary locomotion. Shorter LEFT reaches
    # the wall with a weak boost that falls back to y267.
    if int(session.state.pose) in LAND_POSES:
        for _ in range(12):
            session.step(buttons("LEFT"), "ceres_elev_top_stand")

    for _ in range(4):
        if int(session.state.pose) in STAND_LOCOMOTION_POSES:
            break
        session.step(buttons("LEFT"), "ceres_elev_top_stand")

    if not _ceres_elev_ship_band(session.state) and int(session.state.samus_y) < 280:
        for _ in range(40):
            st = session.state
            if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
                return
            if is_knockback(st):
                break
            session.step(buttons("RIGHT"), "ceres_elev_top_seek_kb")

        for _ in range(3):
            st = session.state
            if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
                return
            if not is_knockback(st) and int(st.samus_x) >= _CERES_ELEV_TOP_X - 2:
                session.step(idle_action(), "ceres_elev_top_kb")
                continue
            if is_knockback(st):
                session.step(idle_action(), "ceres_elev_top_kb")
            else:
                break

        if is_knockback(session.state):
            for _ in range(38):
                st = session.state
                if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
                    return
                session.step(buttons("LEFT", "A"), "ceres_elev_top_boost")

            for _ in range(80):
                st = session.state
                if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
                    return
                session.step(buttons("LEFT"), "ceres_elev_top_walk")

    for _ in range(80):
        st = session.state
        if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
            return
        names = ship_pad_action(st)
        session.step(
            buttons(*names) if names else idle_action(),
            "ceres_elev_ship",
        )

    if (
        session.state.room_id == ROOM_CERES_ELEVATOR
        and not _ceres_elev_leaving(session.state)
    ):
        raise TimeoutError(f"ceres elev ship leave failed: {session.state}")


__all__ = [
    "CERES_ELEV_BENCH_FRAMES",
    "ship_pad_action",
    "_ceres_elev_ship_band",
    "_ceres_elev_leaving",
    "_ceres_elev_top_seat",
    "_ceres_fast_entry_window",
    "_ceres_475_to_363",
    "_ceres_door_blocks_wj",
    "_ceres_any_wall_latch",
    "_ceres_reactive_elev_climb",
    "_ceres_elev_top_to_ship",
]
