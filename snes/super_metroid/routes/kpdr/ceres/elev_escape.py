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
    _CERES_ELEV_LEDGE_Y,
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
_CERES_STEAM_DBOOST_FRAMES = 24
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
    """Elev after Falling → ship leave from a validated rising handoff.

    Native TAS-height entries use the direct 475 wall jump. The product's later
    doorway jump remaps rising at y≈651; settle it, seat y=571, then rejoin the
    measured 475→363→267→171 checkpoint chain.
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

    direct_475 = int(session.state.samus_y) <= 641 and _ceres_entry_to_475(session)
    if direct_475:
        _ceres_475_to_363(session)
    else:
        for _ in range(120):
            st = session.state
            if (
                int(st.samus_y) >= _CERES_ELEV_BOTTOM_Y - 20
                and abs(int(st.velocity_y)) <= 1
                and (
                    int(st.pose) in STAND_LOCOMOTION_POSES
                    or int(st.pose) in LAND_POSES
                )
            ):
                break
            session.step(idle_action(), "ceres_elev_bottom_settle")
        _ceres_seat_ledge(session)

    if not _ceres_checkpoint_shaft(session):
        raise TimeoutError(f"ceres checkpoint shaft missed: {session.state}")
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


def _ceres_at_checkpoint(state, target_y: int, *, slack: int = 18) -> bool:
    """Grounded checkpoint, including a debris-knockback plant."""
    return (
        int(state.game_state) == GS_ORDINARY
        and abs(int(state.samus_y) - target_y) <= slack
        and abs(int(state.velocity_y)) <= 1
        and (
            int(state.pose) in STAND_LOCOMOTION_POSES
            or int(state.pose) in LAND_POSES
            or int(state.pose) in POSE_KNOCKBACK
        )
    )


def _ceres_on_elev_ledge(state) -> bool:
    """Planted on the y=571 recovery ledge."""
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and _ceres_planted_at(state, _CERES_ELEV_LEDGE_Y, slack=6)
        and int(state.samus_x) < 200
    )


def _ceres_seat_ledge(session: RouteSession) -> None:
    """Settle a lower remap and recover to the measured y=571 ledge."""
    if (
        int(session.state.game_state) == GS_ORDINARY
        and int(session.state.samus_y) >= _CERES_ELEV_BOTTOM_Y - 30
    ):
        for _ in range(4):
            session.step(idle_action(), "ceres_elev_bottom_plant")
        session.span(ActionSpan(("LEFT", "A"), 70, "ceres_elev_ledge_jump"))

    for _ in range(400):
        st = session.state
        if _ceres_elev_leaving(st):
            return
        if _ceres_on_elev_ledge(st) and int(st.samus_x) <= 55:
            return
        if int(st.samus_y) <= _CERES_ELEV_SHIP_Y:
            return
        if int(st.game_state) != GS_ORDINARY:
            session.step(idle_action(), "ceres_elev_ledge_wait_gs")
        elif is_knockback(st):
            session.step(idle_action(), "ceres_elev_ledge_kb")
        elif int(st.samus_x) < 90 and int(st.samus_y) >= _CERES_ELEV_BOTTOM_Y - 20:
            session.step(buttons("RIGHT", "A"), "ceres_elev_pit")
        elif int(st.samus_y) > _CERES_ELEV_LEDGE_Y + 16:
            session.step(buttons("LEFT", "A"), "ceres_elev_ledge_recover")
        elif _ceres_on_elev_ledge(st):
            session.step(buttons("LEFT"), "ceres_elev_ledge_walk")
        else:
            session.step(idle_action(), "ceres_elev_ledge_settle")


def _ceres_fast_entry_window(state) -> bool:
    """Validated rising door-jump phase for direct or checkpoint recovery."""
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and int(state.game_state) == GS_ORDINARY
        and 210 <= int(state.samus_x) <= 220
        # The late product jump first remaps at y=651; accept it only while it
        # still carries the TAS rise, horizontal speed, and damage boost.
        and 624 <= int(state.samus_y) <= 655
        and int(state.pose) in SPIN_POSES
        and int(state.vertical_direction) == 1
        and int(state.velocity_y) > 0
        and int(state.momentum_x) >= 2
        and int(state.invincibility_timer) > 0
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


def _ceres_steam_dboost(session: RouteSession, side: str) -> None:
    """Hold the intended hop through a shaft-steam knockback."""
    for _ in range(_CERES_STEAM_DBOOST_FRAMES):
        st = session.state
        if (
            int(st.knockback_timer) <= 0
            and int(st.movement_type) not in (10, 25)
            and not is_knockback(st)
        ):
            return
        session.step(buttons(side, "A"), "ceres_elev_steam_dboost")


def _ceres_checkpoint_hop(
    session: RouteSession,
    *,
    side: str,
    runup: int,
    target_y: int,
    start_x: int | None = None,
) -> bool:
    """Replay one measured platform hop and require its natural landing."""
    checkpoints = (571, 475, 363, 267, 171)
    for _ in range(16):
        pose = int(session.state.pose)
        if pose in STAND_LOCOMOTION_POSES or pose in LAND_POSES:
            break
        if pose in CROUCH_POSES:
            session.step(buttons("UP"), "ceres_elev_checkpoint_uncrouch")
        else:
            session.step(buttons("LEFT"), "ceres_elev_checkpoint_stand")

    if start_x is not None:
        for _ in range(40):
            names = walk_toward_x(int(session.state.samus_x), start_x, slack=4)
            if not names:
                break
            session.step(buttons(*names), "ceres_elev_checkpoint_position")

    for _ in range(runup):
        session.step(buttons(side, "B"), "ceres_elev_checkpoint_runup")
    for _ in range(40):
        session.step(buttons(side, "B", "A"), "ceres_elev_checkpoint_jump")
        if int(session.state.knockback_timer) > 0:
            boost_side = "LEFT" if side == "RIGHT" else "RIGHT"
            _ceres_steam_dboost(session, boost_side)
            break

    for _ in range(220):
        st = session.state
        if _ceres_elev_leaving(st) or _ceres_elev_ship_band(st):
            return True
        if is_knockback(st):
            _ceres_steam_dboost(session, side)
            st = session.state
        if _ceres_planted_at(st, target_y):
            return True
        seats = [y for y in checkpoints if _ceres_at_checkpoint(st, y)]
        if seats:
            nearest = min(seats, key=lambda y: abs(int(st.samus_y) - y))
            return nearest == target_y
        session.step(idle_action(), "ceres_elev_checkpoint_land")
    return _ceres_planted_at(session.state, target_y)


def _ceres_checkpoint_shaft(session: RouteSession) -> bool:
    """Climb the measured y571→475→363→267→171 checkpoint chain."""
    recipes = {
        571: ("RIGHT", 0, 475, None),
        475: ("RIGHT", 4, 363, None),
        363: ("LEFT", 0, 267, None),
        267: ("RIGHT", 0, 171, 131),
    }
    history: list[str] = []
    for _ in range(16):
        if _ceres_at_checkpoint(session.state, _CERES_ELEV_TOP_Y):
            return True
        seat_y = next(
            (y for y in recipes if _ceres_at_checkpoint(session.state, y)),
            None,
        )
        if seat_y is None:
            if int(session.state.samus_y) >= _CERES_ELEV_BOTTOM_Y - 20:
                history.append(f"floor:{int(session.state.samus_y)}")
                _ceres_seat_ledge(session)
                continue
            raise TimeoutError(
                f"ceres elev lost checkpoint history={history}: {session.state}"
            )

        if seat_y == 475 and int(session.state.samus_x) >= 140:
            landed = _ceres_475_to_363(session)
            target_y = 363
        else:
            side, runup, target_y, start_x = recipes[seat_y]
            if seat_y == 363 and int(session.state.samus_x) < 100:
                side, runup = "RIGHT", 4
            landed = _ceres_checkpoint_hop(
                session,
                side=side,
                runup=runup,
                target_y=target_y,
                start_x=start_x,
            )
        history.append(
            f"{seat_y}->{target_y}:{int(landed)}@"
            f"{int(session.state.samus_x)},{int(session.state.samus_y)},"
            f"p{int(session.state.pose)}"
        )
    raise TimeoutError(
        f"ceres elev checkpoint retries exhausted history={history}: {session.state}"
    )


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
