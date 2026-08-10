"""Hi-Jump platform climb in Business Center (elevator return).

Continuous natural-entry hardening keeps pure hop geometry and adds:
- Standing/vy=0 gates before the 1339→1227 charge (no lip walk-off mid-air).
- Longer 987→907 run-up for colder continuous landings on 987.
- Landing checks after each hop so failures raise early with y/x evidence.
- Elevator platform y=683 gate before holding UP.
"""

from __future__ import annotations

from pathlib import Path

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import (
    ITEM_HI_JUMP,
    ROOM_BUSINESS,
    ROOM_WAREHOUSE,
)
from super_metroid.routes.runtime import ControllerSession

# Standing / crouch / turn poses that can start a charged Hi-Jump.
_STANDING = frozenset({1, 2, 9, 10, 25, 26, 27, 28, 37, 38, 137, 138})

_CLIMB_DUMP_DIR = (
    Path(__file__).resolve().parents[2]
    / "custom_integrations"
    / "SuperMetroid-Snes"
    / "scratch"
)

def _maybe_dump_climb_state(session: ControllerSession, label: str) -> None:
    """Best-effort save-state dump when the session owns an emulator env."""
    env = getattr(session, "env", None)
    if env is None:
        return
    try:
        from retro_harness.env import write_state_bytes

        path = _CLIMB_DUMP_DIR / f"{label}.state"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_state_bytes(path, env.em.get_state())
    except Exception:
        # Climb dump is diagnostic only — never abort the route for it.
        return

def _wait_standing_y(
    session: ControllerSession,
    y: int,
    *,
    timeout: int = 90,
    reason: str = "business_standing",
) -> SuperMetroidState:
    """Idle until Samus is standing on platform ``y`` with zero vertical speed."""
    for _ in range(timeout):
        state = session.state
        if (
            state.samus_y == y
            and state.pose in _STANDING
            and state.velocity_y == 0
        ):
            return state
        hold(session, 1, reason=reason)
    raise TimeoutError(f"{reason}: expected y={y}: {session.state}")

def _business_high_jump_platforms(
    session: ControllerSession,
    *,
    runup_907: int = 14,
    pos_1339: int = 84,
    bound_floor_left: bool = False,
) -> None:
    """Bottom Business Center floor → center elevator (Hi-Jump route).

    ``runup_907``: RIGHT+B frames before the 987→907 hop. Continuous natural
    entry needs 14; pure probe prefers 8 (used on floor re-climb fallback).

    ``pos_1339``: LEFT walk target on y1339 before 1227 hop (pure=84;
    continuous Ice floor pin needs ~90 — rr-kxge offline grid).

    ``bound_floor_left``: when True, soft-bound LEFT setup near the HJ door
    (continuous Ice retries only — pure open-loop stays unbound).
    """
    # Four setup jumps land on the first left platform (~y=1339).
    # If already standing there (pure mid-climb states), skip the open-loop setup.
    unmorph(session)
    already = (
        session.state.samus_y == 1339
        and session.state.pose in _STANDING
        and session.state.velocity_y == 0
    )
    if not already:
        for direction in ("LEFT", "LEFT", "RIGHT"):
            hold(session, 12, reason="business_climb_release")
            if not bound_floor_left:
                hold(session, 85, direction, "B", "A", reason="business_climb_setup")
            else:
                for _ in range(85):
                    st = session.state
                    if st.room_id != ROOM_BUSINESS:
                        raise TimeoutError(
                            f"business_climb_setup: left Business: {session.state}"
                        )
                    if (
                        direction == "LEFT"
                        and int(st.samus_y) >= 1300
                        and int(st.samus_x) <= 80
                    ):
                        hold(
                            session,
                            1,
                            "RIGHT",
                            "B",
                            "A",
                            reason="business_climb_setup_bound",
                        )
                    else:
                        hold(
                            session,
                            1,
                            direction,
                            "B",
                            "A",
                            reason="business_climb_setup",
                        )
            if session.state.room_id != ROOM_BUSINESS:
                raise TimeoutError(
                    f"business_climb_setup: left Business: {session.state}"
                )
            hold(session, 30, reason="business_climb_setup_land")
            if session.state.room_id != ROOM_BUSINESS:
                raise TimeoutError(
                    f"business_climb_setup_land: left Business: {session.state}"
                )

    # y1339 → y1227.
    # Continuous failure mode: walk left while not grounded → lip fall (pose 41)
    # so A never charges. Gate on standing; pure needs ~x84 for the LEFT+A arc,
    # but stop at 86 if the second-climb path is edgy after floor recover.
    unmorph(session)
    hold(session, 12, reason="business_1339_settle")
    _wait_standing_y(session, 1339, timeout=60, reason="business_1339_ground")
    for _ in range(80):
        state = session.state
        if state.samus_x <= pos_1339:
            break
        if state.samus_y != 1339 or state.pose not in _STANDING:
            hold(session, 1, reason="business_1339_replant")
            if session.state.samus_y != 1339:
                # Walked off — re-setup first platform and re-enter this hop.
                unmorph(session)
                for direction in ("LEFT", "LEFT", "RIGHT"):
                    hold(session, 12, reason="business_climb_release")
                    hold(
                        session, 85, direction, "B", "A", reason="business_climb_setup"
                    )
                    hold(session, 30, reason="business_climb_setup_land")
                _wait_standing_y(
                    session, 1339, timeout=60, reason="business_1339_ground_retry"
                )
                break
            continue
        hold(session, 1, "LEFT", reason="business_1339_position")
    hold(session, 4, "RIGHT", reason="business_1339_brake")
    hold(session, 8, reason="business_1339_release")
    _wait_standing_y(session, 1339, timeout=40, reason="business_1339_prejump")
    for frame in range(120):
        if frame < 14:
            buttons = ("LEFT", "A")
        elif frame < 24:
            buttons = ("A",)
        else:
            buttons = ("RIGHT", "A")
        state = hold(session, 1, *buttons, reason="business_to_1227")
        if frame > 45 and state.samus_y == 1227 and state.samus_x >= 120:
            break
    hold(session, 3, "LEFT", reason="business_1227_brake")
    hold(session, 12, reason="business_1227_settle")
    _wait_standing_y(session, 1227, timeout=50, reason="business_1227_land")

    # y1227 → right platform y1147.
    unmorph(session)
    hold(session, 15, reason="business_1227_release")
    for _ in range(80):
        if session.state.samus_x <= 105:
            break
        hold(session, 1, "LEFT", reason="business_1227_back")
    hold(session, 4, "RIGHT", reason="business_1227_brake2")
    hold(session, 4, reason="business_1227_run_release")
    hold(session, 8, "RIGHT", "B", reason="business_1227_runup")
    for frame in range(140):
        buttons = ("RIGHT", "B", "A") if frame < 90 else ("LEFT", "A")
        state = hold(session, 1, *buttons, reason="business_to_1147")
        if frame > 88 and state.samus_y == 1147 and state.samus_x >= 192:
            break
    hold(session, 3, "LEFT", reason="business_1147_brake")
    hold(session, 12, reason="business_1147_settle")
    _wait_standing_y(session, 1147, timeout=50, reason="business_1147_land")

    # y1147 → center platform y1067.
    unmorph(session)
    hold(session, 16, reason="business_1147_release")
    for frame in range(150):
        buttons = ("LEFT", "B", "A") if frame < 85 else ("RIGHT", "A")
        state = hold(session, 1, *buttons, reason="business_to_1067")
        if frame > 100 and state.samus_y == 1067 and 95 <= state.samus_x <= 160:
            break
    hold(session, 30, reason="business_1067_settle")
    _wait_standing_y(session, 1067, timeout=50, reason="business_1067_land")

    # y1067 → y987 through the left edge of the overhead platform.
    unmorph(session)
    hold(session, 12, reason="business_1067_release")
    for _ in range(80):
        if session.state.samus_x <= 92:
            break
        hold(session, 1, "LEFT", reason="business_1067_position")
    hold(session, 4, "RIGHT", reason="business_1067_brake")
    hold(session, 8, reason="business_1067_jump_release")
    for frame in range(100):
        buttons = ("A",) if frame < 14 else ("RIGHT", "B", "A")
        state = hold(session, 1, *buttons, reason="business_to_987")
        if frame > 25 and state.samus_y == 987 and state.pose in (1, 2, 9, 10):
            break
    # Landing is on the extreme left pixel of the three-block platform;
    # nudge inward instead of braking back off its edge.
    hold(session, 4, "RIGHT", reason="business_987_brake")
    hold(session, 12, reason="business_987_settle")
    _wait_standing_y(session, 987, timeout=50, reason="business_987_land")
    # Capture continuous natural-entry at this hop for offline iteration.
    _maybe_dump_climb_state(session, "business_987_pre_907")

    # y987 → right platform y907.
    # Continuous natural-entry needs ~14f run-up (8/12 fall past the right
    # ledge). Pure probe prefers 8 (passed as runup_907 on re-climb fallback).
    unmorph(session)
    hold(session, 12, reason="business_987_release")
    _wait_standing_y(session, 987, timeout=40, reason="business_987_pre_907")
    hold(session, runup_907, "RIGHT", "B", reason="business_987_runup")
    for frame in range(100):
        state = hold(session, 1, "RIGHT", "B", "A", reason="business_to_907")
        if frame > 35 and state.samus_y == 907 and state.samus_x >= 160:
            break
    for _ in range(60):
        if session.state.samus_x <= 165:
            break
        hold(session, 1, "LEFT", reason="business_907_brake")
    hold(session, 2, "RIGHT", reason="business_907_brake")
    hold(session, 12, reason="business_907_settle")
    try:
        _wait_standing_y(session, 907, timeout=50, reason="business_907_land")
    except TimeoutError:
        _maybe_dump_climb_state(session, "business_907_miss")
        raise

    # y907 → center y843.
    unmorph(session)
    hold(session, 12, reason="business_907_release")
    for _ in range(80):
        if session.state.samus_x >= 205:
            break
        hold(session, 1, "RIGHT", reason="business_907_back")
    hold(session, 3, "LEFT", reason="business_907_brake2")
    hold(session, 5, reason="business_907_run_release")
    hold(session, 8, "LEFT", "B", reason="business_907_runup")
    for frame in range(90):
        state = hold(session, 1, "LEFT", "B", "A", reason="business_to_843")
        if frame > 35 and state.samus_y == 843 and 108 <= state.samus_x <= 160:
            break
    hold(session, 2, "RIGHT", reason="business_843_brake")
    hold(session, 12, reason="business_843_settle")
    _wait_standing_y(session, 843, timeout=50, reason="business_843_land")

    # y843 → left y779.
    unmorph(session)
    hold(session, 12, reason="business_843_release")
    for _ in range(80):
        if session.state.samus_x >= 145:
            break
        hold(session, 1, "RIGHT", reason="business_843_position")
    hold(session, 3, "LEFT", reason="business_843_brake2")
    hold(session, 6, reason="business_843_jump_release")
    for frame in range(90):
        buttons = ("A",) if frame < 10 else ("LEFT", "B", "A")
        state = hold(session, 1, *buttons, reason="business_to_779")
        if frame > 25 and state.samus_y == 779 and state.samus_x <= 115:
            break
    hold(session, 2, "RIGHT", reason="business_779_brake")
    hold(session, 12, reason="business_779_settle")
    _wait_standing_y(session, 779, timeout=50, reason="business_779_land")

    # y779 → center elevator y683.
    # Continuous dump: walk to x≤76 steps off the left lip (miny≈771, fall to
    # y907). Offline: setup band x≤80 lands on the elevator platform.
    unmorph(session)
    hold(session, 12, reason="business_779_release")
    _wait_standing_y(session, 779, timeout=40, reason="business_779_pre_elev")
    for _ in range(80):
        state = session.state
        if state.samus_x <= 80:
            break
        if state.samus_y != 779 or state.pose not in _STANDING:
            hold(session, 1, reason="business_779_replant")
            if session.state.samus_y != 779:
                raise TimeoutError(
                    f"business_779_position: walked off platform: {session.state}"
                )
            continue
        hold(session, 1, "LEFT", reason="business_779_position")
    hold(session, 3, "RIGHT", reason="business_779_brake2")
    hold(session, 6, reason="business_779_jump_release")
    _wait_standing_y(session, 779, timeout=30, reason="business_779_prejump")
    for frame in range(120):
        buttons = ("A",) if frame < 18 else ("RIGHT", "B", "A")
        state = hold(session, 1, *buttons, reason="business_to_elevator")
        if frame > 45 and state.samus_y == 683 and 95 <= state.samus_x <= 160:
            break
    hold(session, 2, "LEFT", reason="business_elevator_brake")
    hold(session, 12, reason="business_elevator_settle")
    _wait_standing_y(session, 683, timeout=50, reason="business_elevator_land")
    if not (95 <= session.state.samus_x <= 160):
        for _ in range(40):
            x = session.state.samus_x
            if 100 <= x <= 150:
                break
            hold(
                session,
                1,
                "LEFT" if x > 150 else "RIGHT",
                reason="business_elevator_center",
            )
        hold(session, 8, reason="business_elevator_center_settle")
        _wait_standing_y(session, 683, timeout=40, reason="business_elevator_recenter")

def _fall_to_business_floor(session: ControllerSession) -> None:
    """Drop from mid-stack platforms to the Business Center floor climb anchor."""
    direction = "RIGHT"
    for frame in range(600):
        state = session.state
        if (
            state.pose in _STANDING
            and state.velocity_y == 0
            and state.samus_y >= 1405
        ):
            break
        if state.samus_x <= 50:
            direction = "RIGHT"
        elif state.samus_x >= 210:
            direction = "LEFT"
        phase = frame % 70
        buttons = (direction, "B") if phase < 45 else (direction, "B", "A")
        hold(session, 1, *buttons, reason="business_floor_recover")
    else:
        raise TimeoutError(f"business_floor_recover: {session.state}")
    # Match hj_return_business climb anchor (~x88+) so setup jumps re-acquire.
    for _ in range(80):
        state = session.state
        if state.samus_x >= 88:
            break
        hold(session, 1, "RIGHT", reason="business_floor_anchor")
    hold(session, 4, "LEFT", reason="business_floor_anchor_brake")
    hold(session, 15, reason="business_floor_recover_settle")

def play_business_to_warehouse(session: ControllerSession) -> SuperMetroidState:
    """Hi-Jump-assisted Business Center climb and elevator to Warehouse."""
    require_room(session, ROOM_BUSINESS, "business_to_warehouse")
    if not session.state.collected_items & ITEM_HI_JUMP:
        raise RuntimeError(
            f"business_to_warehouse: Hi-Jump not collected: {session.state}"
        )
    try:
        # Continuous natural-entry: 14f run-up (verified continuous kraid_entry).
        _business_high_jump_platforms(session, runup_907=14)
    except TimeoutError:
        # Pure probe prefers 8f; also a second continuous attempt from floor.
        _maybe_dump_climb_state(session, "business_climb_retry_before")
        _fall_to_business_floor(session)
        _business_high_jump_platforms(session, runup_907=8)
    if session.state.samus_y != 683 or session.state.pose not in _STANDING:
        raise TimeoutError(
            f"business_to_warehouse: not on elevator platform: {session.state}"
        )
    for _ in range(1000):
        state = hold(session, 1, "UP", reason="business_elevator_up")
        if state.room_id == ROOM_WAREHOUSE:
            break
    else:
        raise TimeoutError(f"business_to_warehouse: elevator failed: {state}")
    state = wait_ordinary_room(
        session, ROOM_WAREHOUSE, settle_frames=360, label="business_to_warehouse"
    )
    # Let the Warehouse platform finish rising, then step back to the same
    # upper-left anchor used by the natural East Tunnel entry.
    hold(session, 30, reason="warehouse_elevator_top")
    for _ in range(160):
        state = session.state
        if state.samus_x <= 40 and state.samus_y <= 145:
            break
        hold(session, 1, "LEFT", reason="warehouse_elevator_exit")
    hold(session, 30, reason="warehouse_elevator_exit_settle")
    return session.state
