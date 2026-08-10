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

def _on_business_1339(state: SuperMetroidState) -> bool:
    return (
        int(state.samus_y) == 1339
        and int(state.pose) in _STANDING
        and int(state.velocity_y) == 0
    )


def _recenter_business_floor(session: ControllerSession, *, label: str) -> None:
    """Walk to mid-right floor band before LEFT-heavy setup (avoid HJ door)."""
    unmorph(session)
    for _ in range(160):
        st = session.state
        if st.room_id != ROOM_BUSINESS:
            raise TimeoutError(f"{label}: left Business while recentering: {st}")
        x = int(st.samus_x)
        y = int(st.samus_y)
        grounded = int(st.velocity_y) == 0 and int(st.pose) in _STANDING
        if y < 1350:
            # Mid-air / mid-platform — let gravity settle; bias RIGHT away from door.
            hold(session, 1, "RIGHT", "B", reason=f"{label}_recenter_air")
            continue
        if grounded and 170 <= x <= 230:
            hold(session, 8, reason=f"{label}_recenter_settle")
            return
        if x < 170:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_recenter_r")
        elif x > 230 and grounded:
            hold(session, 1, "LEFT", reason=f"{label}_recenter_l")
        else:
            hold(session, 1, reason=f"{label}_recenter_idle")
    # Soft fail — setup may still work from imperfect pin.


def _setup_business_to_1339(
    session: ControllerSession,
    *,
    bound_floor_left: bool = True,
    attempts: int = 4,
) -> None:
    """Floor / mid → standing y1339 without kissing the left HJ door.

    Continuous Ice natural entry is enemy-noisy: open-loop LEFT+B+A from a
    left-biased pin exits into ``0xAA41``. Always soft-bound the door lip
    (x≲48 on the floor band) and re-center between attempts. ``bound_floor_left``
    adds an earlier RIGHT bias (x≲72) used by continuous Ice retries.
    """
    unmorph(session)
    if _on_business_1339(session.state):
        return

    door_x = 48
    soft_x = 72 if bound_floor_left else door_x

    for attempt in range(attempts):
        if session.state.room_id != ROOM_BUSINESS:
            raise TimeoutError(
                f"business_climb_setup: left Business: {session.state}"
            )
        # Mid-right start keeps the second LEFT from eating the HJ door.
        if int(session.state.samus_y) >= 1350:
            _recenter_business_floor(session, label="business_climb_setup")

        for direction in ("LEFT", "LEFT", "RIGHT"):
            hold(session, 12, reason="business_climb_release")
            for _ in range(85):
                st = session.state
                if st.room_id != ROOM_BUSINESS:
                    raise TimeoutError(
                        f"business_climb_setup: left Business: {session.state}"
                    )
                x = int(st.samus_x)
                y = int(st.samus_y)
                # Never hold LEFT into the HJ door band on the lower screens.
                if direction == "LEFT" and y >= 1280 and x <= soft_x:
                    hold(
                        session,
                        1,
                        "RIGHT",
                        "B",
                        "A",
                        reason="business_climb_setup_bound",
                    )
                elif direction == "LEFT" and y >= 1350 and x <= door_x + 8:
                    hold(
                        session,
                        1,
                        "RIGHT",
                        "B",
                        reason="business_climb_setup_door",
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
            if _on_business_1339(session.state):
                return

        if _on_business_1339(session.state):
            return
        # Knocked or missed — drop / re-center and retry the trio.
        if int(session.state.samus_y) < 1350:
            for _ in range(90):
                st = session.state
                if st.room_id != ROOM_BUSINESS:
                    raise TimeoutError(
                        f"business_climb_setup: left Business: {session.state}"
                    )
                if int(st.samus_y) >= 1405 and int(st.velocity_y) == 0:
                    break
                hold(session, 1, "RIGHT", "B", reason="business_climb_setup_drop")
        hold(session, 10, reason="business_climb_setup_retry_pause")

    if not _on_business_1339(session.state):
        raise TimeoutError(
            f"business_climb_setup: failed to reach y1339: {session.state}"
        )


def _business_high_jump_platforms(
    session: ControllerSession,
    *,
    runup_907: int = 14,
    pos_1339: int = 84,
    bound_floor_left: bool = False,
) -> None:
    """Bottom Business Center floor → center elevator (Hi-Jump route).

    ``runup_907``: RIGHT+B frames before the 987→907 hop. Continuous natural
    entry often needs 14–20; pure probe prefers 8 then 14 on retry.

    ``pos_1339``: LEFT walk target on y1339 before 1227 hop (pure≈84;
    continuous Ice floor pin often prefers ≈90 — rr-kxge offline grid).

    ``bound_floor_left``: earlier RIGHT bias during setup (continuous Ice).
    Door lip is always soft-bound regardless.
    """
    # Setup jumps land on the first left platform (~y=1339).
    # IMPORTANT: default (bound_floor_left=False) must keep the classic open-loop
    # LEFT/LEFT/RIGHT timing — warehouse continuous spine is frame-locked to it.
    # Safe/re-centering setup is only for continuous Ice floor retries.
    unmorph(session)
    already = _on_business_1339(session.state)
    if not already:
        if bound_floor_left:
            _setup_business_to_1339(session, bound_floor_left=True)
        else:
            # Classic open-loop (warehouse spine). Minimal door lip guard only:
            # never hold LEFT at x≤40 on the floor band — earlier frames unchanged.
            for direction in ("LEFT", "LEFT", "RIGHT"):
                hold(session, 12, reason="business_climb_release")
                for _ in range(85):
                    st = session.state
                    if st.room_id != ROOM_BUSINESS:
                        raise TimeoutError(
                            f"business_climb_setup: left Business: {session.state}"
                        )
                    if (
                        direction == "LEFT"
                        and int(st.samus_y) >= 1350
                        and int(st.samus_x) <= 40
                    ):
                        hold(
                            session,
                            1,
                            "RIGHT",
                            "B",
                            "A",
                            reason="business_climb_setup_door",
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
    # so A never charges. Gate on standing. Prejump x band ~78–90 is the
    # working window (pos=84 overshoots to ~76 on colder continuous pins).
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
                if bound_floor_left:
                    _setup_business_to_1339(
                        session, bound_floor_left=True, attempts=3
                    )
                else:
                    unmorph(session)
                    for direction in ("LEFT", "LEFT", "RIGHT"):
                        hold(session, 12, reason="business_climb_release")
                        hold(
                            session,
                            85,
                            direction,
                            "B",
                            "A",
                            reason="business_climb_setup",
                        )
                        hold(session, 30, reason="business_climb_setup_land")
                _wait_standing_y(
                    session, 1339, timeout=60, reason="business_1339_ground_retry"
                )
                break
            continue
        hold(session, 1, "LEFT", reason="business_1339_position")
    # Classic pure/warehouse: 4f RIGHT brake. Continuous Ice: shorter brake +
    # nudge so prejump x stays in the 78–90 working band.
    if bound_floor_left:
        hold(session, 3, "RIGHT", reason="business_1339_brake")
        for _ in range(12):
            x = int(session.state.samus_x)
            if session.state.samus_y != 1339 or session.state.pose not in _STANDING:
                break
            if 78 <= x <= 90:
                break
            if x < 78:
                hold(session, 1, "RIGHT", reason="business_1339_nudge_r")
            else:
                break
    else:
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

def _on_business_elevator_platform(state: SuperMetroidState) -> bool:
    return (
        int(state.samus_y) == 683
        and int(state.pose) in _STANDING
        and int(state.velocity_y) == 0
    )


def _anchor_business_floor_midright(session: ControllerSession, *, label: str) -> None:
    """Walk to floor band ~(170–230, ≥1405) before LEFT-heavy setup jumps.

    Post-Ice Super fall often lands left of the classic pure pin; re-center so
    open-loop LEFT/LEFT/RIGHT setup does not kiss the HJ door.
    """
    unmorph(session)
    for _ in range(200):
        st = session.state
        if st.room_id != ROOM_BUSINESS:
            raise TimeoutError(f"{label}: left Business during floor anchor: {st}")
        x = int(st.samus_x)
        y = int(st.samus_y)
        grounded = int(st.velocity_y) == 0 and int(st.pose) in _STANDING
        if y < 1350:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_anchor_air")
            continue
        if grounded and 170 <= x <= 230:
            hold(session, 10, reason=f"{label}_anchor_settle")
            return
        if x < 170:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_anchor_r")
        elif x > 230 and grounded:
            if x > 80:
                hold(session, 1, "LEFT", reason=f"{label}_anchor_l")
            else:
                hold(session, 1, "RIGHT", reason=f"{label}_anchor_bounce")
        else:
            hold(session, 1, reason=f"{label}_anchor_idle")
    # Soft fail — climb may still clear from imperfect pin.


def _climb_business_to_elevator(
    session: ControllerSession,
    *,
    label: str = "business_climb",
) -> None:
    """Floor → elevator platform y683 with Charge-aware multi-attempt ladder.

    Mirrors the Ice floor→elev attempt order (rr-kxge): Wave+/Charge loadouts
    prefer longer 907 runups; pure pre-Charge keeps 8→14 / pos 84 first.
    Classic warehouse continuous still first-hits runup 14/pos 84 (attempt 0 or
    early row) — extra rows only fire after TimeoutError.
    """
    unmorph(session)
    if session.state.room_id != ROOM_BUSINESS:
        raise TimeoutError(f"{label}: not in Business: {session.state}")
    if _on_business_elevator_platform(session.state):
        return

    y0 = int(session.state.samus_y)
    midshaft = y0 < 1350 and not _on_business_elevator_platform(session.state)
    # Super band / mid platforms: fall before open-loop floor→1339 setup.
    # Re-anchor only after midshaft fall / retries — classic floor first-try
    # must keep historical open-loop geometry (warehouse continuous frame-lock).
    if midshaft:
        _maybe_dump_climb_state(session, f"{label}_midshaft_before_fall")
        _fall_to_business_floor(session)
        _anchor_business_floor_midright(session, label=label)

    beams = int(session.state.collected_beams)
    has_charge = bool(beams & 0x1000)
    # (runup_907, pos_1339, bound_floor_left)
    # Always lead with classic 14 then 8 — warehouse continuous spine is
    # frame-sensitive to first-try geometry (do not front-load cont 18/20).
    attempts: list[tuple[int, int, bool]] = [
        (14, 84, False),
        (8, 84, False),
    ]
    if has_charge:
        attempts.extend(
            [
                (18, 90, False),
                (20, 90, False),
                (22, 90, False),
                (18, 90, True),
                (20, 90, True),
                (14, 84, True),
            ]
        )
    else:
        attempts.extend(
            [
                (18, 90, False),
                (20, 90, False),
                (18, 90, True),
                (14, 84, True),
            ]
        )

    last_err: TimeoutError | None = None
    for i, (runup, pos_1339, bound) in enumerate(attempts):
        try:
            if i > 0:
                if session.state.room_id != ROOM_BUSINESS:
                    raise TimeoutError(
                        f"{label}: left Business during climb retry: {session.state}"
                    )
                _fall_to_business_floor(session)
                _anchor_business_floor_midright(session, label=f"{label}_retry{i}")
            _business_high_jump_platforms(
                session,
                runup_907=runup,
                pos_1339=pos_1339,
                bound_floor_left=bound,
            )
            last_err = None
            break
        except TimeoutError as exc:
            last_err = exc
            _maybe_dump_climb_state(session, f"{label}_fail_{i}")
            if session.state.room_id != ROOM_BUSINESS:
                raise TimeoutError(
                    f"{label}: left Business during climb: {session.state}"
                ) from exc
            continue
    if last_err is not None:
        raise last_err
    if not _on_business_elevator_platform(session.state):
        # Allow slight y drift on elevator pad (standing gate already in ladder).
        if session.state.samus_y != 683 or session.state.pose not in _STANDING:
            raise TimeoutError(
                f"{label}: not on elevator platform: {session.state}"
            )


def play_business_to_warehouse(session: ControllerSession) -> SuperMetroidState:
    """Hi-Jump-assisted Business Center climb and elevator to Warehouse.

    Floor start (classic pure / continuous): multi-attempt platform ladder.
    Mid-shaft Super lip handoff (post-Ice pure ~(41,907) p25): fall to the
    floor climb anchor first — the ladder assumes floor→y1339 setup.
    Charge/Wave loadouts use cont-tuned 907 runups (rr-kxge ladder parity).
    """
    require_room(session, ROOM_BUSINESS, "business_to_warehouse")
    if not session.state.collected_items & ITEM_HI_JUMP:
        raise RuntimeError(
            f"business_to_warehouse: Hi-Jump not collected: {session.state}"
        )
    _climb_business_to_elevator(session, label="business_to_warehouse")
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
