"""Ceres Magnet Stairs + Falling Tile reverse (WRAM-reactive escape)."""

from __future__ import annotations

from retro_harness.actions import buttons, idle_action
from super_metroid.routes.kpdr.ceres.arm_pump import (
    _ceres_arm_pump_step,
    _ceres_clear_knockback,
    _ceres_enemy_near,
)
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.routes.kpdr.ceres.geometry import (
    _CERES_MAGNET_EXIT_Y,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_MAGNET,
)
from super_metroid.routes.runtime import RouteSession

def _ceres_magnet_reached_falling(state) -> bool:
    return int(state.room_id) == ROOM_CERES_FALLING and int(state.game_state) == 8


def _ceres_magnet_step(
    session: RouteSession,
    names: tuple[str, ...],
    reason: str,
) -> bool:
    """One magnet frame. Returns True if Falling ordinary reached."""
    st = session.state
    if _ceres_magnet_reached_falling(st):
        return True
    if int(st.room_id) == ROOM_CERES_FALLING and int(st.game_state) in (9, 11):
        session.step(buttons("LEFT"), "ceres_magnet_exit_trans")
        return _ceres_magnet_reached_falling(session.state)
    if int(st.room_id) != ROOM_CERES_MAGNET:
        return False
    if is_knockback(st):
        # Full spin-escape — single LEFT never leaves pose 137/138.
        _ceres_clear_knockback(session, "LEFT", reason="ceres_magnet")
        return _ceres_magnet_reached_falling(session.state)
    # Abort RIGHT only on the east door lip (Scientist). Stair chain briefly
    # visits x~150–180 mid-climb — do not cut that short.
    if "RIGHT" in names and int(st.samus_x) > 220:
        session.step(buttons("LEFT", "A"), "ceres_magnet_abort_east")
        return False
    session.step(buttons(*names) if names else idle_action(), reason)
    return _ceres_magnet_reached_falling(session.state)


def _ceres_planted_near(state, y: int, *, slack: int = 5) -> bool:
    """Natural Ceres shelf plant; movement type 0 is the one-frame land."""
    return (
        int(state.game_state) == 8
        and abs(int(state.samus_y) - y) <= slack
        and int(state.vertical_direction) == 0
        and abs(int(state.velocity_y)) <= 1
    )


def _ceres_magnet_dboost_escape(session: RouteSession) -> bool:
    """Sniq-shaped Magnet→Falling carry, including the y=278 steam boost.

    The steam contact at x≈129 y≈243 starts a 95f invincibility window. The
    fast line reaches the west door 89 ordinary frames later, leaving 6f
    frozen through the transition for Falling's first steam at (440, 168).
    """

    def step(names: tuple[str, ...], reason: str) -> None:
        session.step(buttons(*names) if names else idle_action(), reason)

    # Run down the entry slope to the x≈85/y347 takeoff.
    for i in range(90):
        st = session.state
        if int(st.samus_y) <= 352 and int(st.samus_x) <= 92:
            break
        _ceres_arm_pump_step(
            session, "LEFT", i, "ceres_magnet_dboost_approach", force_pump=True
        )
    else:
        return False

    # TAS f12491-12511: low, nearly vertical hop onto y=267.
    shelf_hop = (
        ("B", "A"),
        ("RIGHT", "B", "A"),
        ("B", "A"),
        ("B", "A"),
        *(("A",),) * 12,
        ("RIGHT", "A"),
        ("RIGHT", "A"),
        ("LEFT", "RIGHT", "B", "A"),
        ("RIGHT", "B", "A"),
        ("RIGHT", "B", "A"),
    )
    for names in shelf_hop:
        step(names, "ceres_magnet_dboost_shelf_hop")
    for i in range(35):
        if _ceres_planted_near(session.state, 267, slack=8):
            break
        _ceres_arm_pump_step(
            session, "RIGHT", i, "ceres_magnet_dboost_shelf_land", force_pump=True
        )
    if not _ceres_planted_near(session.state, 267, slack=8):
        return False

    # Meet live $E1FF slot 4 at (120,278) from its right side. The contact
    # pushes right; carrying A through movement types 10/25 preserves 5.5px/f.
    for i in range(50):
        if int(session.state.samus_x) >= 108:
            break
        _ceres_arm_pump_step(
            session, "RIGHT", i, "ceres_magnet_dboost_runup", force_pump=True
        )
    for names in (
        ("RIGHT", "B", "A"),
        ("RIGHT", "B", "A"),
        ("A",),
        ("LEFT", "B", "A"),
        ("B", "A", "L"),
        ("RIGHT", "B", "A"),
        ("B", "A"),
        ("B", "A"),
    ):
        step(names, "ceres_magnet_steam_setup")
    contacted = int(session.state.knockback_timer) > 0
    for _ in range(36):
        st = session.state
        contacted = contacted or int(st.knockback_timer) > 0
        if contacted and _ceres_planted_near(st, 219, slack=8):
            break
        step(("RIGHT", "B", "A"), "ceres_magnet_steam_dboost")
    if not contacted or not _ceres_planted_near(session.state, 219, slack=8):
        return False

    # Convert the d-boost land into the y=139 exit shelf, then sprint west.
    for names in (
        ("RIGHT", "B", "A"),
        ("RIGHT", "A"),
        ("RIGHT", "A"),
        *(("A",),) * 6,
        ("LEFT", "A"),
        ("A",),
        ("A",),
        ("LEFT", "A"),
        ("LEFT", "A"),
        ("LEFT", "A"),
    ):
        step(names, "ceres_magnet_dboost_top_hop")
    for _ in range(32):
        if _ceres_planted_near(session.state, 139, slack=8):
            break
        step(("LEFT", "B"), "ceres_magnet_dboost_top_land")
    if not _ceres_planted_near(session.state, 139, slack=8):
        return False

    for i in range(100):
        st = session.state
        if int(st.game_state) in (9, 11) or int(st.room_id) == ROOM_CERES_FALLING:
            break
        _ceres_arm_pump_step(
            session, "LEFT", i, "ceres_magnet_dboost_exit", force_pump=True
        )
    for _ in range(180):
        if _ceres_magnet_reached_falling(session.state):
            return True
        step(("LEFT",), "ceres_magnet_dboost_door")
    return False


def _ceres_reactive_magnet_escape(session: RouteSession) -> None:
    """Magnet Stairs escape — WRAM-gated climb + left exit.

    Reverse arm-pump plants mid/high Magnet. Geometry: seat left (~x37),
    stair chain RIGHT+A then LEFT+A to exit height (~y139), arm-pump left into
    Falling. Every frame aborts on Falling ordinary or east-door x; not a
    blind full-escape restore.
    """
    # LEFT through door settle (idle can drop off upper ledges).
    for _ in range(220):
        st = session.state
        if st.room_id == ROOM_CERES_MAGNET and st.game_state == 8:
            break
        if _ceres_magnet_reached_falling(st):
            return
        session.step(buttons("LEFT"), "ceres_magnet_door")
    else:
        raise TimeoutError(f"ceres magnet ordinary missed: {session.state}")

    if is_knockback(session.state):
        _ceres_clear_knockback(session, "LEFT", reason="ceres_magnet")

    if int(session.state.samus_y) > _CERES_MAGNET_EXIT_Y:
        # The continuous route reaches this room on a different steam phase
        # than Sniq. Lift immediately over the shown (136,404) jet; beginning
        # the hop at x≈216 clears it, whereas beginning at x≈210 is too late.
        for _ in range(6):
            session.step(
                buttons("LEFT", "B", "A"),
                "ceres_magnet_entry_steam_hop",
            )

    # If already on exit band, skip climb and run out.
    if int(session.state.samus_y) <= _CERES_MAGNET_EXIT_Y:
        for i in range(360):
            st = session.state
            if _ceres_magnet_reached_falling(st):
                return
            if st.room_id != ROOM_CERES_MAGNET and st.room_id != ROOM_CERES_FALLING:
                break
            if is_knockback(st):
                _ceres_clear_knockback(session, "LEFT", reason="ceres_magnet")
                continue
            if _ceres_enemy_near(st, dx=40, dy=30):
                session.step(buttons("LEFT", "A"), "ceres_magnet_exit_hop")
            else:
                _ceres_arm_pump_step(
                    session, "LEFT", i, "ceres_magnet_exit", force_pump=True
                )
        if session.state.room_id != ROOM_CERES_FALLING:
            raise TimeoutError(f"ceres magnet high-exit missed Falling: {session.state}")
        return

    # Arm-pump through the scientist-door slope to the y=347 takeoff. Jumping
    # earlier at x~110 overshoots the east door back into Scientist.
    for approach_i in range(280):
        st = session.state
        if _ceres_magnet_reached_falling(st):
            return
        if int(st.room_id) != ROOM_CERES_MAGNET:
            break
        if int(st.samus_y) <= _CERES_MAGNET_EXIT_Y:
            break
        if int(st.samus_x) <= 75:
            break
        _ceres_arm_pump_step(
            session,
            "LEFT",
            approach_i,
            "ceres_magnet_to_seat",
            force_pump=True,
        )
        if _ceres_magnet_reached_falling(session.state):
            return

    # Three short kinetic hops follow the reverse TAS geometry instead of one
    # held-A arc: bottom y=347 -> y=267 shelf -> y=219 shelf -> y=139 exit.
    # Each release is landing-gated so the product handoff, not a restored TAS
    # state, owns the timing.
    def climb_span(names: tuple[str, ...], frames: int, reason: str) -> bool:
        for _ in range(frames):
            if _ceres_magnet_step(session, names, reason):
                return True
        return False

    def wait_for_shelf(y_lo: int, y_hi: int, *, direction: str, timeout: int) -> bool:
        for _ in range(timeout):
            st = session.state
            if _ceres_magnet_reached_falling(st):
                return True
            if int(st.room_id) != ROOM_CERES_MAGNET:
                return False
            if int(st.movement_type) == 1 and y_lo <= int(st.samus_y) <= y_hi:
                return False
            if _ceres_magnet_step(
                session,
                (direction, "B"),
                "ceres_magnet_climb_release",
            ):
                return True
        return False

    if int(session.state.samus_y) > _CERES_MAGNET_EXIT_Y:
        if climb_span(("B", "A"), 9, "ceres_magnet_bottom_jump"):
            return
        if climb_span(("RIGHT", "B", "A"), 12, "ceres_magnet_bottom_jump"):
            return
        if wait_for_shelf(250, 275, direction="RIGHT", timeout=60):
            return

    for i in range(80):
        st = session.state
        if int(st.samus_x) >= 112:
            break
        _ceres_arm_pump_step(
            session, "RIGHT", i, "ceres_magnet_mid_run", force_pump=True
        )

    if int(session.state.samus_y) > 230:
        if climb_span(("RIGHT", "B", "A"), 1, "ceres_magnet_mid_jump"):
            return
        if climb_span(("B", "A"), 10, "ceres_magnet_mid_jump"):
            return
        if wait_for_shelf(205, 230, direction="RIGHT", timeout=60):
            return

    for i in range(80):
        st = session.state
        if int(st.samus_y) <= 230 and int(st.samus_x) >= 191:
            break
        _ceres_arm_pump_step(
            session, "RIGHT", i, "ceres_magnet_top_run", force_pump=True
        )

    if int(session.state.samus_y) > _CERES_MAGNET_EXIT_Y:
        if climb_span(("RIGHT", "B", "A"), 1, "ceres_magnet_top_jump"):
            return
        if climb_span(("B", "A"), 2, "ceres_magnet_top_jump"):
            return
        if climb_span(("LEFT", "B", "A"), 18, "ceres_magnet_top_jump"):
            return
        if wait_for_shelf(120, _CERES_MAGNET_EXIT_Y, direction="LEFT", timeout=80):
            return

    # Exit left until Falling.  Do not repeatedly A-pulse at the west door:
    # enemy0 is near that lip, and the old proximity branch held pose 167 for
    # ~55f after the door was already opening.  Knockback remains reactive.
    for i in range(400):
        st = session.state
        if _ceres_magnet_reached_falling(st):
            return
        if st.room_id != ROOM_CERES_MAGNET and st.room_id != ROOM_CERES_FALLING:
            break
        if st.game_state in (9, 11) and st.room_id == ROOM_CERES_FALLING:
            session.step(buttons("LEFT"), "ceres_magnet_exit_trans")
            continue
        if is_knockback(st):
            _ceres_clear_knockback(session, "LEFT", reason="ceres_magnet")
            continue
        # Mid platform: need height still — hop up rather than wall-walk.
        if int(st.samus_y) > _CERES_MAGNET_EXIT_Y:
            session.step(buttons("LEFT", "A"), "ceres_magnet_up_hop")
            continue
        _ceres_arm_pump_step(
            session, "LEFT", i, "ceres_magnet_exit", force_pump=True
        )

    if session.state.room_id != ROOM_CERES_FALLING:
        raise TimeoutError(f"ceres magnet exit missed Falling: {session.state}")


def _ceres_reactive_falling(session: RouteSession) -> None:
    """Falling Tile reverse → elev door. TAS magnet-feet then pose-25 door.

    Sniq 100% lsnes (gs=8 f12792→door f12908): hop y=187 onto y=171, run
    left, air-turn, then LEFT spin and a RIGHT+A face so the door is pose 25
    at y=120. That is the elev wall-jump phase. Walking the floor remaps to
    y=651 and is not a fallback.
    """
    if session.state.room_id != ROOM_CERES_FALLING:
        raise RuntimeError(f"expected Falling after magnet: {session.state}")
    session.wait_until(
        lambda s: s.room_id == ROOM_CERES_FALLING and s.game_state == 8,
        timeout=120,
        reason="ceres_falling_door",
    )
    def step(names: tuple[str, ...], reason: str) -> None:
        session.step(buttons(*names) if names else idle_action(), reason)

    # The current continuous predecessor reaches Falling without Sniq's 6f
    # invincibility carry. A two-frame running hop clears slot 4 at (440,168)
    # without waiting for its debris phase.
    for _ in range(2):
        step(("LEFT", "B", "A"), "ceres_falling_entry_steam_hop")

    # Run to this product pin's measured short-hop point. Its incoming
    # subpixels launch 7px later than Sniq's x=357 tape state.
    for i in range(90):
        st = session.state
        if int(st.samus_x) <= 350 and int(st.samus_y) >= 180:
            break
        _ceres_arm_pump_step(
            session, "LEFT", i, "ceres_falling_phase_run", force_pump=True
        )
    for names in (("LEFT", "B", "A"), ("B", "A"), ("B", "DOWN", "A")):
        step(names, "ceres_falling_shelf_hop")
    for i in range(28):
        if _ceres_planted_near(session.state, 171, slack=8):
            break
        _ceres_arm_pump_step(
            session, "LEFT", i, "ceres_falling_shelf_land", force_pump=True
        )
    if not _ceres_planted_near(session.state, 171, slack=8):
        raise TimeoutError(f"falling missed y171 shelf: {session.state}")

    for i in range(70):
        if int(session.state.samus_x) <= 305:
            break
        _ceres_arm_pump_step(
            session, "LEFT", i, "ceres_falling_shelf_run", force_pump=True
        )

    # Advance three frames before the turn so the live $E1FF debris catches
    # the rising spin instead of the ground pose.  This keeps movement type
    # 10→25 through the full 5.75px/f arc and plants near x=98 with 60 i-frames.
    for i in range(2):
        _ceres_arm_pump_step(
            session, "LEFT", i, "ceres_falling_steam_phase", force_pump=True
        )
    step(("LEFT", "B"), "ceres_falling_steam_phase")
    for names in (("RIGHT", "B"), ("RIGHT", "B", "A"), ("B", "A", "X")):
        step(names, "ceres_falling_steam_setup")
    contacted = int(session.state.knockback_timer) > 0
    contact_i = 0
    for _ in range(65):
        st = session.state
        if not contacted and int(st.knockback_timer) > 0:
            contacted = True
            contact_i = 0
        if int(st.samus_x) <= 82 and int(st.vertical_direction) == 0:
            break
        # Sniq's second X is 9f after contact, left-facing during the boost.
        names = (
            ("LEFT", "B", "A", "X")
            if contacted and contact_i == 9
            else ("LEFT", "B", "A")
        )
        step(names, "ceres_falling_steam_dboost")
        if contacted:
            contact_i += 1
    if not contacted:
        raise TimeoutError(f"falling missed real steam d-boost: {session.state}")

    # Carry the real steam d-boost to the west slope.  Jumping immediately
    # from this plant keeps the upward arc alive through the elevator door.
    for i in range(110):
        st = session.state
        if int(st.samus_x) <= 83:
            break
        _ceres_arm_pump_step(
            session, "LEFT", i, "ceres_falling_slope_run", force_pump=True
        )

    # Seat against the overlay, then face back into the room only after
    # crossing x=28.  This resumes at y≈633 in a rising pose-25 spin.
    for _ in range(14):
        step(("LEFT", "B"), "ceres_falling_door_approach")
    for _ in range(40):
        if int(session.state.game_state) in (9, 11):
            break
        direction = "RIGHT" if int(session.state.samus_x) <= 28 else "LEFT"
        step((direction, "B", "A"), "ceres_falling_door_jump")
    for _ in range(210):
        if int(session.state.room_id) == ROOM_CERES_ELEVATOR:
            return
        names = ("A",) if int(session.state.game_state) in (9, 11) else ("RIGHT", "B", "A")
        step(names, "ceres_falling_door_trans")
    raise TimeoutError(f"falling missed elev: {session.state}")


__all__ = [
    "_ceres_magnet_reached_falling",
    "_ceres_magnet_step",
    "_ceres_reactive_magnet_escape",
    "_ceres_reactive_falling",
]
