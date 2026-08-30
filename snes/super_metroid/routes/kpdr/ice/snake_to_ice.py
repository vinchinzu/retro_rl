"""Pure Ice Beam Snake Room → Ice Beam PLM collect.

Source: ``post_ice_acid_to_snake_pure`` ~(216, 651) in ``0xA8B9``.
Technique: platform-hop / 2WJ climb bands (not freeze ladder). Tape thrash
f12664–15400 is non-product.

Path sketch (tape + live pure probe)::

    floor y651 → L1..L8 platform hops → top y139
      → right cross past center wall (x>171 only open at top)
        → right platform y267 → mid shelf y507
          → jump-up + mid-air morph into tunnel y377
            → morph roll RIGHT → Ice door ~(494,395)
              → Ice 0xA890 PLM bit 0x0002

**rr-5if:** climb bands proven; tunnel entry = mid-shelf jump-up morph
(human success f15350–15463). Left-wall morph at x=171 is solid at mid height.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    is_morph,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.ice.geometry import (
    ICE_BEAM_MASK,
    ICE_PLM_X,
    ICE_ROOM_SETTLE,
    SNAKE_CLIMB_FRAMES,
    SNAKE_DOOR_X,
    SNAKE_ICE_COLLECT_FRAMES,
    SNAKE_L1_Y,
    SNAKE_L2_Y,
    SNAKE_L3_Y,
    SNAKE_L4_Y,
    SNAKE_L5_Y,
    SNAKE_L6_Y,
    SNAKE_L7_Y,
    SNAKE_MID_SHELF_Y,
    SNAKE_TOP_Y,
    SNAKE_TUNNEL_EXIT_X,
    SNAKE_TUNNEL_FRAMES,
    SNAKE_TUNNEL_X_MIN,
    SNAKE_TUNNEL_Y,
    has_ice,
    in_ice_snake,
    on_snake_false_ledge,
    on_snake_mid_shelf,
    on_snake_top,
    on_snake_tunnel_band,
)
from super_metroid.routes.kpdr.norfair.common import _STANDING_POSES
from super_metroid.routes.kpdr.rooms import ROOM_ICE, ROOM_ICE_SNAKE
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import (
    escape_knockback_spin,
    is_knockback,
)

_LEDGE = _STANDING_POSES | frozenset({1, 2, 9, 10, 37, 38})


def _settle_ground(session: ControllerSession, label: str, *, max_frames: int = 40) -> None:
    for _ in range(max_frames):
        st = session.state
        if (
            int(st.velocity_y) == 0
            and int(st.pose) in _LEDGE
            and int(st.door_transition) == 0
        ):
            return
        if is_knockback(st):
            escape_knockback_spin(
                session,
                prefer_dir="LEFT",
                run_frames=2,
                spin_frames=10,
                label=f"{label}_kb",
                ensure_beam=True,
                break_on_motion_clear=True,
            )
            continue
        if int(st.pose) in (31, 39, 40, 41, 42, 65):
            hold(session, 1, "UP", reason=f"{label}_unmorph")
        else:
            hold(session, 1, reason=f"{label}_settle")


def _y_band(state: SuperMetroidState, band: tuple[int, int]) -> bool:
    return band[0] <= int(state.samus_y) <= band[1]


def _snake_platform_climb(session: ControllerSession, label: str) -> SuperMetroidState:
    """Floor handoff → top shelf via alternating platform hops (2WJ-style).

    Live pure pin sequence (post_ice_acid_to_snake_pure)::

        L1 y587 ← left run+jump
        L2 y523 ← right spin
        L3 y459 ← left
        L4 y395 ← A then RIGHT+A from left wall
        L5 y331 ← RIGHT then LEFT+A
        L6 y267 ← left wall, A, RIGHT+A
        L7 y203 ← RIGHT then LEFT+A
        top y139 ← left wall, A, RIGHT+A
    """
    require_room(session, ROOM_ICE_SNAKE, label)
    unmorph(session)
    if int(session.state.selected_item) != 0:
        select_weapon(session, 0)

    # --- L1: left run + left jump ---
    if int(session.state.samus_y) > SNAKE_L1_Y[1]:
        hold(session, 8, reason=f"{label}_l1_idle")
        hold(session, 11, "LEFT", reason=f"{label}_l1_walk")
        hold(session, 11, "LEFT", "B", reason=f"{label}_l1_run")
        hold(session, 34, "LEFT", "B", "A", reason=f"{label}_l1_jump")
        hold(session, 12, "LEFT", "B", reason=f"{label}_l1_coast")
        _settle_ground(session, f"{label}_l1")

    # --- L2: right spin hop ---
    if int(session.state.samus_y) > SNAKE_L2_Y[1]:
        hold(session, 3, "B", reason=f"{label}_l2_b")
        hold(session, 4, "B", "A", reason=f"{label}_l2_ba")
        hold(session, 23, "RIGHT", "B", "A", reason=f"{label}_l2_jump")
        hold(session, 27, "RIGHT", "B", reason=f"{label}_l2_coast")
        _settle_ground(session, f"{label}_l2")

    # --- L3: left hop ---
    if int(session.state.samus_y) > SNAKE_L3_Y[1]:
        hold(session, 3, "LEFT", "A", reason=f"{label}_l3_a")
        hold(session, 21, "LEFT", "A", reason=f"{label}_l3_jump")
        hold(session, 12, "LEFT", reason=f"{label}_l3_coast")
        hold(session, 8, "LEFT", "B", reason=f"{label}_l3_run")
        hold(session, 12, "LEFT", reason=f"{label}_l3_walk")
        _settle_ground(session, f"{label}_l3")

    # --- L4: left wall, A, RIGHT+A → mid door height ---
    if int(session.state.samus_y) > SNAKE_L4_Y[1]:
        hold(session, 5, "A", reason=f"{label}_l4_a")
        hold(session, 12, "RIGHT", "A", reason=f"{label}_l4_ra")
        hold(session, 12, "RIGHT", "B", "A", reason=f"{label}_l4_spin")
        hold(session, 25, "RIGHT", reason=f"{label}_l4_coast")
        for _ in range(50):
            st = session.state
            if (
                _y_band(st, SNAKE_L4_Y)
                and int(st.velocity_y) == 0
                and int(st.pose) in _LEDGE
            ):
                break
            hold(session, 1, reason=f"{label}_l4_land")
        _settle_ground(session, f"{label}_l4")

    # --- L5: RIGHT then LEFT+A ---
    if int(session.state.samus_y) > SNAKE_L5_Y[1]:
        hold(session, 12, "RIGHT", reason=f"{label}_l5_right")
        hold(session, 28, "LEFT", "A", reason=f"{label}_l5_jump")
        hold(session, 15, "LEFT", reason=f"{label}_l5_coast")
        _settle_ground(session, f"{label}_l5")

    # --- L6: left wall, A, RIGHT+A ---
    if int(session.state.samus_y) > SNAKE_L6_Y[1]:
        hold(session, 18, "LEFT", reason=f"{label}_l6_wall")
        hold(session, 4, "A", reason=f"{label}_l6_a")
        hold(session, 18, "RIGHT", "A", reason=f"{label}_l6_ra")
        hold(session, 10, "RIGHT", "B", "A", reason=f"{label}_l6_spin")
        hold(session, 15, "RIGHT", reason=f"{label}_l6_coast")
        _settle_ground(session, f"{label}_l6")

    # --- L7: RIGHT then LEFT+A ---
    if int(session.state.samus_y) > SNAKE_L7_Y[1]:
        hold(session, 20, "RIGHT", reason=f"{label}_l7_right")
        hold(session, 24, "LEFT", "A", reason=f"{label}_l7_jump")
        hold(session, 14, "LEFT", reason=f"{label}_l7_coast")
        _settle_ground(session, f"{label}_l7")

    # --- top: left wall, A, RIGHT+A ---
    if int(session.state.samus_y) > SNAKE_TOP_Y[1]:
        for _ in range(30):
            if int(session.state.samus_x) <= 68:
                break
            hold(session, 1, "LEFT", reason=f"{label}_top_wall")
        hold(session, 3, reason=f"{label}_top_pause")
        hold(session, 5, "A", reason=f"{label}_top_a")
        hold(session, 20, "RIGHT", "A", reason=f"{label}_top_ra")
        hold(session, 12, "RIGHT", "B", "A", reason=f"{label}_top_spin")
        hold(session, 25, "RIGHT", reason=f"{label}_top_coast")
        for _ in range(50):
            st = session.state
            if on_snake_top(st):
                break
            hold(session, 1, reason=f"{label}_top_land")
        _settle_ground(session, f"{label}_top")

    return session.state


def _snake_top_to_right_column(session: ControllerSession, label: str) -> None:
    """Top shelf → past center wall → right platform ~y267."""
    # Cross right on top (avoid Tutorial door x≳230 y<160).
    for _ in range(50):
        st = session.state
        if not in_ice_snake(st):
            return
        if int(st.samus_x) >= 200:
            break
        if int(st.samus_y) > 180:
            break
        hold(session, 1, "RIGHT", "B", reason=f"{label}_top_cross")

    # Jump right then fall to y267 platform (human f12582–12643).
    # Shoot-down opens solid shelf ~y155; do not hold DOWN (morph thrash).
    hold(session, 10, "RIGHT", "A", reason=f"{label}_top_jump")
    shelf_shots = 0
    for _ in range(120):
        st = session.state
        if not in_ice_snake(st):
            return
        y = int(st.samus_y)
        if y >= 250 and int(st.velocity_y) == 0:
            break
        if 145 <= y <= 180 and int(st.velocity_y) == 0:
            # Solid shelf: unmorph if needed, shoot down a few times, then walk off.
            if is_morph(st.pose) or int(st.pose) in (39, 40, 41, 42):
                hold(session, 1, "UP", reason=f"{label}_shelf_up")
                continue
            if shelf_shots < 6:
                hold(session, 1, "DOWN", "X", reason=f"{label}_shelf_shot")
                hold(session, 2, reason=f"{label}_shelf_wait")
                shelf_shots += 1
                continue
            # Walk/fall off shelf to the right column.
            hold(session, 1, "RIGHT", reason=f"{label}_shelf_off")
            continue
        if y < 250:
            # Free-fall — no DOWN (that morphs and sticks on ledges).
            hold(session, 1, reason=f"{label}_fall")
        else:
            hold(session, 1, reason=f"{label}_right_land")

    unmorph(session)
    _settle_ground(session, f"{label}_right_plat")


def _in_tunnel_grounded(state: SuperMetroidState) -> bool:
    """True when morph-capable and sitting on the y377 tunnel floor."""
    if not in_ice_snake(state):
        return False
    if int(state.samus_x) < SNAKE_TUNNEL_X_MIN:
        return False
    if not (SNAKE_TUNNEL_Y[0] <= int(state.samus_y) <= SNAKE_TUNNEL_Y[1]):
        return False
    return int(state.velocity_y) == 0


def _morph_drop_into_tunnel(session: ControllerSession, label: str) -> None:
    """From right platform ~y267: morph, drop into y377 tunnel at x~203.

    Live path: free-fall from y267 at x~203 passes the tunnel band; must be
    morph ball before y~360 or we fall through. Standing morph on the platform
    then roll/drop left is the clean one-knob (not freeze, not left-wall x=171).
    """
    if session.state.room_id == ROOM_ICE:
        return
    if _in_tunnel_grounded(session.state) and is_morph(session.state.pose):
        return

    # Stand on y267 platform, walk to left edge (~x205–210).
    unmorph(session)
    _settle_ground(session, f"{label}_plat")
    for _ in range(40):
        st = session.state
        if not in_ice_snake(st):
            return
        y = int(st.samus_y)
        if y > 300:
            break  # already dropping
        if int(st.samus_x) <= 208 and int(st.velocity_y) == 0:
            break
        hold(session, 1, "LEFT", reason=f"{label}_edge")

    # Morph on platform (double-tap DOWN).
    if int(session.state.samus_y) <= 300 and int(session.state.velocity_y) == 0:
        try:
            ensure_morph(session)
        except TimeoutError:
            hold(session, 2, "DOWN", reason=f"{label}_morph_tap")
            hold(session, 2, reason=f"{label}_morph_rel")
            hold(session, 4, "DOWN", reason=f"{label}_morph_tap2")

    # Roll / nudge left off the platform edge, then fall as morph toward tunnel.
    for _ in range(20):
        st = session.state
        if not in_ice_snake(st):
            return
        if int(st.samus_y) > 280:
            break
        if is_morph(st.pose):
            hold(session, 1, "LEFT", reason=f"{label}_roll_off")
        else:
            hold(session, 1, "LEFT", "DOWN", reason=f"{label}_nudge_off")

    # Fall: keep morph + slight RIGHT to sit at tunnel mouth x~203.
    for _ in range(100):
        st = session.state
        if st.room_id == ROOM_ICE:
            return
        if not in_ice_snake(st):
            return
        x, y = int(st.samus_x), int(st.samus_y)

        # Grounded in tunnel band → success.
        if _in_tunnel_grounded(st):
            if not is_morph(st.pose):
                try:
                    ensure_morph(session)
                except TimeoutError:
                    hold(session, 4, "DOWN", reason=f"{label}_tunnel_morph")
            return

        # Fell past tunnel without landing — stop; recovery handles mid shelf.
        if y > 430 and int(st.velocity_y) == 0:
            return

        if not is_morph(st.pose) and int(st.pose) not in (23, 55, 41, 42, 45, 49, 50):
            # Air morph attempt approaching tunnel height.
            if y >= 300:
                hold(session, 1, "DOWN", reason=f"{label}_air_morph")
                continue

        # Align x near tunnel mouth while falling.
        if x < 200:
            hold(session, 1, "RIGHT", reason=f"{label}_fall_r")
        elif x > 210:
            hold(session, 1, "LEFT", reason=f"{label}_fall_l")
        else:
            hold(session, 1, "RIGHT" if y >= 340 else "LEFT", reason=f"{label}_fall_in")


# Human tape f15354–15470 open-loop (from stand ~(197,507) into tunnel roll).
# Prefer this over inventing freeze/thrash once mid-shelf pin is reached.
_MID_SHELF_TUNNEL_RLE: list[tuple[int, tuple[str, ...]]] = [
    (1, ("B",)),
    (6, ("B", "RIGHT", "A")),
    (5, ("B", "UP", "RIGHT", "A")),
    (5, ("B", "UP", "A")),
    (3, ("B", "UP", "LEFT", "A")),
    (3, ("B", "LEFT", "A")),
    (2, ("B", "UP", "LEFT", "A")),
    (1, ("B", "UP", "A")),
    (1, ("B", "UP", "RIGHT", "A")),
    (19, ("B", "RIGHT", "A")),
    (3, ("B", "A")),
    (2, ("B",)),
    (1, ("B", "DOWN")),
    (5, ("DOWN",)),
    (4, ()),
    (4, ("DOWN", "RIGHT")),
    (52, ("RIGHT",)),
]


def _jump_morph_from_mid_shelf(session: ControllerSession, label: str) -> None:
    """Recovery: from mid shelf ~(197,507) Hi-Jump + morph into tunnel.

    Plays human-tape RLE (f15354–15470) after x-align. Only used if morph-drop
    from y267 misses.
    """
    if session.state.room_id == ROOM_ICE:
        return
    if _in_tunnel_grounded(session.state) and is_morph(session.state.pose):
        return

    unmorph(session)
    # Must launch near x197 — ceiling is low against right wall x219.
    for _ in range(50):
        st = session.state
        if not in_ice_snake(st):
            return
        x, y = int(st.samus_x), int(st.samus_y)
        if y < SNAKE_MID_SHELF_Y[0] - 30:
            break
        if 194 <= x <= 200 and int(st.velocity_y) == 0:
            break
        hold(session, 1, "LEFT" if x > 200 else "RIGHT", reason=f"{label}_align")
    _settle_ground(session, f"{label}_pre_jump")
    # Idle a few frames standing (human pre-jump).
    hold(session, 4, reason=f"{label}_pre_idle")

    for n, buttons in _MID_SHELF_TUNNEL_RLE:
        st = session.state
        if st.room_id == ROOM_ICE or not in_ice_snake(st):
            return
        if _in_tunnel_grounded(st) and is_morph(st.pose) and int(st.samus_x) >= 250:
            # Already deep in tunnel — let roll handler finish.
            return
        hold(session, n, *buttons, reason=f"{label}_rle")

    # If RLE left us near tunnel, ensure morph on floor.
    if _in_tunnel_grounded(session.state) and not is_morph(session.state.pose):
        try:
            ensure_morph(session)
        except TimeoutError:
            pass


def _roll_tunnel_to_ice_door(session: ControllerSession, label: str) -> SuperMetroidState:
    """Morph roll through y377 tunnel then unmorph → Ice blue door RIGHT.

    Only call when already on/near tunnel floor — do not thrash from low y.
    """
    if (
        in_ice_snake(session.state)
        and _in_tunnel_grounded(session.state)
        and not is_morph(session.state.pose)
    ):
        try:
            ensure_morph(session)
        except TimeoutError:
            hold(session, 6, "DOWN", reason=f"{label}_morph_retry")

    for i in range(SNAKE_TUNNEL_FRAMES):
        st = session.state
        if st.room_id == ROOM_ICE:
            return st
        if not in_ice_snake(st):
            return st
        if is_knockback(st):
            hold(session, 1, "DOWN", "RIGHT", reason=f"{label}_kb_morph")
            continue

        x, y = int(st.samus_x), int(st.samus_y)

        # Fell out of tunnel zone without exit — abort roll (outer loop recovers).
        if y > 450 and x < SNAKE_TUNNEL_EXIT_X:
            return st

        # In tunnel: morph + RIGHT only on real tunnel floor (not y409 ledge).
        if x < SNAKE_TUNNEL_EXIT_X and SNAKE_TUNNEL_Y[0] - 5 <= y <= SNAKE_TUNNEL_Y[1] + 5:
            if not is_morph(st.pose):
                if int(st.velocity_y) == 0:
                    try:
                        ensure_morph(session)
                    except TimeoutError:
                        hold(session, 1, "DOWN", reason=f"{label}_remorph")
                else:
                    hold(session, 1, "DOWN", reason=f"{label}_air_remorph")
                continue
            hold(session, 1, "RIGHT", reason=f"{label}_tunnel_roll")
            continue

        # Past tunnel mouth (x≥320): unmorph and run to Ice door.
        if is_morph(st.pose) or int(st.pose) in (39, 40, 41, 42, 61):
            hold(session, 1, "UP", reason=f"{label}_unmorph_exit")
            continue

        if x >= SNAKE_DOOR_X:
            phase = i % 16
            if phase < 4:
                hold(session, 1, "RIGHT", "X", reason=f"{label}_door_shot")
            elif phase < 11:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_door_run")
            else:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_door_hop")
            continue

        # Open right column after tunnel: hop/run right toward door y~395.
        phase = i % 18
        if phase < 10:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_right_hop")
        elif phase < 14:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_right_run")
        else:
            hold(session, 1, "RIGHT", "X", reason=f"{label}_right_shot")

    return session.state


def _snake_top_to_tunnel(session: ControllerSession, label: str) -> SuperMetroidState:
    """Top shelf → morph-drop into tunnel → Ice door.

    Primary: morph on y267 right platform, drop into y377 tunnel at x~203.
    Recovery: mid-shelf Hi-Jump morph (human f15350) if drop misses.
    """
    _snake_top_to_right_column(session, label)
    if session.state.room_id == ROOM_ICE:
        return session.state

    for attempt in range(4):
        if session.state.room_id == ROOM_ICE:
            return session.state
        if not in_ice_snake(session.state):
            return session.state

        if is_knockback(session.state):
            escape_knockback_spin(
                session,
                prefer_dir="RIGHT",
                run_frames=2,
                spin_frames=10,
                label=f"{label}_kb_a{attempt}",
                ensure_beam=True,
                break_on_motion_clear=True,
            )

        y = int(session.state.samus_y)
        x = int(session.state.samus_x)

        # Already grounded in tunnel → roll out.
        if _in_tunnel_grounded(session.state):
            st = _roll_tunnel_to_ice_door(session, f"{label}_a{attempt}")
            if st.room_id == ROOM_ICE:
                return st
            continue

        # High right shelf (~y155) — unmorph, shoot-open, fall to y267.
        if 140 <= y <= 200 and x >= 190 and int(session.state.velocity_y) == 0:
            unmorph(session)
            for _ in range(8):
                hold(session, 1, "DOWN", "X", reason=f"{label}_shelf_clear")
                hold(session, 1, reason=f"{label}_shelf_clear")
            hold(session, 6, "RIGHT", reason=f"{label}_shelf_off")
            for _ in range(40):
                st = session.state
                if int(st.samus_y) >= 250:
                    break
                hold(session, 1, reason=f"{label}_shelf_fall")
            unmorph(session)
            _settle_ground(session, f"{label}_shelf_land")
            continue

        # Primary: from upper right platform (y~250–300) morph-drop into tunnel.
        if 240 <= y <= 320 and x >= 180:
            _morph_drop_into_tunnel(session, f"{label}_a{attempt}")
            if _in_tunnel_grounded(session.state):
                st = _roll_tunnel_to_ice_door(session, f"{label}_a{attempt}")
                if st.room_id == ROOM_ICE:
                    return st
            continue

        # Above y240 but not on shelf band — continue fall / walk to platform.
        if y < 240 and x >= 180:
            unmorph(session)
            hold(session, 8, "RIGHT", "A", reason=f"{label}_to_plat")
            for _ in range(50):
                if int(session.state.samus_y) >= 250:
                    break
                hold(session, 1, reason=f"{label}_to_plat_fall")
            unmorph(session)
            _settle_ground(session, f"{label}_to_plat")
            continue

        # Mid shelf recovery launch.
        if on_snake_mid_shelf(session.state) or (
            SNAKE_MID_SHELF_Y[0] - 20 <= y <= SNAKE_MID_SHELF_Y[1] + 30 and x >= 180
        ):
            _jump_morph_from_mid_shelf(session, f"{label}_a{attempt}")
            if _in_tunnel_grounded(session.state):
                st = _roll_tunnel_to_ice_door(session, f"{label}_a{attempt}")
                if st.room_id == ROOM_ICE:
                    return st
            continue

        # Too low (floor / below mid): climb back toward mid shelf.
        if y > SNAKE_MID_SHELF_Y[1]:
            unmorph(session)
            hold(session, 25, "LEFT" if x > 210 else "RIGHT", "B", "A", reason=f"{label}_floor_up")
            hold(session, 15, "A", reason=f"{label}_floor_up2")
            _settle_ground(session, f"{label}_floor")
            continue

        # False ledge ~y409 (just below tunnel floor): drop to mid shelf and
        # recover with Hi-Jump morph (human thrash→success pattern).
        if on_snake_false_ledge(session.state) or (320 < y < SNAKE_MID_SHELF_Y[0]):
            unmorph(session)
            # Nudge left of wall then drop to mid shelf y507.
            for _ in range(20):
                st = session.state
                if int(st.samus_x) <= 200:
                    break
                hold(session, 1, "LEFT", reason=f"{label}_ledge_left")
            hold(session, 6, "LEFT", "A", reason=f"{label}_ledge_drop")
            for _ in range(60):
                st = session.state
                if on_snake_mid_shelf(st) or int(st.samus_y) >= SNAKE_MID_SHELF_Y[0]:
                    break
                if int(st.samus_y) > 580:
                    break
                hold(session, 1, reason=f"{label}_to_mid")
            unmorph(session)
            _settle_ground(session, f"{label}_mid_from_ledge")
            # Immediately attempt jump-morph recovery this attempt.
            if on_snake_mid_shelf(session.state) or (
                SNAKE_MID_SHELF_Y[0] - 20
                <= int(session.state.samus_y)
                <= SNAKE_MID_SHELF_Y[1] + 30
            ):
                _jump_morph_from_mid_shelf(session, f"{label}_a{attempt}_rec")
                if _in_tunnel_grounded(session.state):
                    st = _roll_tunnel_to_ice_door(session, f"{label}_a{attempt}_rec")
                    if st.room_id == ROOM_ICE:
                        return st
            continue

    return session.state


def _ice_collect_plm(session: ControllerSession, label: str) -> SuperMetroidState:
    """Ice Beam room left entry → chozo PLM (beam bit 0x0002)."""
    require_room(session, ROOM_ICE, label)
    if has_ice(session.state):
        return session.state

    unmorph(session)
    if int(session.state.selected_item) != 0:
        select_weapon(session, 0)
    for _ in range(30):
        st = hold(session, 1, reason=f"{label}_stand")
        if int(st.velocity_y) == 0 and int(st.pose) in _STANDING_POSES:
            break

    for frame in range(SNAKE_ICE_COLLECT_FRAMES):
        st = session.state
        if has_ice(st):
            break
        if st.room_id != ROOM_ICE:
            raise TimeoutError(
                f"{label}: left Ice during collect; "
                f"room=0x{int(st.room_id):04X} xy=({st.samus_x},{st.samus_y})"
            )
        if int(st.pose) in (137, 138, 39, 40, 41, 42):
            hold(session, 1, "UP", reason=f"{label}_unmorph")
            continue
        if int(st.samus_x) < ICE_PLM_X - 10:
            phase = frame % 20
            if phase < 8:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_chozo_hop")
            elif phase < 14:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_chozo_run")
            else:
                hold(session, 1, "RIGHT", "X", reason=f"{label}_chozo_shot")
        else:
            if frame % 10 == 0:
                hold(session, 1, "X", reason=f"{label}_plm_shot")
            else:
                hold(session, 1, "RIGHT", reason=f"{label}_plm_walk")
    else:
        st = session.state
        raise TimeoutError(
            f"{label}: Ice PLM not collected; beams=0x{int(st.collected_beams):04X} "
            f"pose={st.pose} xy=({st.samus_x},{st.samus_y})"
        )

    hold(session, 80, reason=f"{label}_fanfare")
    unmorph(session)
    for _ in range(40):
        st = hold(session, 1, reason=f"{label}_post_stand")
        if int(st.velocity_y) == 0 and int(st.pose) in _STANDING_POSES:
            break
    return session.state


def play_ice_snake_to_ice(session: ControllerSession) -> SuperMetroidState:
    """Snake floor pin → Ice Beam PLM (beam bit 0x0002).

    Source: pure Acid→Snake handoff ``post_ice_acid_to_snake_pure`` ~(216, 651).
    """
    label = "ice_snake_to_ice"
    require_room(session, ROOM_ICE_SNAKE, label)
    start = session.frame

    if has_ice(session.state) and session.state.room_id == ROOM_ICE:
        return session.state

    if is_knockback(session.state):
        escape_knockback_spin(
            session,
            prefer_dir="LEFT",
            run_frames=3,
            spin_frames=12,
            label=f"{label}_kb0",
            ensure_beam=True,
            break_on_motion_clear=True,
        )

    if session.state.room_id == ROOM_ICE_SNAKE:
        _snake_platform_climb(session, label)

    if session.state.room_id == ROOM_ICE_SNAKE and not has_ice(session.state):
        if not on_snake_top(session.state) and int(session.state.samus_y) > SNAKE_TOP_Y[1]:
            # Climb budget exceeded — one more open attempt window.
            for _ in range(min(400, SNAKE_CLIMB_FRAMES // 4)):
                if on_snake_top(session.state) or session.state.room_id != ROOM_ICE_SNAKE:
                    break
                hold(session, 1, "LEFT", "A", reason=f"{label}_climb_push")
        _snake_top_to_tunnel(session, label)

    if session.state.room_id != ROOM_ICE:
        st = session.state
        raise TimeoutError(
            f"{label}: Ice door missed; room=0x{int(st.room_id):04X} "
            f"pose={st.pose} xy=({st.samus_x},{st.samus_y}) "
            f"door_transition={st.door_transition} "
            f"beams=0x{int(st.collected_beams):04X} "
            f"frames={session.frame - start} "
            f"(prefer 2WJ climb + right-column tunnel; not freeze ladder)"
        )

    wait_ordinary_room(
        session, ROOM_ICE, settle_frames=ICE_ROOM_SETTLE, label=label
    )
    state = _ice_collect_plm(session, label)

    if not has_ice(state):
        raise TimeoutError(
            f"{label}: finished without Ice bit; "
            f"beams=0x{int(state.collected_beams):04X} room=0x{int(state.room_id):04X} "
            f"xy=({state.samus_x},{state.samus_y}) "
            f"frames={session.frame - start}"
        )
    return state


__all__ = ["play_ice_snake_to_ice", "ICE_BEAM_MASK"]
