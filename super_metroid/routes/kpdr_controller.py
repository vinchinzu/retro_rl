"""Controller-only KPDR segments through Hi-Jump and Kraid entry.

Seed with natural/dev entry states from predecessor; continuous acceptance
must compose after the verified Super prefix. No RAM writes, state loads, or
infinite bomb jumps are used by these controllers.

Living boards: ``docs/routes/ROUTE_KPDR.md``, ``docs/routes/KPDR_TRACKER.csv``.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.post_spore_controller import (
    _hold,
    _require_room,
    _select_weapon,
    _unmorph,
    ensure_morph,
)
from super_metroid.routes.runtime import ControllerSession

ROOM_GHZ = 0x9E52
ROOM_NOOB = 0x9FBA
ROOM_RED_TOWER = 0xA253
ROOM_BAT = 0xA3DD
ROOM_BIG_PINK = 0x9D19
ROOM_BELOW_SPAZER = 0xA408
ROOM_WEST_TUNNEL = 0xCF54
ROOM_GLASS = 0xCEFB
ROOM_EAST_TUNNEL = 0xCF80
ROOM_WAREHOUSE = 0xA6A1
ROOM_BUSINESS = 0xA7DE
ROOM_HJ_SHAFT = 0xAA41
ROOM_HJ = 0xA9E5
ROOM_ZEELA = 0xA471
ROOM_WAREHOUSE_KIHUNTER = 0xA4DA
ROOM_BABY_KRAID = 0xA521
ROOM_KRAID_EYE = 0xA56B
ROOM_KRAID = 0xA59F

ITEM_HI_JUMP = 0x0100


def _wait_ordinary_room(
    session: ControllerSession,
    room_id: int,
    *,
    settle_frames: int = 200,
    label: str,
) -> SuperMetroidState:
    for frame in range(settle_frames):
        state = _hold(session, 1, reason=f"{label}_settle")
        if (
            state.room_id == room_id
            and state.game_state == 8
            and state.door_transition == 0
            and frame > 15
        ):
            return state
    state = session.state
    if state.room_id != room_id:
        raise RuntimeError(
            f"{label}: expected 0x{room_id:04X}, got 0x{state.room_id:04X} @ {state}"
        )
    return state


def play_run_shoot_exit(
    session: ControllerSession,
    *,
    from_room: int,
    to_room: int,
    direction: str,
    label: str,
    run_frames: int = 40,
    shoot_frames: int = 6,
    spin_frames: int = 40,
    hold_frames: int = 160,
    settle_frames: int = 200,
    super_door: bool = False,
) -> SuperMetroidState:
    """Generic horizontal door exit: run + shoot (+ optional Super) + spin through.

    ``direction`` is ``LEFT`` or ``RIGHT``.
    """
    _require_room(session, from_room, label)
    if super_door:
        try:
            _select_weapon(session, 2)  # supers
        except RuntimeError:
            pass
    else:
        try:
            _select_weapon(session, 0)
        except RuntimeError:
            pass
    _hold(session, run_frames, direction, "B", reason=f"{label}_run")
    shoot_btns = (direction, "B", "X") if not super_door else (direction, "X")
    _hold(session, shoot_frames, *shoot_btns, reason=f"{label}_shoot")
    if super_door:
        _hold(session, 20, reason=f"{label}_super_fuse")
        _hold(session, 8, direction, "X", reason=f"{label}_super2")
        _hold(session, 20, reason=f"{label}_super_fuse2")
    _hold(session, spin_frames, direction, "B", "A", reason=f"{label}_spin")
    entered = False
    for _ in range(hold_frames):
        state = _hold(session, 1, direction, reason=f"{label}_hold")
        if state.room_id == to_room:
            entered = True
            break
    if not entered:
        raise TimeoutError(f"{label}: did not reach 0x{to_room:04X}: {session.state}")
    return _wait_ordinary_room(
        session, to_room, settle_frames=settle_frames, label=label
    )


def play_big_pink_to_ghz(session: ControllerSession) -> SuperMetroidState:
    """Natural Big Pink main-shaft anchor → Green Hill Zone.

    The GHZ door is the lower-right green door, not the upper-right wall beside
    the main-shaft anchor.  This descends through the lower winding morph
    tunnel, unmorphs to fire a Super from the left, and enters the door without
    placement or room/progression writes.

    Charge Beam is a separate side trip below the mass at x≈683. Its natural
    collect is known, but a conventional return is not yet route-ready, so this
    function takes the direct KPDR exit. The active route does not require an
    infinite bomb jump here.
    """
    _require_room(session, ROOM_BIG_PINK, "big_pink_to_ghz")
    ensure_morph(session)

    for _ in range(500):
        state = _hold(session, 1, "LEFT", reason="big_pink_lower_left")
        if state.samus_x <= 560 and state.samus_y >= 1540:
            break
    else:
        raise TimeoutError(f"big_pink_to_ghz: missed lower-left shelf: {state}")

    _unmorph(session)
    for _ in range(220):
        state = _hold(session, 1, "RIGHT", "B", "A", reason="big_pink_lower_drop")
        if state.samus_x >= 665 and state.samus_y >= 1660:
            break
    else:
        raise TimeoutError(f"big_pink_to_ghz: missed lower mass: {state}")

    _hold(session, 30, "RIGHT", "B", reason="big_pink_mass_run")
    _hold(session, 10, reason="big_pink_mass_settle")
    _hold(session, 12, "LEFT", reason="big_pink_mass_brake")
    _hold(session, 8, "A", reason="big_pink_mass_vertical")
    for _ in range(160):
        state = _hold(session, 1, "RIGHT", "A", reason="big_pink_tunnel_mount")
        if state.samus_x >= 705 and 1590 <= state.samus_y <= 1630:
            break
    else:
        raise TimeoutError(f"big_pink_to_ghz: missed morph-tunnel lip: {state}")
    _hold(session, 50, reason="big_pink_tunnel_lip_settle")
    ensure_morph(session)

    for frame in range(500):
        buttons = ("RIGHT", "X") if frame % 45 < 3 else ("RIGHT",)
        state = _hold(session, 1, *buttons, reason="big_pink_bomb_roll")
        if state.samus_x >= 900:
            break
    else:
        raise TimeoutError(f"big_pink_to_ghz: lower bomb-roll stalled: {state}")

    for _ in range(220):
        state = _hold(session, 1, "RIGHT", reason="big_pink_door_roll")
        if state.samus_x >= 970 and state.samus_y >= 1670:
            break
    else:
        raise TimeoutError(f"big_pink_to_ghz: missed green-door pocket: {state}")

    _unmorph(session)
    _select_weapon(session, 2)
    _hold(session, 25, "LEFT", reason="big_pink_super_standoff")
    _hold(session, 3, "RIGHT", reason="big_pink_face_door")
    _hold(session, 3, reason="big_pink_face_door_release")
    _hold(session, 2, "RIGHT", "X", reason="big_pink_green_door_super")
    _hold(session, 60, reason="big_pink_green_door_fuse")
    for _ in range(300):
        state = _hold(session, 1, "RIGHT", "B", reason="big_pink_enter_ghz")
        if state.room_id == ROOM_GHZ:
            break
    else:
        raise TimeoutError(f"big_pink_to_ghz: green door did not open: {state}")
    return _wait_ordinary_room(
        session, ROOM_GHZ, settle_frames=240, label="big_pink_to_ghz"
    )


def play_ghz_to_noob(session: ControllerSession) -> SuperMetroidState:
    """Green Hill Zone ``0x9E52`` → Noob Bridge ``0x9FBA``.

    Exit door is **bottom-right** block ``[127,55]`` (~x=2032, y=880) after a
    full diagonal descent.  At the bottom, a five-block pillar at x≈1456 and
    the blue gate at x≈1600 need a deliberate sequence: jump in place, shoot
    right around y≈888 to open the gate, land, then jump in place again before
    drifting right over the pillar.
    """
    _require_room(session, ROOM_GHZ, "ghz_to_noob")
    try:
        _select_weapon(session, 0)
    except RuntimeError:
        pass

    # Descend the diagonal room while biasing right.  Stop at the central
    # pillar instead of blindly repeating jumps against it.
    reached_pillar = False
    for _ in range(120):
        state = session.state
        if state.room_id == ROOM_NOOB:
            break
        if state.samus_x >= 1445 and state.samus_y >= 885:
            reached_pillar = True
            break
        if state.samus_y < 700:
            # Drop through platforms: walk right + occasional down-look.
            _hold(session, 8, "RIGHT", "B", reason="ghz_run")
            _hold(session, 4, "RIGHT", "B", "A", reason="ghz_hop")
            if state.samus_x > 200 and state.samus_y < 500:
                _hold(session, 6, "RIGHT", reason="ghz_edge")
        else:
            _hold(session, 10, "RIGHT", "B", reason="ghz_low_run")
            _hold(session, 3, "RIGHT", "B", "X", reason="ghz_shoot")
            _hold(session, 20, "RIGHT", "B", "A", reason="ghz_spin")

    if session.state.room_id != ROOM_NOOB and not reached_pillar:
        raise TimeoutError(f"ghz_to_noob: did not reach bottom pillar: {session.state}")

    if session.state.room_id != ROOM_NOOB:
        # Settle against the pillar at x≈1451, then shoot the blue-gate switch
        # while rising.  The useful beam line is only a few pixels tall.
        for _ in range(60):
            state = session.state
            if state.samus_y >= 935:
                break
            _hold(session, 1, reason="ghz_pillar_settle")
        # Let the landing/spin pose finish.  Firing from the transient 0xA4
        # landing pose produces the same coordinates but does not trip the
        # gate switch.
        _hold(session, 20, reason="ghz_pillar_stand")

        gate_line = False
        for _ in range(32):
            state = _hold(session, 1, "A", reason="ghz_gate_jump")
            if 886 <= state.samus_y <= 889:
                gate_line = True
                break
        if not gate_line:
            raise TimeoutError(
                "ghz_to_noob: missed blue-gate shot line "
                f"at ({state.samus_x},{state.samus_y})"
            )
        _hold(session, 3, "RIGHT", "X", reason="ghz_gate_shot")
        _hold(session, 60, reason="ghz_gate_open")

        # Clear the pillar only after the gate has opened.  Adding RIGHT too
        # early clips the pillar face and produces the old x≈1451 loop.
        _hold(session, 24, "A", reason="ghz_pillar_vertical_jump")
        for _ in range(220):
            state = _hold(session, 1, "RIGHT", "B", "A", reason="ghz_pillar_clear")
            if state.samus_x >= 1700:
                break
        if state.samus_x < 1700:
            raise TimeoutError(
                "ghz_to_noob: blue gate/pillar clear failed "
                f"at ({state.samus_x},{state.samus_y}) pose={state.pose}"
            )

        # Bottom-right corridor and blue door.
        for frame in range(500):
            if frame % 24 < 6:
                buttons = ("RIGHT", "B", "X")
            elif frame % 40 >= 28:
                buttons = ("RIGHT", "B", "A")
            else:
                buttons = ("RIGHT", "B")
            state = _hold(session, 1, *buttons, reason="ghz_exit_run")
            if state.room_id == ROOM_NOOB:
                break

    if session.state.room_id != ROOM_NOOB:
        raise TimeoutError(f"ghz_to_noob: still in GHZ: {session.state}")
    return _wait_ordinary_room(
        session, ROOM_NOOB, settle_frames=200, label="ghz_to_noob"
    )


def play_noob_to_red_tower(session: ControllerSession) -> SuperMetroidState:
    """Noob Bridge ``0x9FBA`` → Red Tower ``0xA253`` (green Super door).

    Layout (6 screens wide):
    - The intended route is the upper pit-block bridge.  The lower floor ends
      at a full-height wall around x=1131; it is not a scroll lock.
    - From the left door/lower floor, jump toward x≈200, brake, jump mostly
      vertically past the underside of the ledge, then hold RIGHT+B+A across
      the upper bridge.  A shallow running jump hits the ledge underside and
      leaves Samus on the dead-end lower floor.
    - Green Super door is on the far right (~x=1467–1516, floor y≈171).
      Door PLM sits near block (92, 13); docs' ``[79,7]`` was approximate.

    Door open: select Supers → run to x≳1380 →
    pulse Super (RIGHT+X) → brief fuse → spin (RIGHT+B+A) into the open door.
    Crouch-Super alone is unreliable; spin-through is what lands Red Tower.
    """
    _require_room(session, ROOM_NOOB, "noob_to_red")
    try:
        _select_weapon(session, 2)
    except RuntimeError:
        pass

    # Climb onto the upper pit-block bridge.  The apparent x≈1131 "barrier"
    # is simply the end of the lower chamber.  Horizontal momentum must be
    # killed before the ledge jump; otherwise Samus reaches x≈251 while still
    # too low, clips its underside, and remains trapped below.
    if session.state.samus_x < 1150:
        for frame in range(150):
            state = _hold(session, 1, "RIGHT", "B", "A", reason="noob_bridge_setup_hop")
            if frame > 45 and state.samus_x >= 190 and state.samus_y >= 155:
                break
        else:
            raise TimeoutError(
                "noob_to_red: could not reach upper-bridge jump setup "
                f"from ({state.samus_x},{state.samus_y})"
            )

        _hold(session, 10, "LEFT", reason="noob_bridge_brake")
        _hold(session, 24, "A", reason="noob_bridge_vertical_jump")
        for _ in range(330):
            state = _hold(session, 1, "RIGHT", "B", "A", reason="noob_bridge_dash")
            if state.samus_x >= 1200:
                break
        if state.samus_x < 1200:
            raise TimeoutError(
                "noob_to_red: failed upper pit-block bridge "
                f"(samus=({state.samus_x},{state.samus_y}) pose={state.pose})"
            )

    stuck = 0
    last_x = float(session.state.samus_x)
    for i in range(1400):
        state = session.state
        if state.room_id == ROOM_RED_TOWER:
            break

        # Leave morph if a long DOWN accidentally ball'd us.
        if state.pose in (39, 40, 137, 138):
            _hold(session, 2, "UP", reason="noob_unmorph")
            _hold(session, 4, reason="noob_unmorph")
            _hold(session, 2, "A", reason="noob_unmorph")
            _hold(session, 6, reason="noob_unmorph")
            continue

        # Door zone: Super pulse + spin-through (do not crouch-hold).
        if state.samus_x >= 1380:
            _hold(session, 2, "RIGHT", "X", reason="noob_super")
            _hold(session, 12, reason="noob_fuse")
            _hold(session, 18, "RIGHT", "B", "A", reason="noob_spin")
            for _ in range(30):
                state = _hold(session, 1, "RIGHT", "B", reason="noob_push")
                if state.room_id == ROOM_RED_TOWER:
                    break
            if state.room_id == ROOM_RED_TOWER:
                break
            continue

        # Right corridor after the upper bridge: run + shoot + hop.
        phase = i % 24
        if state.samus_y > 250:
            state = _hold(session, 2, "LEFT", "A", reason="noob_recover")
        elif phase < 14:
            state = _hold(session, 1, "RIGHT", "B", reason="noob_run")
        elif phase < 18:
            state = _hold(session, 1, "RIGHT", "B", "X", reason="noob_shoot")
        else:
            state = _hold(session, 1, "RIGHT", "B", "A", reason="noob_hop")

        if state.samus_x > last_x + 0.5:
            stuck = 0
            last_x = float(state.samus_x)
        else:
            stuck += 1

        # Generic recovery for enemies / door geometry on the right side.
        if stuck > 25 and state.samus_x >= 1050:
            for _ in range(15):
                _hold(session, 1, "LEFT", "B", reason="noob_backup")
            for _ in range(18):
                _hold(session, 1, "RIGHT", "B", reason="noob_runup")
            for _ in range(45):
                state = _hold(session, 1, "RIGHT", "B", "A", reason="noob_longjump")
                if state.samus_x >= 1150:
                    stuck = 0
                    last_x = float(state.samus_x)
                    break
            if state.samus_x < 1150 and stuck > 50:
                raise TimeoutError(
                    "noob_to_red: stalled before right corridor "
                    f"(samus=({state.samus_x},{state.samus_y}) pose={state.pose})"
                )

    if session.state.room_id != ROOM_RED_TOWER:
        raise TimeoutError(f"noob_to_red: {session.state}")
    return _wait_ordinary_room(
        session, ROOM_RED_TOWER, settle_frames=220, label="noob_to_red"
    )


def play_red_tower_to_bat(session: ControllerSession) -> SuperMetroidState:
    """Natural Noob-door spawn → Red Tower bottom → Bat Room ``0xA3DD``."""
    _require_room(session, ROOM_RED_TOWER, "red_to_bat")
    _unmorph(session)

    # Zigzag down the tall upper shaft.  x<=45 is intentional: switching at
    # x<=35 leaves Samus perched at x≈37/y≈1499.
    direction = "RIGHT"
    for _ in range(1600):
        state = session.state
        if state.samus_y >= 1600:
            break
        if state.samus_x >= 225:
            direction = "LEFT"
        elif state.samus_x <= 45:
            direction = "RIGHT"
        _hold(session, 1, direction, "B", reason="red_upper_zigzag")
    else:
        raise TimeoutError(f"red_to_bat: upper descent stalled: {state}")
    _hold(session, 40, reason="red_floor_settle")

    # Open the temporary floor and cross it during the bomb timer.
    ensure_morph(session)
    for _ in range(100):
        state = session.state
        if 148 <= state.samus_x <= 152:
            break
        direction = "LEFT" if state.samus_x > 152 else "RIGHT"
        _hold(session, 1, direction, reason="red_floor_bomb_position")
    _hold(session, 2, "X", reason="red_floor_bomb")
    _hold(session, 8, "LEFT", reason="red_floor_bomb_retreat")
    _hold(session, 40, reason="red_floor_bomb_wait")
    for _ in range(180):
        state = _hold(session, 1, "RIGHT", reason="red_floor_cross")
        if state.samus_y >= 1650:
            break
    else:
        raise TimeoutError(f"red_to_bat: timed floor crossing failed: {state}")

    for frame in range(500):
        buttons = ("LEFT", "X") if frame % 45 < 2 else ("LEFT",)
        state = _hold(session, 1, *buttons, reason="red_tunnel_bomb_roll")
        if state.samus_y >= 1880:
            break
    else:
        raise TimeoutError(f"red_to_bat: upper tunnel descent stalled: {state}")

    for _ in range(300):
        state = _hold(session, 1, "RIGHT", reason="red_tunnel_right")
        if state.samus_y >= 2090:
            break
    else:
        raise TimeoutError(f"red_to_bat: lower tunnel entry stalled: {state}")

    direction = "LEFT"
    for _ in range(600):
        state = session.state
        if state.samus_y >= 2440:
            break
        if state.samus_x >= 220:
            direction = "LEFT"
        elif state.samus_x <= 40:
            direction = "RIGHT"
        _hold(session, 1, direction, reason="red_lower_zigzag")
    else:
        raise TimeoutError(f"red_to_bat: lower descent stalled: {state}")

    _hold(session, 40, reason="red_bottom_settle")
    _unmorph(session)
    _select_weapon(session, 0)
    for frame in range(420):
        phase = frame % 30
        if phase < 5:
            buttons = ("RIGHT", "B", "X")
        elif phase >= 21:
            buttons = ("RIGHT", "B", "A")
        else:
            buttons = ("RIGHT", "B")
        state = _hold(session, 1, *buttons, reason="red_bottom_exit")
        if state.room_id == ROOM_BAT:
            break
    else:
        raise TimeoutError(f"red_to_bat: Bat Room door not reached: {state}")
    return _wait_ordinary_room(session, ROOM_BAT, settle_frames=240, label="red_to_bat")


def play_bat_to_below_spazer(session: ControllerSession) -> SuperMetroidState:
    """Cross Bat Room's three dry platforms and enter Below Spazer."""
    _require_room(session, ROOM_BAT, "bat_to_below_spazer")
    _unmorph(session)
    _select_weapon(session, 0)
    # Preserve five frames of the natural door-exit glide (x39→x≈48); isolated
    # anchors already include these boot-settle frames.
    _hold(session, 5, reason="bat_entry_glide")

    # The natural Red Tower door spawn is x≈39/y≈108 (above the older staged
    # x≈65/y≈155 anchor), so preserve a longer run-up to the first platform.
    _hold(session, 35, "RIGHT", "B", reason="bat_first_runup")
    _hold(session, 60, "RIGHT", "B", "A", reason="bat_first_jump")
    _hold(session, 30, reason="bat_first_land")
    if session.state.samus_x < 210:
        raise TimeoutError(
            f"bat_to_below_spazer: missed first platform: {session.state}"
        )

    _hold(session, 8, "RIGHT", "B", reason="bat_second_runup")
    _hold(session, 20, "RIGHT", "B", "A", reason="bat_second_jump")
    _hold(session, 80, reason="bat_second_land")
    state = session.state
    if not (345 <= state.samus_x <= 380 and 165 <= state.samus_y <= 180):
        raise TimeoutError(f"bat_to_below_spazer: missed middle platform: {state}")

    _hold(session, 48, "RIGHT", "B", "A", reason="bat_third_jump")
    _hold(session, 60, reason="bat_third_land")
    if session.state.samus_x < 405:
        raise TimeoutError(
            f"bat_to_below_spazer: missed right platform: {session.state}"
        )
    return play_run_shoot_exit(
        session,
        from_room=ROOM_BAT,
        to_room=ROOM_BELOW_SPAZER,
        direction="RIGHT",
        label="bat_to_below_spazer",
        run_frames=20,
        shoot_frames=4,
        spin_frames=30,
        hold_frames=240,
        settle_frames=260,
    )


def play_below_spazer_to_west(session: ControllerSession) -> SuperMetroidState:
    """Below Spazer water room → West Tunnel."""
    _require_room(session, ROOM_BELOW_SPAZER, "below_spazer_to_west")
    # Let the door-exit running pose settle before `_unmorph`; pose 9/10 is
    # intentionally handled by that shared helper and would otherwise turn
    # this ordinary entry glide into an unwanted jump.
    _hold(session, 6, reason="below_spazer_entry_glide")
    _unmorph(session)
    _select_weapon(session, 0)
    for frame in range(2000):
        buttons = ("RIGHT", "B", "X") if frame % 35 < 10 else ("RIGHT", "B", "A")
        state = _hold(session, 1, *buttons, reason="below_spazer_right")
        if state.room_id == ROOM_WEST_TUNNEL:
            break
    else:
        raise TimeoutError(f"below_spazer_to_west: West Tunnel not reached: {state}")
    return _wait_ordinary_room(
        session, ROOM_WEST_TUNNEL, settle_frames=260, label="below_spazer_to_west"
    )


def play_west_to_glass(session: ControllerSession) -> SuperMetroidState:
    """West Tunnel → Glass Tunnel."""
    return play_run_shoot_exit(
        session,
        from_room=ROOM_WEST_TUNNEL,
        to_room=ROOM_GLASS,
        direction="RIGHT",
        label="west_to_glass",
        run_frames=80,
        shoot_frames=5,
        spin_frames=50,
        hold_frames=300,
        settle_frames=260,
    )


def play_glass_to_east(session: ControllerSession) -> SuperMetroidState:
    """Glass Tunnel → East Tunnel."""
    return play_run_shoot_exit(
        session,
        from_room=ROOM_GLASS,
        to_room=ROOM_EAST_TUNNEL,
        direction="RIGHT",
        label="glass_to_east",
        run_frames=80,
        shoot_frames=5,
        spin_frames=50,
        hold_frames=300,
        settle_frames=260,
    )


def play_east_to_warehouse(session: ControllerSession) -> SuperMetroidState:
    """East Tunnel → natural Warehouse Entrance spawn."""
    _require_room(session, ROOM_EAST_TUNNEL, "east_to_warehouse")
    for frame in range(1600):
        phase = frame % 25
        if phase < 5:
            buttons = ("RIGHT", "B", "X")
        elif phase >= 18:
            buttons = ("RIGHT", "B", "A")
        else:
            buttons = ("RIGHT", "B")
        state = _hold(session, 1, *buttons, reason="east_tunnel_right")
        if state.room_id == ROOM_WAREHOUSE:
            break
    else:
        raise TimeoutError(f"east_to_warehouse: Warehouse not reached: {state}")
    # This multi-screen room commonly remains in game state 11 for >100f.
    return _wait_ordinary_room(
        session, ROOM_WAREHOUSE, settle_frames=900, label="east_to_warehouse"
    )


def play_red_tower_to_warehouse(session: ControllerSession) -> SuperMetroidState:
    """Compose the verified controller-only Red Tower → Warehouse prefix."""
    play_red_tower_to_bat(session)
    play_bat_to_below_spazer(session)
    play_below_spazer_to_west(session)
    play_west_to_glass(session)
    play_glass_to_east(session)
    return play_east_to_warehouse(session)


def play_warehouse_wall_to_lower_lip(
    session: ControllerSession,
) -> SuperMetroidState:
    """Open Warehouse's three Super blocks and reach the lower-right lip.

    The stack at block x=15 is vertical: crouch-Super hits y=9, standing-Super
    hits y=8, and a five-frame hop-Super hits y=7.  This crosses the stack
    controller-only, but deliberately stops at x≈507/y≈315.  The no-Hi-Jump
    climb from that lower lip to the upper-right ledge is still open, so this
    is not a Warehouse→Zeela clearance.
    """
    _require_room(session, ROOM_WAREHOUSE, "warehouse_wall")
    _unmorph(session)
    _select_weapon(session, 2)
    for _ in range(160):
        state = _hold(session, 1, "RIGHT", "B", reason="warehouse_wall_runup")
        if state.samus_x >= 75:
            break
    _hold(session, 30, reason="warehouse_super_cooldown")

    _hold(session, 8, "DOWN", reason="warehouse_crouch")
    _hold(session, 1, "X", reason="warehouse_bottom_super")
    _hold(session, 30, reason="warehouse_bottom_open")
    _hold(session, 5, "UP", reason="warehouse_stand")
    _hold(session, 4, reason="warehouse_stand_settle")
    _hold(session, 1, "X", reason="warehouse_middle_super")
    _hold(session, 30, reason="warehouse_middle_open")
    _hold(session, 5, "A", reason="warehouse_tiny_hop")
    _hold(session, 1, "RIGHT", "X", reason="warehouse_top_super")
    _hold(session, 24, reason="warehouse_top_open")

    for _ in range(500):
        state = _hold(session, 1, "RIGHT", "B", "A", reason="warehouse_cross_stack")
        if state.samus_x >= 500 and state.samus_y >= 300:
            break
    else:
        raise TimeoutError(f"warehouse_wall: lower lip not reached: {state}")
    _hold(session, 30, reason="warehouse_lower_lip_settle")
    state = session.state
    if state.samus_x < 500 or state.samus_y < 300:
        raise TimeoutError(f"warehouse_wall: unstable lower lip: {state}")
    return state


def play_warehouse_to_business(session: ControllerSession) -> SuperMetroidState:
    """Warehouse Entrance elevator → natural Business Center spawn."""
    _require_room(session, ROOM_WAREHOUSE, "warehouse_to_business")
    _unmorph(session)
    for _ in range(180):
        state = session.state
        if state.samus_x >= 126:
            break
        _hold(session, 1, "RIGHT", reason="warehouse_elevator_position")
    _hold(session, 5, "LEFT", reason="warehouse_elevator_brake")
    _hold(session, 20, reason="warehouse_elevator_settle")
    for _ in range(700):
        state = _hold(session, 1, "DOWN", reason="warehouse_elevator_down")
        if state.room_id == ROOM_BUSINESS:
            break
    else:
        raise TimeoutError(f"warehouse_to_business: {state}")
    return _wait_ordinary_room(
        session, ROOM_BUSINESS, settle_frames=320, label="warehouse_to_business"
    )


def play_business_to_hj_shaft(session: ControllerSession) -> SuperMetroidState:
    """Descend Business Center and enter the lower-left red Hi-Jump door."""
    _require_room(session, ROOM_BUSINESS, "business_to_hj_shaft")
    # The room becomes ordinary while Samus is still riding the incoming
    # elevator.  Wait for its center stop, then walk off before descending.
    for _ in range(500):
        state = _hold(session, 1, reason="business_incoming_elevator")
        if state.pose == 0 and 675 <= state.samus_y <= 690:
            break
    for _ in range(120):
        state = _hold(session, 1, "RIGHT", reason="business_elevator_dismount")
        if state.pose != 0 and state.samus_x >= 145:
            break
    _unmorph(session)

    # Descend the lower shaft.  Run off each alternating shelf; Hi-Jump is not
    # owned yet, so do not rely on jump height here.
    direction = "LEFT"
    for frame in range(4200):
        state = session.state
        if state.samus_y >= 1390:
            break
        if state.pose in (137, 138):
            _unmorph(session)
        if state.samus_x <= 45:
            direction = "RIGHT"
        elif state.samus_x >= 215:
            direction = "LEFT"
        phase = frame % 90
        if phase < 58:
            buttons = (direction, "B")
        else:
            buttons = (direction, "B", "A")
        _hold(session, 1, *buttons, reason="business_descend")
    else:
        raise TimeoutError(f"business_to_hj_shaft: descent stalled: {state}")
    _hold(session, 60, reason="business_bottom_settle")

    # The Sova can be tanked.  Approach from its left, then fire the red-door
    # Super explicitly facing left (direction+shot on the same first frame can
    # otherwise use the previous facing).
    for _ in range(320):
        state = session.state
        if state.samus_x <= 70:
            break
        _hold(session, 1, "LEFT", "B", reason="business_red_door_approach")
    for _ in range(100):
        state = session.state
        if state.samus_x >= 92:
            break
        _hold(session, 1, "RIGHT", reason="business_red_door_standoff")
    _hold(session, 5, "LEFT", reason="business_red_door_brake")
    _hold(session, 20, reason="business_red_door_settle")
    _select_weapon(session, 2)
    _hold(session, 3, "LEFT", reason="business_face_red_door")
    _hold(session, 3, reason="business_face_red_door_release")
    _hold(session, 2, "LEFT", "X", reason="business_red_door_super")
    _hold(session, 80, reason="business_red_door_fuse")
    for _ in range(500):
        state = _hold(session, 1, "LEFT", "B", "A", reason="business_enter_hj_shaft")
        if state.room_id == ROOM_HJ_SHAFT:
            break
    else:
        raise TimeoutError(f"business_to_hj_shaft: red door failed: {state}")
    return _wait_ordinary_room(
        session, ROOM_HJ_SHAFT, settle_frames=280, label="business_to_hj_shaft"
    )


def play_hj_shaft_to_hj_room(session: ControllerSession) -> SuperMetroidState:
    """Hi-Jump E-Tank room right door → lower-left Hi-Jump Boots door."""
    _require_room(session, ROOM_HJ_SHAFT, "hj_shaft_to_hj")
    _unmorph(session)
    _select_weapon(session, 0)

    # Cross the E-Tank plinth, collecting it naturally, and enter the low
    # morph tunnel.
    for _ in range(220):
        state = session.state
        if state.samus_x <= 390:
            break
        _hold(session, 1, "LEFT", "B", reason="hj_shaft_etank_approach")
    # The item fanfare holds Samus against the right face of the plinth.
    # Morph back to the right after it finishes, then use that clear runway
    # for the leftward jump over the statue.
    _hold(session, 480, reason="hj_shaft_etank_fanfare")
    ensure_morph(session)
    for _ in range(120):
        state = _hold(session, 1, "RIGHT", reason="hj_shaft_etank_backoff")
        if state.samus_x >= 470:
            break
    _unmorph(session)
    _hold(session, 20, reason="hj_shaft_etank_stand")
    _hold(session, 20, "LEFT", "B", reason="hj_shaft_etank_runup")
    for _ in range(140):
        state = _hold(session, 1, "LEFT", "B", "A", reason="hj_shaft_etank_jump")
        if state.samus_x <= 310 and state.samus_y >= 180:
            break
    _hold(session, 30, reason="hj_shaft_etank_jump_land")
    for _ in range(160):
        state = session.state
        if state.samus_x <= 310 and state.samus_y >= 180:
            break
        _hold(session, 1, "LEFT", "B", reason="hj_shaft_low_tunnel")
    ensure_morph(session)
    for _ in range(700):
        state = _hold(session, 1, "LEFT", reason="hj_shaft_morph_left")
        if state.samus_x <= 40 and state.samus_y >= 450:
            break
    else:
        raise TimeoutError(f"hj_shaft_to_hj: lower tunnel stalled: {state}")

    # Jump the short shaft beside the blue door and shoot it while rising.
    _unmorph(session)
    _select_weapon(session, 0)
    _hold(session, 12, reason="hj_shaft_door_release")
    for _ in range(80):
        state = _hold(session, 1, "A", reason="hj_shaft_door_jump")
        if state.samus_y <= 390:
            break
    _hold(session, 2, "LEFT", "A", "X", reason="hj_shaft_blue_door_shot")
    for _ in range(420):
        state = _hold(session, 1, "LEFT", "A", reason="hj_shaft_enter_hj")
        if state.room_id == ROOM_HJ:
            break
    else:
        raise TimeoutError(f"hj_shaft_to_hj: blue door failed: {state}")
    return _wait_ordinary_room(
        session, ROOM_HJ, settle_frames=260, label="hj_shaft_to_hj"
    )


def play_hj_room_collect(session: ControllerSession) -> SuperMetroidState:
    """Destroy both pillar shot-block sets and collect Hi-Jump naturally."""
    _require_room(session, ROOM_HJ, "hj_room_collect")
    _unmorph(session)
    _select_weapon(session, 0)
    _hold(session, 20, reason="hj_room_entry_settle")

    # Left-facing down-shot opens the first half of the pillar.
    _hold(session, 12, "LEFT", "B", reason="hj_room_first_runup")
    for _ in range(70):
        state = _hold(session, 1, "LEFT", "B", "A", reason="hj_room_first_jump")
        if state.samus_y <= 52:
            break
    _hold(session, 1, "DOWN", reason="hj_room_first_aim_down")
    _hold(session, 1, "X", reason="hj_room_first_downshot")
    _hold(session, 80, reason="hj_room_first_land")

    # Face right, jump vertically, and down-shoot the other orientation.
    _hold(session, 2, "RIGHT", reason="hj_room_face_right")
    _hold(session, 10, reason="hj_room_face_right_settle")
    for _ in range(80):
        state = _hold(session, 1, "A", reason="hj_room_second_jump")
        if state.samus_y <= 53:
            break
    _hold(session, 1, "DOWN", reason="hj_room_second_aim_down")
    _hold(session, 1, "X", reason="hj_room_second_downshot")
    _hold(session, 80, reason="hj_room_second_land")

    # Rebuild a leftward run and cross the now-open pillar.
    _hold(session, 12, "RIGHT", "B", reason="hj_room_cross_backoff")
    _hold(session, 15, "LEFT", "B", reason="hj_room_cross_runup")
    for _ in range(100):
        state = _hold(session, 1, "LEFT", "B", "A", reason="hj_room_cross_pillar")
        if state.samus_x < 120:
            break
    _hold(session, 80, reason="hj_room_left_land")

    # Shoot the Chozo statue from the right, then walk into the real PLM.
    for _ in range(80):
        state = session.state
        if state.samus_x >= 115:
            break
        _hold(session, 1, "RIGHT", reason="hj_room_statue_approach")
    _hold(session, 12, "LEFT", reason="hj_room_statue_brake")
    _hold(session, 8, reason="hj_room_statue_settle")
    _hold(session, 3, "LEFT", reason="hj_room_statue_face")
    _hold(session, 3, reason="hj_room_statue_face_release")
    _hold(session, 1, "X", reason="hj_room_statue_shot")
    _hold(session, 60, reason="hj_room_statue_open")
    for _ in range(180):
        state = _hold(session, 1, "LEFT", reason="hj_room_collect_item")
        if state.collected_items & ITEM_HI_JUMP:
            break
    else:
        raise TimeoutError(f"hj_room_collect: Hi-Jump PLM not collected: {state}")
    # Item-room controls remain locked substantially longer than the visible
    # pickup flash.  Let the full fanfare finish so the return composes without
    # relying on a save/load input reset.
    _hold(session, 480, reason="hj_room_item_fanfare")
    return session.state


def play_warehouse_to_hijump(session: ControllerSession) -> SuperMetroidState:
    """Natural Warehouse entry → real Hi-Jump Boots collection."""
    play_warehouse_to_business(session)
    play_business_to_hj_shaft(session)
    play_hj_shaft_to_hj_room(session)
    return play_hj_room_collect(session)


def play_hj_room_to_shaft(session: ControllerSession) -> SuperMetroidState:
    """Collected Hi-Jump alcove → natural E-Tank-room left-door spawn."""
    _require_room(session, ROOM_HJ, "hj_room_to_shaft")
    _unmorph(session)
    _hold(session, 20, reason="hj_room_return_settle")
    for _ in range(80):
        state = session.state
        if state.samus_x <= 80:
            break
        _hold(session, 1, "LEFT", "B", reason="hj_room_return_backoff")
    _hold(session, 8, "RIGHT", reason="hj_room_return_brake")
    _hold(session, 10, reason="hj_room_return_release")
    _hold(session, 12, "RIGHT", "B", reason="hj_room_return_runup")
    for _ in range(120):
        state = _hold(session, 1, "RIGHT", "B", "A", reason="hj_room_return_cross")
        if state.samus_x >= 181:
            break
    _hold(session, 80, reason="hj_room_return_land")

    _unmorph(session)
    _select_weapon(session, 0)
    for _ in range(80):
        state = session.state
        if state.samus_x <= 185:
            break
        _hold(session, 1, "LEFT", reason="hj_room_return_door_backoff")
    _hold(session, 8, "RIGHT", reason="hj_room_return_door_brake")
    _hold(session, 8, reason="hj_room_return_door_settle")
    _hold(session, 3, "RIGHT", reason="hj_room_return_face_door")
    _hold(session, 3, reason="hj_room_return_face_release")
    _hold(session, 1, "X", reason="hj_room_return_door_shot")
    _hold(session, 40, reason="hj_room_return_door_open")
    for _ in range(420):
        state = _hold(session, 1, "RIGHT", "B", "A", reason="hj_room_return_enter")
        if state.room_id == ROOM_HJ_SHAFT:
            break
    else:
        raise TimeoutError(f"hj_room_to_shaft: {state}")
    return _wait_ordinary_room(
        session, ROOM_HJ_SHAFT, settle_frames=280, label="hj_room_to_shaft"
    )


def play_hj_shaft_to_business(session: ControllerSession) -> SuperMetroidState:
    """Use Hi-Jump's intended left climb and bomb tunnel back to Business."""
    _require_room(session, ROOM_HJ_SHAFT, "hj_shaft_to_business")
    _unmorph(session)
    _hold(session, 50, reason="hj_return_bottom_land")

    # Bottom floor → right shelf.
    _hold(session, 10, reason="hj_return_jump_release")
    for frame in range(125):
        buttons = ("A",) if frame < 18 else ("RIGHT", "A")
        _hold(session, 1, *buttons, reason="hj_return_first_jump")
    _hold(session, 80, reason="hj_return_first_land")

    # Right shelf → upper-left slope.
    _unmorph(session)
    _hold(session, 50, reason="hj_return_shelf_stand")
    for _ in range(80):
        state = session.state
        if state.samus_x <= 82:
            break
        _hold(session, 1, "LEFT", reason="hj_return_shelf_position")
    _hold(session, 6, "RIGHT", reason="hj_return_shelf_brake")
    _hold(session, 8, reason="hj_return_shelf_release")
    for frame in range(130):
        buttons = ("A",) if frame < 65 else ("LEFT", "A")
        _hold(session, 1, *buttons, reason="hj_return_second_jump")
    _hold(session, 50, reason="hj_return_second_land")

    # Upper-left slope → one-tile morph tunnel.
    _unmorph(session)
    _hold(session, 40, reason="hj_return_slope_stand")
    for frame in range(110):
        buttons = ("A",) if frame < 18 else ("RIGHT", "B", "A")
        state = _hold(session, 1, *buttons, reason="hj_return_top_jump")
        if frame > 55 and state.samus_y <= 95 and state.pose in (1, 2, 9, 10, 137, 138):
            break
    _hold(session, 40, reason="hj_return_top_land")

    # Bomb through the missile tunnel.  The explosions also naturally kill
    # the Sova, satisfying the gray-door lock.
    ensure_morph(session)
    for frame in range(1100):
        buttons = ("RIGHT", "X") if frame % 45 < 2 else ("RIGHT",)
        state = _hold(session, 1, *buttons, reason="hj_return_bomb_tunnel")
        if state.samus_x >= 350:
            break
    else:
        raise TimeoutError(f"hj_shaft_to_business: tunnel stalled: {state}")
    if state.enemies_killed < 1:
        for frame in range(500):
            buttons = ("RIGHT", "X") if frame % 40 < 2 else ("RIGHT",)
            state = _hold(session, 1, *buttons, reason="hj_return_sova_cleanup")
            if state.enemies_killed >= 1:
                break

    _hold(session, 80, "RIGHT", reason="hj_return_gray_approach")
    _unmorph(session)
    _select_weapon(session, 0)
    for frame in range(600):
        buttons = ("RIGHT", "B", "X") if frame % 30 < 4 else ("RIGHT", "B", "A")
        state = _hold(session, 1, *buttons, reason="hj_return_gray_exit")
        if state.room_id == ROOM_BUSINESS:
            break
    else:
        raise TimeoutError(f"hj_shaft_to_business: gray door failed: {state}")
    state = _wait_ordinary_room(
        session, ROOM_BUSINESS, settle_frames=280, label="hj_shaft_to_business"
    )
    for _ in range(120):
        state = _hold(session, 1, reason="hj_return_business_floor")
        if state.samus_y >= 1419 and state.pose in (1, 2, 9, 10, 137, 138):
            break
    for _ in range(100):
        state = session.state
        if state.samus_x >= 88:
            break
        _hold(session, 1, "RIGHT", reason="hj_return_business_climb_anchor")
    _hold(session, 4, "LEFT", reason="hj_return_business_anchor_brake")
    _hold(session, 20, reason="hj_return_business_anchor_settle")
    return session.state


def _business_high_jump_platforms(session: ControllerSession) -> None:
    """Bottom Business Center floor → center elevator (Hi-Jump route)."""
    # Four forgiving setup jumps land on the first left platform (~y=1339).
    _unmorph(session)
    for direction in ("RIGHT", "LEFT", "LEFT", "RIGHT"):
        _hold(session, 12, reason="business_climb_release")
        _hold(session, 85, direction, "B", "A", reason="business_climb_setup")
        _hold(session, 30, reason="business_climb_setup_land")

    # y1339 → y1227.
    _unmorph(session)
    _hold(session, 20, reason="business_1339_settle")
    for _ in range(80):
        if session.state.samus_x <= 84:
            break
        _hold(session, 1, "LEFT", reason="business_1339_position")
    _hold(session, 4, "RIGHT", reason="business_1339_brake")
    _hold(session, 8, reason="business_1339_release")
    for frame in range(120):
        if frame < 14:
            buttons = ("LEFT", "A")
        elif frame < 24:
            buttons = ("A",)
        else:
            buttons = ("RIGHT", "A")
        state = _hold(session, 1, *buttons, reason="business_to_1227")
        if frame > 45 and state.samus_y == 1227 and state.samus_x >= 120:
            break
    _hold(session, 3, "LEFT", reason="business_1227_brake")
    _hold(session, 20, reason="business_1227_settle")

    # y1227 → right platform y1147.
    _unmorph(session)
    _hold(session, 15, reason="business_1227_release")
    for _ in range(80):
        if session.state.samus_x <= 105:
            break
        _hold(session, 1, "LEFT", reason="business_1227_back")
    _hold(session, 4, "RIGHT", reason="business_1227_brake2")
    _hold(session, 4, reason="business_1227_run_release")
    _hold(session, 8, "RIGHT", "B", reason="business_1227_runup")
    for frame in range(140):
        buttons = ("RIGHT", "B", "A") if frame < 90 else ("LEFT", "A")
        state = _hold(session, 1, *buttons, reason="business_to_1147")
        if frame > 88 and state.samus_y == 1147 and state.samus_x >= 192:
            break
    _hold(session, 3, "LEFT", reason="business_1147_brake")
    _hold(session, 20, reason="business_1147_settle")

    # y1147 → center platform y1067.
    _unmorph(session)
    _hold(session, 16, reason="business_1147_release")
    for frame in range(150):
        buttons = ("LEFT", "B", "A") if frame < 85 else ("RIGHT", "A")
        state = _hold(session, 1, *buttons, reason="business_to_1067")
        if frame > 100 and state.samus_y == 1067 and 95 <= state.samus_x <= 160:
            break
    _hold(session, 30, reason="business_1067_settle")

    # y1067 → y987 through the left edge of the overhead platform.
    _unmorph(session)
    _hold(session, 12, reason="business_1067_release")
    for _ in range(80):
        if session.state.samus_x <= 92:
            break
        _hold(session, 1, "LEFT", reason="business_1067_position")
    _hold(session, 4, "RIGHT", reason="business_1067_brake")
    _hold(session, 8, reason="business_1067_jump_release")
    for frame in range(100):
        buttons = ("A",) if frame < 14 else ("RIGHT", "B", "A")
        state = _hold(session, 1, *buttons, reason="business_to_987")
        if frame > 25 and state.samus_y == 987 and state.pose in (1, 2, 9, 10):
            break
    # This landing is on the extreme left pixel of the three-block platform;
    # nudge inward instead of braking back off its edge.
    _hold(session, 4, "RIGHT", reason="business_987_brake")
    _hold(session, 20, reason="business_987_settle")

    # y987 → right platform y907.
    _unmorph(session)
    _hold(session, 12, reason="business_987_release")
    _hold(session, 8, "RIGHT", "B", reason="business_987_runup")
    for frame in range(90):
        state = _hold(session, 1, "RIGHT", "B", "A", reason="business_to_907")
        if frame > 35 and state.samus_y == 907 and state.samus_x >= 160:
            break
    for _ in range(60):
        if session.state.samus_x <= 165:
            break
        _hold(session, 1, "LEFT", reason="business_907_brake")
    _hold(session, 2, "RIGHT", reason="business_907_brake")
    _hold(session, 20, reason="business_907_settle")

    # y907 → center y843.
    _unmorph(session)
    _hold(session, 12, reason="business_907_release")
    for _ in range(80):
        if session.state.samus_x >= 205:
            break
        _hold(session, 1, "RIGHT", reason="business_907_back")
    _hold(session, 3, "LEFT", reason="business_907_brake2")
    _hold(session, 5, reason="business_907_run_release")
    _hold(session, 8, "LEFT", "B", reason="business_907_runup")
    for frame in range(90):
        state = _hold(session, 1, "LEFT", "B", "A", reason="business_to_843")
        if frame > 35 and state.samus_y == 843 and 108 <= state.samus_x <= 160:
            break
    _hold(session, 2, "RIGHT", reason="business_843_brake")
    _hold(session, 20, reason="business_843_settle")

    # y843 → left y779.
    _unmorph(session)
    _hold(session, 12, reason="business_843_release")
    for _ in range(80):
        if session.state.samus_x >= 145:
            break
        _hold(session, 1, "RIGHT", reason="business_843_position")
    _hold(session, 3, "LEFT", reason="business_843_brake2")
    _hold(session, 6, reason="business_843_jump_release")
    for frame in range(90):
        buttons = ("A",) if frame < 10 else ("LEFT", "B", "A")
        state = _hold(session, 1, *buttons, reason="business_to_779")
        if frame > 25 and state.samus_y == 779 and state.samus_x <= 115:
            break
    _hold(session, 2, "RIGHT", reason="business_779_brake")
    _hold(session, 20, reason="business_779_settle")

    # y779 → center elevator y683.
    _unmorph(session)
    _hold(session, 12, reason="business_779_release")
    for _ in range(80):
        if session.state.samus_x <= 76:
            break
        _hold(session, 1, "LEFT", reason="business_779_position")
    _hold(session, 3, "RIGHT", reason="business_779_brake2")
    _hold(session, 6, reason="business_779_jump_release")
    for frame in range(120):
        buttons = ("A",) if frame < 18 else ("RIGHT", "B", "A")
        state = _hold(session, 1, *buttons, reason="business_to_elevator")
        if frame > 45 and state.samus_y == 683 and 95 <= state.samus_x <= 160:
            break
    _hold(session, 2, "LEFT", reason="business_elevator_brake")
    _hold(session, 20, reason="business_elevator_settle")


def play_business_to_warehouse(session: ControllerSession) -> SuperMetroidState:
    """Hi-Jump-assisted Business Center climb and elevator to Warehouse."""
    _require_room(session, ROOM_BUSINESS, "business_to_warehouse")
    _business_high_jump_platforms(session)
    for _ in range(1000):
        state = _hold(session, 1, "UP", reason="business_elevator_up")
        if state.room_id == ROOM_WAREHOUSE:
            break
    else:
        raise TimeoutError(f"business_to_warehouse: elevator failed: {state}")
    state = _wait_ordinary_room(
        session, ROOM_WAREHOUSE, settle_frames=360, label="business_to_warehouse"
    )
    # Let the Warehouse platform finish rising, then step back to the same
    # upper-left anchor used by the natural East Tunnel entry.
    _hold(session, 30, reason="warehouse_elevator_top")
    for _ in range(160):
        state = session.state
        if state.samus_x <= 40 and state.samus_y <= 145:
            break
        _hold(session, 1, "LEFT", reason="warehouse_elevator_exit")
    _hold(session, 30, reason="warehouse_elevator_exit_settle")
    return session.state


def play_hijump_to_warehouse(session: ControllerSession) -> SuperMetroidState:
    """Natural collected Hi-Jump state → Warehouse upper-left anchor."""
    play_hj_room_to_shaft(session)
    play_hj_shaft_to_business(session)
    return play_business_to_warehouse(session)


def play_warehouse_to_zeela_with_hijump(
    session: ControllerSession,
) -> SuperMetroidState:
    """Open Warehouse Super stack, Hi-Jump to the ledge, and enter Zeela."""
    play_warehouse_wall_to_lower_lip(session)
    _unmorph(session)
    _select_weapon(session, 0)
    _hold(session, 12, reason="warehouse_hj_release")
    for _ in range(120):
        if session.state.samus_x <= 445:
            break
        _hold(session, 1, "LEFT", reason="warehouse_hj_backoff")
    _hold(session, 5, "RIGHT", reason="warehouse_hj_brake")
    _hold(session, 8, reason="warehouse_hj_jump_release")
    for frame in range(180):
        buttons = ("A",) if frame < 25 else ("RIGHT", "B", "A")
        state = _hold(session, 1, *buttons, reason="warehouse_hj_climb")
        if state.samus_x >= 720 and state.samus_y <= 160:
            break
    _hold(session, 30, reason="warehouse_hj_door_settle")
    _unmorph(session)
    _select_weapon(session, 0)
    _hold(session, 3, "RIGHT", reason="warehouse_face_zeela")
    _hold(session, 3, reason="warehouse_face_zeela_release")
    _hold(session, 2, "RIGHT", "X", reason="warehouse_zeela_door_shot")
    _hold(session, 30, reason="warehouse_zeela_door_open")
    for _ in range(420):
        state = _hold(session, 1, "RIGHT", "B", "A", reason="warehouse_enter_zeela")
        if state.room_id == ROOM_ZEELA:
            break
    else:
        raise TimeoutError(f"warehouse_to_zeela: {state}")
    return _wait_ordinary_room(
        session, ROOM_ZEELA, settle_frames=280, label="warehouse_to_zeela"
    )


def play_zeela_to_kihunter(session: ControllerSession) -> SuperMetroidState:
    """Warehouse Zeela Room top-left → upper door to Kihunter room."""
    _require_room(session, ROOM_ZEELA, "zeela_to_kihunter")
    _unmorph(session)
    _select_weapon(session, 0)
    _hold(session, 10, reason="zeela_entry_release")
    _hold(session, 10, "A", reason="zeela_first_drop_jump")
    _hold(session, 1, "DOWN", reason="zeela_first_drop_aim")
    _hold(session, 2, "X", reason="zeela_first_drop_shot")
    _hold(session, 80, reason="zeela_first_drop")
    ensure_morph(session)
    for _ in range(300):
        state = _hold(session, 1, "RIGHT", reason="zeela_middle_roll")
        if state.samus_x >= 105 and state.samus_y >= 325:
            break
    _unmorph(session)
    _hold(session, 30, reason="zeela_middle_land")
    _select_weapon(session, 0)
    _hold(session, 8, "A", reason="zeela_second_drop_jump")
    _hold(session, 1, "DOWN", reason="zeela_second_drop_aim")
    _hold(session, 2, "X", reason="zeela_second_drop_shot")
    for _ in range(180):
        state = _hold(session, 1, "LEFT", reason="zeela_second_drop")
        if state.samus_y >= 395:
            break
    _hold(session, 40, reason="zeela_bottom_land")
    ensure_morph(session)
    for frame in range(700):
        buttons = ("RIGHT", "X") if frame % 45 < 2 else ("RIGHT",)
        state = _hold(session, 1, *buttons, reason="zeela_bottom_bomb_roll")
        if state.samus_x >= 400:
            break
    else:
        raise TimeoutError(f"zeela_to_kihunter: tunnel stalled: {state}")
    _unmorph(session)
    _hold(session, 40, reason="zeela_up_door_stand")
    _select_weapon(session, 0)
    _hold(session, 2, "UP", reason="zeela_up_door_aim")
    _hold(session, 2, "UP", "X", reason="zeela_up_door_shot")
    _hold(session, 35, reason="zeela_up_door_open")
    for _ in range(400):
        state = _hold(session, 1, "A", reason="zeela_enter_kihunter")
        if state.room_id == ROOM_WAREHOUSE_KIHUNTER:
            break
    else:
        raise TimeoutError(f"zeela_to_kihunter: up door failed: {state}")
    return _wait_ordinary_room(
        session,
        ROOM_WAREHOUSE_KIHUNTER,
        settle_frames=280,
        label="zeela_to_kihunter",
    )


def play_kihunter_to_baby_kraid(session: ControllerSession) -> SuperMetroidState:
    """Drop through the Warehouse Kihunter floor and take the lower door."""
    _require_room(session, ROOM_WAREHOUSE_KIHUNTER, "kihunter_to_baby")
    _hold(session, 80, reason="kihunter_entry_floor")
    _unmorph(session)
    _select_weapon(session, 0)
    for _ in range(300):
        if session.state.samus_x >= 350:
            break
        _hold(session, 1, "RIGHT", "B", reason="kihunter_drop_position")
    _hold(session, 6, "LEFT", reason="kihunter_drop_brake")
    _hold(session, 10, reason="kihunter_drop_settle")
    _hold(session, 3, "RIGHT", reason="kihunter_drop_exact")
    _hold(session, 2, "LEFT", reason="kihunter_drop_exact_brake")
    _hold(session, 10, reason="kihunter_drop_exact_settle")
    ensure_morph(session)
    _hold(session, 2, "X", reason="kihunter_floor_bomb")
    _hold(session, 55, reason="kihunter_floor_bomb_wait")
    _hold(session, 2, "X", reason="kihunter_floor_bomb2")
    for _ in range(180):
        state = _hold(session, 1, reason="kihunter_floor_drop")
        if state.samus_y >= 310:
            break
    ensure_morph(session)
    for _ in range(160):
        state = _hold(session, 1, "LEFT", reason="kihunter_shaft_align")
        if state.samus_y >= 350:
            break
    for _ in range(360):
        state = _hold(session, 1, "RIGHT", reason="kihunter_lower_roll")
        if state.samus_x >= 470:
            break
    _unmorph(session)
    _select_weapon(session, 0)
    for frame in range(500):
        buttons = ("RIGHT", "B", "X") if frame % 25 < 5 else ("RIGHT", "B", "A")
        state = _hold(session, 1, *buttons, reason="kihunter_enter_baby")
        if state.room_id == ROOM_BABY_KRAID:
            break
    else:
        raise TimeoutError(f"kihunter_to_baby: {state}")
    return _wait_ordinary_room(
        session, ROOM_BABY_KRAID, settle_frames=280, label="kihunter_to_baby"
    )


def _baby_kraid_sweep(
    session: ControllerSession,
    direction: str,
    target_x: int,
    *,
    limit: int,
    label: str,
) -> None:
    for frame in range(limit):
        phase = frame % 24
        if phase < 3:
            buttons = (direction, "X")
        elif phase >= 14:
            buttons = (direction, "B", "A")
        else:
            buttons = (direction, "B")
        state = _hold(session, 1, *buttons, reason=label)
        if direction == "RIGHT" and state.samus_x >= target_x:
            return
        if direction == "LEFT" and state.samus_x <= target_x:
            return
    raise TimeoutError(f"{label}: {session.state}")


def play_baby_kraid_to_eye(session: ControllerSession) -> SuperMetroidState:
    """Kill the three pirates and Mini-Kraid, then take the right gray door."""
    _require_room(session, ROOM_BABY_KRAID, "baby_kraid_to_eye")
    _hold(session, 100, reason="baby_kraid_entry_floor")
    _unmorph(session)
    _select_weapon(session, 2)
    _baby_kraid_sweep(session, "RIGHT", 1490, limit=1700, label="baby_kraid_forward")
    if session.state.enemies_killed < session.state.num_enemies:
        _baby_kraid_sweep(session, "LEFT", 50, limit=1900, label="baby_kraid_cleanup")
    _baby_kraid_sweep(session, "RIGHT", 1490, limit=1900, label="baby_kraid_return")
    for _ in range(600):
        state = _hold(session, 1, "RIGHT", "B", "A", reason="baby_kraid_enter_eye")
        if state.room_id == ROOM_KRAID_EYE:
            break
    else:
        raise TimeoutError(f"baby_kraid_to_eye: gray door failed: {state}")
    return _wait_ordinary_room(
        session, ROOM_KRAID_EYE, settle_frames=300, label="baby_kraid_to_eye"
    )


def play_eye_to_kraid(session: ControllerSession) -> SuperMetroidState:
    """Cross Kraid Eye Door Room and open the eye door with Supers."""
    _require_room(session, ROOM_KRAID_EYE, "eye_to_kraid")
    _hold(session, 100, reason="kraid_eye_entry_floor")
    _unmorph(session)
    _select_weapon(session, 2)
    for frame in range(1800):
        phase = frame % 28
        if phase < 3:
            buttons = ("RIGHT", "X")
        elif phase >= 16:
            buttons = ("RIGHT", "B", "A")
        else:
            buttons = ("RIGHT", "B")
        state = _hold(session, 1, *buttons, reason="kraid_eye_run")
        if state.room_id == ROOM_KRAID:
            break
    else:
        raise TimeoutError(f"eye_to_kraid: eye door failed: {state}")
    return _wait_ordinary_room(
        session, ROOM_KRAID, settle_frames=340, label="eye_to_kraid"
    )


def play_warehouse_to_kraid_with_hijump(
    session: ControllerSession,
) -> SuperMetroidState:
    """Natural Warehouse anchor with Hi-Jump → natural Kraid-room entry."""
    if not session.state.collected_items & ITEM_HI_JUMP:
        raise RuntimeError("warehouse_to_kraid_with_hijump: Hi-Jump not collected")
    play_warehouse_to_zeela_with_hijump(session)
    play_zeela_to_kihunter(session)
    play_kihunter_to_baby_kraid(session)
    play_baby_kraid_to_eye(session)
    return play_eye_to_kraid(session)


def play_warehouse_hijump_kraid(session: ControllerSession) -> SuperMetroidState:
    """Composed safer route: Warehouse → Hi-Jump → Warehouse → Kraid."""
    play_warehouse_to_hijump(session)
    play_hijump_to_warehouse(session)
    return play_warehouse_to_kraid_with_hijump(session)
