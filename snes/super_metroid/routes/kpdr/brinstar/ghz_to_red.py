"""Green Hill Zone → Noob Bridge → Red Tower."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    play_run_shoot_exit,
    require_room,
    select_weapon,
    unmorph,
    vertical_hop,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.rooms import (
    ITEM_HI_JUMP,
    ROOM_BABY_KRAID,
    ROOM_BAT,
    ROOM_BELOW_SPAZER,
    ROOM_BIG_PINK,
    ROOM_BUSINESS,
    ROOM_EAST_TUNNEL,
    ROOM_GHZ,
    ROOM_GLASS,
    ROOM_HJ,
    ROOM_HJ_SHAFT,
    ROOM_KRAID,
    ROOM_KRAID_EYE,
    ROOM_NOOB,
    ROOM_RED_TOWER,
    ROOM_WAREHOUSE,
    ROOM_WAREHOUSE_KIHUNTER,
    ROOM_WEST_TUNNEL,
    ROOM_ZEELA,
)
from super_metroid.routes.runtime import ControllerSession

def play_ghz_to_noob(session: ControllerSession) -> SuperMetroidState:
    """Green Hill Zone ``0x9E52`` → Noob Bridge ``0x9FBA``.

    Exit door is **bottom-right** block ``[127,55]`` (~x=2032, y=880) after a
    full diagonal descent.  At the bottom, a five-block pillar at x≈1456 and
    the blue gate at x≈1600 need a deliberate sequence: jump in place, shoot
    right around y≈888 to open the gate, land, then jump in place again before
    drifting right over the pillar.
    """
    require_room(session, ROOM_GHZ, "ghz_to_noob")
    try:
        select_weapon(session, 0)
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
            hold(session, 8, "RIGHT", "B", reason="ghz_run")
            hold(session, 4, "RIGHT", "B", "A", reason="ghz_hop")
            if state.samus_x > 200 and state.samus_y < 500:
                hold(session, 6, "RIGHT", reason="ghz_edge")
        else:
            hold(session, 10, "RIGHT", "B", reason="ghz_low_run")
            hold(session, 3, "RIGHT", "B", "X", reason="ghz_shoot")
            hold(session, 20, "RIGHT", "B", "A", reason="ghz_spin")

    if session.state.room_id != ROOM_NOOB and not reached_pillar:
        raise TimeoutError(f"ghz_to_noob: did not reach bottom pillar: {session.state}")

    if session.state.room_id != ROOM_NOOB:
        # Settle against the pillar at x≈1451, then shoot the blue-gate switch
        # while rising.  The useful beam line is only a few pixels tall.
        for _ in range(60):
            state = session.state
            if state.samus_y >= 935:
                break
            hold(session, 1, reason="ghz_pillar_settle")
        # Let the landing/spin pose finish.  Firing from the transient 0xA4
        # landing pose produces the same coordinates but does not trip the
        # gate switch.
        hold(session, 20, reason="ghz_pillar_stand")

        gate_line = False
        for _ in range(32):
            state = hold(session, 1, "A", reason="ghz_gate_jump")
            if 886 <= state.samus_y <= 889:
                gate_line = True
                break
        if not gate_line:
            raise TimeoutError(
                "ghz_to_noob: missed blue-gate shot line "
                f"at ({state.samus_x},{state.samus_y})"
            )
        hold(session, 3, "RIGHT", "X", reason="ghz_gate_shot")
        hold(session, 60, reason="ghz_gate_open")

        # Clear the pillar only after the gate has opened.  Adding RIGHT too
        # early clips the pillar face and produces the old x≈1451 loop.
        vertical_hop(session, 24, reason="ghz_pillar_vertical_jump")
        for _ in range(220):
            state = hold(session, 1, "RIGHT", "B", "A", reason="ghz_pillar_clear")
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
            state = hold(session, 1, *buttons, reason="ghz_exit_run")
            if state.room_id == ROOM_NOOB:
                break

    if session.state.room_id != ROOM_NOOB:
        raise TimeoutError(f"ghz_to_noob: still in GHZ: {session.state}")
    return wait_ordinary_room(
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
    require_room(session, ROOM_NOOB, "noob_to_red")
    try:
        select_weapon(session, 2)
    except RuntimeError:
        pass

    # Climb onto the upper pit-block bridge.  The apparent x≈1131 "barrier"
    # is simply the end of the lower chamber.  Horizontal momentum must be
    # killed before the ledge jump; otherwise Samus reaches x≈251 while still
    # too low, clips its underside, and remains trapped below.
    if session.state.samus_x < 1150:
        for frame in range(150):
            state = hold(session, 1, "RIGHT", "B", "A", reason="noob_bridge_setup_hop")
            if frame > 45 and state.samus_x >= 190 and state.samus_y >= 155:
                break
        else:
            raise TimeoutError(
                "noob_to_red: could not reach upper-bridge jump setup "
                f"from ({state.samus_x},{state.samus_y})"
            )

        hold(session, 10, "LEFT", reason="noob_bridge_brake")
        vertical_hop(session, 24, reason="noob_bridge_vertical_jump")
        for _ in range(330):
            state = hold(session, 1, "RIGHT", "B", "A", reason="noob_bridge_dash")
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
            hold(session, 2, "UP", reason="noob_unmorph")
            hold(session, 4, reason="noob_unmorph")
            hold(session, 2, "A", reason="noob_unmorph")
            hold(session, 6, reason="noob_unmorph")
            continue

        # Door zone: Super pulse + spin-through (do not crouch-hold).
        if state.samus_x >= 1380:
            hold(session, 2, "RIGHT", "X", reason="noob_super")
            hold(session, 12, reason="noob_fuse")
            hold(session, 18, "RIGHT", "B", "A", reason="noob_spin")
            for _ in range(30):
                state = hold(session, 1, "RIGHT", "B", reason="noob_push")
                if state.room_id == ROOM_RED_TOWER:
                    break
            if state.room_id == ROOM_RED_TOWER:
                break
            continue

        # Right corridor after the upper bridge: run + shoot + hop.
        phase = i % 24
        if state.samus_y > 250:
            state = hold(session, 2, "LEFT", "A", reason="noob_recover")
        elif phase < 14:
            state = hold(session, 1, "RIGHT", "B", reason="noob_run")
        elif phase < 18:
            state = hold(session, 1, "RIGHT", "B", "X", reason="noob_shoot")
        else:
            state = hold(session, 1, "RIGHT", "B", "A", reason="noob_hop")

        if state.samus_x > last_x + 0.5:
            stuck = 0
            last_x = float(state.samus_x)
        else:
            stuck += 1

        # Generic recovery for enemies / door geometry on the right side.
        if stuck > 25 and state.samus_x >= 1050:
            for _ in range(15):
                hold(session, 1, "LEFT", "B", reason="noob_backup")
            for _ in range(18):
                hold(session, 1, "RIGHT", "B", reason="noob_runup")
            for _ in range(45):
                state = hold(session, 1, "RIGHT", "B", "A", reason="noob_longjump")
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
    return wait_ordinary_room(
        session, ROOM_RED_TOWER, settle_frames=220, label="noob_to_red"
    )

