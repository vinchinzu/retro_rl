"""K4.10 Double Chamber → Wave Beam PLM pure controller.

Gate hop/open lives in :mod:`.double_gate`; this module owns past-gate
missile ledge runway, Super door, and Wave chozo collect.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
    walljump_once,
)
from super_metroid.routes.kpdr.norfair.common import _STANDING_POSES
from super_metroid.routes.kpdr.rooms import ROOM_DOUBLE_CHAMBER, ROOM_WAVE
from super_metroid.routes.kpdr.wave.double_gate import (
    dc_hop_to_gate_zone,
    dc_open_blue_gate,
)
from super_metroid.routes.kpdr.wave.geometry import (
    DC_DOOR_X,
    DC_DOOR_Y_MAX,
    DC_EDGE_X,
    DC_LEDGE_Y_MAX,
    DC_MISSILE_X,
    DC_PAST_GATE_X,
    DC_RUNWAY_X,
    DC_WAVE_SETTLE,
    DC_WJ,
    DC_WJ_LEFT_FOLLOW,
    WAVE_BEAM_MASK,
    dc_on_missile_ledge,
    dc_on_sill,
    has_wave,
)
from super_metroid.routes.skills.knockback import escape_kb, is_knockback
from super_metroid.routes.runtime import ControllerSession


def _dc_missiles_and_runway(session: ControllerSession, label: str) -> None:
    """Missile pack on ledge → free PLM pin → backup under gate for runway.

    Human recipe (rr-re9): same ledge as missiles (y≈139), **not** spike floor.
    Backup left toward gate (~x420) so the dash into the right gap is max length.

    One-knob (rr-l0u / SM-K4-TIP-WAVE): **free-then-runway** at the missile PLM
    (~x492–493). Standing collect pins Samus (vx=0); LEFT cannot free while
    seated on the pack (Spazer cont-like hard-pin; pure also stalls). RIGHT+B
    clears the pin band in ~400f (x≥510), then LEFT backup to runway works and
    the edge dash rebuilds speed (vx≈3 @ x600) for door WJ height.
    """
    unmorph(session)
    select_weapon(session, 0)
    mis0 = int(session.state.missiles)

    # Walk right on ledge to missile pack / past-gate solid.
    for _ in range(200):
        state = session.state
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if state.samus_y > DC_LEDGE_Y_MAX + 20:
            hold(session, 1, "LEFT", "A", reason=f"{label}_ledge_recover")
            continue
        if state.missiles > mis0 and dc_on_missile_ledge(state):
            break
        if state.samus_x >= DC_MISSILE_X and dc_on_missile_ledge(state):
            break
        hold(session, 1, "RIGHT", reason=f"{label}_to_missiles")

    # Free PLM pin band before LEFT backup. Live cont-like (beams 0x1004):
    # standing collect freezes ~x492 for ~400f; LEFT never moves; RIGHT+B
    # clears to x≳510 around frame 409. Pure may free earlier. Require past
    # pin before runway so the edge dash rebuilds speed (vx≈3 @ x600).
    for _ in range(520):
        state = session.state
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if state.samus_y > DC_LEDGE_Y_MAX + 20:
            hold(session, 1, "LEFT", "A", reason=f"{label}_plm_free_recover")
            continue
        if state.samus_x >= 510 and dc_on_missile_ledge(state):
            break
        hold(session, 1, "RIGHT", "B", reason=f"{label}_plm_free")

    # Backup LEFT on ledge only (longest runway under gate). PLM spent → LEFT ok.
    for _ in range(280):
        state = session.state
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if state.samus_y > DC_LEDGE_Y_MAX + 20:
            hold(session, 1, "RIGHT", "A", reason=f"{label}_runway_recover")
            continue
        if (
            state.samus_x <= DC_RUNWAY_X
            and dc_on_missile_ledge(state)
        ):
            break
        hold(session, 1, "LEFT", reason=f"{label}_runway_back")

    # Face right and settle on ledge before dash.
    hold(session, 10, "RIGHT", reason=f"{label}_runway_face")
    hold(session, 8, reason=f"{label}_runway_settle")


def _dc_ledge_dash_and_launch(session: ControllerSession, label: str) -> None:
    """Dash missile ledge → spin-launch (~x600) → high door-column WJ → sill.

    Live pure (rr-re9): dash y≈139 to edge x600, launch peaks y≈60, wall
    contact ~(923,238), classic away WJ (LEFT×3 + LEFT+A×6) + left follow,
    RIGHT arc to sill ~(929,116). Never open-loop WJ on spike floor.
    """
    # Dash on ledge to edge (no jump — stay planted).
    for _ in range(220):
        state = session.state
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if dc_on_sill(state) or (
            state.samus_x >= DC_DOOR_X and state.samus_y < DC_DOOR_Y_MAX
        ):
            return
        if state.samus_y > DC_LEDGE_Y_MAX + 20:
            hold(session, 1, "LEFT", "A", reason=f"{label}_dash_recover")
            continue
        if state.samus_x >= DC_EDGE_X and dc_on_missile_ledge(state):
            break
        hold(session, 1, "RIGHT", "B", reason=f"{label}_ledge_dash")

    # Launch toward door column; one high-contact classic WJ, then sill arc.
    did_wj = False
    left_follow = 0
    for frame in range(280):
        state = session.state
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if state.room_id == ROOM_WAVE:
            return
        if dc_on_sill(state) or (
            state.samus_x >= DC_DOOR_X
            and state.samus_y < DC_DOOR_Y_MAX
            and state.velocity_y == 0
        ):
            return
        # Spike floor band — do not thrash WJ down here.
        if state.samus_y > 280 and state.velocity_y == 0:
            return
        if state.samus_y > 320:
            return

        x, y = state.samus_x, state.samus_y
        on_ledge = dc_on_missile_ledge(state)

        # Still on missile ledge short of edge: keep dash / launch.
        if on_ledge and x < DC_EDGE_X:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_ledge_dash")
            continue
        if on_ledge:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_ledge_launch")
            continue

        # High door-column contact: classic away WJ (not floor).
        at_wall = x >= 915 and state.velocity_x == 0 and y < 280
        if at_wall and not did_wj and y <= 260:
            walljump_once(session, DC_WJ, reason=f"{label}_door_wj")
            did_wj = True
            left_follow = DC_WJ_LEFT_FOLLOW
            continue

        # Post-WJ: left spin carry (gain height left of column), then right to sill.
        if left_follow > 0:
            left_follow -= 1
            hold(session, 1, "LEFT", "B", "A", reason=f"{label}_wj_left")
            continue

        if did_wj:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_sill_arc")
            continue

        # Pre-WJ air: hold height while drifting right (peak ~y60 natural).
        if y <= 200:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_high_air")
            continue

        # Mid drop (y 200–280): still try right+height, no floor WJ spam.
        hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_mid_air")


def _dc_super_door_push(session: ControllerSession, label: str) -> None:
    """At Super red door sill: Supers + RIGHT into Wave room."""
    select_weapon(session, 2)
    for frame in range(400):
        state = session.state
        if state.room_id == ROOM_WAVE:
            return
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if is_knockback(state):
            escape_kb(session, label, "RIGHT", stop_room_id=ROOM_WAVE)
            continue
        if state.pose in (137, 138):
            unmorph(session)
            continue
        x, y = state.samus_x, state.samus_y
        if x < DC_DOOR_X - 40 or y > DC_DOOR_Y_MAX + 40:
            # Not on sill — stop (caller may re-approach).
            return
        if state.velocity_y == 0:
            if frame % 36 < 3:
                hold(session, 1, "RIGHT", "X", reason=f"{label}_super")
            elif frame % 36 < 12:
                hold(session, 1, reason=f"{label}_super_fuse")
            else:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_door_push")
        else:
            hold(session, 1, "RIGHT", reason=f"{label}_door_air")


def _dc_to_wave_door(session: ControllerSession, label: str) -> None:
    """Past-gate missile ledge runway → high launch → door WJ → Super → Wave.

    Geometry (rr-re9):
    1. Missiles on upper ledge y≈139 (same ledge as gate exit)
    2. Backup under gate for longest runway on **that** ledge — never spike floor
    3. Dash RIGHT → spin-launch at edge ~x600 (peaks y≈60)
    4. High door-column classic WJ → sill ~(920–940, y≲180)
    5. Super red door into Wave
    """
    if session.state.room_id != ROOM_DOUBLE_CHAMBER:
        return
    if session.state.room_id == ROOM_WAVE:
        return

    # Already at door sill (e.g. post-WJ pin).
    if dc_on_sill(session.state) or (
        session.state.samus_x >= DC_DOOR_X
        and session.state.samus_y < DC_DOOR_Y_MAX
    ):
        _dc_super_door_push(session, label)
        return

    # Need upper path from past-gate / mid-room.
    if session.state.samus_x < DC_PAST_GATE_X and session.state.samus_y < 220:
        # Still left of open gate — should not happen after open phase.
        hold(session, 1, "RIGHT", reason=f"{label}_past_nudge")

    if session.state.samus_y <= DC_LEDGE_Y_MAX + 40 or session.state.samus_x < 650:
        _dc_missiles_and_runway(session, label)
        if session.state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        _dc_ledge_dash_and_launch(session, label)

    # If launch + WJ landed on/near sill, push Super door.
    if (
        session.state.room_id == ROOM_DOUBLE_CHAMBER
        and session.state.samus_x >= DC_DOOR_X - 20
        and session.state.samus_y < DC_DOOR_Y_MAX
    ):
        _dc_super_door_push(session, label)
        return

    # Short sill-seek if high near door column (no spike-floor WJ).
    for frame in range(180):
        state = session.state
        if state.room_id == ROOM_WAVE:
            return
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if state.samus_y > 280:
            return  # refuse spike floor
        if state.samus_x >= DC_DOOR_X - 20 and state.samus_y < DC_DOOR_Y_MAX:
            _dc_super_door_push(session, label)
            return
        if is_knockback(state):
            escape_kb(session, label, "RIGHT", stop_room_id=ROOM_WAVE)
            continue
        if state.pose in (137, 138):
            unmorph(session)
            continue
        # Stay high while seeking right wall/sill.
        if state.samus_y <= 200:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_sill_seek")
        else:
            hold(session, 1, "RIGHT", "A", reason=f"{label}_sill_up")


def _wave_collect_plm(session: ControllerSession, label: str) -> SuperMetroidState:
    """Wave Room left entry → chozo PLM → beam bit 0x0001."""
    require_room(session, ROOM_WAVE, label)
    if has_wave(session.state):
        return session.state

    unmorph(session)
    select_weapon(session, 0)
    for _ in range(30):
        state = hold(session, 1, reason=f"{label}_stand")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break
        if state.pose in (137, 138, 39, 40):
            hold(session, 1, "UP", reason=f"{label}_unmorph")

    for frame in range(500):
        state = session.state
        if has_wave(state):
            break
        if state.room_id != ROOM_WAVE:
            raise TimeoutError(
                f"{label}: left Wave Room during collect; "
                f"room=0x{state.room_id:04X} xy=({state.samus_x},{state.samus_y})"
            )
        if state.pose in (137, 138):
            hold(session, 8, "UP", reason=f"{label}_unmorph")
            continue
        if state.samus_x < 160:
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
        state = session.state
        raise TimeoutError(
            f"{label}: Wave PLM not collected; beams=0x{state.collected_beams:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y})"
        )

    hold(session, 80, reason=f"{label}_fanfare")
    unmorph(session)
    for _ in range(40):
        state = hold(session, 1, reason=f"{label}_post_stand")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break
    return session.state


def play_double_chamber_to_wave(session: ControllerSession) -> SuperMetroidState:
    """Double Chamber (post Single→Double pure) → Wave Beam PLM collect.

    Path: top-left ~(61,139) → upper hop path → blue gate → right Super door
    into Wave ``0xADDE`` → chozo collect (``WAVE_BEAM_MASK`` 0x0001).

    Caps: Morph, Bombs, Missiles, Supers, Hi-Jump, Varia, Speed.
    """
    label = "double_chamber_to_wave"
    require_room(session, ROOM_DOUBLE_CHAMBER, label)
    start = session.frame

    if has_wave(session.state) and session.state.room_id == ROOM_WAVE:
        return session.state

    if session.state.room_id == ROOM_DOUBLE_CHAMBER:
        dc_hop_to_gate_zone(session, label)

    if (
        session.state.room_id == ROOM_DOUBLE_CHAMBER
        and session.state.samus_x < DC_PAST_GATE_X
    ):
        dc_open_blue_gate(session, label)

    if session.state.room_id == ROOM_DOUBLE_CHAMBER:
        _dc_to_wave_door(session, label)

    if session.state.room_id != ROOM_WAVE:
        state = session.state
        raise TimeoutError(
            f"{label}: Wave door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} "
            f"missiles={state.missiles} supers={state.super_missiles} "
            f"selected={state.selected_item} "
            f"beams=0x{state.collected_beams:04X} "
            f"frames={session.frame - start}"
        )

    wait_ordinary_room(
        session, ROOM_WAVE, settle_frames=DC_WAVE_SETTLE, label=label
    )
    state = _wave_collect_plm(session, label)

    if not has_wave(state):
        raise TimeoutError(
            f"{label}: finished without Wave bit; "
            f"beams=0x{state.collected_beams:04X} room=0x{state.room_id:04X} "
            f"xy=({state.samus_x},{state.samus_y}) "
            f"frames={session.frame - start}"
        )
    return state


__all__ = ["play_double_chamber_to_wave", "WAVE_BEAM_MASK"]
