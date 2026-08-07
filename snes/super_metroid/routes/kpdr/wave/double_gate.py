"""K4.10 Double Chamber blue-gate hop + human open (Kamer seat).

Split from double_to_wave so no Wave multi-hop module exceeds ~500 lines.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    is_morph,
    select_weapon,
    unmorph,
)
from super_metroid.routes.kpdr.k4_common import _STANDING_POSES
from super_metroid.routes.kpdr.rooms import ROOM_DOUBLE_CHAMBER, ROOM_WAVE
from super_metroid.routes.kpdr.wave.geometry import (
    DC_GATE_OPEN_SEAT_X,
    DC_GATE_OPEN_SEAT_Y,
    DC_GATE_SEAT_X,
    DC_GATE_SEAT_Y_MAX,
    DC_PAST_GATE_X,
)
from super_metroid.routes.kpdr.wave.scripts import HUMAN_GATE_OPEN_RLE
from super_metroid.routes.rle import play_script
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import escape_kb, is_knockback


def dc_hop_to_gate_zone(session: ControllerSession, label: str) -> None:
    """Top-left continuous leave → upper platforms → gate seat x∈[365,390].

    Continuous natural leave after single→double is ~(39,139) pose=9 on Kamer.
    Launch pin is **x≥61 y≈139**. Kamer ride must finish **before** unmorph /
    select — pose-9 ``unmorph`` burns ~26f and ``select_weapon`` can burn 50f+,
    which drops the fixed-10 idle window (dual RED). Idle-only until the pin;
    then hop_run to x≈210 and spin16_run12 toward gate.
    """
    # Finish Kamer ride first (no unmorph/select — those desync the phase).
    if session.state.samus_y <= 170 and session.state.samus_x < 61:
        for _ in range(40):
            state = session.state
            if state.samus_x >= 61 and state.velocity_y == 0:
                break
            if state.samus_y > 220:
                break
            hold(session, 1, reason=f"{label}_kamer_phase")

    # Unmorph only morph / ball knockback — not pose 9/10 on the Kamer pin
    # (global unmorph treats 9/10 as crouch and burns the launch window).
    pose = int(session.state.pose)
    if is_morph(pose) or pose in (39, 40, 137, 138):
        unmorph(session)
    if int(session.state.selected_item) != 0:
        select_weapon(session, 0)

    for _ in range(20):
        state = hold(session, 1, reason=f"{label}_top_stand")
        if state.velocity_y == 0 and state.pose in _STANDING_POSES:
            break

    # hop_run toward mid platforms
    for frame in range(160):
        state = session.state
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if state.samus_x >= 210 and state.velocity_y == 0 and state.samus_y < 200:
            break
        if is_knockback(state):
            escape_kb(session, label, "RIGHT", stop_room_id=ROOM_WAVE)
            continue
        if state.pose in (137, 138):
            unmorph(session)
            continue
        phase = frame % 30
        if phase < 4:
            hold(session, 1, "RIGHT", "X", reason=f"{label}_hop_shot")
        elif phase < 12:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_hop_spin")
        elif phase < 22:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_hop_run")
        else:
            hold(session, 1, "RIGHT", reason=f"{label}_hop_walk")

    # spin16_run12 toward gate / high platforms
    for frame in range(280):
        state = session.state
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if (
            DC_GATE_SEAT_X[0] <= state.samus_x <= DC_GATE_SEAT_X[1]
            and state.samus_y < DC_GATE_SEAT_Y_MAX
            and state.velocity_y == 0
        ):
            return
        if state.samus_x >= DC_PAST_GATE_X and state.samus_y < 220:
            return
        if state.samus_y > 360 and state.velocity_y == 0:
            return  # fell; door phase may still recover poorly
        if is_knockback(state):
            escape_kb(session, label, "RIGHT", stop_room_id=ROOM_WAVE)
            continue
        if state.pose in (137, 138):
            unmorph(session)
            continue
        # Near seat band: brake / short walk rather than spin past switch.
        if (
            state.samus_x >= DC_GATE_SEAT_X[0] - 20
            and state.samus_y < DC_GATE_SEAT_Y_MAX
            and state.velocity_y == 0
        ):
            if state.samus_x < DC_GATE_SEAT_X[0]:
                hold(session, 1, "RIGHT", reason=f"{label}_seat_in")
            elif state.samus_x > DC_GATE_SEAT_X[1]:
                hold(session, 1, "LEFT", reason=f"{label}_seat_back")
            else:
                hold(session, 1, reason=f"{label}_seat_brake")
            continue
        phase = frame % 28
        if phase < 16:
            hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_gate_spin")
        else:
            hold(session, 1, "RIGHT", "B", reason=f"{label}_gate_run")


def _dc_wait_kamer_open_seat(session: ControllerSession, label: str) -> bool:
    """Ride Kamer to human open seat: x∈[370,375], y≤139, standing.

    Tighter than hop delivery band — pure human-button replay only greens
    from this pin (rr-dbu.10). y≤145 is too low; shoot starts must be peak.
    """
    x_lo, x_hi = DC_GATE_OPEN_SEAT_X
    for _ in range(800):
        state = session.state
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return False
        if state.samus_x >= DC_PAST_GATE_X and state.samus_y < 220:
            return True
        if state.samus_y > 360 and state.velocity_y == 0:
            return False
        if is_knockback(state):
            escape_kb(session, label, "RIGHT", stop_room_id=ROOM_WAVE)
            continue
        if state.pose in (137, 138):
            unmorph(session)
            continue
        seated = (
            state.velocity_y == 0
            and state.samus_y <= DC_GATE_OPEN_SEAT_Y
            and x_lo <= state.samus_x <= x_hi
            and state.pose in _STANDING_POSES
        )
        if seated:
            return True
        if state.samus_x < x_lo:
            hold(session, 1, "RIGHT", reason=f"{label}_seat_r")
        elif state.samus_x > x_hi:
            hold(session, 1, "LEFT", reason=f"{label}_seat_l")
        else:
            hold(session, 1, reason=f"{label}_seat_wait")
    return False


def _dc_select_missiles_for_open(session: ControllerSession, label: str) -> None:
    """SELECT missiles with fixed Kamer cost (26f even when already selected).

    Continuous single→double leaves selected=1 (missile door). Pure cont-like
    often has selected=0. ``select_weapon`` is a no-op when already on 1, which
    desyncs Kamer vs the 26f SELECT+settle path the human RLE was timed for.
    """
    if int(session.state.selected_item) != 1:
        select_weapon(session, 1)
    else:
        hold(session, 1, reason=f"{label}_select_missiles_pad")
        hold(session, 25, reason=f"{label}_select_missiles_settle")


def _dc_floor_recover_to_gate(session: ControllerSession, label: str) -> bool:
    """From spike/floor band y≳300, Hi-Jump climb back toward gate seat.

    Human take04 red path when P1/open dumps. Best-effort — returns True if
    back on upper band y≲200 x∈[300,450].
    """
    unmorph(session)
    if int(session.state.selected_item) != 0:
        select_weapon(session, 0)
    for frame in range(900):
        state = session.state
        if state.room_id != ROOM_DOUBLE_CHAMBER:
            return False
        if (
            state.samus_y <= 200
            and 300 <= state.samus_x <= 450
            and state.velocity_y == 0
        ):
            return True
        if is_knockback(state) or int(state.pose) in (20, 83, 84):
            escape_kb(session, label, "RIGHT", stop_room_id=ROOM_WAVE)
            continue
        x, y = state.samus_x, state.samus_y
        # Floor: run to climb column ~x330–360 then HJ up.
        if y >= 360 and state.velocity_y == 0:
            if x < 320:
                hold(session, 1, "RIGHT", "B", reason=f"{label}_floor_r")
            elif x > 380:
                hold(session, 1, "LEFT", "B", reason=f"{label}_floor_l")
            else:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_floor_hj")
            continue
        # Mid air: bias toward gate x and hold height.
        if y > 200:
            if x < 340:
                hold(session, 1, "RIGHT", "B", "A", reason=f"{label}_climb_r")
            elif x > 400:
                hold(session, 1, "LEFT", "B", "A", reason=f"{label}_climb_l")
            else:
                hold(session, 1, "RIGHT", "A", reason=f"{label}_climb_up")
            continue
        # High band: walk into seat window.
        if x < 370:
            hold(session, 1, "RIGHT", reason=f"{label}_reseat_r")
        elif x > 390:
            hold(session, 1, "LEFT", reason=f"{label}_reseat_l")
        else:
            hold(session, 1, reason=f"{label}_reseat")
    return (
        session.state.samus_y <= 200
        and 300 <= session.state.samus_x <= 450
    )


def dc_open_blue_gate(session: ControllerSession, label: str) -> None:
    """Open mid blue gate and walk onto past-gate platform (x≳480 y≲200).

    Human tape buttons f4650–5200 from ``speed_to_ice_moat_human.json`` after
    Kamer seat x∈[370,375] y≤139 (rr-dbu.10). Do not re-invent shot cadence.

    Continuous natural entry has live Multiviola; cont-like pure often does
    not. Mid-RLE hit desyncs Kamer and dumps to floor. On hit/fall: escape,
    floor-recover if needed, re-seat, retry (up to 3). No pre-RLE beam thrash
    (desyncs cont-like Kamer phase).
    """
    for _attempt in range(3):
        unmorph(session)

        if session.state.samus_y > 300:
            if not _dc_floor_recover_to_gate(session, label):
                return

        if not _dc_wait_kamer_open_seat(session, label):
            return
        if session.state.samus_x >= DC_PAST_GATE_X:
            return

        _dc_select_missiles_for_open(session, label)
        hold(session, 8, reason=f"{label}_seat_settle")
        if session.state.samus_x >= DC_PAST_GATE_X:
            return

        aborted = False
        hit_abort = False
        health0 = int(session.state.health)

        def _rle_stop(state: SuperMetroidState) -> bool:
            nonlocal aborted, hit_abort, health0
            if state.samus_x >= DC_PAST_GATE_X and state.samus_y < 220:
                return True
            if state.samus_y > 360 and state.velocity_y == 0:
                aborted = True
                return True
            hp = int(state.health)
            hit = (
                is_knockback(state)
                or int(state.pose) in (20, 83, 84)
                or hp < health0
            )
            if hp >= health0:
                health0 = hp
            if hit:
                hit_abort = True
                aborted = True
                return True
            return False

        play_script(
            session,
            HUMAN_GATE_OPEN_RLE,
            reason=f"{label}_human_open",
            room_id=ROOM_DOUBLE_CHAMBER,
            stop_when=_rle_stop,
            on_lag="ignore",
        )
        if session.state.room_id != ROOM_DOUBLE_CHAMBER:
            return
        if session.state.samus_x >= DC_PAST_GATE_X and session.state.samus_y < 220:
            return
        if hit_abort:
            escape_kb(session, label, "RIGHT", stop_room_id=ROOM_WAVE)

        if not aborted:
            for _ in range(40):
                state = session.state
                if state.room_id != ROOM_DOUBLE_CHAMBER:
                    return
                if state.samus_x >= DC_PAST_GATE_X and state.samus_y < 220:
                    return
                if state.samus_y > 300:
                    aborted = True
                    break
                if is_knockback(state) or int(state.pose) in (20, 83, 84):
                    escape_kb(session, label, "RIGHT", stop_room_id=ROOM_WAVE)
                    aborted = True
                    break
                hold(session, 1, "RIGHT", reason=f"{label}_past_walk")
            if not aborted:
                return
        # Hit/fall — recover and retry (RLE shots may have killed pests).



__all__ = [
    "dc_hop_to_gate_zone",
    "dc_open_blue_gate",
]
