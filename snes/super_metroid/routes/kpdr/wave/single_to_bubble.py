"""K4 Single Chamber → Bubble Mountain pure return (rr-u0y8 / rr-vqv3 stack).

Source: ``post_double_to_single_chamber_pure`` ~(216,630) deep shaft after
Double→Single. Human tape Phase B hop 10 (f6908–7496):

1. Deep floor settle y651; LEFT + LEFT+A hop over morph slope → wall ~x61
2. A → RIGHT+A wall flip → mid-low platform ~y523
3. RIGHT across mid-low → LEFT+A up to floor band y395
4. LEFT on y395 → B+LEFT+A wall climb through mid-hi ~y267
5. RIGHT/up hops → LEFT+A to top y139
6. LEFT (+ shot) through top-left blue door → Bubble ``0xACB3``

Live recon (2026-08-09): pure LEFT from pin morph-falls the slope at x≈167
into y779+; human LEFT+A hop is required. Open-loop tape segments gated by
y-band. Do not invent a top-only pin.
Tape recon: ``docs/tasks/SM-SPEED-ICE-MOAT-HUMAN.md`` Phase B hop 10.
"""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    unmorph,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.k4_common import _STANDING_POSES
from super_metroid.routes.kpdr.rooms import ROOM_BUBBLE, ROOM_SINGLE_CHAMBER
from super_metroid.routes.kpdr.wave.geometry import (
    STB_BUBBLE_SETTLE,
    STB_CLIMB_FRAMES,
    STB_DEEP_Y_MIN,
    STB_DOOR_FRAMES,
    STB_DOOR_X,
    STB_DOOR_Y,
    STB_FLOOR_Y,
    STB_MID_HI_Y,
    STB_MID_LOW_Y,
    STB_TOP_Y_MAX,
)
from super_metroid.routes.runtime import ControllerSession
from super_metroid.routes.skills.knockback import escape_kb, is_knockback

_LEDGE = _STANDING_POSES | frozenset({1, 2, 9, 10, 37, 38})


def _on_top(state: SuperMetroidState) -> bool:
    return (
        state.room_id == ROOM_SINGLE_CHAMBER
        and state.samus_y <= STB_TOP_Y_MAX
        and int(state.velocity_y) == 0
    )


def _on_door_sill(state: SuperMetroidState) -> bool:
    return (
        state.room_id == ROOM_SINGLE_CHAMBER
        and state.samus_x <= STB_DOOR_X + 10
        and STB_DOOR_Y[0] <= state.samus_y <= STB_DOOR_Y[1]
        and int(state.velocity_y) == 0
    )


def _y_band(state: SuperMetroidState, band: tuple[int, int]) -> bool:
    return band[0] <= int(state.samus_y) <= band[1]


def _settle_ground(
    session: ControllerSession, label: str, *, max_frames: int = 40
) -> None:
    for _ in range(max_frames):
        st = session.state
        if st.room_id != ROOM_SINGLE_CHAMBER:
            return
        if is_knockback(st):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_BUBBLE)
            continue
        if int(st.pose) in (31, 39, 40, 41, 42, 65, 137, 138):
            hold(session, 1, "UP", reason=f"{label}_unmorph")
            continue
        if (
            int(st.velocity_y) == 0
            and int(st.pose) in _LEDGE
            and int(st.door_transition) == 0
        ):
            return
        hold(session, 1, reason=f"{label}_settle")


def _play_rle(
    session: ControllerSession,
    label: str,
    steps: list[tuple[int, tuple[str, ...]]],
    *,
    stop_when=None,
) -> None:
    for n, buttons in steps:
        for _ in range(n):
            st = session.state
            if st.room_id == ROOM_BUBBLE:
                return
            if st.room_id != ROOM_SINGLE_CHAMBER:
                return
            if stop_when is not None and stop_when(st):
                return
            if is_knockback(st):
                escape_kb(session, label, "LEFT", stop_room_id=ROOM_BUBBLE)
                continue
            if int(st.pose) in (31, 39, 40, 41, 42, 65, 137, 138) and "UP" not in buttons:
                hold(session, 1, "UP", reason=f"{label}_unmorph")
                continue
            hold(session, 1, *buttons, reason=label)


def _deep_to_mid_low(session: ControllerSession, label: str) -> None:
    """Deep pin → mid-low y523 via human deep walk+hop + wall A/RIGHT+A.

    Live pure: LEFT-only morph-falls slope ~x167; LEFT+A hop is required.
    Proven RLE lands ~(77,523) pose 9 from post_double_to_single pin.
    """
    if int(session.state.samus_y) <= STB_MID_LOW_Y[1] + 15:
        return

    unmorph(session)
    select_weapon(session, 0)
    _settle_ground(session, f"{label}_deep")

    # Human f6908–7065 (from ~x216 y651 standing).
    # Do NOT stop_when on first y≈507 vy=0 contact — that is a brief mid-air
    # seat (pose 81) before RIGHT lands solid mid-low ~(78,523). Early stop
    # idles off the lip and slides back to deep floor.
    _play_rle(
        session,
        f"{label}_deep_rle",
        [
            (10, ()),
            (16, ("LEFT",)),
            (36, ("LEFT", "A")),  # hop over morph slope
            (34, ("LEFT",)),
            (20, ("A",)),  # wall launch
            (13, ("RIGHT", "A")),
            (4, ("RIGHT", "A", "X")),
            (8, ("RIGHT", "X")),
            (16, ("RIGHT",)),
        ],
        stop_when=lambda st: (
            _y_band(st, STB_MID_LOW_Y)
            and int(st.velocity_y) == 0
            and int(st.pose) in _LEDGE
            and int(st.samus_x) >= 70
        ),
    )
    _settle_ground(session, f"{label}_ml_land", max_frames=50)

    # One recovery if still deep (shortened hop + launch).
    if int(session.state.samus_y) >= STB_DEEP_Y_MIN - 30:
        unmorph(session)
        _play_rle(
            session,
            f"{label}_deep_retry",
            [
                (8, ("LEFT",)),
                (28, ("LEFT", "A")),
                (20, ("LEFT",)),
                (18, ("A",)),
                (14, ("RIGHT", "A")),
                (12, ("RIGHT",)),
            ],
            stop_when=lambda st: int(st.samus_y) < STB_DEEP_Y_MIN - 40,
        )
        _settle_ground(session, f"{label}_deep_retry_land", max_frames=50)


def _mid_low_to_floor(session: ControllerSession, label: str) -> None:
    """Mid-low y523 RIGHT → LEFT+A up → floor band y395 (human f7065–7169)."""
    if int(session.state.samus_y) <= STB_FLOOR_Y[1] + 10:
        return
    if int(session.state.samus_y) > STB_MID_LOW_Y[1] + 40:
        return

    unmorph(session)
    _settle_ground(session, f"{label}_ml")

    # RIGHT to launch ~x150–160 (human uses B+Y sometimes; B is enough).
    for _ in range(55):
        st = session.state
        if st.room_id != ROOM_SINGLE_CHAMBER:
            return
        if int(st.samus_y) < STB_MID_LOW_Y[0] - 10:
            break
        if is_knockback(st):
            escape_kb(session, label, "RIGHT", stop_room_id=ROOM_BUBBLE)
            continue
        if int(st.pose) in (31, 39, 40, 41, 42, 65, 137, 138):
            hold(session, 2, "UP", reason=f"{label}_ml_unmorph")
            continue
        if int(st.samus_x) >= 152 and int(st.velocity_y) == 0:
            break
        hold(session, 1, "RIGHT", "B", reason=f"{label}_ml_run")

    _play_rle(
        session,
        f"{label}_ml_up",
        [
            (17, ("LEFT", "A")),
            (18, ("A",)),
            (11, ("LEFT", "A")),
            (7, ("LEFT",)),
            (15, ("B", "LEFT")),
        ],
        stop_when=lambda st: _y_band(st, STB_FLOOR_Y) and int(st.velocity_y) == 0,
    )
    _settle_ground(session, f"{label}_fl_land", max_frames=50)

    y = int(session.state.samus_y)
    if y > STB_FLOOR_Y[1] and y <= STB_MID_LOW_Y[1] + 20:
        _play_rle(
            session,
            f"{label}_ml_up_retry",
            [
                (8, ("RIGHT", "B")),
                (20, ("LEFT", "A")),
                (14, ("A",)),
                (10, ("LEFT",)),
            ],
            stop_when=lambda st: int(st.samus_y) <= STB_FLOOR_Y[1],
        )
        _settle_ground(session, f"{label}_fl_retry")


def _floor_to_mid_hi(session: ControllerSession, label: str) -> None:
    """Floor y395 LEFT → B+LEFT+A climb → mid-hi ~y267 (human f7169–7259)."""
    if int(session.state.samus_y) <= STB_MID_HI_Y[1] + 10:
        return
    if int(session.state.samus_y) > STB_FLOOR_Y[1] + 40:
        return

    unmorph(session)
    _settle_ground(session, f"{label}_fl")

    # Human launches near x≈85; x≥95 wall-misses and falls back to y395.
    for _ in range(50):
        st = session.state
        if st.room_id != ROOM_SINGLE_CHAMBER:
            return
        if int(st.samus_y) < STB_FLOOR_Y[0] - 10:
            break
        if is_knockback(st):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_BUBBLE)
            continue
        if int(st.samus_x) <= 88 and int(st.velocity_y) == 0:
            break
        hold(session, 1, "B", "LEFT", reason=f"{label}_fl_left")

    _play_rle(
        session,
        f"{label}_fl_climb",
        [
            (17, ("B", "LEFT", "A")),
            (9, ("B", "A")),
            (18, ("B", "RIGHT", "A")),
            (2, ("B", "RIGHT")),
            (9, ("RIGHT",)),
            (26, ("B", "RIGHT")),
        ],
        # Solid mid-hi land needs x≳90 (human ~(104,267)); early x50 y270 is air.
        stop_when=lambda st: (
            _y_band(st, STB_MID_HI_Y)
            and int(st.velocity_y) == 0
            and int(st.pose) in _LEDGE
            and int(st.samus_x) >= 90
        ),
    )
    _settle_ground(session, f"{label}_mh_land", max_frames=50)


def _mid_hi_to_top(session: ControllerSession, label: str) -> None:
    """Mid-hi y267 → top shelf y139 (human f7259–7395)."""
    if _on_top(session.state) or int(session.state.samus_y) <= STB_TOP_Y_MAX:
        return
    if int(session.state.samus_y) > STB_MID_HI_Y[1] + 50:
        return

    unmorph(session)
    _settle_ground(session, f"{label}_mh")

    _play_rle(
        session,
        f"{label}_mh_up",
        [
            (11, ("B", "RIGHT")),
            (6, ("B", "RIGHT", "A")),
            (23, ("B", "A")),
            (6, ("B", "RIGHT", "A")),
            (15, ("RIGHT",)),
            (2, ("RIGHT", "A")),
            (5, ("A",)),
            (5, ("A",)),
            (31, ("LEFT", "A")),
            (6, ("LEFT",)),
            (10, ("LEFT", "X")),
            (7, ("LEFT",)),
            (6, ("LEFT", "X")),
            (3, ("LEFT",)),
        ],
        stop_when=lambda st: int(st.samus_y) <= STB_TOP_Y_MAX and int(st.velocity_y) == 0,
    )
    _settle_ground(session, f"{label}_top_land", max_frames=50)

    for frame in range(140):
        st = session.state
        if st.room_id != ROOM_SINGLE_CHAMBER:
            return
        if _on_top(st):
            return
        if is_knockback(st):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_BUBBLE)
            continue
        y = int(st.samus_y)
        x = int(st.samus_x)
        if y <= STB_TOP_Y_MAX + 15 and int(st.velocity_y) == 0:
            return
        if int(st.velocity_y) != 0:
            if x > 160:
                hold(session, 1, "LEFT", "A", reason=f"{label}_top_air_l")
            else:
                hold(session, 1, "LEFT", "B", "A", reason=f"{label}_top_air")
            continue
        if x > 70:
            hold(session, 1, "LEFT", "B", "A", reason=f"{label}_top_reclimb")
        else:
            phase = frame % 12
            if phase < 7:
                hold(session, 1, "LEFT", "B", "A", reason=f"{label}_top_wall")
            else:
                hold(session, 1, "RIGHT", "A", reason=f"{label}_top_flip")


def _top_left_door(session: ControllerSession, label: str) -> None:
    """Top shelf y139 → LEFT blue door into Bubble (human f7395–7496)."""
    unmorph(session)
    select_weapon(session, 0)

    for frame in range(STB_DOOR_FRAMES):
        state = session.state
        if state.room_id == ROOM_BUBBLE:
            return
        if state.room_id != ROOM_SINGLE_CHAMBER:
            return
        if is_knockback(state):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_BUBBLE)
            continue
        if int(state.pose) in (137, 138, 39, 40, 31, 41, 42):
            hold(session, 4, "UP", reason=f"{label}_door_unmorph")
            continue

        x = int(state.samus_x)
        y = int(state.samus_y)
        vy = int(state.velocity_y)

        if y > STB_TOP_Y_MAX + 50:
            return

        if _on_door_sill(state) or (
            x <= STB_DOOR_X + 8 and y <= STB_TOP_Y_MAX and vy == 0
        ):
            phase = frame % 14
            if phase < 4:
                hold(session, 1, "LEFT", "X", reason=f"{label}_door_shot")
            elif phase < 11:
                hold(session, 1, "LEFT", "B", reason=f"{label}_door_push")
            else:
                hold(session, 1, "LEFT", reason=f"{label}_door_walk")
            continue

        if y <= STB_TOP_Y_MAX + 10:
            if x > STB_DOOR_X + 5:
                phase = frame % 16
                if phase < 6:
                    hold(session, 1, "LEFT", "X", reason=f"{label}_top_shot")
                elif phase < 12:
                    hold(session, 1, "LEFT", "B", reason=f"{label}_top_run")
                else:
                    hold(session, 1, "LEFT", reason=f"{label}_top_walk")
            else:
                hold(session, 1, "LEFT", reason=f"{label}_top_nudge")
            continue

        hold(session, 1, "LEFT", "A", reason=f"{label}_top_rehop")


def _reactive_climb_budget(session: ControllerSession, label: str) -> None:
    """Last-resort reactive climb if open-loop left us mid-room."""
    for frame in range(STB_CLIMB_FRAMES // 2):
        st = session.state
        if st.room_id == ROOM_BUBBLE:
            return
        if st.room_id != ROOM_SINGLE_CHAMBER:
            return
        if _on_top(st):
            return
        if is_knockback(st):
            escape_kb(session, label, "LEFT", stop_room_id=ROOM_BUBBLE)
            continue
        if int(st.pose) in (137, 138, 39, 40, 31, 41, 42):
            hold(session, 3, "UP", reason=f"{label}_rx_unmorph")
            continue

        x = int(st.samus_x)
        y = int(st.samus_y)
        vy = int(st.velocity_y)
        grounded = vy == 0 and int(st.pose) in _LEDGE

        # Deep morph-slope trap: hop LEFT+A, never pure LEFT morph.
        if grounded and y >= STB_DEEP_Y_MIN - 20:
            if x > 170:
                hold(session, 1, "LEFT", "A", reason=f"{label}_rx_deep_hop")
            elif x > 60:
                hold(session, 1, "LEFT", "A", reason=f"{label}_rx_deep_spin")
            else:
                phase = frame % 20
                if phase < 12:
                    hold(session, 1, "A", reason=f"{label}_rx_deep_a")
                else:
                    hold(session, 1, "RIGHT", "A", reason=f"{label}_rx_deep_r")
            continue

        if not grounded:
            if y >= 500 and x < 50:
                hold(session, 1, "RIGHT", "A", reason=f"{label}_rx_air_r")
            elif x > 100:
                hold(session, 1, "LEFT", "A", reason=f"{label}_rx_air_l")
            else:
                hold(session, 1, "LEFT", "B", "A", reason=f"{label}_rx_air")
            continue

        if x > 60 and y > STB_TOP_Y_MAX:
            hold(session, 1, "LEFT", "B", "A", reason=f"{label}_rx_up")
        else:
            phase = frame % 18
            if phase < 10:
                hold(session, 1, "LEFT", "B", "A", reason=f"{label}_rx_climb")
            elif phase < 14:
                hold(session, 1, "A", reason=f"{label}_rx_peak")
            else:
                hold(session, 1, "RIGHT", "A", reason=f"{label}_rx_flip")


def play_single_to_bubble(session: ControllerSession) -> SuperMetroidState:
    """Single Chamber deep pin → ordinary Bubble Mountain.

    Expects post-Double→Single pure ~(216,630). Climbs left wall/platforms
    and leaves via top-left blue door into Bubble ``0xACB3``.
    """
    label = "single_to_bubble"
    require_room(session, ROOM_SINGLE_CHAMBER, label)
    start = session.frame

    unmorph(session)
    select_weapon(session, 0)

    for _attempt in range(3):
        if session.state.room_id != ROOM_SINGLE_CHAMBER:
            break
        if _on_top(session.state):
            break
        y = int(session.state.samus_y)
        if y > STB_MID_LOW_Y[1]:
            _deep_to_mid_low(session, label)
        if session.state.room_id != ROOM_SINGLE_CHAMBER:
            break
        y = int(session.state.samus_y)
        if y > STB_FLOOR_Y[1] and y <= STB_MID_LOW_Y[1] + 40:
            _mid_low_to_floor(session, label)
        if session.state.room_id != ROOM_SINGLE_CHAMBER:
            break
        y = int(session.state.samus_y)
        if STB_FLOOR_Y[0] - 30 <= y <= STB_FLOOR_Y[1] + 30:
            _floor_to_mid_hi(session, label)
        if session.state.room_id != ROOM_SINGLE_CHAMBER:
            break
        y = int(session.state.samus_y)
        if y <= STB_MID_HI_Y[1] + 50 and y > STB_TOP_Y_MAX:
            _mid_hi_to_top(session, label)
        if _on_top(session.state) or session.state.room_id == ROOM_BUBBLE:
            break

    if (
        session.state.room_id == ROOM_SINGLE_CHAMBER
        and not _on_top(session.state)
        and int(session.state.samus_y) > STB_TOP_Y_MAX
    ):
        _reactive_climb_budget(session, label)

    if session.state.room_id == ROOM_SINGLE_CHAMBER:
        _top_left_door(session, label)

    if session.state.room_id == ROOM_SINGLE_CHAMBER and not _on_top(session.state):
        _reactive_climb_budget(session, label)
        if session.state.room_id == ROOM_SINGLE_CHAMBER:
            _top_left_door(session, label)

    if session.state.room_id != ROOM_BUBBLE:
        state = session.state
        raise TimeoutError(
            f"{label}: Bubble door missed; room=0x{state.room_id:04X} "
            f"pose={state.pose} xy=({state.samus_x},{state.samus_y}) "
            f"door_transition={state.door_transition} "
            f"frames={session.frame - start}"
        )

    return wait_ordinary_room(
        session,
        ROOM_BUBBLE,
        settle_frames=STB_BUBBLE_SETTLE,
        label=label,
    )


__all__ = ["play_single_to_bubble"]
