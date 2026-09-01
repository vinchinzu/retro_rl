"""Ceres outbound (elev→Ridley) and escape (Ridley→Landing) play callables."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from retro_harness.actions import buttons, idle_action
from super_metroid.ram import GS_ORDINARY, SuperMetroidState, parse_state, set_moonwalk
from super_metroid.routes.kpdr.ceres.arm_pump import (
    _ceres_clear_knockback,
    _ceres_wait_ordinary,
)
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.routes.kpdr.ceres.elev_escape import _ceres_reactive_elev_climb
from super_metroid.routes.kpdr.ceres.geometry import (
    CERES_FALLING_EXIT_HOP,
    CERES_FALLING_FLOOR_HOP,
    CERES_MAGNET_MID_HOP,
    CERES_MAGNET_TOP_HOP,
    _CERES_FALLING_OUT_DOOR_X,
    _CERES_FALLING_OUT_FLOOR_Y,
    _CERES_FALLING_OUT_PLAT_Y,
    _CERES_FIRST_DOOR_X,
    _CERES_FIRST_PAD_Y,
    _CERES_MAGNET_BOT_Y,
    _CERES_MAGNET_DOOR_Y,
    _CERES_MAGNET_MID_Y,
    _CERES_MAGNET_OUT_DOOR_X,
    _CERES_MAGNET_TOP_Y,
)
from super_metroid.routes.kpdr.ceres.magnet import (
    play_ceres_falling_to_elev,
    play_ceres_magnet_to_falling,
)
from super_metroid.routes.kpdr.ceres.scientist import (
    CeresScientistCross,
    play_ceres_scientist_to_flat,
    play_ceres_scientist_to_magnet,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_FLAT,
    ROOM_CERES_MAGNET,
    ROOM_CERES_RIDLEY,
    ROOM_CERES_SCIENTIST,
    ROOM_LANDING_SITE,
)
from super_metroid.routes.runtime import ActionSpan, ControllerSession, RouteSession
from super_metroid.routes.skills.moonfall import (
    WIKI_URL,
    is_airborne,
    require_moonwalk_on,
)
from super_metroid.takeoff import shoulder_pump_button, spin_jump


# Wiki Ceres 1: moonfall. Sniq 100% lsnes (pad f8639→door f8788) short-hops
# RIGHT off the pad, air-turns pose 25→26, lands y=75 (pose 229), then a
# spinning moonfall whose idle weave is past 171/267. Same buttons skip
# those ledges on this pin (vd=0, |vy| to 12). Do not d-pad-steer a RAM
# corridor — WJ-check lands 267/363.
# https://wiki.supermetroid.run/KPDR_Room_Strategies
CeresFirstPhase = Literal[
    "ride",
    "hop",
    "air_turn",
    "moon_arm",
    "fall",
    "land",
    "exit",
    "done",
]
FALLING_SETTLE = 180
# PJBoy $0A1F: 2 jump, 3 spin, 6 fall, 23 used-item / gun-jump.
_AIR_MT = frozenset({2, 3, 6, 23})


def _expand_button_spans(
    spans: tuple[tuple[tuple[str, ...], int], ...],
) -> tuple[tuple[str, ...], ...]:
    out: list[tuple[str, ...]] = []
    for names, n in spans:
        out.extend([names] * n)
    return tuple(out)


# Sniq 100% lsnes pad→door (movie f8639–8788). Flattened in action().
CERES_FIRST_TAS_PAD_SPANS: tuple[tuple[tuple[str, ...], int], ...] = (
    (("B", "RIGHT"), 1),
    (("B", "Y", "RIGHT", "A"), 1),
    (("B", "RIGHT"), 10),
    (("B", "LEFT"), 1),
    (("B", "L"), 1),
    (("B",), 1),
    (("B", "RIGHT", "X"), 1),
    (("B", "RIGHT", "A"), 1),
    (("B",), 1),
    (("B", "RIGHT"), 7),
    (("RIGHT",), 7),
    ((), 4),
    (("LEFT",), 1),
    ((), 7),
    (("LEFT",), 15),
    ((), 1),
    (("RIGHT",), 1),
    ((), 13),
    (("RIGHT",), 1),
    ((), 7),
    (("LEFT",), 1),
    ((), 5),
    (("LEFT",), 1),
    ((), 7),
    (("RIGHT",), 1),
    ((), 4),
    (("LEFT", "RIGHT"), 1),
    ((), 14),
    (("RIGHT",), 3),
    (("L",), 1),
    (("RIGHT",), 4),
    (("B", "RIGHT"), 3),
    (("B", "LEFT", "X"), 1),
    (("RIGHT",), 1),
    (("B",), 1),
    (("B", "RIGHT"), 1),
    (("B", "RIGHT", "L"), 12),
    (("B", "RIGHT"), 1),
    (("B", "RIGHT", "L"), 1),
    (("B", "RIGHT"), 1),
    (("B", "RIGHT", "L"), 1),
    (("B", "RIGHT"), 1),
    (("B", "RIGHT", "L"), 1),
    (("B", "RIGHT"), 1),
)
CERES_FIRST_TAS_PAD: tuple[tuple[str, ...], ...] = _expand_button_spans(
    CERES_FIRST_TAS_PAD_SPANS
)


def _ceres_first_tas_phase(held: int) -> CeresFirstPhase:
    if held < 12:
        return "hop"
    if held < 13:
        return "air_turn"
    if held < 25:
        return "moon_arm"
    return "fall"


@dataclass(frozen=True)
class CeresFirstMoonfallTrack:
    phase: CeresFirstPhase = "ride"
    held: int = 0
    pump_i: int = 0


def _ceres_first_airborne(state) -> bool:
    if is_airborne(state):
        return True
    if int(state.movement_type) in _AIR_MT:
        return True
    if int(state.vertical_direction) in (1, 2):
        return True
    return False


def ceres_first_moonfall_action(
    state,
    track: CeresFirstMoonfallTrack,
) -> tuple[tuple[str, ...], CeresFirstMoonfallTrack]:
    """One-frame Ceres elevator → Falling moonfall policy (ROM-free)."""
    x = int(state.samus_x)
    y = int(state.samus_y)
    room = int(state.room_id)
    phase = track.phase
    held = track.held
    grounded = not _ceres_first_airborne(state)

    if room == ROOM_CERES_FALLING and int(state.game_state) == 8:
        return (), replace(track, phase="done", held=0)
    # TAS idles the fade (gs 9 then 11). Holding B+RIGHT does not shorten it.
    if int(state.game_state) in (9, 10, 11):
        return (), replace(track, phase="exit", held=held + 1)
    if room != ROOM_CERES_ELEVATOR and phase != "exit":
        return ("RIGHT", "B"), replace(track, phase="exit", held=0)

    if is_knockback(state) and phase not in ("exit", "done", "land"):
        away = "LEFT" if x >= 190 else "RIGHT"
        return (away,), replace(track, phase="fall", held=max(held, 25))

    if phase == "ride":
        if y >= _CERES_FIRST_PAD_Y or is_airborne(state):
            return CERES_FIRST_TAS_PAD[0], replace(track, phase="hop", held=1)
        return (), track

    if phase in ("hop", "air_turn", "moon_arm", "fall"):
        if room != ROOM_CERES_ELEVATOR:
            return ("RIGHT", "B"), replace(track, phase="exit", held=0)
        if held < len(CERES_FIRST_TAS_PAD):
            names = CERES_FIRST_TAS_PAD[held]
            nxt = held + 1
            return names, replace(track, phase=_ceres_first_tas_phase(held), held=nxt)
        if grounded and y >= 650 and x >= 175:
            return ("RIGHT", "B"), replace(track, phase="land", held=0, pump_i=0)
        if y >= 640:
            return ("RIGHT", "B"), replace(track, phase="land", held=0, pump_i=0)
        return ("RIGHT", "B"), replace(track, phase="fall", held=held + 1)

    if phase == "land":
        if room != ROOM_CERES_ELEVATOR:
            return ("RIGHT", "B"), replace(track, phase="exit", held=0)
        if is_knockback(state) and held < 16:
            return (), replace(track, held=held + 1)
        if y > 655:
            return ("RIGHT", "A"), replace(track, held=held + 1)
        pump = shoulder_pump_button(track.pump_i)
        if x >= _CERES_FIRST_DOOR_X:
            return ("RIGHT", "B", pump), replace(
                track, phase="exit", held=0, pump_i=track.pump_i + 1
            )
        return ("RIGHT", "B", pump), replace(
            track, held=held + 1, pump_i=track.pump_i + 1
        )

    if phase == "exit":
        if room == ROOM_CERES_FALLING and int(state.game_state) == 8:
            return (), replace(track, phase="done", held=0)
        return ("RIGHT", "B"), replace(track, held=held + 1)

    return (), track


def play_ceres_first_room_moonfall(
    session: ControllerSession,
    *,
    max_frames: int = 900,
    restore_moonwalk: bool = True,
) -> None:
    """TAS-shaped Ceres 1 moonfall. Pokes $09E4 on, off after Falling settle."""
    env = getattr(session, "env", None)
    if env is None:
        raise RuntimeError("ceres first moonfall needs session.env for $09E4 poke")
    set_moonwalk(env, True)
    session.state = parse_state(env.get_ram(), frame=session.frame)
    require_moonwalk_on(session.state, label="ceres_first_moonfall")
    if session.state.room_id != ROOM_CERES_ELEVATOR and session.state.game_state not in (
        8,
        11,
    ):
        raise RuntimeError(
            f"ceres first moonfall: expected elev 0x{ROOM_CERES_ELEVATOR:04X}, "
            f"got {session.state}"
        )

    track = CeresFirstMoonfallTrack()
    for _ in range(max_frames):
        names, track = ceres_first_moonfall_action(session.state, track)
        action = buttons(*names) if names else idle_action()
        session.step(action, f"ceres_first_{track.phase}")
        if track.phase == "done" or (
            session.state.room_id == ROOM_CERES_FALLING
            and int(session.state.game_state) == 8
        ):
            break
    else:
        raise TimeoutError(
            f"ceres first moonfall missed Falling after {max_frames}f: "
            f"{session.state} phase={track.phase} ({WIKI_URL})"
        )

    if session.state.room_id != ROOM_CERES_FALLING or session.state.game_state == 11:
        _ceres_wait_ordinary(
            session,
            ROOM_CERES_FALLING,
            reason="ceres_first_falling_settle",
            timeout=FALLING_SETTLE,
        )
    if restore_moonwalk:
        set_moonwalk(env, False)
        session.state = parse_state(env.get_ram(), frame=session.frame)


CeresFallingPhase = Literal[
    "door",
    "ledge",
    "floor_hop",
    "magnet_feet",
    "plat",
    "exit_hop",
    "exit",
    "done",
]


@dataclass(frozen=True)
class CeresFallingTrack:
    """TAS Ceres 2: arm-pump the slope, magnet-feet onto y=171, fly the exit.

    Sniq 100% lsnes (gs=8 f8950→door f9071): B+RIGHT with L every other
    frame down the y=139 slope, then a 3f short hop at x≈152 y=187
    (B+RIGHT+A, B+A, B+LEFT+RIGHT+A) so magnet feet plant y=171. A 3f
    spin jump hangs at y=178 and misses the shelf. Run the shelf, 4f
    B+RIGHT+A + 2f A, idle-spin so y stays ~148 into the east door.
    Door is LEFT+B+X, not knockback.
    """

    phase: CeresFallingPhase = "ledge"
    pump_i: int = 0
    floor_hopped: bool = False
    exit_hopped: bool = False
    hop_held: int = 0
    steam_shot: bool = False


def _ceres_falling_air(state) -> bool:
    if is_airborne(state):
        return True
    if int(state.movement_type) in _AIR_MT:
        return True
    return int(state.vertical_direction) in (1, 2)


def _ceres_pump(direction: str, i: int) -> tuple[str, ...]:
    return (direction, "B", shoulder_pump_button(i))


def _ceres_shelf_pump(direction: str, i: int) -> tuple[str, ...]:
    """TAS Ceres 2 y=171 shelf: B+dir, L on odd frames. Not the entry slope."""
    if i % 2:
        return (direction, "B", "L")
    return (direction, "B")


def _ceres_magnet_feet_jump(held: int) -> tuple[str, ...]:
    """Wiki Ceres 2 short hop + L/R magnet feet onto y=171.

    Sniq 100% lsnes f8973–8975: 1f B+RIGHT+A, 1f B+A, 1f B+LEFT+RIGHT+A.
    Spin-jump (dir+B+A held) never plants the shelf. Do not X during the
    hop — that unspins into the y=171 face.
    """
    if held <= 1:
        return ("RIGHT", "B", "A")
    if held == 2:
        return ("B", "A")
    return ("LEFT", "RIGHT", "B", "A")


def ceres_falling_magnet_feet_action(
    state,
    track: CeresFallingTrack,
) -> tuple[tuple[str, ...], CeresFallingTrack]:
    """One-frame Falling Tile → Magnet policy (ROM-free)."""
    room = int(state.room_id)
    gs = int(state.game_state)
    x = int(state.samus_x)
    y = int(state.samus_y)
    air = _ceres_falling_air(state)

    if room == ROOM_CERES_MAGNET and gs == 8:
        return (), replace(track, phase="done")
    if gs != GS_ORDINARY:
        return (), replace(track, phase="door")
    if room != ROOM_CERES_FALLING:
        return ("RIGHT", "B"), replace(track, phase="exit")

    floor = CERES_FALLING_FLOOR_HOP
    exit_hop = CERES_FALLING_EXIT_HOP

    # Entry ledge is also y=139 ≤ 176. Only the y=187 floor (x≳148) is the
    # magnet-feet hop. An air+floor_hopped jump on the slope burns run speed.
    on_floor = y >= _CERES_FALLING_OUT_FLOOR_Y - 2
    on_shelf = (not air) and y <= _CERES_FALLING_OUT_PLAT_Y + 2 and x >= 148

    if air and track.hop_held >= 1 and not track.exit_hopped and not on_shelf:
        if track.hop_held < 3:
            nxt = track.hop_held + 1
            return _ceres_magnet_feet_jump(nxt), replace(
                track, phase="floor_hop", hop_held=nxt
            )
        if y <= _CERES_FALLING_OUT_PLAT_Y + 2:
            return _ceres_shelf_pump("RIGHT", track.pump_i), replace(
                track, phase="plat", hop_held=0, pump_i=track.pump_i + 1
            )
        if on_floor and track.hop_held >= 6:
            return _ceres_pump("RIGHT", track.pump_i), replace(
                track,
                phase="ledge",
                floor_hopped=False,
                hop_held=0,
                pump_i=track.pump_i + 1,
            )
        return ("RIGHT", "B"), replace(
            track, phase="magnet_feet", hop_held=track.hop_held + 1
        )

    if (not air) and on_floor:
        if (
            x <= 168
            and int(state.momentum_x) >= 1
            and (floor.ready(state) or floor.at_ledge_end(x) or x >= 145)
        ):
            return _ceres_magnet_feet_jump(1), replace(
                track, phase="floor_hop", floor_hopped=True, hop_held=1
            )
        return _ceres_pump("RIGHT", track.pump_i), replace(
            track, phase="ledge", hop_held=0, pump_i=track.pump_i + 1
        )

    if air and track.exit_hopped:
        # TAS: 4f B+RIGHT+A, 2f A, ~19f idle-spin, then B+LEFT+X and run.
        # Idling until y>175 lands the door lip (pose 156 stall at x=462).
        if track.hop_held < 4:
            return spin_jump("RIGHT"), replace(
                track, phase="exit_hop", hop_held=track.hop_held + 1
            )
        if track.hop_held < 6:
            return ("A",), replace(track, phase="exit_hop", hop_held=track.hop_held + 1)
        if track.hop_held < 25 and y <= 170:
            return (), replace(track, phase="exit_hop", hop_held=track.hop_held + 1)
        return ("RIGHT", "B"), replace(track, phase="exit")

    if (
        (not air)
        and exit_hop.covers_y(y, slack=12)
        and (exit_hop.ready(state) or exit_hop.at_ledge_end(x) or x >= 320)
        and not track.exit_hopped
    ):
        return spin_jump("RIGHT"), replace(
            track, phase="exit_hop", exit_hopped=True, hop_held=1
        )

    if x >= 450 and y <= 150 and not track.steam_shot:
        return ("RIGHT", "B", "X"), replace(track, phase="exit", steam_shot=True)

    if x >= 430:
        return ("RIGHT", "B"), replace(track, phase="exit")

    if air:
        return ("RIGHT", "B"), replace(
            track, phase="exit_hop" if track.exit_hopped else "ledge"
        )

    pump_names = (
        _ceres_shelf_pump("RIGHT", track.pump_i)
        if on_shelf
        else _ceres_pump("RIGHT", track.pump_i)
    )
    return pump_names, replace(
        track,
        phase="plat" if on_shelf else "ledge",
        pump_i=track.pump_i + 1,
        floor_hopped=track.floor_hopped or on_shelf,
        hop_held=0 if on_shelf else track.hop_held,
    )


def _play_track(
    session,
    *,
    src: int,
    dest: int,
    track,
    action,
    door_kb,
    tag: str,
    missed: str,
    max_frames: int,
) -> None:
    if int(session.state.room_id) == dest and int(session.state.game_state) == 8:
        return
    if int(session.state.room_id) != src or int(session.state.game_state) != 8:
        _ceres_wait_ordinary(
            session, src, reason=f"{tag}_ordinary", timeout=FALLING_SETTLE
        )
    for _ in range(max_frames):
        st = session.state
        if int(st.room_id) == dest and int(st.game_state) == 8:
            return
        if is_knockback(st):
            if door_kb(st):
                session.step(buttons("RIGHT"), f"{tag}_door_kb")
                continue
            _ceres_clear_knockback(session, "RIGHT", reason=f"{tag}_out")
            continue
        names, track = action(st, track)
        session.step(
            buttons(*names) if names else idle_action(),
            f"{tag}_{track.phase}",
        )
        if track.phase == "done":
            break
    else:
        raise TimeoutError(
            f"{missed} after {max_frames}f: {session.state} phase={track.phase}"
        )
    if session.state.room_id != dest or session.state.game_state == 11:
        _ceres_wait_ordinary(
            session, dest, reason=f"{tag}_settle", timeout=FALLING_SETTLE
        )


def play_ceres_falling_to_magnet(
    session: ControllerSession,
    *,
    max_frames: int = 700,
) -> None:
    """Wiki Ceres 2 magnet-feet hop. Waits dest gs=8."""
    _play_track(
        session,
        src=ROOM_CERES_FALLING,
        dest=ROOM_CERES_MAGNET,
        track=CeresFallingTrack(),
        action=ceres_falling_magnet_feet_action,
        door_kb=lambda s: int(s.samus_x) >= 400,
        tag="ceres_falling",
        missed="ceres falling magnet-feet missed Magnet",
        max_frames=max_frames,
    )


CeresMagnetPhase = Literal[
    "top",
    "jump1",
    "drop1",
    "mid",
    "jump2",
    "drop2",
    "bot",
    "exit",
    "done",
]


@dataclass(frozen=True)
class CeresMagnetTrack:
    """Wiki Ceres 3: jump before each magnet-stair ledge.

    Leave pin is door-height (39, 139) p9 — the LEFT 120 tape walks back
    into Falling from here. L/R on the stairs bounces pose 41. Bare run,
    short jump the y=139 ledge, DOWN/LEFT onto y=219, jump the slope at
    ~x131 y255, release A on vd=1, RIGHT to the east door. Steam at
    x~177: 1f LEFT+B+X, then RIGHT — do not tank pose 137.
    """

    phase: CeresMagnetPhase = "top"
    hop_held: int = 0
    steam_shot: bool = False


def ceres_magnet_to_scientist_action(
    state,
    track: CeresMagnetTrack,
) -> tuple[tuple[str, ...], CeresMagnetTrack]:
    """One-frame Magnet Stairs → Scientist policy (ROM-free)."""
    room = int(state.room_id)
    gs = int(state.game_state)
    x = int(state.samus_x)
    y = int(state.samus_y)
    air = _ceres_falling_air(state)

    if room == ROOM_CERES_SCIENTIST and gs == 8:
        return (), replace(track, phase="done")
    if gs != GS_ORDINARY:
        return (), replace(track, phase="exit", hop_held=track.hop_held + 1)
    if room != ROOM_CERES_MAGNET:
        return ("RIGHT", "B"), replace(track, phase="exit")

    top = CERES_MAGNET_TOP_HOP
    mid = CERES_MAGNET_MID_HOP

    if (
        (not air)
        and y >= _CERES_MAGNET_DOOR_Y - 8
        and 168 <= x <= 190
        and not track.steam_shot
    ):
        # TAS: 1f LEFT+B+X at x~177, then RIGHT. Hitbox/timing dodge, not tank.
        return ("LEFT", "B", "X"), replace(track, phase="exit", steam_shot=True)

    if (not air) and y <= _CERES_MAGNET_TOP_Y + 16 and track.phase in (
        "top",
        "jump1",
        "drop1",
    ):
        if top.ready(state) or top.at_ledge_end(x) or x >= 126:
            return spin_jump("RIGHT"), replace(track, phase="jump1", hop_held=1)
        return _ceres_pump("RIGHT", track.hop_held), replace(
            track, phase="top", hop_held=track.hop_held + 1
        )

    if track.phase == "jump1":
        if (not air) and y >= _CERES_MAGNET_MID_Y - 8:
            return ("LEFT", "B"), replace(track, phase="mid", hop_held=0)
        if (not air) and y <= _CERES_MAGNET_TOP_Y + 16:
            return ("RIGHT", "B"), replace(track, phase="top", hop_held=0)
        # 3f spin then DOWN — idle-coast on this pin plants the 139 ledge (pose 41).
        if track.hop_held < 3:
            return spin_jump("RIGHT"), replace(
                track, phase="jump1", hop_held=track.hop_held + 1
            )
        return ("DOWN",), replace(track, phase="drop1", hop_held=track.hop_held + 1)

    if track.phase == "drop1":
        if (not air) and y >= _CERES_MAGNET_MID_Y - 8:
            return ("LEFT", "B"), replace(track, phase="mid", hop_held=0)
        if (not air) and y <= _CERES_MAGNET_TOP_Y + 16:
            return ("RIGHT", "B"), replace(track, phase="top", hop_held=0)
        if y < 155:
            return ("DOWN",), replace(track, hop_held=track.hop_held + 1)
        return ("LEFT",), replace(track, hop_held=track.hop_held + 1)

    if track.phase in ("mid", "jump2", "drop2") or (
        (not air) and _CERES_MAGNET_MID_Y - 8 <= y < _CERES_MAGNET_BOT_Y - 8
    ):
        if (not air) and y >= _CERES_MAGNET_BOT_Y - 8:
            return ("RIGHT", "B"), replace(track, phase="bot", hop_held=0)
        if air and track.phase == "jump2":
            if track.hop_held < 3:
                return spin_jump("LEFT"), replace(
                    track, hop_held=track.hop_held + 1
                )
            # TAS: 1f LEFT (release A) then 15f A, not an immediate DOWN.
            return ("LEFT",), replace(track, phase="drop2", hop_held=track.hop_held + 1)
        if air and track.phase == "drop2":
            if y >= 290:
                return ("RIGHT",), replace(track, hop_held=track.hop_held + 1)
            if track.hop_held < 19:
                return ("A",), replace(track, hop_held=track.hop_held + 1)
            return ("DOWN", "A"), replace(track, hop_held=track.hop_held + 1)
        if (
            (not air)
            and track.phase in ("mid", "drop1")
            and y >= 255
            and y <= 280
            and x <= 135
            and (mid.ready(state) or mid.at_ledge_end(x) or x <= 128)
        ):
            return spin_jump("LEFT"), replace(track, phase="jump2", hop_held=1)
        if not air:
            if track.phase == "drop2":
                return ("LEFT", "B"), track
            return ("LEFT", "B"), replace(track, phase="mid", hop_held=0)
        return ("LEFT",), replace(track, hop_held=track.hop_held + 1)

    if y >= _CERES_MAGNET_BOT_Y - 16 or track.phase in ("bot", "exit"):
        if x >= _CERES_MAGNET_OUT_DOOR_X or y >= _CERES_MAGNET_DOOR_Y - 8:
            return ("RIGHT", "B"), replace(track, phase="exit")
        return ("RIGHT", "B"), replace(track, phase="bot")

    if air:
        return ("RIGHT", "B"), track
    return ("RIGHT", "B"), replace(track, phase="top")


def play_ceres_magnet_to_scientist(
    session: ControllerSession,
    *,
    max_frames: int = 700,
) -> None:
    """Wiki Ceres 3 jump-before-ledge. Waits dest gs=8."""
    _play_track(
        session,
        src=ROOM_CERES_MAGNET,
        dest=ROOM_CERES_SCIENTIST,
        track=CeresMagnetTrack(),
        action=ceres_magnet_to_scientist_action,
        door_kb=lambda s: int(s.samus_y) >= _CERES_MAGNET_DOOR_Y - 20,
        tag="ceres_magnet",
        missed="ceres magnet stairs missed Scientist",
        max_frames=max_frames,
    )


def play_ceres_to_ridley_door(session: RouteSession) -> None:
    """Ceres elevator → Ridley room ordinary settle (no fight)."""
    play_ceres_first_room_moonfall(session)
    play_ceres_falling_to_magnet(session)
    play_ceres_magnet_to_scientist(session)
    # Dead Scientist is its own hop: TAS-style arm-pump run, never jump.
    play_ceres_scientist_to_flat(session)
    play_ceres_flat_to_ridley(session)
    _ceres_wait_ordinary(
        session, ROOM_CERES_RIDLEY, reason="ceres_ridley_door", timeout=200
    )


def play_ceres_outbound_to_ridley(session: RouteSession) -> None:
    """Ceres elevator → Ridley + countdown (classic L↔R arm-pump).

    Elev→Falling uses spinning moonfall (wiki Ceres 1 / Sniq pad body)
    then wiki Ceres 2 magnet-feet and Ceres 3 jump-before-ledge. Dead Scientist
    arm-pumps the pit and stairs (no jump);
    Flat→Ridley is room-gated arm-pump. Fight body is tail-tank
    :func:`play_ceres_ridley_fight`. Escape re-solves magnet / falling / elev
    reactively.
    """
    play_ceres_to_ridley_door(session)
    # Late import: combat.__init__ → progression → early_spine → this module.
    from super_metroid.combat.ceres_ridley import (
        CeresRidleyStrategy,
        play_ceres_ridley_fight,
        require_ceres_ridley_countdown,
    )

    # Energy assist is already suspended on Ceres.
    evidence = play_ceres_ridley_fight(session, strategy=CeresRidleyStrategy())
    require_ceres_ridley_countdown(evidence)
    # Tail-tank often ends in KB (pose 137/138). Escape LEFT+A needs standing.
    for _ in range(40):
        if not is_knockback(session.state):
            break
        session.step(idle_action(), "ceres_ridley_settle")


class CeresFlatEscape:
    """One-frame reverse Ceres 5 (Flat → Scientist). Never jump.

    Sniq 100% lsnes (gs=8 f11939→door f12036) never presses A: LEFT+B+L/R
    across y=139 from (472,139) p18. stuck-jump on this corridor is leftover.
    """

    def __init__(self) -> None:
        self.pump_i = 0

    def action(self, state: SuperMetroidState) -> tuple[str, ...]:
        if int(state.game_state) != GS_ORDINARY:
            return ("LEFT",)
        names = ("LEFT", "B", shoulder_pump_button(self.pump_i))
        self.pump_i += 1
        return names


def _flat_escape_past(state: SuperMetroidState) -> bool:
    """True in Scientist/Magnet ordinary — not the Flat→Scientist door."""
    if int(state.game_state) != GS_ORDINARY:
        return False
    return int(state.room_id) in (ROOM_CERES_SCIENTIST, ROOM_CERES_MAGNET)


def _flat_outbound_past(state: SuperMetroidState) -> bool:
    if int(state.game_state) != GS_ORDINARY:
        return False
    return int(state.room_id) == ROOM_CERES_RIDLEY


def play_ceres_flat_to_ridley(session: RouteSession) -> None:
    """Flat ordinary → Ridley. TAS dwell never presses A."""
    if _flat_outbound_past(session.state):
        return
    for _ in range(180):
        st = session.state
        if _flat_outbound_past(st):
            return
        if int(st.room_id) == ROOM_CERES_FLAT and int(st.game_state) == GS_ORDINARY:
            break
        session.step(buttons("RIGHT"), "ceres_flat_out_door")
    else:
        st = session.state
        if int(st.room_id) != ROOM_CERES_FLAT:
            raise TimeoutError(f"ceres flat outbound ordinary missed: {st}")

    cross = CeresScientistCross("RIGHT")
    for _ in range(400):
        st = session.state
        if _flat_outbound_past(st):
            return
        if is_knockback(st):
            _ceres_clear_knockback(session, "RIGHT", reason="ceres_flat_out")
            continue
        names = cross.action(st)
        reason = "ceres_flat_out_fade" if int(st.game_state) != GS_ORDINARY else "ceres_flat_out"
        session.step(buttons(*names) if names else idle_action(), reason)
    raise TimeoutError(f"ceres flat missed Ridley: {session.state}")


def play_ceres_flat_to_scientist(session: RouteSession) -> None:
    """Flat ordinary → Scientist (or Magnet if the door overshoots).

    No-op when already past the room. Waits out the Ridley→Flat door.
    TAS dwell never jumps; product stuck-jump is the leftover versus 259f.
    """
    if _flat_escape_past(session.state):
        return
    for _ in range(220):
        st = session.state
        if _flat_escape_past(st):
            return
        if int(st.room_id) == ROOM_CERES_FLAT and int(st.game_state) == GS_ORDINARY:
            break
        session.step(buttons("LEFT"), "ceres_flat_door")
    else:
        st = session.state
        if int(st.room_id) != ROOM_CERES_FLAT:
            raise TimeoutError(f"ceres flat ordinary missed: {st}")

    cross = CeresFlatEscape()
    for _ in range(400):
        st = session.state
        if _flat_escape_past(st):
            return
        if is_knockback(st):
            _ceres_clear_knockback(session, "LEFT", reason="ceres_flat")
            continue
        names = cross.action(st)
        if int(st.game_state) != GS_ORDINARY:
            reason = "ceres_flat_fade"
        else:
            reason = "ceres_flat"
        session.step(buttons(*names) if names else idle_action(), reason)
    raise TimeoutError(f"ceres flat missed Scientist: {session.state}")


def play_ceres_escape_to_landing(session: RouteSession) -> None:
    """Ceres reverse + elev → Zebes Landing (arm-pump + WRAM-reactive).

    Magnet escape is TAS 347→267→steam→219→139.
    Falling / elev still re-solve from room, y, pose, knockback.
    Ridley exit is still the product LEFT+A (not tuned). Reverse Ceres 5
    (Flat) and reverse Ceres 4 (Scientist) never jump.
    """
    # Leave Ridley left (jump clear of platform). Not tuned this sitting.
    session.span(ActionSpan(("LEFT", "A"), 24, "ceres_ridley_exit"))
    play_ceres_flat_to_scientist(session)
    play_ceres_scientist_to_magnet(session)
    play_ceres_magnet_to_falling(session)
    play_ceres_falling_to_elev(session)
    _ceres_reactive_elev_climb(session)

    session.wait_until(
        lambda state: state.room_id == ROOM_LANDING_SITE and state.game_state == 8,
        timeout=3_000,
        reason="zebes_landing_transition",
    )
    stable = 0
    for _ in range(1_200):
        if session.state.samus_y == 1088:
            stable += 1
            if stable >= 30:
                break
        else:
            stable = 0
        session.step(idle_action(), "zebes_ship_final_settle")
    else:
        raise TimeoutError(f"Zebes ship never reached final settle: {session.state}")


__all__ = [
    "CERES_FIRST_TAS_PAD",
    "CERES_FIRST_TAS_PAD_SPANS",
    "CeresFirstMoonfallTrack",
    "CeresFallingTrack",
    "CeresMagnetTrack",
    "CeresFlatEscape",
    "ceres_first_moonfall_action",
    "ceres_falling_magnet_feet_action",
    "ceres_magnet_to_scientist_action",
    "play_ceres_first_room_moonfall",
    "play_ceres_falling_to_magnet",
    "play_ceres_magnet_to_scientist",
    "play_ceres_flat_to_ridley",
    "play_ceres_flat_to_scientist",
    "play_ceres_to_ridley_door",
    "play_ceres_outbound_to_ridley",
    "play_ceres_escape_to_landing",
]
