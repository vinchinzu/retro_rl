"""Ceres reverse: Magnet Stairs, Falling Tile, Elevator to ship.

Magnet: hop east steam, jump 347 at x≈74, air-turn RIGHT onto 267, run
RIGHT, jump 219 then 139. L-every-other while running. Wait on 267 when
the jet spritemap is idle.

Falling reverse: run off 139 onto 187, hop 347 onto 171, turn RIGHT at
x≤314 leftover LEFT mx, 1f p25, B-only fall, LEFT+B+A so pose 80 rides
LEFT. Door: TAS jumps x≈37 y=139, air-turns pose 25 at x≲28.

Elevator: one TAS wall-jump climb. Fast entry is x216 y624–641 spin.
Missed wall jump is a hard fail. No checkpoint recovery. Shaft over
2500f is a hard fail.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from retro_harness.actions import buttons, idle_action
from super_metroid.combat.enemies import CERES_DOOR_ID, list_enemies
from super_metroid.combat.enemies.species import enemy_overlaps, steam_jet_shown
from super_metroid.ram import GS_CERES_LEAVE, GS_ORDINARY
from super_metroid.routes.controller_common import POSE_WALL_LATCH
from super_metroid.routes.kpdr.ceres.geometry import (
    CERES_FALLING_REV_FLOOR_HOP,
    CERES_MAGNET_HIGH_HOP,
    CERES_MAGNET_MID_ESCAPE_HOP,
    CERES_MAGNET_STEAM_HOP,
    _CERES_ELEV_BOTTOM_Y,
    _CERES_ELEV_SHIP_X,
    _CERES_ELEV_SHIP_Y,
    _CERES_ELEV_TOP_X,
    _CERES_ELEV_TOP_Y,
    _CERES_FALLING_DOOR_JUMP_X,
    _CERES_FALLING_DOOR_LEDGE_Y,
    _CERES_FALLING_DOOR_TURN_X,
    _CERES_FALLING_REV_FLOOR_Y,
    _CERES_FALLING_REV_SHELF_Y,
    _CERES_FALLING_REV_TILE_X,
    _CERES_FALLING_REV_TURN_X,
    _CERES_MAGNET_BOT_Y,
    _CERES_MAGNET_DOOR_STEAM_FRAMES,
    _CERES_MAGNET_SHELF_Y,
    _CERES_MAGNET_TOP_Y,
)
from super_metroid.routes.kpdr.room_ids import (
    ROOM_CERES_ELEVATOR,
    ROOM_CERES_FALLING,
    ROOM_CERES_MAGNET,
)
from super_metroid.routes.runtime import ActionSpan, RouteSession
from super_metroid.routes.skills.geometry import (
    CROUCH_POSES,
    LAND_POSES,
    POSE_KNOCKBACK,
    SPIN_POSES,
    STAND_LOCOMOTION_POSES,
)
from super_metroid.routes.skills.knockback import is_knockback
from super_metroid.takeoff import walk_toward_x


CeresMagnetEscapePhase = Literal[
    "door",
    "slope",
    "shelf_hop",
    "shelf",
    "steam_hop",
    "mid_hop",
    "exit",
    "done",
]
CeresFallingEscapePhase = Literal[
    "door",
    "run_off",
    "floor_hop",
    "shelf",
    "dboost",
    "slope",
    "exit",
    "done",
]
_POSE_DBOOST = 80


@dataclass(frozen=True)
class CeresMagnetEscapeTrack:
    """One-frame Magnet→Falling policy state."""

    phase: CeresMagnetEscapePhase = "door"
    held: int = 0
    pump_i: int = 0
    contacted: bool = False
    steam_shown: bool = True


@dataclass(frozen=True)
class CeresFallingEscapeTrack:
    """One-frame Falling→elev policy state."""

    phase: CeresFallingEscapePhase = "door"
    held: int = 0
    pump_i: int = 0
    contacted: bool = False
    boosted: bool = False


def _ceres_magnet_reached_falling(state) -> bool:
    return int(state.room_id) == ROOM_CERES_FALLING and int(state.game_state) == 8


def _ceres_falling_reached_elev(state) -> bool:
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and int(state.game_state) == 8
        and int(state.vertical_direction) == 1
        and int(state.velocity_y) > 0
        and int(state.momentum_x) >= 2
        and int(state.invincibility_timer) > 0
    )


def _ceres_grounded(state) -> bool:
    return int(state.vertical_direction) == 0 and abs(int(state.velocity_y)) <= 1


def _ceres_planted_near(state, y: int, *, slack: int = 5) -> bool:
    """Natural Ceres shelf plant; movement type 0 is the one-frame land."""
    return (
        int(state.game_state) == 8
        and abs(int(state.samus_y) - y) <= slack
        and _ceres_grounded(state)
    )


def _tas_l_pump(direction: str, i: int, state) -> tuple[str, ...]:
    """TAS magnet run: dir+B, L on odd frames, only once already running.

    Not L↔R period-2. Not a force-pump while accelerating.
    """
    running = int(state.speed_flag) != 0 or abs(int(state.momentum_x)) >= 1
    if running and i % 2 == 1:
        return (direction, "B", "L")
    return (direction, "B")


def _steam_kb(state) -> bool:
    """Ceres steam knockback is mt=10 / timer, not Zebes pose 137/138."""
    return int(state.knockback_timer) > 0 or int(state.movement_type) == 10


def ceres_magnet_escape_action(
    state,
    track: CeresMagnetEscapeTrack,
) -> tuple[tuple[str, ...], CeresMagnetEscapeTrack]:
    """One-frame Magnet Stairs → Falling policy."""
    room = int(state.room_id)
    gs = int(state.game_state)
    x = int(state.samus_x)
    y = int(state.samus_y)
    bot = CERES_MAGNET_HIGH_HOP
    steam = CERES_MAGNET_STEAM_HOP
    mid = CERES_MAGNET_MID_ESCAPE_HOP

    if _ceres_magnet_reached_falling(state):
        return (), replace(track, phase="done")
    if room == ROOM_CERES_FALLING:
        return ("LEFT",), replace(track, phase="exit")
    if gs != 8:
        return ("LEFT",), replace(track, phase="door", held=track.held + 1)
    if room != ROOM_CERES_MAGNET:
        return ("LEFT", "B"), replace(track, phase="exit")

    kb = _steam_kb(state)
    grounded = _ceres_grounded(state)

    planted_347 = grounded and abs(y - _CERES_MAGNET_BOT_Y) <= 8

    if track.phase in ("door", "slope"):
        if kb:
            return ("LEFT", "B", "A"), replace(track, phase="slope")
        if track.phase == "door" and track.held < _CERES_MAGNET_DOOR_STEAM_FRAMES:
            return ("LEFT", "B", "A"), replace(
                track, phase="door", held=track.held + 1
            )
        if planted_347 and (bot.ready(state) or x <= 70):
            return ("LEFT", "B", "A"), replace(track, phase="shelf_hop", held=1)
        # Door hop shifts subpixel vs TAS; start L on the other phase so
        # the 347 east corner is not the magnet-stop tile.
        pump_i = 1 if track.phase == "door" else track.pump_i
        names = _tas_l_pump("LEFT", pump_i, state)
        return names, replace(track, phase="slope", pump_i=pump_i + 1)

    if track.phase == "shelf_hop":
        if grounded and steam.covers_y(y):
            names = _tas_l_pump("RIGHT", 0, state)
            return names, replace(track, phase="shelf", pump_i=1, held=0)
        if grounded and abs(y - _CERES_MAGNET_BOT_Y) <= 8:
            names = ("B", "A") if bot.ready(state) else ("LEFT", "B", "A")
            return names, replace(track, held=track.held + 1)
        if not grounded:
            held = track.held + 1
            # 267 underside is x≳65 y=332. Reach the shaft (x≲60), then
            # RIGHT onto 267. Release A near y=267 so we plant, not fly over.
            if x <= 66:
                if y <= 275:
                    return ("RIGHT", "B"), replace(track, held=held)
                return ("RIGHT", "A"), replace(track, held=held)
            if held <= 3:
                return ("LEFT", "B", "A"), replace(track, held=held)
            return ("A",), replace(track, held=held)
        names = _tas_l_pump("LEFT", track.pump_i, state)
        return names, replace(track, phase="slope", pump_i=track.pump_i + 1)

    if track.phase == "shelf":
        if grounded and steam.covers_y(y) and 60 <= x <= 140:
            if steam.ready(state) and (track.steam_shown or track.held >= 24):
                return ("RIGHT", "B", "A"), replace(track, phase="steam_hop", held=1)
            if not track.steam_shown:
                direction = "LEFT" if x >= 122 else "RIGHT"
                names = _tas_l_pump(direction, track.pump_i, state)
                return names, replace(
                    track, pump_i=track.pump_i + 1, held=track.held + 1
                )
        if grounded and steam.ready(state):
            return ("RIGHT", "B", "A"), replace(track, phase="steam_hop", held=1)
        names = _tas_l_pump("RIGHT", track.pump_i, state)
        return names, replace(track, pump_i=track.pump_i + 1)

    if track.phase == "steam_hop":
        if grounded and mid.covers_y(y):
            if x < 188:
                names = _tas_l_pump("RIGHT", track.pump_i, state)
                return names, replace(track, pump_i=track.pump_i + 1, held=0)
            return ("LEFT", "B", "A"), replace(track, phase="mid_hop", held=1)
        if grounded and y <= _CERES_MAGNET_TOP_Y + 8:
            names = _tas_l_pump("LEFT", 0, state)
            return names, replace(track, phase="exit", pump_i=1, held=0)
        contacted = track.contacted or kb
        if kb:
            return ("RIGHT", "B", "A"), replace(
                track, contacted=True, held=track.held + 1
            )
        if not grounded:
            held = track.held + 1
            if y <= 225 and x >= 160:
                return ("RIGHT", "B"), replace(
                    track, held=held, contacted=contacted
                )
            return ("RIGHT", "B", "A"), replace(
                track, held=held, contacted=contacted
            )
        names = _tas_l_pump("RIGHT", track.pump_i, state)
        return names, replace(
            track, phase="shelf", pump_i=track.pump_i + 1, contacted=contacted
        )

    if track.phase == "mid_hop":
        if grounded and y <= _CERES_MAGNET_TOP_Y + 8:
            names = _tas_l_pump("LEFT", 0, state)
            return names, replace(track, phase="exit", pump_i=1, held=0)
        if grounded and mid.covers_y(y) and x < 188:
            names = _tas_l_pump("RIGHT", track.pump_i, state)
            return names, replace(
                track, phase="steam_hop", pump_i=track.pump_i + 1, held=0
            )
        if grounded:
            return ("LEFT", "B", "A"), replace(track, held=track.held + 1)
        if y <= 150:
            return ("LEFT", "B"), replace(track, held=track.held + 1)
        if y <= 180:
            return ("LEFT", "B", "A"), replace(track, held=track.held + 1)
        return ("A",), replace(track, held=track.held + 1)

    # 139 west magnet-stop is pose 138 at x≈45. Hop the planted corner.
    pose = int(state.pose)
    if grounded and y <= _CERES_MAGNET_TOP_Y + 8 and pose in (137, 138):
        return ("LEFT", "B", "A"), replace(track, phase="exit", held=track.held + 1)
    names = _tas_l_pump("LEFT", track.pump_i, state)
    return names, replace(track, phase="exit", pump_i=track.pump_i + 1)


def _magnet_steam_shown(session: RouteSession) -> bool:
    """True when a 267-height jet is on the live spritemap cycle."""
    for enemy in list_enemies(session):
        if not steam_jet_shown(enemy):
            continue
        if abs(int(enemy.y) - _CERES_MAGNET_SHELF_Y) <= 40:
            return True
        if abs(int(enemy.x) - 62) <= 16 and abs(int(enemy.y) - 304) <= 16:
            return True
    return False


def play_ceres_magnet_to_falling(
    session: RouteSession, *, max_frames: int = 500
) -> None:
    """Magnet Stairs escape. One trajectory. Raises if Falling gs=8 misses."""
    track = CeresMagnetEscapeTrack()
    for _ in range(max_frames):
        st = session.state
        if _ceres_magnet_reached_falling(st):
            return
        track = replace(track, steam_shown=_magnet_steam_shown(session))
        names, track = ceres_magnet_escape_action(st, track)
        reason = f"ceres_magnet_{track.phase}"
        session.step(buttons(*names) if names else idle_action(), reason)
        if track.phase == "done" or _ceres_magnet_reached_falling(session.state):
            return
        if track.phase == "shelf_hop" and track.held > 50:
            raise TimeoutError(
                f"ceres magnet 347 hop stalled: {session.state}"
            )
        if track.phase == "steam_hop" and track.held > 60:
            raise TimeoutError(
                f"ceres magnet missed 219 from 267: {session.state}"
            )
        if track.phase == "mid_hop" and track.held > 55:
            raise TimeoutError(
                f"ceres magnet 219 hop stalled: {session.state}"
            )
    raise TimeoutError(
        f"ceres magnet escape missed Falling after {max_frames}f: "
        f"{session.state} phase={track.phase} inv={int(session.state.invincibility_timer)}"
    )


def ceres_falling_escape_action(
    state,
    track: CeresFallingEscapeTrack,
) -> tuple[tuple[str, ...], CeresFallingEscapeTrack]:
    """One-frame Falling Tile → elev policy."""
    room = int(state.room_id)
    gs = int(state.game_state)
    x = int(state.samus_x)
    y = int(state.samus_y)
    pose = int(state.pose)
    floor = CERES_FALLING_REV_FLOOR_HOP

    if _ceres_falling_reached_elev(state):
        return (), replace(track, phase="done")
    if room == ROOM_CERES_ELEVATOR:
        return ("A",), replace(track, phase="exit")
    if room == ROOM_CERES_FALLING and gs in (9, 11) and track.phase == "exit":
        return ("A",), replace(track, held=track.held + 1)
    if gs != 8:
        return ("LEFT",), replace(track, phase="door", held=track.held + 1)
    if room != ROOM_CERES_FALLING:
        return ("LEFT", "B"), replace(track, phase="exit")

    kb = _steam_kb(state)
    grounded = _ceres_grounded(state)
    contacted = track.contacted or kb
    boosted = track.boosted or pose == _POSE_DBOOST
    held = track.held + 1

    if track.phase == "door":
        if grounded and y <= _CERES_FALLING_DOOR_LEDGE_Y + 8:
            if int(state.invincibility_timer) < 6 and track.held < 2:
                return ("LEFT", "B", "A"), replace(
                    track, phase="door", held=held
                )
            return _tas_l_pump("LEFT", 0, state), replace(
                track, phase="run_off", pump_i=1, held=0
            )
        return ("LEFT", "B"), replace(track, phase="run_off", pump_i=1, held=0)

    planted_187 = _ceres_planted_near(state, _CERES_FALLING_REV_FLOOR_Y, slack=8)
    planted_171 = _ceres_planted_near(state, _CERES_FALLING_REV_SHELF_Y, slack=8)

    if track.phase == "run_off":
        if planted_187:
            if floor.ready(state) or 342 <= x <= 352:
                return ("LEFT", "B", "A"), replace(
                    track, phase="floor_hop", held=1
                )
            names = _tas_l_pump("LEFT", track.pump_i, state)
            return names, replace(track, pump_i=track.pump_i + 1)
        if planted_171:
            names = _tas_l_pump("LEFT", 0, state)
            return names, replace(track, phase="shelf", pump_i=1, held=0)
        names = _tas_l_pump("LEFT", track.pump_i, state)
        return names, replace(track, pump_i=track.pump_i + 1)

    if track.phase == "floor_hop":
        if planted_171:
            names = _tas_l_pump("LEFT", 0, state)
            return names, replace(track, phase="shelf", pump_i=1, held=0)
        if not grounded:
            # TAS unspins (p24) then plants 165. Air: B+A, then B+DOWN+A.
            if held == 2:
                return ("B", "A"), replace(track, held=held)
            if held == 3:
                return ("B", "DOWN", "A"), replace(track, held=held)
            return ("LEFT", "B"), replace(track, held=held)
        if planted_187:
            if 342 <= x <= 352:
                return ("LEFT", "B", "A"), replace(track, held=1)
            names = _tas_l_pump("LEFT", track.pump_i, state)
            return names, replace(track, pump_i=track.pump_i + 1)
        names = _tas_l_pump("LEFT", track.pump_i, state)
        return names, replace(track, phase="run_off", pump_i=track.pump_i + 1)

    if track.phase == "shelf":
        if kb:
            return ("LEFT", "A"), replace(
                track, phase="dboost", contacted=True, held=1
            )
        # TAS: p38, 1f p25, 1f B+A+X, LEFT+A on the hit. Do not RIGHT at 294.
        if planted_171 and x <= _CERES_FALLING_REV_TURN_X:
            if x <= _CERES_FALLING_REV_TILE_X:
                return ("LEFT", "A"), replace(track, phase="dboost", held=1)
            return ("RIGHT",), replace(track, phase="dboost", held=1)
        names = _tas_l_pump("LEFT", track.pump_i, state)
        return names, replace(track, pump_i=track.pump_i + 1)

    if track.phase == "dboost":
        if kb:
            return ("LEFT", "B", "A"), replace(
                track, contacted=True, boosted=True, held=held
            )
        if pose in (_POSE_DBOOST, 83, 84) or (boosted and not grounded):
            return ("LEFT", "B", "A"), replace(
                track, contacted=contacted, boosted=True, held=held
            )
        if contacted and not grounded and y < _CERES_FALLING_REV_SHELF_Y:
            return ("LEFT", "B", "A"), replace(
                track, contacted=True, boosted=True, held=held
            )
        if boosted and grounded and y <= _CERES_FALLING_DOOR_LEDGE_Y + 8:
            names = _tas_l_pump("LEFT", 0, state)
            return names, replace(
                track,
                phase="exit",
                pump_i=1,
                held=0,
                contacted=contacted,
                boosted=True,
            )
        if boosted and planted_171 and x <= 90:
            names = _tas_l_pump("LEFT", 0, state)
            return names, replace(
                track,
                phase="slope",
                pump_i=1,
                held=0,
                contacted=contacted,
                boosted=True,
            )
        if not contacted and planted_171:
            # RIGHT at 294 while facing left is p84. Stay facing right; no X.
            if x <= _CERES_FALLING_REV_TILE_X:
                return ("B", "A"), replace(track, held=held)
            if held <= 2:
                return ("RIGHT", "B", "A"), replace(track, held=held)
            return ("B", "A"), replace(track, held=held)
        if not contacted and not grounded:
            # X-unspin is p47. LEFT before contact air-turns p25 → p84.
            # Release A at y<=165 so the 294 jet is met near TAS y=162.
            if y <= 110:
                return ("LEFT",), replace(track, held=held)
            if y <= 165:
                return ("B",), replace(track, held=held)
            return ("B", "A"), replace(track, held=held)
        if planted_171:
            names = _tas_l_pump("LEFT", track.pump_i, state)
            return names, replace(
                track,
                phase="slope",
                pump_i=track.pump_i + 1,
                contacted=contacted,
                boosted=boosted,
            )
        if not grounded:
            return ("LEFT", "B"), replace(
                track, contacted=contacted, boosted=boosted, held=held
            )
        names = _tas_l_pump("LEFT", track.pump_i, state)
        return names, replace(
            track,
            phase="slope",
            pump_i=track.pump_i + 1,
            contacted=contacted,
            boosted=boosted,
        )

    if track.phase == "slope":
        if grounded and y <= _CERES_FALLING_DOOR_LEDGE_Y + 8:
            names = _tas_l_pump("LEFT", 0, state)
            return names, replace(track, phase="exit", pump_i=1, held=0)
        names = _tas_l_pump("LEFT", track.pump_i, state)
        return names, replace(track, pump_i=track.pump_i + 1)

    # Door: TAS jumps x≈37 y=139 p26, air-turns p25 at (26,120). Hold A
    # through fade. Knockback on the ledge still jumps; do not run to x=23.
    if gs in (9, 11):
        return ("A",), replace(track, phase="exit", held=held)
    if grounded and y <= _CERES_FALLING_DOOR_LEDGE_Y + 8 and (
        x <= _CERES_FALLING_DOOR_JUMP_X or pose in (137, 138)
    ):
        return ("LEFT", "B", "A"), replace(track, phase="exit", held=1)
    if not grounded:
        if x <= _CERES_FALLING_DOOR_TURN_X:
            return ("RIGHT", "B", "A"), replace(track, held=held)
        return ("LEFT", "B", "A"), replace(track, held=held)
    names = _tas_l_pump("LEFT", track.pump_i, state)
    return names, replace(track, pump_i=track.pump_i + 1)


def play_ceres_falling_to_elev(
    session: RouteSession, *, max_frames: int = 500
) -> None:
    """Falling Tile reverse. Raises if elev gs=8 misses."""
    if int(session.state.room_id) not in (ROOM_CERES_FALLING, ROOM_CERES_ELEVATOR):
        raise RuntimeError(f"expected Falling after magnet: {session.state}")
    track = CeresFallingEscapeTrack()
    for _ in range(max_frames):
        st = session.state
        if _ceres_falling_reached_elev(st):
            return
        names, track = ceres_falling_escape_action(st, track)
        session.step(
            buttons(*names) if names else idle_action(),
            f"ceres_falling_{track.phase}",
        )
        if track.phase == "done" or _ceres_falling_reached_elev(session.state):
            return
        if track.phase == "dboost" and track.held > 70:
            raise TimeoutError(
                f"falling d-boost stalled: {session.state} "
                f"contacted={track.contacted} boosted={track.boosted}"
            )
        if track.phase == "floor_hop" and track.held > 40:
            raise TimeoutError(f"falling missed y171 shelf: {session.state}")
    raise TimeoutError(
        f"falling missed elev after {max_frames}f: {session.state} "
        f"phase={track.phase} contacted={track.contacted} "
        f"boosted={track.boosted}"
    )


_POSE_WALL_LATCH_LEFT = 131
# 475→363: LEFT into the wall, RIGHT away; latch pose 131.
_CERES_475_TO_363_INTO = "LEFT"
_CERES_475_TO_363_AWAY = "RIGHT"
# Hard fail. TAS elev_to_landing is 2246f. Anything over this is a missed WJ.
CERES_ELEV_MAX_FRAMES = 2500
CERES_ELEV_BENCH_FRAMES = 2246


def _ceres_elev_ship_band(state) -> bool:
    """Grounded on ship pad (product leave ~x145 y75 pose 2/10 → gs 32)."""
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and int(state.game_state) == GS_ORDINARY
        and int(state.samus_y) <= _CERES_ELEV_SHIP_Y
        and abs(int(state.velocity_y)) <= 1
    )


def ship_pad_action(state) -> tuple[str, ...]:
    """Walk through the Ceres pad x that starts gs 32."""
    return walk_toward_x(int(state.samus_x), _CERES_ELEV_SHIP_X)


def _ceres_elev_leaving(state) -> bool:
    """True ship leave: left the elevator, or Ceres success / Zebes load.

    Inbound Falling→elev door (gs 9/11, often fake y≈139) is not leave.
    """
    if int(state.room_id) != ROOM_CERES_ELEVATOR:
        return True
    return int(state.game_state) in GS_CERES_LEAVE


def _ceres_elev_top_seat(state) -> bool:
    """s10 land / right-wall KB — the only shaft→ship handoff."""
    if int(state.room_id) != ROOM_CERES_ELEVATOR:
        return False
    if int(state.game_state) != GS_ORDINARY:
        return False
    if abs(int(state.samus_y) - _CERES_ELEV_TOP_Y) > 16:
        return False
    x = int(state.samus_x)
    if int(state.pose) in POSE_KNOCKBACK and x >= _CERES_ELEV_TOP_X - 30:
        return True
    return x >= _CERES_ELEV_TOP_X - 20


def _ceres_elev_entry_action(state) -> tuple[str, ...] | None:
    """Inputs until the shaft may start. ``None`` means the window is ready.

    Hold LEFT only while the Falling door is still settling. Ordinary y≈628
    spin is the TAS wall-jump phase. Walking LEFT there leaves x=216 and
    dumps the well. A floor remap (y≈651) is a missed wall jump.
    """
    if _ceres_elev_leaving(state):
        return None
    if int(state.room_id) != ROOM_CERES_ELEVATOR:
        return ()
    if int(state.game_state) != GS_ORDINARY:
        # Door-transition inputs are not movement frames. Preserve the
        # predecessor's pose-25 rise; the first ordinary frame owns the WJ.
        return ()
    if _ceres_fast_entry_window(state):
        return None
    if int(state.samus_y) >= _CERES_ELEV_BOTTOM_Y - 20:
        return None
    return ()


def _ceres_elev_budget(session: RouteSession, start: int) -> None:
    used = int(session.frame) - start
    if used > CERES_ELEV_MAX_FRAMES:
        raise TimeoutError(
            f"ceres elev_to_landing {used}f exceeded {CERES_ELEV_MAX_FRAMES}f: "
            f"{session.state}"
        )


def _ceres_reactive_elev_climb(session: RouteSession) -> None:
    """Elev after Falling → ship leave through one TAS wall-jump climb.

    Fast entry is x216 y624–641 spin. Missed fast-entry, 475, 363, 267, or
    171 plants raise. There is no checkpoint recover. Over 2500f raises.
    """
    start = int(session.frame)
    session.info["ceres_elev_start"] = start
    session.wait_until(
        lambda s: s.room_id == ROOM_CERES_ELEVATOR,
        timeout=300,
        reason="ceres_elev_door",
    )
    _ceres_elev_budget(session, start)
    for _ in range(160):
        names = _ceres_elev_entry_action(session.state)
        if names is None:
            break
        session.step(
            buttons(*names) if names else idle_action(),
            "ceres_elev_entry",
        )
        _ceres_elev_budget(session, start)
    if not _ceres_fast_entry_window(session.state):
        raise TimeoutError(f"ceres elev wall jump missed: {session.state}")
    overlay = _ceres_door_blocks_wj(session)
    if not _ceres_entry_to_475(session):
        raise TimeoutError(
            f"ceres 475 plant missed overlay={overlay}: {session.state}"
        )
    _ceres_elev_budget(session, start)
    if not _ceres_475_to_363(session):
        raise TimeoutError(f"ceres 363 plant missed: {session.state}")
    _ceres_elev_budget(session, start)
    _ceres_wj_plant(session, 267, "LEFT", "RIGHT")
    _ceres_elev_budget(session, start)
    _ceres_wj_plant(session, 171, "RIGHT", "LEFT")
    _ceres_elev_budget(session, start)
    _ceres_elev_top_to_ship(session)
    _ceres_elev_budget(session, start)


def _ceres_planted_at(state, target_y: int, *, slack: int = 18) -> bool:
    return (
        int(state.game_state) == GS_ORDINARY
        and abs(int(state.samus_y) - target_y) <= slack
        and abs(int(state.velocity_y)) <= 1
        and (
            int(state.pose) in STAND_LOCOMOTION_POSES
            or int(state.pose) in LAND_POSES
        )
    )


def _ceres_dumped_well(state) -> bool:
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and int(state.game_state) == GS_ORDINARY
        and int(state.samus_y) >= _CERES_ELEV_BOTTOM_Y - 20
        and abs(int(state.velocity_y)) <= 1
    )


def _ceres_fast_entry_window(state) -> bool:
    """Natural door-jump phase from which the y=475 wall jump is repeatable.

    y=651 floor remap is a missed wall jump. Do not widen this band.
    """
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and int(state.game_state) == GS_ORDINARY
        and 210 <= int(state.samus_x) <= 220
        and 624 <= int(state.samus_y) <= 641
        and int(state.pose) in SPIN_POSES
        and int(state.vertical_direction) == 1
        and int(state.velocity_y) > 0
        and int(state.momentum_x) >= 2
        and int(state.invincibility_timer) > 0
    )


def _ceres_any_wall_latch(state) -> bool:
    """Right-wall latch 132 or left-wall latch 131 (TAS 475→363 contact)."""
    return int(state.pose) in (POSE_WALL_LATCH, _POSE_WALL_LATCH_LEFT)


def _ceres_entry_to_475(session: RouteSession) -> bool:
    """Two measured wall jumps from the natural door phase to y=475."""
    if not _ceres_fast_entry_window(session.state):
        return False

    # Reach the lower edge of $E23F, release A for one frame, then wall-jump
    # at x≈208/y620. This predecessor is lower than Sniq's native entry, so
    # the first wall jump needs a 40-frame hold before the reverse.
    for names, frames in (
        (("LEFT", "A"), 1),
        (("A",), 1),
        (("LEFT", "A"), 1),
        (("A",), 3),
        (("LEFT",), 1),
    ):
        session.span(ActionSpan(names, frames, "ceres_elev_direct_entry"))

    first_latch = False
    for _ in range(40):
        session.step(buttons("LEFT", "A"), "ceres_elev_direct_first_wj")
        first_latch = first_latch or int(session.state.pose) == POSE_WALL_LATCH
    if not first_latch:
        return False

    # Unlatch at x≈165/y524, reverse for one frame, and catch the left wall.
    # The second jump clears the underside and plants x≈164/y475 naturally.
    tail = (
        (("RIGHT", "A"), 1),
        (("RIGHT",), 1),
        (("RIGHT", "A"), 1),
        (("LEFT", "A"), 7),
        (("DOWN", "RIGHT", "A"), 1),
        (("LEFT",), 1),
        ((), 1),
        (("LEFT",), 1),
        (("LEFT", "A"), 3),
    )
    second_latch = False
    for names, frames in tail:
        for _ in range(frames):
            session.step(
                buttons(*names) if names else idle_action(),
                "ceres_elev_direct_second_wj",
            )
            second_latch = second_latch or int(session.state.pose) == _POSE_WALL_LATCH_LEFT
            if _ceres_planted_at(session.state, 475, slack=4):
                return second_latch
            if _ceres_dumped_well(session.state):
                raise TimeoutError(
                    f"ceres elev dumped well at entry→475: {session.state}"
                )
    return False


def _ceres_475_to_363(session: RouteSession) -> bool:
    """Left-wall jump from the planted 475 seat onto y=363.

    TAS (lsnes sniq_100): plant x=163 pose 165/10, LEFT+A, pose 131 at
    y≈404, land x=156 y=363. Do not RIGHT-run from this seat.
    """
    if not _ceres_planted_at(session.state, 475, slack=4):
        return False
    spans = (
        ((_CERES_475_TO_363_INTO, "A"), 3),
        (("A",), 2),
        ((_CERES_475_TO_363_AWAY, "A"), 1),
        ((_CERES_475_TO_363_INTO, "A"), 1),
        ((_CERES_475_TO_363_AWAY, "A"), 1),
        (("A",), 9),
        ((_CERES_475_TO_363_AWAY, "A"), 1),
        (("A",), 1),
        ((_CERES_475_TO_363_INTO, _CERES_475_TO_363_AWAY), 1),
        ((_CERES_475_TO_363_INTO, _CERES_475_TO_363_AWAY, "A"), 1),
        (("A",), 5),
        (("A", "X"), 1),
        (("DOWN", "A"), 1),
        ((_CERES_475_TO_363_AWAY,), 1),
        ((), 1),
        ((_CERES_475_TO_363_AWAY,), 1),
    )
    latched = False
    for names, frames in spans:
        for _ in range(frames):
            session.step(
                buttons(*names) if names else idle_action(),
                "ceres_elev_475_363",
            )
            latched = latched or int(session.state.pose) == _POSE_WALL_LATCH_LEFT
            if _ceres_planted_at(session.state, 363, slack=4):
                return latched
            if _ceres_dumped_well(session.state):
                raise TimeoutError(f"ceres elev dumped well at 475→363: {session.state}")
    return False


def _ceres_wj_plant(
    session: RouteSession,
    target_y: int,
    into: str,
    away: str,
) -> None:
    """One precise wall jump onto ``target_y``. Miss or well dump raises."""
    if _ceres_elev_top_seat(session.state) or _ceres_elev_ship_band(session.state):
        return
    if _ceres_planted_at(session.state, target_y, slack=8):
        return
    if _ceres_dumped_well(session.state):
        raise TimeoutError(f"ceres elev dumped well before {target_y}: {session.state}")
    spans = (
        ((into, "A"), 3),
        (("A",), 2),
        ((away, "A"), 1),
        ((into, "A"), 1),
        ((away, "A"), 1),
        (("A",), 9),
        ((away, "A"), 1),
        (("A",), 1),
        ((into, away), 1),
        ((into, away, "A"), 1),
        (("A",), 5),
        (("A", "X"), 1),
        (("DOWN", "A"), 1),
        ((away,), 1),
        ((), 1),
        ((away,), 1),
    )
    latched = False
    for names, frames in spans:
        for _ in range(frames):
            session.step(
                buttons(*names) if names else idle_action(),
                f"ceres_elev_wj_{target_y}",
            )
            latched = latched or _ceres_any_wall_latch(session.state)
            if (
                _ceres_planted_at(session.state, target_y, slack=8)
                or _ceres_elev_top_seat(session.state)
                or _ceres_elev_ship_band(session.state)
                or _ceres_elev_leaving(session.state)
            ):
                if not latched and _ceres_planted_at(session.state, target_y, slack=8):
                    raise TimeoutError(
                        f"ceres {target_y} plant without wall latch: {session.state}"
                    )
                return
            if _ceres_dumped_well(session.state):
                raise TimeoutError(
                    f"ceres elev dumped well aiming {target_y}: {session.state}"
                )
    if not (
        _ceres_planted_at(session.state, target_y, slack=8)
        or _ceres_elev_top_seat(session.state)
        or _ceres_elev_ship_band(session.state)
        or _ceres_elev_leaving(session.state)
    ):
        raise TimeoutError(f"ceres {target_y} wall jump missed: {session.state}")


def _ceres_door_blocks_wj(session: RouteSession) -> bool:
    """True when a Ceres-door overlay occupies the right-wall WJ contact."""
    st = session.state
    sx, sy = int(st.samus_x), int(st.samus_y)
    for enemy in list_enemies(session):
        if int(enemy.enemy_id) != CERES_DOOR_ID:
            continue
        if enemy_overlaps(enemy, sx, sy, samus_r=16):
            return True
    return False


def _ceres_elev_top_to_ship(session: RouteSession) -> None:
    """From s10 land (~y171) force right-wall pose 137 then LEFT+A to ship pad.

    Product (open-loop s10 tail): land x211 y171 pose 9 → idle → pose 137 →
    LEFT+A 38 peaks ~y65 → LEFT 25 walks to x≈145 y75 → Ceres success (gs 32).
    Already on the pad still walks through ``_CERES_ELEV_SHIP_X`` until gs 32.
    """
    if session.state.room_id != ROOM_CERES_ELEVATOR:
        return
    if _ceres_elev_leaving(session.state):
        return

    if int(session.state.pose) in CROUCH_POSES:
        for _ in range(10):
            if int(session.state.pose) not in CROUCH_POSES:
                break
            session.step(buttons("UP"), "ceres_elev_uncrouch")

    # Pose 166 needs ~12f LEFT to ordinary locomotion. Shorter LEFT reaches
    # the wall with a weak boost that falls back to y267.
    if int(session.state.pose) in LAND_POSES:
        for _ in range(12):
            session.step(buttons("LEFT"), "ceres_elev_top_stand")

    for _ in range(4):
        if int(session.state.pose) in STAND_LOCOMOTION_POSES:
            break
        session.step(buttons("LEFT"), "ceres_elev_top_stand")

    if not _ceres_elev_ship_band(session.state) and int(session.state.samus_y) < 280:
        for _ in range(40):
            st = session.state
            if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
                return
            if is_knockback(st):
                break
            session.step(buttons("RIGHT"), "ceres_elev_top_seek_kb")

        for _ in range(3):
            st = session.state
            if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
                return
            if not is_knockback(st) and int(st.samus_x) >= _CERES_ELEV_TOP_X - 2:
                session.step(idle_action(), "ceres_elev_top_kb")
                continue
            if is_knockback(st):
                session.step(idle_action(), "ceres_elev_top_kb")
            else:
                break

        if is_knockback(session.state):
            for _ in range(38):
                st = session.state
                if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
                    return
                session.step(buttons("LEFT", "A"), "ceres_elev_top_boost")

            for _ in range(80):
                st = session.state
                if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
                    return
                session.step(buttons("LEFT"), "ceres_elev_top_walk")

    for _ in range(80):
        st = session.state
        if st.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(st):
            return
        names = ship_pad_action(st)
        session.step(
            buttons(*names) if names else idle_action(),
            "ceres_elev_ship",
        )

    if (
        session.state.room_id == ROOM_CERES_ELEVATOR
        and not _ceres_elev_leaving(session.state)
    ):
        raise TimeoutError(f"ceres elev ship leave failed: {session.state}")


__all__ = [
    "CERES_ELEV_BENCH_FRAMES",
    "CERES_ELEV_MAX_FRAMES",
    "CeresFallingEscapeTrack",
    "CeresMagnetEscapeTrack",
    "_ceres_475_to_363",
    "_ceres_any_wall_latch",
    "_ceres_door_blocks_wj",
    "_ceres_elev_leaving",
    "_ceres_elev_ship_band",
    "_ceres_elev_top_seat",
    "_ceres_elev_top_to_ship",
    "_ceres_falling_reached_elev",
    "_ceres_fast_entry_window",
    "_ceres_magnet_reached_falling",
    "_ceres_reactive_elev_climb",
    "ceres_falling_escape_action",
    "ceres_magnet_escape_action",
    "play_ceres_falling_to_elev",
    "play_ceres_magnet_to_falling",
    "ship_pad_action",
]
