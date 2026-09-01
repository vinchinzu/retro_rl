"""Ceres Magnet Stairs escape + Falling Tile reverse.

Magnet: hop east steam, jump 347 at x≈74, air-turn RIGHT onto 267, run
RIGHT, jump 219 then 139. L-every-other while running. Wait on 267 when
the jet spritemap is idle.

Falling reverse: run off 139 onto 187, hop 347 onto 171, turn RIGHT at
x≤314 leftover LEFT mx, 1f p25, B-only fall, LEFT+B+A so pose 80 rides
LEFT. Do not hold A from x≈318 (ceiling). Do not add X (p47).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from retro_harness.actions import buttons, idle_action
from super_metroid.routes.kpdr.ceres.geometry import (
    CERES_FALLING_REV_FLOOR_HOP,
    CERES_MAGNET_HIGH_HOP,
    CERES_MAGNET_MID_ESCAPE_HOP,
    CERES_MAGNET_STEAM_HOP,
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
from super_metroid.routes.runtime import RouteSession


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
    """One-frame Magnet→Falling track. ROM-free tests drive this."""

    phase: CeresMagnetEscapePhase = "door"
    held: int = 0
    pump_i: int = 0
    contacted: bool = False
    steam_shown: bool = True


@dataclass(frozen=True)
class CeresFallingEscapeTrack:
    """One-frame Falling→elev track. ROM-free tests drive this."""

    phase: CeresFallingEscapePhase = "door"
    held: int = 0
    pump_i: int = 0
    contacted: bool = False
    boosted: bool = False


def _ceres_magnet_reached_falling(state) -> bool:
    return int(state.room_id) == ROOM_CERES_FALLING and int(state.game_state) == 8


def _ceres_falling_reached_elev(state) -> bool:
    return int(state.room_id) == ROOM_CERES_ELEVATOR and int(state.game_state) == 8


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
    """One-frame Magnet Stairs → Falling policy (ROM-free)."""
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
    from super_metroid.combat.enemies import list_enemies
    from super_metroid.combat.enemies.species import steam_jet_shown

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
    """One-frame Falling Tile → elev policy (ROM-free)."""
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

    # Door: TAS jumps x≈37 y=139 p26, air-turns p25 at (26,120).
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


__all__ = [
    "CeresFallingEscapeTrack",
    "CeresMagnetEscapeTrack",
    "_ceres_falling_reached_elev",
    "_ceres_magnet_reached_falling",
    "ceres_falling_escape_action",
    "ceres_magnet_escape_action",
    "play_ceres_falling_to_elev",
    "play_ceres_magnet_to_falling",
]
