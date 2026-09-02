"""Ceres reverse: Magnet Stairs, Falling Tile, Elevator to ship.

Magnet: hop east steam, jump 347 at x≈74, air-turn RIGHT onto 267, run
RIGHT, jump 219 then 139. L-every-other while running. Wait on 267 when
the jet spritemap is idle.

Falling reverse: run off 139 onto 187, hop 347 onto 171, turn RIGHT at
x≤314 leftover LEFT mx, 1f p25, B-only fall, LEFT+B+A so pose 80 rides
LEFT. Door: TAS jumps x≈37 y=139, air-turns pose 25 at x≲28.

Elevator: one TAS wall-jump climb. Fast entry is x216 y624–641 spin.
Missed wall jump is a hard fail. No checkpoint recovery. Shaft over
2500f is a hang-cap fail, not a TAS pass.
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
    CERES_FALLING_DOOR_HOP,
    CERES_FALLING_REV_FLOOR_HOP,
    CERES_MAGNET_HIGH_HOP,
    CERES_MAGNET_MID_ESCAPE_HOP,
    CERES_MAGNET_STEAM_HOP,
    _CERES_ELEV_171_LAUNCH_X,
    _CERES_ELEV_267_LAUNCH_X,
    _CERES_ELEV_363_LAUNCH_X,
    _CERES_ELEV_475_LAUNCH_X,
    _CERES_ELEV_BOTTOM_Y,
    _CERES_ELEV_ENTRY_RISE_FRAMES,
    _CERES_ELEV_WJ_KICK_FRAMES,
    _CERES_ELEV_WJ_RELEASE_FRAMES,
    _CERES_ELEV_WJ_RIDE_FRAMES,
    _CERES_ELEV_SHIP_X,
    _CERES_ELEV_SHIP_Y,
    _CERES_ELEV_TOP_Y,
    _CERES_FALLING_DOOR_LEDGE_Y,
    _CERES_FALLING_DOOR_SHUTTER_FRAMES,
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
from super_metroid.routes.skills.geometry import (
    CROUCH_POSES,
    LAND_POSES,
    SPIN_POSES,
    STAND_LOCOMOTION_POSES,
)
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
    """TAS dest gs=8: (216, 632) pose 25 mx=2 vy=+4 inv=36.

    Same predicate as the elevator fast-entry window. y=651 / mx=0 / inv=0
    is a missed door jump, not a leave.
    """
    return _ceres_fast_entry_window(state)


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
        # gs=8 without the TAS window is a miss. Do not hold A into y=651.
        if gs == 8:
            return (), replace(track, phase="exit")
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

    # Door: crouch out the $E23F shutter east of x=45, then run LEFT and
    # jump at x<=33 so the leave is the 4th air frame — pose 26 rising vy=4
    # at (19, 121), elev (216, 633) against TAS (216, 632). Walking the shut
    # door is pose-138 knockback (momentum 0, y=108 ceiling); the old x≈50
    # takeoff flew straight into it. Do not jump standing or on knockback.
    door = CERES_FALLING_DOOR_HOP
    if gs in (9, 11):
        return ("A",), replace(track, phase="exit", held=held)
    if grounded and y <= _CERES_FALLING_DOOR_LEDGE_Y + 8:
        if track.held < _CERES_FALLING_DOOR_SHUTTER_FRAMES:
            return ("DOWN",), replace(track, held=held)
        if (
            door.ready(state)
            and int(state.invincibility_timer) > 0
            and pose not in (137, 138)
        ):
            return ("LEFT", "B", "A"), replace(track, phase="exit", held=held)
        names = _tas_l_pump("LEFT", track.pump_i, state)
        return names, replace(track, pump_i=track.pump_i + 1, held=held)
    if not grounded:
        # Height is the only lever left on the band: momentum_x halves once
        # on the second air frame, so an air-turn only costs rise.
        return ("LEFT", "B", "A"), replace(track, held=held)
    names = _tas_l_pump("LEFT", track.pump_i, state)
    return names, replace(track, pump_i=track.pump_i + 1, held=held)


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
        if int(st.room_id) == ROOM_CERES_ELEVATOR and int(st.game_state) == 8:
            raise TimeoutError(
                f"falling leave missed TAS elev window: {st} "
                f"phase={track.phase} contacted={track.contacted} "
                f"boosted={track.boosted}"
            )
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


# Hang cap. TAS elev_to_landing is CERES_ELEV_BENCH_FRAMES. Do not grow this
# for checkpoint recover (that run was 3349f).
CERES_ELEV_BENCH_FRAMES = 2246
CERES_ELEV_MAX_FRAMES = 2500


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
    """Elev after Falling → ship leave: one wall jump, then three ledge hops.

    Fast entry is x216 y624–641 spin. Missed fast-entry, 475, 363, 267, or
    171 plants raise. There is no checkpoint recover. Over 2500f raises.
    """
    start = int(session.frame)
    # Local ``start`` is the live cap; the info key is not. RouteSession.step
    # overwrites ``session.info`` with the env step info every frame, so the
    # elev_to_landing guard in play_ceres_escape_to_landing reads None and
    # never fires. Left as the planner's call (see docs/plan.md).
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
    for target_y, launch_x, side in (
        (363, _CERES_ELEV_475_LAUNCH_X, "RIGHT"),
        (267, _CERES_ELEV_363_LAUNCH_X, "LEFT"),
        (_CERES_ELEV_TOP_Y, _CERES_ELEV_267_LAUNCH_X, "LEFT"),
    ):
        if not _ceres_ledge_hop(session, target_y, launch_x, side):
            raise TimeoutError(f"ceres {target_y} plant missed: {session.state}")
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
    Falling leave uses this same predicate.

    ``momentum_x >= 1`` is the measured floor, not a relaxation of a >= 2 that
    ever held: ground momentum caps at 2.75 and halves once on the second
    airborne frame, and the y band needs air frame 3, so no takeoff out of the
    Falling door can carry 2 into this window.
    """
    return (
        int(state.room_id) == ROOM_CERES_ELEVATOR
        and int(state.game_state) == GS_ORDINARY
        and 210 <= int(state.samus_x) <= 220
        and 624 <= int(state.samus_y) <= 641
        and int(state.pose) in SPIN_POSES
        and int(state.vertical_direction) == 1
        and int(state.velocity_y) > 0
        and int(state.momentum_x) >= 1
        and int(state.invincibility_timer) > 0
    )


def _ceres_elev_grounded(state) -> bool:
    """Standing or walking on a shaft ledge (movement types 0/1)."""
    return int(state.movement_type) in (0, 1) and abs(int(state.velocity_y)) <= 1


def _ceres_elev_walk_to(
    session: RouteSession, target_x: int, *, limit: int = 90
) -> None:
    """Walk the current ledge onto a measured launch x."""
    for _ in range(limit):
        names = walk_toward_x(int(session.state.samus_x), target_x, slack=1)
        if not names:
            return
        session.step(buttons(*names), f"ceres_elev_walk_{target_x}")


def _ceres_entry_to_475(session: RouteSession) -> bool:
    """One precise wall jump off the shaft right wall onto the y=475 ledge.

    The door leave arrives four air frames into its spin jump, so the entry
    rise alone tops out at y=608 and no amount of drift reaches 475. Riding
    RIGHT+A up the x=211 wall to y≈571 and kicking off it does: release A for
    two frames pressing LEFT (one frame reads as a jump cut and never
    latches), then LEFT+A for the pose-132 kick, then hold A while movement
    type 20 carries the kick to y=474.
    """
    if not _ceres_fast_entry_window(session.state):
        return False
    for _ in range(_CERES_ELEV_ENTRY_RISE_FRAMES):
        session.step(buttons("RIGHT", "A"), "ceres_elev_entry_rise")
    for _ in range(_CERES_ELEV_WJ_RELEASE_FRAMES):
        session.step(buttons("LEFT"), "ceres_elev_entry_release")
    latched = False
    for _ in range(_CERES_ELEV_WJ_KICK_FRAMES):
        session.step(buttons("LEFT", "A"), "ceres_elev_entry_kick")
        latched = latched or int(session.state.pose) == POSE_WALL_LATCH
    for _ in range(_CERES_ELEV_WJ_RIDE_FRAMES):
        session.step(buttons("A"), "ceres_elev_entry_ride")
        latched = latched or int(session.state.pose) == POSE_WALL_LATCH
    if not latched:
        return False
    for _ in range(40):
        session.step(idle_action(), "ceres_elev_entry_land")
        if _ceres_dumped_well(session.state):
            raise TimeoutError(
                f"ceres elev dumped well at entry→475: {session.state}"
            )
        if _ceres_planted_at(session.state, 475, slack=4):
            return True
    return False


def _ceres_ledge_hop(
    session: RouteSession,
    target_y: int,
    launch_x: int,
    side: str,
    *,
    limit: int = 120,
) -> bool:
    """Walk a shaft ledge to ``launch_x`` and spin-jump onto ``target_y``.

    Above y=475 the rungs are ground spin jumps, not wall jumps: a full ground
    spin jump rises 111px against gaps of 112/96/96. What has to be right is
    the launch x — off its band the jump clips a ledge lip and drops back down
    the shaft, which raises here rather than remapping to a lower floor.
    """
    _ceres_elev_walk_to(session, launch_x)
    for _ in range(2):
        session.step(buttons(side), f"ceres_elev_turn_{target_y}")
    airborne = False
    for _ in range(limit):
        session.step(buttons(side, "A"), f"ceres_elev_hop_{target_y}")
        state = session.state
        if _ceres_elev_leaving(state):
            return True
        if _ceres_dumped_well(state):
            raise TimeoutError(
                f"ceres elev dumped well aiming {target_y}: {state}"
            )
        grounded = _ceres_elev_grounded(state)
        airborne = airborne or not grounded
        if airborne and grounded:
            return abs(int(state.samus_y) - target_y) <= 4
    return False


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
    """From the y=171 seat: walk into the left wall, spin-jump onto the pad.

    The climb lands 171 on its west end (x≈66), not the s10 east seat the old
    right-wall knockback boost started from. From the x=45 wall a plain RIGHT
    spin jump peaks at y=60 and drops straight onto the ship pad, which is
    where Ceres success (game state 32) fires. ``ship_pad_action`` is the tail
    for a landing that is already on the pad but short of the trigger x.
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

    if not _ceres_elev_ship_band(session.state):
        _ceres_elev_walk_to(session, _CERES_ELEV_171_LAUNCH_X)
        for _ in range(2):
            session.step(buttons("RIGHT"), "ceres_elev_ship_turn")
        for _ in range(120):
            state = session.state
            if state.room_id != ROOM_CERES_ELEVATOR or _ceres_elev_leaving(state):
                return
            if _ceres_elev_ship_band(state):
                break
            session.step(buttons("RIGHT", "A"), "ceres_elev_ship_hop")

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
    "_ceres_ledge_hop",
    "_ceres_door_blocks_wj",
    "_ceres_elev_leaving",
    "_ceres_elev_ship_band",
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
