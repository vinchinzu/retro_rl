"""Mission Impossible / Pink PB maze wall open and collect."""

from __future__ import annotations

from collections.abc import Callable

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    is_morph,
    require_room,
    select_weapon,
    unmorph,
    wait_until,
)
from super_metroid.routes.post_spore.morph_bomb_roll import bomb_roll_left_safe
from super_metroid.routes.post_spore.rooms import (
    ROOM_BIG_PINK,
    ROOM_FARMING,
    ROOM_PINK_PB,
    ROOM_SUPER,
    SuperCollectEvidence,
)
from super_metroid.routes.runtime import ControllerSession

_hold = hold
_require_room = require_room
_select_weapon = select_weapon
_unmorph = unmorph

def _double_tap_morph(session: ControllerSession) -> SuperMetroidState:
    """Deprecated alias for ``ensure_morph`` (pose-confirmed double-tap)."""
    return ensure_morph(session)



def play_pink_pb_break_maze_wall(
    session: ControllerSession,
    *,
    max_frames: int = 640,
) -> SuperMetroidState:
    """Open bottom-spawn maze wall at x≈437 with morph bombs.

    From natural bottom-door spawn ~(460, 395): walk left into the wall,
    ``ensure_morph``, y-safe bomb-roll past the wall (~x=405, upper band
    y≈395–410). Missiles/supers do **not** open this wall; crouch-bombs do
    not either — morph is required.
    """
    _require_room(session, ROOM_PINK_PB, "pink_pb_break_maze_wall")
    if session.state.max_power_bombs > 0:
        return session.state
    if session.state.samus_x <= 410 and session.state.samus_y <= 415:
        return session.state
    # Approach wall.
    for _ in range(40):
        if session.state.samus_x <= 440:
            break
        _hold(session, 1, "LEFT", reason="pb_maze_to_wall")
    ensure_morph(session)
    state = bomb_roll_left_safe(
        session,
        410,
        max_y=412,
        pit_y=430,
        max_frames=max_frames,
    )
    if state.max_power_bombs > 0:
        return state
    if state.samus_x > 420:
        raise TimeoutError(
            f"pink_pb_break_maze_wall: still blocked: {session.state}"
        )
    return state



def play_pink_pb_morph_bomb_collect(
    session: ControllerSession,
    *,
    max_frames: int = 400,
) -> SuperMetroidState:
    """Morph-bomb-roll left from collect pocket (x≤225, y≈395) to the PB PLM.

    Proven from place ``(220–225, 395)`` after bottom-door entry: ensure morph,
    bomb-roll left; capacity 0→5 near the item (~x=100–120, y≈370–400).
    Unmorph+walk fallback if we overshoot the PLM in ball form.

    Expects Pink PB room ``0x9E11`` already west of mid-maze barriers (not
    merely past wall@437 — that only reaches ~x=405). Use
    ``play_pink_pb_mid_maze_to_collect`` when starting near x≈405.
    """
    _require_room(session, ROOM_PINK_PB, "pink_pb_morph_collect")
    if session.state.max_power_bombs > 0:
        return session.state
    ensure_morph(session)
    state = bomb_roll_left_safe(
        session,
        100,
        max_y=415,
        pit_y=430,
        max_frames=max_frames,
    )
    if state.max_power_bombs > 0:
        return state
    # Unmorph + walk fallback near PLM (item may need standing contact).
    if state.samus_x < 160:
        _hold(session, 5, "UP", reason="pb_collect_unmorph")
        _hold(session, 10, reason="pb_collect_unmorph")
        for _ in range(100):
            state = _hold(session, 1, "LEFT", reason="pb_collect_walk")
            if state.max_power_bombs > 0:
                return state
        for _ in range(40):
            state = _hold(session, 1, "RIGHT", reason="pb_collect_walk_back")
            if state.max_power_bombs > 0:
                return state
    if session.state.max_power_bombs <= 0:
        raise TimeoutError(
            f"pink_pb_morph_collect: still 0 PB capacity: {session.state}"
        )
    return session.state



def play_pink_pb_from_left_zone(
    session: ControllerSession,
) -> SuperMetroidState:
    """Left free volume (~x≤220, y≈310–380) → drop into pocket → collect.

    Proven from place ``(180, 360)`` after wall@437 open: walk/fall into the
    collect band (y≳385, x≲220) then ``play_pink_pb_morph_bomb_collect``.
    Pure entry into this left volume from the bottom door is still open.
    """
    _require_room(session, ROOM_PINK_PB, "from_left_zone")
    if session.state.max_power_bombs > 0:
        return session.state
    # Drop / walk into pocket band if still elevated in left volume.
    if session.state.samus_y < 385 and session.state.samus_x <= 230:
        for i in range(80):
            d = "LEFT" if (i // 8) % 2 == 0 else "RIGHT"
            state = _hold(session, 1, d, reason="pb_leftzone_drop")
            if state.samus_y >= 385 or state.max_power_bombs > 0:
                break
            if state.samus_x > 230:
                _hold(session, 5, "LEFT", reason="pb_leftzone_back")
    if session.state.max_power_bombs > 0:
        return session.state
    return play_pink_pb_morph_bomb_collect(session)



def play_pink_pb_mid_maze_to_collect(
    session: ControllerSession,
    *,
    max_frames: int = 500,
    log_every: int = 0,
) -> SuperMetroidState:
    """After wall break (~408,398) → collect pocket without place (OPEN).

    Walkthrough / map notes (wiki.supermetroid.run “Mission Impossible Room”,
    room map ``PinkBrinstarPowerBombRoom.png``):

    - Room is two-tier: pink upper (top door, sidehoppers) + metal lower maze
      (bottom door, item). 100% often Quick-Drops a **crumble** from above.
    - After wall@437 open, continuous morph-roll sampling shows **no mid
      bridge** at band y: door-side ledge ~x=412 and left volume x≲228 are
      rollable; mid x≈230–400 is solid. Pit y≈455 is continuous x=90–420 but
      morph headroom ~2px (cannot unmorph/climb to item). Top corridor y≈171
      spans full x but is sealed from below (bombs do not open the floor).
    - **Working suffix:** once in left volume (~180,360), walk/fall into
      pocket and collect (``play_pink_pb_from_left_zone``).
    - **Still open:** pure door-side → left volume (or pure top → crumble).

    Tries y-safe bomb-roll (strict band-keeping + pit recovery toward the
    door ledge) then left-zone collect; times out with geometry notes if
    still east of the left volume or stuck in the pit.
    """
    _require_room(session, ROOM_PINK_PB, "mid_maze")
    if session.state.max_power_bombs > 0:
        return session.state
    # Already in left volume or pocket (not deep pit).
    if session.state.samus_x <= 230 and session.state.samus_y <= 420:
        if session.state.samus_y < 385 and session.state.samus_x <= 220:
            return play_pink_pb_from_left_zone(session)
        return play_pink_pb_morph_bomb_collect(session)
    ensure_morph(session)
    start_x = session.state.samus_x
    start_y = session.state.samus_y
    bomb_roll_left_safe(
        session,
        225,
        max_y=412,
        pit_y=420,
        max_frames=max_frames,
        elev_y=400,
        log_every=log_every,
        stall_frames=50,
    )
    s = session.state
    if s.max_power_bombs > 0:
        return s
    if s.samus_x <= 230 and s.samus_y <= 420:
        return play_pink_pb_from_left_zone(session)
    # Stuck in pit after transit: still not collectable (no climb-out).
    pit_note = ""
    if s.samus_y > 440:
        pit_note = (
            " deep-pit trap y≈457 (rollable under mid but ~2px headroom — "
            "no climb to item band y≈360–395);"
        )
    raise TimeoutError(
        f"pink_pb_mid_maze: no pure path yet "
        f"(start=({start_x},{start_y}) → x={s.samus_x} y={s.samus_y} "
        f"pose={s.pose});{pit_note} "
        f"mid solid at band — need door→left-volume or top→crumble "
        f"(see Mission Impossible Room)"
    )



