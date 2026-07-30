"""Super Missile room collect and exit to farming / Big Pink."""

from __future__ import annotations

from super_metroid.ram import GameplayPhase, SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    require_room,
    select_weapon,
    unmorph,
    wait_until,
)
from super_metroid.routes.post_spore.rooms import (
    ROOM_BIG_PINK,
    ROOM_FARMING,
    ROOM_SUPER,
    SuperCollectEvidence,
)
from super_metroid.routes.runtime import ControllerSession

_hold = hold
_require_room = require_room
_select_weapon = select_weapon
_unmorph = unmorph

from super_metroid.routes.post_spore.big_pink_shaft import (
    play_big_pink_crest_pocket,
)

def play_super_room_collect(session: ControllerSession) -> SuperCollectEvidence:
    """From natural Super-room entry, descend and collect Super Missiles.

    Entry expectation: ordinary gameplay in ``0x9B5B``, ``max_super_missiles==0``.
    """
    entry_frame = session.frame
    state = session.state
    if state.room_id != ROOM_SUPER:
        raise RuntimeError(
            f"Super collect entry: room 0x{state.room_id:04X} != 0x{ROOM_SUPER:04X}"
        )
    if state.max_super_missiles > 0:
        raise RuntimeError("Super collect entry: supers already collected")

    # Walk to shaft entrance.
    for _ in range(80):
        state = _hold(session, 1, "RIGHT", "B", reason="super_shaft_approach")
        if state.samus_x >= 140:
            break

    # Crouch-shot floor blocks.
    for _ in range(15):
        _hold(session, 3, "DOWN", "X", reason="super_shaft_shot")
        _hold(session, 2, "DOWN", reason="super_shaft_shot")
    _hold(session, 10, "DOWN", reason="super_shaft_morph")
    for _ in range(8):
        _hold(session, 2, "A", reason="super_shaft_bomb")
        _hold(session, 30, reason="super_shaft_bomb_wait")

    # Jump right into the cleared shaft path.
    for _ in range(50):
        state = _hold(session, 2, "RIGHT", "A", "B", reason="super_shaft_jump")
        if state.samus_x > 250:
            break

    # Explore right/down until free-fall begins.
    for i in range(200):
        phase = i % 10
        if phase < 3:
            state = _hold(session, 4, "RIGHT", "B", "X", reason="super_shaft_explore")
        elif phase < 5:
            state = _hold(session, 4, "RIGHT", "A", "B", reason="super_shaft_explore")
        elif phase < 7:
            state = _hold(session, 3, "DOWN", "X", reason="super_shaft_explore")
        elif phase < 8:
            _hold(session, 8, "DOWN", reason="super_shaft_explore")
            _hold(session, 2, "A", reason="super_shaft_explore")
            state = _hold(session, 20, reason="super_shaft_explore")
        else:
            state = _hold(session, 4, "RIGHT", "B", reason="super_shaft_explore")
        if state.samus_y > 500:
            break

    for _ in range(800):
        state = _hold(session, 2, reason="super_shaft_fall")
        if state.samus_y > 2100:
            break
    if state.samus_y <= 2000:
        raise TimeoutError(f"Super shaft fall failed: {state}")

    # Approach Chozo Super and collect.
    collect_frame: int | None = None
    for i in range(400):
        if state.samus_x < 412:
            state = _hold(session, 2, "RIGHT", "B", reason="super_item_approach")
        elif state.samus_x > 428:
            state = _hold(session, 2, "LEFT", "B", reason="super_item_approach")
        else:
            state = _hold(session, 2, reason="super_item_approach")
        if i % 12 == 0:
            state = _hold(session, 4, "X", reason="super_item_shoot")
        if i % 40 == 20:
            state = _hold(session, 6, "A", reason="super_item_jump")
        if state.max_super_missiles > 0:
            collect_frame = session.frame
            break
    if collect_frame is None or state.max_super_missiles <= 0:
        raise TimeoutError(f"Super Missile PLM never collected: {state}")

    # Fanfare / control return.
    for i in range(300):
        state = _hold(session, 1, reason="super_item_fanfare")
        if (
            state.game_state == 8
            and state.phase is GameplayPhase.ORDINARY_GAMEPLAY
            and state.max_super_missiles > 0
            and i > 80
        ):
            break

    return SuperCollectEvidence(
        entry_frame=entry_frame,
        collect_frame=collect_frame,
        exit_frame=None,
        max_super_missiles=state.max_super_missiles,
        final_room_id=state.room_id,
        samus_x=state.samus_x,
        samus_y=state.samus_y,
    )




def play_super_room_to_farming(session: ControllerSession) -> SuperMetroidState:
    """After Super collect at bottom, bomb left gate and enter farming."""
    state = session.state
    _require_room(session, ROOM_SUPER, "to_farming")
    if state.max_super_missiles <= 0:
        raise RuntimeError("to_farming requires Super Missiles")

    # Prefer missiles for blue door shot; SELECT may be flaky mid-pose.
    try:
        _select_weapon(session, 1)
    except RuntimeError:
        pass

    for _ in range(80):
        state = _hold(session, 2, "LEFT", "B", reason="super_gate_approach")
        if state.samus_x <= 320:
            break

    _hold(session, 12, "DOWN", reason="super_gate_bomb")
    for _ in range(15):
        _hold(session, 2, "A", reason="super_gate_bomb")
        _hold(session, 35, reason="super_gate_bomb")
        state = _hold(session, 4, "LEFT", reason="super_gate_bomb")
        if state.samus_x < 200:
            break
    _unmorph(session)

    for _ in range(120):
        state = _hold(session, 2, "LEFT", "B", reason="super_door_approach")
        if state.samus_x < 50:
            break

    for _ in range(50):
        _hold(session, 3, "LEFT", "X", reason="super_door_shot")
        state = _hold(session, 5, "LEFT", "B", reason="super_door_enter")
        if state.room_id == ROOM_FARMING:
            break
    for _ in range(200):
        state = _hold(session, 1, reason="farming_settle")
        if (
            state.room_id == ROOM_FARMING
            and state.game_state == 8
            and state.phase is GameplayPhase.ORDINARY_GAMEPLAY
        ):
            break
    _require_room(session, ROOM_FARMING, "farming entry")
    return state




def play_farming_to_big_pink(session: ControllerSession) -> SuperMetroidState:
    """Cross farming left and Super the green door into Big Pink."""
    state = session.state
    _require_room(session, ROOM_FARMING, "farming_to_pink")
    try:
        _select_weapon(session, 2)
    except RuntimeError:
        pass
    _unmorph(session)

    for i in range(500):
        if session.state.pose in (39, 40, 137, 138):
            _unmorph(session)
        state = _hold(session, 3, "LEFT", "A", "B", reason="farming_cross")
        if i % 5 == 0:
            _hold(session, 2, "LEFT", "X", reason="farming_super")
        if i % 25 == 12:
            _hold(session, 8, "DOWN", reason="farming_bomb")
            _hold(session, 2, "A", reason="farming_bomb")
            _hold(session, 30, reason="farming_bomb")
            _unmorph(session)
        if state.room_id == ROOM_BIG_PINK:
            break
    for _ in range(150):
        state = _hold(session, 1, reason="big_pink_settle")
        if (
            state.room_id == ROOM_BIG_PINK
            and state.game_state == 8
            and state.phase is GameplayPhase.ORDINARY_GAMEPLAY
        ):
            break
    _require_room(session, ROOM_BIG_PINK, "big pink entry")
    return state




def play_post_spore_supers(
    session: ControllerSession,
    *,
    continue_to_farming: bool = True,
    continue_to_big_pink: bool = True,
    continue_to_crest: bool = False,
) -> SuperCollectEvidence:
    """Collect Supers and optionally reach Big Pink / pocket crest."""
    evidence = play_super_room_collect(session)
    exit_frame = None
    if continue_to_farming:
        play_super_room_to_farming(session)
        if continue_to_big_pink:
            play_farming_to_big_pink(session)
            if continue_to_crest:
                play_big_pink_crest_pocket(session)
        exit_frame = session.frame
        state = session.state
        return SuperCollectEvidence(
            entry_frame=evidence.entry_frame,
            collect_frame=evidence.collect_frame,
            exit_frame=exit_frame,
            max_super_missiles=state.max_super_missiles,
            final_room_id=state.room_id,
            samus_x=state.samus_x,
            samus_y=state.samus_y,
        )
    return evidence



