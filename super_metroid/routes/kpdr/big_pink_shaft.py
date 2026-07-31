"""Big Pink crest, Super-block clear, morph tunnel, and main shaft."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    ensure_morph,
    hold,
    require_room,
    select_weapon,
    unmorph,
    wait_until,
)
from super_metroid.routes.kpdr.rooms import (
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

def play_big_pink_crest_pocket(session: ControllerSession) -> SuperMetroidState:
    """Crest the farm-pocket lip at x≈1157 (run-right, spin-jump left).

    Expects ordinary Big Pink near the farm door (~x≥1180). On success Samus
    is roughly at **(1125, 1387)** standing on the raised platform (same floor
    as morph y≈1401 — pose height). Double-tap DOWN morphs into the y87 tunnel.
    Raises ``TimeoutError`` if the lip is not crossed.
    """
    state = session.state
    _require_room(session, ROOM_BIG_PINK, "crest_pocket")
    _unmorph(session)
    try:
        _select_weapon(session, 2)
    except RuntimeError:
        pass

    # Walk into the pocket wall, then reverse for run speed.
    for _ in range(40):
        state = _hold(session, 1, "LEFT", "B", reason="big_pink_pocket_approach")
        if state.samus_x <= 1160:
            break
    _hold(session, 25, "RIGHT", "B", reason="big_pink_pocket_runup")

    for _ in range(40):
        state = _hold(session, 12, "LEFT", "A", "B", reason="big_pink_pocket_crest")
        state = _hold(session, 6, "LEFT", "B", reason="big_pink_pocket_crest")
        if state.samus_x <= 1135:
            break
    if state.samus_x > 1135:
        raise TimeoutError(
            f"Big Pink pocket crest failed (still x={state.samus_x}, y={state.samus_y})"
        )
    _hold(session, 12, reason="big_pink_pocket_crest_settle")
    return session.state




def play_big_pink_clear_super_block(session: ControllerSession) -> SuperMetroidState:
    """Clear the permanent Super-only shot block at tile (69, 87) from crest.

    The second barrier is level type ``0xC`` / BTS ``0x0B`` (Super Missile only,
    permanent). Standing/angle supers from the wall-top often miss; crouch +
    Super left from the crest ledge clears it in one shot.

    Expects Big Pink after ``play_big_pink_crest_pocket`` (~x≤1135, y≈1387).
    Does **not** by itself reach the open main shaft — Samus must still enter
    the raised morph tunnel (~y=1401) and clear bomb blocks (62–63, 87).
    """
    state = session.state
    _require_room(session, ROOM_BIG_PINK, "clear_super_block")
    # Unspin from crest land (pose often 0x8A ran-into-wall).
    _hold(session, 4, "A", reason="big_pink_unspin")
    _hold(session, 40, reason="big_pink_unspin_settle")
    try:
        _select_weapon(session, 2)
    except RuntimeError:
        pass
    _hold(session, 15, "DOWN", reason="big_pink_crouch_super")
    for _ in range(8):
        _hold(session, 3, "LEFT", "X", reason="big_pink_crouch_super")
        state = _hold(session, 18, "LEFT", "DOWN", reason="big_pink_crouch_super")
    _hold(session, 10, reason="big_pink_super_block_settle")
    return session.state




def play_big_pink_morph_to_tunnel(session: ControllerSession) -> SuperMetroidState:
    """Double-tap DOWN to morph into the y87 tunnel on the raised platform.

    Standing center is ~y=1387 on the same floor as morph ~y=1401 (pose height).
    Holding DOWN only crouches (pose 40) and cannot enter the 1-tile tunnel;
    a short tap–release–tap completes morph ball (pose 65) so Samus fits under
    the y86 ceiling and can roll west.

    Expects Big Pink after Super block clear, roughly x≤1140 on the platform.
    Raises ``TimeoutError`` if morph tunnel height is not reached.
    """
    _require_room(session, ROOM_BIG_PINK, "morph_to_tunnel")
    # Leave crouch/super pose if still holding down from the Super clear.
    _hold(session, 12, reason="big_pink_morph_stand")
    # Double-tap DOWN: crouch then morph (hold-DOWN alone stays crouch).
    _hold(session, 1, "DOWN", reason="big_pink_morph_tap1")
    _hold(session, 4, reason="big_pink_morph_gap")
    _hold(session, 1, "DOWN", reason="big_pink_morph_tap2")
    state = _hold(session, 18, "DOWN", reason="big_pink_morph_hold")
    on_tunnel = 1395 <= state.samus_y <= 1410 and state.samus_x <= 1155
    if not on_tunnel:
        # One retry with a slightly longer second tap.
        _hold(session, 8, reason="big_pink_morph_retry_stand")
        _hold(session, 2, "DOWN", reason="big_pink_morph_retry_tap1")
        _hold(session, 4, reason="big_pink_morph_retry_gap")
        _hold(session, 8, "DOWN", reason="big_pink_morph_retry_tap2")
        state = _hold(session, 20, "DOWN", reason="big_pink_morph_retry_hold")
        on_tunnel = 1395 <= state.samus_y <= 1410 and state.samus_x <= 1155
    if not on_tunnel:
        raise TimeoutError(
            "Big Pink morph-to-tunnel failed: "
            f"({state.samus_x}, {state.samus_y}) pose={state.pose}; "
            "expected morph on raised floor y≈1401 after double-tap DOWN"
        )
    return session.state




def play_big_pink_tunnel_west(
    session: ControllerSession,
    *,
    target_x: int = 750,
    max_frames: int = 400,
) -> SuperMetroidState:
    """Morph-roll west through the y87 tunnel into the open main shaft.

    Expects Big Pink on the **raised tunnel floor** (~y 1395–1410, x≲1140) with
    the Super shot block at (69, 87) already cleared. Sequence:

    1. Morph and roll left (opens scroll PLM at (64, 87) → screen (3,5)).
    2. Lay morph bombs with **X** (not A) on permanent bomb blocks (62–63, 87).
    3. Continue west into open main-shaft volume (default x≤``target_x``).

    Proven from crest after Super clear + ``play_big_pink_morph_to_tunnel``.
    Raises ``TimeoutError`` if the target is not reached.
    """
    state = session.state
    _require_room(session, ROOM_BIG_PINK, "tunnel_west")
    try:
        _select_weapon(session, 0)
    except RuntimeError:
        pass
    # Already morphed after morph_to_tunnel; short DOWN keeps ball if needed.
    _hold(session, 8, "DOWN", reason="big_pink_tunnel_morph")

    for i in range(max_frames):
        state = _hold(session, 1, "LEFT", "B", reason="big_pink_tunnel_roll")
        if i % 18 == 5:
            # Morph bombs: X (A is unreliable here against these BTS-4 blocks).
            _hold(session, 2, "X", reason="big_pink_tunnel_bomb")
            _hold(session, 50, reason="big_pink_tunnel_bomb_wait")
        if state.room_id != ROOM_BIG_PINK:
            raise RuntimeError(
                f"tunnel west left Big Pink at frame {session.frame}: {state}"
            )
        if state.samus_x <= target_x:
            _hold(session, 10, reason="big_pink_tunnel_west_settle")
            return session.state

    raise TimeoutError(
        f"Big Pink tunnel west failed: x={state.samus_x}, y={state.samus_y} "
        f"(target x≤{target_x})"
    )




def play_big_pink_drop_to_pocket(session: ControllerSession) -> SuperMetroidState:
    """From crest wall-top, walk off east into the deep farm pocket floor.

    Lands roughly x≥1157, y≈1419. Used before bomb-jump attempts onto the
    raised tunnel ledge.
    """
    _require_room(session, ROOM_BIG_PINK, "drop_to_pocket")
    _hold(session, 10, reason="big_pink_pocket_drop_settle")
    _hold(session, 4, "A", reason="big_pink_pocket_drop_unspin")
    _hold(session, 25, reason="big_pink_pocket_drop_unspin")
    _hold(session, 40, "RIGHT", "B", reason="big_pink_pocket_drop")
    _hold(session, 50, reason="big_pink_pocket_drop_land")
    _unmorph(session)
    return session.state




def play_big_pink_bomb_to_walkway_edge(
    session: ControllerSession,
    *,
    fuse_frames: int = 15,
    jump_frames: int = 14,
) -> SuperMetroidState:
    """From deep pocket lip, morph-bomb jump onto the floating walkway edge.

    Lands near **(1151, 1387)** — the east edge of the wall-top walkway above
    the raised tunnel floor. Does **not** yet drop the last ~9px onto tunnel
    floor y≈1401 (that hop remains open: morph-right falls past x=1152 into
    the deep pocket).

    Expects Super block already cleared and Samus near the lip (~x 1160–1210).
    """
    state = session.state
    _require_room(session, ROOM_BIG_PINK, "bomb_to_edge")
    # Approach lip if still east.
    for _ in range(60):
        if state.samus_x <= 1168:
            break
        state = _hold(session, 1, "LEFT", "B", reason="big_pink_edge_approach")
    try:
        _select_weapon(session, 0)
    except RuntimeError:
        pass
    _hold(session, 12, "DOWN", reason="big_pink_edge_morph")
    _hold(session, 3, "LEFT", reason="big_pink_edge_press")
    _hold(session, 2, "X", reason="big_pink_edge_bomb")
    _hold(session, fuse_frames, "LEFT", reason="big_pink_edge_fuse")
    _hold(session, jump_frames, "LEFT", "A", reason="big_pink_edge_boost")
    # Idle fall/land on walkway edge (no left — left pulls back to crest).
    for _ in range(50):
        state = _hold(session, 1, reason="big_pink_edge_land")
    _hold(session, 10, reason="big_pink_edge_settle")
    return session.state




def play_big_pink_into_main_shaft(
    session: ControllerSession,
    *,
    target_x: int = 750,
) -> SuperMetroidState:
    """Crest → Super clear → double-tap morph → tunnel west into main shaft.

    Fully controller (no place/WRAM). Standing y≈1387 and morph y≈1401 are the
    same raised floor (pose height); the former “hop” was a morph-input issue.
    """
    play_big_pink_crest_pocket(session)
    play_big_pink_clear_super_block(session)
    play_big_pink_morph_to_tunnel(session)
    return play_big_pink_tunnel_west(session, target_x=target_x)




