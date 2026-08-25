"""Wrecked Ship Main Shaft → Basement (rr-4btp).

Unpowered first visit toward Phantoon. Human s21 hop body from the Main
Shaft pin dual-greens: dash switchbacks, morph PB the floor pipes,
jump-down Super, hold DOWN through the hatch. Lands ordinary ``gs=8``.
"""

from __future__ import annotations

from pathlib import Path

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    is_morph,
    require_room,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.room_ids import ROOM_WS_BASEMENT, ROOM_WS_MAIN
from super_metroid.routes.rle import load_rle_json, play_script
from super_metroid.routes.runtime import ControllerSession

_DATA = Path(__file__).resolve().parents[1] / "data"
_WS_MAIN_RLE = load_rle_json(_DATA / "ws_main_to_basement_rle.json")

ROOM_WS_ATTIC = 0xCA52
ROOM_WS_SAVE = 0xCE8A
WEAPON_SUPER = 2

# Main Shaft descent (rr-4btp dumps from post_ws_entrance_to_main).
# Pin (1063,907) p9 is the entry ledge. Save is x≳1240 at this y — do not enter.
# Morph-only hole in the grated floor around x∈[1140,1176] drops onto stairs
# at y≳950. Ping-pong morph-roll reaches the node-3 platform y≳1650.
# Green floor hatch is below x∈[1135,1165]; shoot pipes, Super, drop.
WS_MAIN_SAVE_X = 1240
WS_MAIN_HOLE_Y = 950
WS_MAIN_BOTTOM_Y = 1650
WS_MAIN_HATCH_X_MIN = 1135
WS_MAIN_HATCH_X_MAX = 1165
WS_MAIN_ATTIC_Y = 850
_WS_MAIN_SETTLE = 200


def ws_main_basement_settled(state: SuperMetroidState) -> bool:
    """Ordinary Basement handoff: room ``0xCC6F`` gs=8 door_transition=0."""
    return (
        int(state.room_id) == ROOM_WS_BASEMENT
        and int(state.game_state) == 8
        and int(state.door_transition) == 0
    )


def at_ws_main_green_floor(state: SuperMetroidState) -> bool:
    """True over the green floor-hatch band of unpowered Main Shaft."""
    return (
        int(state.room_id) == ROOM_WS_MAIN
        and WS_MAIN_HATCH_X_MIN <= int(state.samus_x) <= WS_MAIN_HATCH_X_MAX
        and int(state.samus_y) >= WS_MAIN_BOTTOM_Y
    )


def ws_main_to_basement_action(state: SuperMetroidState) -> tuple[str, ...]:
    """One-frame buttons. Never UP (attic). Super the green floor door, not save."""
    room = int(state.room_id)
    if room == ROOM_WS_BASEMENT:
        return ()
    if room != ROOM_WS_MAIN:
        return ()
    y = int(state.samus_y)
    x = int(state.samus_x)
    if y < WS_MAIN_ATTIC_Y:
        return ("DOWN",)
    if y < WS_MAIN_HOLE_Y:
        if not is_morph(int(state.pose)):
            return ("DOWN",)
        return ("RIGHT",)
    if y < WS_MAIN_BOTTOM_Y:
        return ("DOWN", "RIGHT") if int(state.samus_x) < 1150 else ("DOWN", "LEFT")
    if not at_ws_main_green_floor(state):
        return ("RIGHT",) if x < WS_MAIN_HATCH_X_MIN else ("LEFT",)
    if int(state.selected_item) != WEAPON_SUPER:
        return ("SELECT",)
    # Shoulder angle-down Super — DOWN+X from standing double-tap morphs.
    return ("L", "X")


def play_ws_main_to_basement(session: ControllerSession) -> SuperMetroidState:
    """Unpowered first visit toward Phantoon. Descend stairs. Coverns/Sbugs only.

    Pin is mid-height on the right of a wide room (x=1063; save is across —
    do NOT enter the save). Descend the stairs (do NOT go UP = Attic 0xCA52).
    Ignore grey locked doors. Skip the optional left-wall missile. At the
    bottom: shoot the floor pipes (aim down), morph, Super the green floor
    hatch, drop into Basement ``0xCC6F``. Unpowered Atomics stay in glass —
    ignore. Energy assist on — tank.

    https://wiki.supermetroid.run/Wrecked_Ship_Main_Shaft

    Human s21 hop body (1091f) from this pin dual-greens: dash switchbacks,
    morph PB the floor pipes, jump-down Super, hold DOWN through the hatch.
    Lands ordinary ``gs=8`` (game state 11 can last 50–100+f).
    """
    label = "ws_main_to_basement"
    require_room(session, ROOM_WS_MAIN, label)
    if ws_main_basement_settled(session.state):
        return session.state
    play_script(
        session,
        _WS_MAIN_RLE,
        reason=f"{label}_body",
        room_id=ROOM_WS_MAIN,
        stop_when=lambda state: int(state.room_id) != ROOM_WS_MAIN,
    )
    return wait_ordinary_room(
        session, ROOM_WS_BASEMENT, settle_frames=_WS_MAIN_SETTLE, label=label
    )


__all__ = [
    "ROOM_WS_ATTIC",
    "ROOM_WS_BASEMENT",
    "ROOM_WS_MAIN",
    "ROOM_WS_SAVE",
    "WEAPON_SUPER",
    "WS_MAIN_BOTTOM_Y",
    "WS_MAIN_HATCH_X_MAX",
    "WS_MAIN_HATCH_X_MIN",
    "WS_MAIN_HOLE_Y",
    "WS_MAIN_SAVE_X",
    "at_ws_main_green_floor",
    "play_ws_main_to_basement",
    "ws_main_basement_settled",
    "ws_main_to_basement_action",
]
