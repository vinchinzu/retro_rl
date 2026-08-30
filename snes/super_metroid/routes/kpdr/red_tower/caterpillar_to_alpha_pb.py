"""Spazer-safe Caterpillar descent to the Alpha Power Bomb room (K5 hop 14)."""

from __future__ import annotations

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import (
    hold,
    require_room,
    select_weapon,
    wait_ordinary_room,
)
from super_metroid.routes.kpdr.alpha_pb import play_alpha_pb_collect
from super_metroid.routes.runtime import ControllerSession

ROOM_CATERPILLAR = 0xA322
ROOM_ALPHA_PB = 0xA3AE

_DESCENT_BUDGET = 1800
_ENTRY_SHAFT_MIN_X = 78
_ENTRY_SHAFT_MAX_X = 100
_ENTRY_FLOOR_Y = 1405


def _entry_shelf_dir(x: int) -> str | None:
    """Walk back to the first shaft. x>100 is the right ledge after a Cacatac hit."""
    if int(x) > _ENTRY_SHAFT_MAX_X:
        return "LEFT"
    if int(x) < _ENTRY_SHAFT_MIN_X:
        return "RIGHT"
    return None


def _downshot(session: ControllerSession, frame: int, *horizontal: str) -> None:
    """Aim down continuously and multi-tap beam for Spazer shot blocks."""
    buttons = [*horizontal, "DOWN"]
    if frame % 3 == 0:
        buttons.append("X")
    hold(session, 1, *buttons, reason="caterpillar_spazer_downshot")


def play_caterpillar_to_alpha_pb(session: ControllerSession) -> SuperMetroidState:
    """Descend Caterpillar reactively and leave through its bottom-left door."""
    require_room(session, ROOM_CATERPILLAR, "caterpillar_to_alpha_pb")
    select_weapon(session, 0)
    bottom_frames = 0

    for frame in range(_DESCENT_BUDGET):
        state = session.state
        if int(state.room_id) == ROOM_ALPHA_PB:
            wait_ordinary_room(
                session,
                ROOM_ALPHA_PB,
                settle_frames=260,
                label="caterpillar_to_alpha_pb",
            )
            return play_alpha_pb_collect(session)
        if int(state.room_id) != ROOM_CATERPILLAR:
            raise RuntimeError(
                "caterpillar_to_alpha_pb: unexpected room "
                f"0x{int(state.room_id):04X}"
            )

        x = int(state.samus_x)
        y = int(state.samus_y)

        # Entry shelf: run to the first shaft and jump in. Recenter only
        # when standing on the right ledge — not mid-jump (x can pass 100).
        if y < 1490:
            grounded = (
                int(state.velocity_y) == 0 and int(state.vertical_direction) == 0
            )
            if grounded and _entry_shelf_dir(x) == "LEFT":
                hold(session, 1, "LEFT", reason="caterpillar_entry_recenter")
            elif x < _ENTRY_SHAFT_MIN_X:
                hold(session, 1, "RIGHT", reason="caterpillar_entry_runup")
            elif y >= _ENTRY_FLOOR_Y and (
                grounded or int(state.vertical_direction) == 1
            ):
                hold(session, 1, "A", reason="caterpillar_entry_jump")
            else:
                _downshot(session, frame)
            continue

        # The narrow shelves alternate sides. Recenter before each down-shot
        # so Spazer's side pellets cannot leave the block directly below intact.
        if y < 1585:
            horizontal = ("LEFT",) if x > 70 else ()
            _downshot(session, frame, *horizontal)
            continue
        if y < 1810:
            if y >= 1660 and int(state.velocity_y) == 0:
                hold(session, 1, "LEFT", reason="caterpillar_third_shelf_left")
                continue
            horizontal = ("RIGHT",) if x < 78 else (("LEFT",) if x > 100 else ())
            _downshot(session, frame, *horizontal)
            continue
        if y < 1910:
            horizontal = ("LEFT",) if x > 72 else ()
            _downshot(session, frame, *horizontal)
            continue

        # Open the blue door from a stable stand, give its animation time to
        # clear, then walk through. Holding a charged beam while walking can
        # pin Samus in the firing pose short of the threshold.
        if y < 1928 or int(state.velocity_y) != 0:
            hold(session, 1, reason="caterpillar_bottom_land")
            continue
        if bottom_frames == 0:
            select_weapon(session, 2)
            hold(session, 1, reason="caterpillar_bottom_super_selected")
        elif bottom_frames < 3:
            hold(session, 1, "LEFT", reason="caterpillar_bottom_door_face")
        elif bottom_frames < 8:
            hold(session, 1, reason="caterpillar_bottom_door_release")
        elif bottom_frames == 8:
            hold(session, 1, "X", reason="caterpillar_bottom_door_shot")
        elif bottom_frames < 78:
            hold(session, 1, reason="caterpillar_bottom_door_open")
        else:
            hold(session, 1, "LEFT", reason="caterpillar_bottom_left_exit")
        bottom_frames += 1

    state = session.state
    raise TimeoutError(
        "caterpillar_to_alpha_pb: descent timeout: "
        f"room=0x{int(state.room_id):04X} pose={state.pose} "
        f"xy=({state.samus_x},{state.samus_y}) beams=0x{state.collected_beams:04X}"
    )


__all__ = [
    "ROOM_ALPHA_PB",
    "ROOM_CATERPILLAR",
    "_entry_shelf_dir",
    "play_caterpillar_to_alpha_pb",
]
