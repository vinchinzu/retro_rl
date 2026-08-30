"""Crateria Kihunter traversal to the Moat handoff."""

from __future__ import annotations

from pathlib import Path

from super_metroid.ram import SuperMetroidState
from super_metroid.routes.controller_common import require_room, wait_ordinary_room
from super_metroid.routes.rle import load_rle_json, play_script
from super_metroid.routes.runtime import ControllerSession

ROOM_KIHUNTER = 0x948C
ROOM_MOAT = 0x95FF

_DATA = Path(__file__).resolve().parents[1] / "data"
_KIHUNTER_RLE = load_rle_json(_DATA / "kihunter_to_moat_human_rle.json")


def play_kihunter_to_moat(session: ControllerSession) -> SuperMetroidState:
    """Traverse Kihunter from its natural elevator entry and settle in Moat."""
    require_room(session, ROOM_KIHUNTER, "kihunter_to_moat")
    play_script(
        session,
        _KIHUNTER_RLE,
        reason="kihunter_to_moat_body",
        room_id=ROOM_KIHUNTER,
        stop_when=lambda state: int(state.room_id) != ROOM_KIHUNTER,
    )
    if int(session.state.room_id) != ROOM_MOAT:
        raise TimeoutError(
            f"kihunter_to_moat: body did not reach Moat: {session.state}"
        )
    return wait_ordinary_room(
        session,
        ROOM_MOAT,
        settle_frames=260,
        label="kihunter_to_moat",
    )


__all__ = ["ROOM_KIHUNTER", "ROOM_MOAT", "play_kihunter_to_moat"]
