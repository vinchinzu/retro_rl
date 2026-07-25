"""Minimal ALTTP GameSpec seam: title → controllable castle grounds."""

from __future__ import annotations

from alttp.paths import FIRST_ACTION_STATE, GAME_SPEC
from alttp.ram import on_hyrule_castle_grounds, player_has_control
from alttp.startup import boot_past_title_to_castle, create_castle_grounds_state

GAME = GAME_SPEC


def create_first_action_state() -> None:
    """Persist FirstAction / HyruleCastleGrounds after a clean title boot."""
    result = create_castle_grounds_state(also_first_action=True)
    if not result.snapshot.on_castle_grounds:
        raise RuntimeError(
            f"startup failed after {result.frames} steps "
            f"(screen={result.snapshot.screen_id:#04x})"
        )
    print(
        f"Saved {FIRST_ACTION_STATE}.state on castle grounds "
        f"(screen=0x{result.snapshot.screen_id:02X}, "
        f"xy=({result.snapshot.link_x},{result.snapshot.link_y}))"
    )


__all__ = [
    "GAME",
    "boot_past_title_to_castle",
    "create_first_action_state",
    "on_hyrule_castle_grounds",
    "player_has_control",
]
