"""Hammer/axe attempt policy extracted from the farm-clear state machine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from harvest.core.stamina import Stamina
from harvest.tasks.farm_ops import use_tool
from harvest.tasks.nav import make_action

if TYPE_CHECKING:
    from harvest.tasks.farm_clearer import FarmClearer

Tile = Tuple[int, int]


def _reset_target(clearer: "FarmClearer") -> None:
    clearer.current_target = None
    clearer.approach_tile = None
    clearer.clearing_start_frame = 0
    clearer.target_hits = 0


def handle_tool_clear(
    clearer: "FarmClearer",
    ram: np.ndarray,
    *,
    player: Tile,
    target: Tile,
) -> Optional[str]:
    """Queue one tool attempt and require tile disappearance for success."""
    current = clearer.current_target
    if current is None:
        return "scanning"
    tool = current.required_tool
    if tool is None:
        clearer.failed_tiles.add(target)
        _reset_target(clearer)
        return "scanning"

    if clearer.tool_manager.current != tool:
        print(
            f"[CLEARER] Need {tool.name}, "
            f"have 0x{clearer.tool_manager.current:02X}"
        )
        clearer.searching_tool = tool
        clearer.tool_manager.start_search()
        clearer.tool_search_frames = 0
        return "tool_switch"

    stamina = Stamina.from_ram(ram)
    if clearer.target_hits == 0 and not clearer._can_afford_target(ram, current):
        print(
            f"[CLEARER] Skip {current.debris_type.name} at {target}: need "
            f"{stamina.cost_to_clear(current.required_hits)} stam for "
            f"{current.required_hits}+miss budget, have {stamina}"
        )
        clearer.stamina_exhausted = True
        _reset_target(clearer)
        return "scanning"
    if clearer.target_hits > 0 and stamina < 2:
        clearer.stamina_exhausted = True
        _reset_target(clearer)
        return "complete"

    if clearer.target_hits == 0:
        tile_key = (target[0], target[1], current.tile_id)
        attempts = clearer.tile_attempts.get(tile_key, 0)
        if attempts >= 3:
            print(
                f"[CLEARER] Giving up on {current.debris_type.name} at "
                f"{target} tile=0x{current.tile_id:02X} (3 failed attempts)"
            )
            clearer.failed_tiles.add(target)
            _reset_target(clearer)
            return "scanning"
        clearer.tile_attempts[tile_key] = attempts + 1
        direction = clearer._face_dir(player, target)
        verb = "Clearing" if attempts == 0 else "Re-targeting"
        print(
            f"[CLEARER] {verb} {current.debris_type.name} at {target} "
            f"tile=0x{current.tile_id:02X} from {player} facing {direction} "
            f"({current.required_hits} hits, attempt {attempts + 1}/3)"
        )

    # Tile disappearance is checked by the caller before this module runs.
    if clearer.target_hits >= current.required_hits + 3:
        print(f"[CLEARER] Hits exhausted but tile remains at {target}")
        clearer.failed_tiles.add(target)
        _reset_target(clearer)
        return "scanning"

    direction = clearer._face_dir(player, target)
    clearer.action_queue.append(make_action(**{direction: True}))
    clearer.action_queue.extend([make_action() for _ in range(8)])
    clearer.action_queue.extend(use_tool(frames=20, cooldown=20))
    clearer.target_hits += 1
    return None


__all__ = ["handle_tool_clear"]
