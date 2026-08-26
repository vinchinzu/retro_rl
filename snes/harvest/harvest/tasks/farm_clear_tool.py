"""Hammer/axe attempt policy extracted from the farm-clear state machine.

$096D (tool_hit_counter) accumulates toward 6 only while the farmer stays
planted. A d-pad walk or re-center between swings STZs the counter, so the
first face is the last movement of a multi-hit. Hit credit comes from a live
RAM edge after a short post-swing wait, never from the queued attempt count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from harvest.core.stamina import SWING_STAMINA_COST, Stamina
from harvest.tasks.farm_ops import use_tool
from harvest.tasks.nav import make_action

if TYPE_CHECKING:
    from harvest.tasks.farm_clearer import FarmClearer

Tile = Tuple[int, int]

FACE_SETTLE_FRAMES = 8
Y_HOLD_FRAMES = 20
SWING_COOLDOWN_FRAMES = 20
POST_SWING_OBSERVE_FRAMES = 2
MAX_OBSERVE_EXTRA = 4
MAX_STAND_MISSES = 3


def _reset_tool_seq(clearer: "FarmClearer") -> None:
    clearer._tool_swing_pending = False
    clearer._tool_last_hits = 0
    clearer._tool_last_stam = None
    clearer._tool_misses = 0
    clearer._tool_observe_extra = 0
    clearer._tool_faced = False
    clearer._tool_seq_key = None


def _reset_target(clearer: "FarmClearer") -> None:
    clearer.current_target = None
    clearer.approach_tile = None
    clearer.clearing_start_frame = 0
    clearer.target_hits = 0
    _reset_tool_seq(clearer)


def _seq_key(target: Tile, player: Tile) -> Tuple[int, int, int, int]:
    return (target[0], target[1], player[0], player[1])


def _nearest_face_tile(player: Tile, current) -> Tile:
    cells = tuple(current.footprint) or (current.tile,)
    return min(
        cells,
        key=lambda cell: abs(cell[0] - player[0]) + abs(cell[1] - player[1]),
    )


def _hit_edge(clearer: "FarmClearer", ram: np.ndarray) -> bool:
    """True when RAM shows a registered swing after the observe wait."""
    stam = Stamina.from_ram(ram)
    last_hits = int(getattr(clearer, "_tool_last_hits", 0) or 0)
    last_stam = getattr(clearer, "_tool_last_stam", None)
    if stam.tool_hits > last_hits:
        clearer.target_hits = stam.tool_hits
        return True
    if last_stam is not None and stam.current <= int(last_stam) - SWING_STAMINA_COST:
        # $096D can lag the stamina debit by one frame.
        clearer.target_hits = max(int(clearer.target_hits), last_hits + 1)
        return True
    return False


def _retry_other_stand(
    clearer: "FarmClearer",
    ram: np.ndarray,
    *,
    target: Tile,
) -> str:
    current = clearer.current_target
    player = clearer.navigator.current_tile
    clearer.failed_approaches.add((target, player))
    from harvest.tasks.farm_clear_nav import find_unfailed_approach

    nxt = find_unfailed_approach(clearer, ram, current) if current is not None else None
    print(
        f"[CLEARER] Three planted misses at {player} vs {target}; "
        f"next stand={nxt}"
    )
    if nxt is None:
        clearer.failed_tiles.add(target)
        _reset_target(clearer)
        return "scanning"
    clearer.approach_tile = nxt
    clearer.target_hits = 0
    clearer.clearing_start_frame = 0
    _reset_tool_seq(clearer)
    return "navigating"


def _observe_pending_swing(
    clearer: "FarmClearer",
    ram: np.ndarray,
    *,
    target: Tile,
) -> Optional[str]:
    if not getattr(clearer, "_tool_swing_pending", False):
        return None
    if _hit_edge(clearer, ram):
        clearer._tool_swing_pending = False
        clearer._tool_observe_extra = 0
        clearer._tool_misses = 0
        clearer.clearing_start_frame = clearer.frame_count or 1
        stam = Stamina.from_ram(ram)
        print(
            f"[CLEARER] Hit registered {clearer.target_hits} "
            f"stam={stam} (planted)"
        )
        return None
    extra = int(getattr(clearer, "_tool_observe_extra", 0) or 0)
    if extra < MAX_OBSERVE_EXTRA:
        clearer._tool_observe_extra = extra + 1
        clearer.action_queue.append(make_action())
        return None
    clearer._tool_swing_pending = False
    clearer._tool_observe_extra = 0
    clearer._tool_misses = int(getattr(clearer, "_tool_misses", 0) or 0) + 1
    print(
        f"[CLEARER] Swing miss {clearer._tool_misses}/{MAX_STAND_MISSES} "
        f"at {target} from {clearer.navigator.current_tile}; stay planted"
    )
    if clearer._tool_misses >= MAX_STAND_MISSES:
        return _retry_other_stand(clearer, ram, target=target)
    return None


def _queue_planted_swing(
    clearer: "FarmClearer",
    ram: np.ndarray,
    *,
    player: Tile,
    target: Tile,
) -> None:
    current = clearer.current_target
    stam = Stamina.from_ram(ram)
    if not getattr(clearer, "_tool_faced", False):
        face_tile = _nearest_face_tile(player, current) if current is not None else target
        direction = clearer._face_dir(player, face_tile)
        clearer.action_queue.append(make_action(**{direction: True}))
        clearer.action_queue.extend(
            [make_action() for _ in range(FACE_SETTLE_FRAMES)]
        )
        clearer._tool_faced = True
        print(
            f"[CLEARER] Face {direction} at {target} from {player}, "
            "then Y-only until it breaks"
        )
    clearer._tool_last_hits = stam.tool_hits
    clearer._tool_last_stam = stam.current
    clearer.action_queue.extend(
        use_tool(frames=Y_HOLD_FRAMES, cooldown=SWING_COOLDOWN_FRAMES)
    )
    clearer.action_queue.extend(
        [make_action() for _ in range(POST_SWING_OBSERVE_FRAMES)]
    )
    clearer._tool_swing_pending = True


def handle_tool_clear(
    clearer: "FarmClearer",
    ram: np.ndarray,
    *,
    player: Tile,
    target: Tile,
) -> Optional[str]:
    """Queue one planted tool attempt; credit hits only from a RAM edge."""
    current = clearer.current_target
    if current is None:
        return "scanning"
    tool = current.required_tool
    if tool is None:
        clearer.failed_tiles.add(target)
        _reset_target(clearer)
        return "scanning"

    key = _seq_key(target, player)
    if getattr(clearer, "_tool_seq_key", None) != key:
        _reset_tool_seq(clearer)
        clearer._tool_seq_key = key
        clearer.target_hits = 0

    if clearer.tool_manager.current != tool:
        print(
            f"[CLEARER] Need {tool.name}, "
            f"have 0x{clearer.tool_manager.current:02X}"
        )
        clearer.searching_tool = tool
        clearer.tool_manager.start_search()
        clearer.tool_search_frames = 0
        _reset_tool_seq(clearer)
        return "tool_switch"

    observe = _observe_pending_swing(clearer, ram, target=target)
    if observe is not None:
        return observe

    stamina = Stamina.from_ram(ram)
    if stamina.tool_hits > int(clearer.target_hits):
        clearer.target_hits = stamina.tool_hits

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

    if clearer.target_hits == 0 and not getattr(clearer, "_tool_faced", False):
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
        verb = "Clearing" if attempts == 0 else "Re-targeting"
        print(
            f"[CLEARER] {verb} {current.debris_type.name} at {target} "
            f"tile=0x{current.tile_id:02X} from {player} "
            f"({current.required_hits} hits, attempt {attempts + 1}/3)"
        )

    _queue_planted_swing(clearer, ram, player=player, target=target)
    return None


def tool_clear_is_planted(clearer: "FarmClearer") -> bool:
    """True once the first face is committed; d-pad must stay off."""
    if clearer.current_target is None:
        return False
    if clearer._should_lift(clearer.current_target):
        return False
    return bool(
        getattr(clearer, "_tool_faced", False)
        or getattr(clearer, "_tool_swing_pending", False)
        or int(clearer.target_hits) > 0
    )


__all__ = [
    "FACE_SETTLE_FRAMES",
    "MAX_STAND_MISSES",
    "POST_SWING_OBSERVE_FRAMES",
    "handle_tool_clear",
    "tool_clear_is_planted",
]
