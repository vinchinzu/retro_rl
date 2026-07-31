"""Continuous routes and play controllers.

- ``continuous`` — power-on chain (Morph → … → Red → Bat → Below Spazer → Warehouse)
- ``catalog`` — named routes, continuous tips, segment registry
- ``runtime`` — shared session / report harness
- ``*_controller`` — movement/combat only (no env ownership)
- ``controller_common`` — shared Samus primitives (morph, weapon, wait)
- ``kpdr/`` — pure controllers from Spore Super through Kraid entry
  (``post_spore_controller`` is a thin re-export for older imports)
- ``spore_spawn_route`` — editor-backed leg table for planning

Record via one CLI: ``scripts/record/continuous.py --to <tip>`` (not
per-tip ``start_to_*.py`` files).
"""

from super_metroid.routes.catalog import (
    CONTINUOUS_SEGMENTS,
    CONTINUOUS_TIPS,
    DEFAULT_CONTINUOUS_TIP,
    get_continuous_tip,
    get_named_route,
    list_continuous_tips,
    list_named_routes,
)
from super_metroid.routes.continuous import (
    play_start_to_bat,
    play_start_to_below_spazer,
    play_start_to_bombs,
    play_start_to_morph,
    play_start_to_red_tower,
    play_start_to_spore_spawn,
    play_start_to_supers,
    play_start_to_warehouse,
    run_start_to_bat,
    run_start_to_below_spazer,
    run_start_to_bombs,
    run_start_to_morph,
    run_start_to_red_tower,
    run_start_to_spore_spawn,
    run_start_to_supers,
    run_start_to_warehouse,
    run_to,
)

__all__ = [
    "CONTINUOUS_SEGMENTS",
    "CONTINUOUS_TIPS",
    "DEFAULT_CONTINUOUS_TIP",
    "get_continuous_tip",
    "get_named_route",
    "list_continuous_tips",
    "list_named_routes",
    "play_start_to_morph",
    "run_start_to_morph",
    "play_start_to_bombs",
    "run_start_to_bombs",
    "play_start_to_spore_spawn",
    "run_start_to_spore_spawn",
    "play_start_to_supers",
    "run_start_to_supers",
    "play_start_to_red_tower",
    "run_start_to_red_tower",
    "play_start_to_bat",
    "run_start_to_bat",
    "play_start_to_below_spazer",
    "run_start_to_below_spazer",
    "play_start_to_warehouse",
    "run_start_to_warehouse",
    "run_to",
]
