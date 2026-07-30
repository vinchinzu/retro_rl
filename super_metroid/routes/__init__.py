"""Continuous routes and play controllers.

- ``continuous`` — power-on chain (Morph → Bombs → Spore → Supers)
- ``runtime`` — shared session / report harness
- ``*_controller`` — movement/combat only (no env ownership)
- ``controller_common`` — shared Samus primitives (morph, weapon, wait)
- ``kpdr/`` / ``post_spore/`` — split segment packages (controllers re-export)
- ``spore_spawn_route`` — editor-backed leg table for planning
"""

from super_metroid.routes.continuous import (
    play_start_to_bombs,
    play_start_to_morph,
    play_start_to_spore_spawn,
    play_start_to_supers,
    run_start_to_bombs,
    run_start_to_morph,
    run_start_to_spore_spawn,
    run_start_to_supers,
)

__all__ = [
    "play_start_to_morph",
    "run_start_to_morph",
    "play_start_to_bombs",
    "run_start_to_bombs",
    "play_start_to_spore_spawn",
    "run_start_to_spore_spawn",
    "play_start_to_supers",
    "run_start_to_supers",
]
