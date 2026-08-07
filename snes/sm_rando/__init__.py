"""Super Metroid Randomizer — single-game solver scaffold.

Simpler than SMZ3: one world, item shuffle, room physics shared with vanilla
``super_metroid``. Build item-logic + room skill hooks here, then extend to SMZ3.

Reuse vanilla primitives — do **not** copy the super_metroid tree.
"""

from __future__ import annotations

__all__ = [
    "boot",
    "logic_graph",
    "paths",
    "play",
    "seed",
]
