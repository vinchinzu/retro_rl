"""ALTTP Randomizer — single-game solver scaffold.

Simpler than SMZ3: Light/Dark World + item shuffle without Super Metroid
portals. Build item-logic + dungeon/OW skill hooks here, then extend to SMZ3.

Reuse vanilla ``alttp`` primitives — do **not** copy that tree.
"""

from __future__ import annotations

__all__ = [
    "boot",
    "logic_graph",
    "paths",
    "play",
    "seed",
]
