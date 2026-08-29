"""Hold-each-token dungeon maze walker.

L4 maze controllers copy the same ``path_index`` / ``hold_left`` loop.
Drive those loops from ``HoldTokenPath`` instead of a new phase machine.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HoldTokenPath:
    """Walk ``tokens`` holding each cardinal ``hold`` frames."""

    tokens: tuple[str, ...]
    hold: int
    index: int = 0
    held: int = 0

    def reset(self) -> None:
        self.index = 0
        self.held = 0

    def done(self) -> bool:
        return self.index >= len(self.tokens)

    def direction(self) -> str | None:
        if self.done():
            return None
        return self.tokens[self.index]

    def advance(self) -> str | None:
        """Return the current token and consume one hold frame."""
        direction = self.direction()
        if direction is None:
            return None
        self.held += 1
        if self.held >= self.hold:
            self.index += 1
            self.held = 0
        return direction
