"""A Link to the Past — opening-route workspace.

Core runtime (``ram``, ``primitives``, ``startup``, ``overworld``,
``session``) lives at package root. Continuous trunk lives in
``alttp.opening_route``. See ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

from alttp.paths import GAME, GAME_DIR, INTEGRATION

__all__ = ["GAME", "GAME_DIR", "INTEGRATION"]
