"""ALTTP spine package.

The clean ladder is a status table (AP-003), not a run dispatcher.
"""

from __future__ import annotations

from alttp.spine.clean_ladder import CLEAN_LADDER, CleanRung, LadderStatus, next_open

__all__ = ["CLEAN_LADDER", "CleanRung", "LadderStatus", "next_open"]
