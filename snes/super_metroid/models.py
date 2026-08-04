"""DEPRECATED compat shim — prefer ``super_metroid.legacy`` for new code.

**Do not grow this surface.** Frozen vision/RL registry re-export.
Continuous routes must not import this module. Prefer
``super_metroid.legacy.models`` (or ``super_metroid.legacy``).
"""

from __future__ import annotations

from super_metroid.legacy.models import LegacyModel, load_model_registry

__all__ = ["LegacyModel", "load_model_registry"]
