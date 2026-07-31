"""Compat shim — prefer ``super_metroid.legacy`` for new code.

Frozen vision/RL registry. Continuous routes must not import this module.
"""

from __future__ import annotations

from super_metroid.legacy.models import LegacyModel, load_model_registry

__all__ = ["LegacyModel", "load_model_registry"]
