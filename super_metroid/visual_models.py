"""Compat shim — prefer ``super_metroid.legacy.visual_models`` for new code.

Frozen vision BC adapters. Continuous routes must not import this module.
"""

from __future__ import annotations

from super_metroid.legacy.visual_models import (
    LegacyBCPolicy,
    ModelContract,
    ModelPrediction,
)

__all__ = ["LegacyBCPolicy", "ModelContract", "ModelPrediction"]
