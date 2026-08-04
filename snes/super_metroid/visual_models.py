"""DEPRECATED compat shim — prefer ``super_metroid.legacy.visual_models``.

**Do not grow this surface.** Frozen vision BC adapters re-export.
Continuous routes must not import this module.
"""

from __future__ import annotations

from super_metroid.legacy.visual_models import (
    LegacyBCPolicy,
    ModelContract,
    ModelPrediction,
)

__all__ = ["LegacyBCPolicy", "ModelContract", "ModelPrediction"]
