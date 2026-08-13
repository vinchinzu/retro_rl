"""Frozen Super Metroid legacy surface (vision BC / imported model registry).

**Do not add new continuous-route or controller imports through this package.**

Active product path is scripted continuous KPDR (`routes/`, `combat/` feature
strategies, hash-pinned early policies). Vision BC is parked until gold; see
``docs/research/LEGACY_MODEL_REUSE.md`` and ``docs/ARCHITECTURE.md``.

Import from ``super_metroid.legacy.models`` / ``legacy.visual_models``.
"""

from __future__ import annotations

from super_metroid.legacy.models import LegacyModel, load_model_registry

__all__ = ["LegacyModel", "load_model_registry"]
