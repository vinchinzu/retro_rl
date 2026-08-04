"""Continuous routes and play controllers.

- ``continuous`` — power-on chain via tip-id functions + ``run_to``
- ``catalog`` — named routes, continuous tips, segment registry
- ``runtime`` — shared session / report harness
- ``kpdr/`` — pure controllers (no env ownership)
- ``controller_common`` — shared Samus primitives

Record via one CLI: ``scripts/record/continuous.py --to <tip>``.

Public names are resolved lazily so ``progression.data`` can import Super+
DoorEdges from ``routes.kpdr.spine`` without a circular import through
``continuous`` / ``early_continuous``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CONTINUOUS_SEGMENTS",
    "CONTINUOUS_TIPS",
    "DEFAULT_CONTINUOUS_TIP",
    "get_continuous_tip",
    "get_named_route",
    "list_continuous_tips",
    "list_named_routes",
    "play_morph",
    "run_morph",
    "play_bombs",
    "run_bombs",
    "play_spore",
    "run_spore",
    "play_supers",
    "run_supers",
    "run_to",
]

_CATALOG_EXPORTS = frozenset(
    {
        "CONTINUOUS_SEGMENTS",
        "CONTINUOUS_TIPS",
        "DEFAULT_CONTINUOUS_TIP",
        "get_continuous_tip",
        "get_named_route",
        "list_continuous_tips",
        "list_named_routes",
    }
)
_CONTINUOUS_EXPORTS = frozenset(
    {
        "play_bombs",
        "play_morph",
        "play_spore",
        "play_supers",
        "run_bombs",
        "run_morph",
        "run_spore",
        "run_supers",
        "run_to",
    }
)


def __getattr__(name: str) -> Any:
    if name in _CATALOG_EXPORTS:
        from super_metroid.routes import catalog as _catalog

        return getattr(_catalog, name)
    if name in _CONTINUOUS_EXPORTS:
        from super_metroid.routes import continuous as _continuous

        return getattr(_continuous, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
