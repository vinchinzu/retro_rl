"""KPDR package surface: room ids, evidence types, registry.

Hop controllers live in owner modules. Import ``play_*`` from the owner
(``registry.get_segment``, spine hops, ``k4_cathedral``, ``wave``, …).
This package does not re-export every hop callable.

Exports load lazily so ``room_ids`` (pure constants) can be imported from
``progression`` without pulling ``controller_common`` / ``runtime``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ITEM_HI_JUMP",
    "KPDR_SEGMENTS",
    "MORPH_POSES",
    "MorphBombRollPhase",
    "PowerBombEvidence",
    "ROOM_BABY_KRAID",
    "ROOM_BAT",
    "ROOM_BELOW_SPAZER",
    "ROOM_BIG_PINK",
    "ROOM_BUSINESS",
    "ROOM_EAST_TUNNEL",
    "ROOM_FARMING",
    "ROOM_GHZ",
    "ROOM_GLASS",
    "ROOM_HJ",
    "ROOM_HJ_SHAFT",
    "ROOM_KRAID",
    "ROOM_KRAID_EYE",
    "ROOM_NOOB",
    "ROOM_PINK_PB",
    "ROOM_RED_TOWER",
    "ROOM_SUPER",
    "ROOM_VARIA",
    "ROOM_WAREHOUSE",
    "ROOM_WAREHOUSE_KIHUNTER",
    "ROOM_WEST_TUNNEL",
    "ROOM_ZEELA",
    "SuperCollectEvidence",
    "bomb_roll_left_safe",
    "ensure_morph",
    "get_segment",
    "is_morph",
    "wait_until",
]

# export_name -> (module, attr)
_EXPORTS: dict[str, tuple[str, str]] = {
    "ITEM_HI_JUMP": ("super_metroid.routes.kpdr.rooms", "ITEM_HI_JUMP"),
    "KPDR_SEGMENTS": ("super_metroid.routes.kpdr.registry", "KPDR_SEGMENTS"),
    "MORPH_POSES": ("super_metroid.routes.controller_common", "MORPH_POSES"),
    "MorphBombRollPhase": ("super_metroid.routes.kpdr.morph_bomb_roll", "MorphBombRollPhase"),
    "PowerBombEvidence": ("super_metroid.routes.kpdr.rooms", "PowerBombEvidence"),
    "ROOM_BABY_KRAID": ("super_metroid.routes.kpdr.rooms", "ROOM_BABY_KRAID"),
    "ROOM_BAT": ("super_metroid.routes.kpdr.rooms", "ROOM_BAT"),
    "ROOM_BELOW_SPAZER": ("super_metroid.routes.kpdr.rooms", "ROOM_BELOW_SPAZER"),
    "ROOM_BIG_PINK": ("super_metroid.routes.kpdr.rooms", "ROOM_BIG_PINK"),
    "ROOM_BUSINESS": ("super_metroid.routes.kpdr.rooms", "ROOM_BUSINESS"),
    "ROOM_EAST_TUNNEL": ("super_metroid.routes.kpdr.rooms", "ROOM_EAST_TUNNEL"),
    "ROOM_FARMING": ("super_metroid.routes.kpdr.rooms", "ROOM_FARMING"),
    "ROOM_GHZ": ("super_metroid.routes.kpdr.rooms", "ROOM_GHZ"),
    "ROOM_GLASS": ("super_metroid.routes.kpdr.rooms", "ROOM_GLASS"),
    "ROOM_HJ": ("super_metroid.routes.kpdr.rooms", "ROOM_HJ"),
    "ROOM_HJ_SHAFT": ("super_metroid.routes.kpdr.rooms", "ROOM_HJ_SHAFT"),
    "ROOM_KRAID": ("super_metroid.routes.kpdr.rooms", "ROOM_KRAID"),
    "ROOM_KRAID_EYE": ("super_metroid.routes.kpdr.rooms", "ROOM_KRAID_EYE"),
    "ROOM_NOOB": ("super_metroid.routes.kpdr.rooms", "ROOM_NOOB"),
    "ROOM_PINK_PB": ("super_metroid.routes.kpdr.rooms", "ROOM_PINK_PB"),
    "ROOM_RED_TOWER": ("super_metroid.routes.kpdr.rooms", "ROOM_RED_TOWER"),
    "ROOM_SUPER": ("super_metroid.routes.kpdr.rooms", "ROOM_SUPER"),
    "ROOM_VARIA": ("super_metroid.routes.kpdr.rooms", "ROOM_VARIA"),
    "ROOM_WAREHOUSE": ("super_metroid.routes.kpdr.rooms", "ROOM_WAREHOUSE"),
    "ROOM_WAREHOUSE_KIHUNTER": ("super_metroid.routes.kpdr.rooms", "ROOM_WAREHOUSE_KIHUNTER"),
    "ROOM_WEST_TUNNEL": ("super_metroid.routes.kpdr.rooms", "ROOM_WEST_TUNNEL"),
    "ROOM_ZEELA": ("super_metroid.routes.kpdr.rooms", "ROOM_ZEELA"),
    "SuperCollectEvidence": ("super_metroid.routes.kpdr.rooms", "SuperCollectEvidence"),
    "bomb_roll_left_safe": ("super_metroid.routes.kpdr.morph_bomb_roll", "bomb_roll_left_safe"),
    "ensure_morph": ("super_metroid.routes.controller_common", "ensure_morph"),
    "get_segment": ("super_metroid.routes.kpdr.registry", "get_segment"),
    "is_morph": ("super_metroid.routes.controller_common", "is_morph"),
    "wait_until": ("super_metroid.routes.controller_common", "wait_until"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
