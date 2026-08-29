"""Thin session: sequence existing controllers via NamedRoute / RouteLeg ids.

No room phase machines and no path geometry. Scripts stay env+assist+report;
this module only binds factory strings to existing L1/L2 legs (and optional
door_graph bindings when that helper exists).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from zelda_i.overworld.graph import LEVEL1_PATH_SCREENS

ControllerFactory = Callable[..., Any]


@dataclass(frozen=True)
class ControllerBinding:
    """One existing controller factory attached to a graph edge or RouteLeg."""

    edge_or_leg_id: str
    factory: str
    stop_predicate: str
    assist_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_or_leg_id": self.edge_or_leg_id,
            "factory": self.factory,
            "stop_predicate": self.stop_predicate,
            "assist_ok": self.assist_ok,
        }


# Existing L1/L2 controllers (dotted import path or registry key).
_SWORD = "zelda_i.overworld.sword_cave.SwordCaveController"
_L1_OW = "zelda_i.overworld.nav.OverworldToLevel1Controller"
_L1_KEY = "zelda_i.level1.path.Level1FirstKeyController"
_L1_NORTH = "zelda_i.level1.path.Level1UnlockNorthController"
_L1_C63 = "zelda_i.level1.path.Level1Clear63Controller"
_L1_C53 = "zelda_i.level1.path.Level1Clear53Controller"
_L1_C54 = "zelda_i.dungeon.engine.GenericDungeonRoomController"
_L1_DONE = "level1_complete_chain"
_L2_SETTLE = "zelda_i.level2.overworld.PostTriforceSettleController"
_L2_PREFIX = "zelda_i.level2.overworld.OverworldToLevel2Controller"
_L2_DOOR = "zelda_i.level2.overworld.OverworldToLevel2Controller"
_L2_ENTER = "zelda_i.level2.overworld.OverworldToLevel2Controller"

# (leg_id, factory, stop_predicate) — first consumer of NamedRoute + route_legs.
_L1_L2_ROWS: tuple[tuple[str, str, str], ...] = (
    ("enter_sword_cave", _SWORD, "enter_wooden_sword_cave"),
    ("take_wooden_sword_and_exit", _SWORD, "wooden_sword_on_start_screen"),
    ("enter_level1_dungeon", _L1_OW, "level1_entrance_success"),
    ("settle_level1_entrance", _L1_KEY, "level1_entry_room_ready"),
    ("enter_level1_first_key_room", _L1_KEY, "level1_first_key_room_ready"),
    ("collect_level1_first_key", _L1_KEY, "level1_first_key_success"),
    ("resume_after_level1_first_key", _L1_NORTH, "level1_first_key_room_with_key"),
    ("return_to_level1_entrance", _L1_NORTH, "return_level1_room_73"),
    ("unlock_level1_north", _L1_NORTH, "level1_north_room_success"),
    ("clear_level1_room_63", _L1_C63, "level1_room_63_cleared"),
    ("enter_level1_room_53", _L1_C53, "reach_level1_room_53"),
    ("clear_level1_room_53", _L1_C53, "level1_room_53_cleared"),
    ("enter_level1_room_54", _L1_C54, "reach_level1_room_54"),
    ("clear_level1_room_54", _L1_C54, "dungeon_room_cleared(ROOM_54_SPEC)"),
    ("complete_level1_eagle", _L1_DONE, "triforce & 0x01"),
    ("settle_post_triforce_overworld", _L2_SETTLE, "post_triforce_overworld_ready"),
    ("walk_level2_path_prefix", _L2_PREFIX, "level2_path_prefix_success"),
    ("walk_level2_door_path", _L2_DOOR, "level2_door_screen_3c"),
    ("enter_level2_dungeon", _L2_ENTER, "level2_entrance_success"),
)

# Existing L3–L5 / L9-fixture controllers (no new path logic).
_L3_OW = "zelda_i.level3.overworld.OverworldPostL2ToLevel3Controller"
_L3_KEY = "zelda_i.level3.path.Level3WestKeyController"
_L3_RAFT = "zelda_i.level3.raft_path.Level3RaftPathController"
_L3_BOSS = "zelda_i.level3.boss_path.Level3BossPathController"
_L4_OW = "zelda_i.level4.overworld.OverworldToLevel4Controller"
_L4_LADDER = "zelda_i.level4.stepladder.Level4StepladderController"
_L4_BOSS = "zelda_i.level4.boss_combat.Level4GleeokFightController"
_L5_OW = "zelda_i.level5.overworld.OverworldToLevel5Controller"
_L5_RETURN = "zelda_i.level5.path.Level5Return66Controller"
_L5_POLS = "zelda_i.level5.dungeon.Level5PolsVoiceController"
_L9_PATRA = "zelda_i.level9.patra.FinalPatraFightController"
_L9_GANON = "zelda_i.level9.ganon.GanonFightController"

# (leg_id, factory, stop_predicate, assist_ok)
_LATER_ROWS: tuple[tuple[str, str, str, bool], ...] = (
    ("enter_level3", _L3_OW, "level3_entrance_success", True),
    ("level3_west_key", _L3_KEY, "level3_room_7b_key", True),
    ("level3_raft", _L3_RAFT, "level3_raft_collected", True),
    ("level3_manhandla", _L3_BOSS, "level3_manhandla_room", True),
    ("level3_triforce", _L3_BOSS, "triforce & 0x04", True),
    ("level4_raft_to_island", _L4_OW, "level4_door_screen", True),
    ("enter_level4", _L4_OW, "level4_entrance_success", True),
    ("level4_stepladder", _L4_LADDER, "level4_stepladder_collected", True),
    ("level4_gleeok", _L4_BOSS, "level4_gleeok_room", True),
    ("level4_triforce", _L4_BOSS, "triforce & 0x08", True),
    ("level5_hills_to_door", _L5_OW, "level5_door_screen", True),
    ("enter_level5", _L5_OW, "level5_entrance_success", True),
    ("level5_first_key", _L5_RETURN, "level5_room_66_cleared", True),
    ("level5_east_key", _L5_POLS, "level5_room_77_key", True),
    ("level5_whistle", "zelda_i.level5.path.hunt_whistle", "level5_whistle_collected", True),
    ("level5_digdogger", "zelda_i.level5.tf_path", "level5_digdogger_room", True),
    ("level5_triforce", "zelda_i.level5.tf_path", "triforce & 0x10", True),
    ("level9_41_to_31", "zelda_i.level9.path", "level9_in_room_31", True),
    ("level9_31_to_30", "zelda_i.level9.path", "level9_in_room_30", True),
    ("level9_30_to_67", "zelda_i.level9.path", "level9_in_cellar_67", True),
    ("level9_67_to_04", "zelda_i.level9.path", "level9_in_room_04", True),
    ("level9_04_to_03", "zelda_i.level9.path", "level9_in_room_03", True),
    ("level9_03_to_patra", "zelda_i.level9.path", "level9_patra_room", True),
    ("level9_patra_to_ganon", _L9_PATRA, "level9_ganon_defeated", True),
    ("level9_ganon_to_zelda", _L9_GANON, "level9_ending", True),
)


def _level1_ow_bindings() -> tuple[ControllerBinding, ...]:
    hops = zip(LEVEL1_PATH_SCREENS, LEVEL1_PATH_SCREENS[1:])
    return tuple(
        ControllerBinding(
            edge_or_leg_id=f"ow_{src:02x}_to_{dst:02x}",
            factory=_L1_OW,
            stop_predicate=f"reach_screen_{dst:02X}",
        )
        for src, dst in hops
    )


def default_l1_l2_bindings() -> tuple[ControllerBinding, ...]:
    """Bindings for published L1/L2 RouteLeg ids (no new path logic)."""
    rows = tuple(
        ControllerBinding(leg_id, factory, stop)
        for leg_id, factory, stop in _L1_L2_ROWS
    )
    return _level1_ow_bindings() + rows


def optional_door_graph_bindings() -> tuple[ControllerBinding, ...]:
    """Use ``zelda_i.door_graph.bind`` when a sibling module provides it."""
    try:
        from zelda_i.door_graph import bind as door_graph_bind
    except ImportError:
        return ()
    extra: Any
    if callable(door_graph_bind):
        extra = door_graph_bind()
    else:
        getter = getattr(door_graph_bind, "controller_bindings", None)
        extra = getter() if callable(getter) else getter
    if not extra:
        return ()
    return tuple(extra)


class RouteSession:
    """Match RouteLeg / edge ids onto bound controller factories."""

    def __init__(self, *, route_id: str | None = None) -> None:
        self.route_id = route_id
        self._bindings: dict[str, ControllerBinding] = {}
        self._planned: list[ControllerBinding] = []
        self._unbound: list[str] = []

    def bind(self, bindings: Sequence[ControllerBinding]) -> None:
        for binding in bindings:
            self._bindings[binding.edge_or_leg_id] = binding

    def plan(self, legs: Sequence[Any]) -> list[ControllerBinding]:
        planned: list[ControllerBinding] = []
        unbound: list[str] = []
        for leg in legs:
            key = leg_key(leg)
            binding = self._bindings.get(key)
            if binding is None:
                unbound.append(key)
                continue
            planned.append(binding)
        self._planned = planned
        self._unbound = unbound
        return list(planned)

    def describe(self) -> dict[str, Any]:
        """JSON-serializable plan. Does not touch the emulator."""
        return {
            "route_id": self.route_id,
            "bindings": [binding.to_dict() for binding in self._planned],
            "unbound": list(self._unbound),
            "bound_count": len(self._planned),
            "unbound_count": len(self._unbound),
        }

    def instantiate(
        self,
        registry: Mapping[str, ControllerFactory] | None = None,
    ) -> list[Any]:
        """Build bound factories (tests pass a fake registry; no ROM)."""
        return [resolve_factory(binding.factory, registry)() for binding in self._planned]


def leg_key(leg: Any) -> str:
    """Resolve a RouteLeg, PlannedLeg, mapping, or raw id string."""
    if isinstance(leg, str):
        return leg
    for attr in ("leg_id", "edge_or_leg_id"):
        value = getattr(leg, attr, None)
        if value:
            return str(value)
    if isinstance(leg, Mapping):
        for key in ("leg_id", "edge_or_leg_id", "id"):
            if leg.get(key):
                return str(leg[key])
    inner = getattr(leg, "leg", None)
    if inner is not None and inner is not leg:
        return leg_key(inner)
    raise TypeError(f"cannot resolve leg id from {type(leg)!r}")


def resolve_factory(
    factory: str,
    registry: Mapping[str, ControllerFactory] | None = None,
) -> ControllerFactory:
    """Look up a registry key, else import ``module.attr``."""
    if registry and factory in registry:
        return registry[factory]
    module_name, sep, attr = factory.rpartition(".")
    if not sep:
        raise KeyError(f"unknown factory {factory!r}")
    import importlib

    module = importlib.import_module(module_name)
    resolved = getattr(module, attr)
    if not callable(resolved):
        raise TypeError(f"factory {factory!r} is not callable")
    return resolved


def default_later_bindings() -> tuple[ControllerBinding, ...]:
    """Bindings for published L3–L5 + L9-fixture RouteLeg ids."""
    return tuple(
        ControllerBinding(leg_id, factory, stop, assist_ok=assist)
        for leg_id, factory, stop, assist in _LATER_ROWS
    )


def _resolve_named_route(route_id: str) -> Any:
    from zelda_i.route.catalog import get_route
    from zelda_i.route.catalog_later import get_later_route

    try:
        return get_route(route_id)
    except KeyError:
        return get_later_route(route_id)


def legs_for_named_route(route_id: str) -> tuple[Any, ...]:
    """Return the published RouteLeg table for ``route_id`` / alias."""
    from zelda_i.route.legs import (
        level1_clear53_route_legs,
        level1_clear54_route_legs,
        level1_clear63_route_legs,
        level1_complete_route_legs,
        level1_first_key_route_legs,
        level1_north_route_legs,
        level1_route_legs,
        level2_door_path_route_legs,
        level2_path_prefix_route_legs,
        sword_cave_route_legs,
    )
    from zelda_i.route.legs_later import (
        level3_complete_route_legs,
        level4_complete_route_legs,
        level5_complete_route_legs,
        level9_fixture_route_legs,
    )

    table: dict[str, Callable[[], tuple[Any, ...]]] = {
        "zelda_sword_cave": sword_cave_route_legs,
        "zelda_to_level1": level1_route_legs,
        "zelda_level1_first_key": level1_first_key_route_legs,
        "zelda_level1_north": level1_north_route_legs,
        "zelda_level1_clear63": level1_clear63_route_legs,
        "zelda_level1_clear53": level1_clear53_route_legs,
        "zelda_level1_clear54": level1_clear54_route_legs,
        "zelda_level1_complete": level1_complete_route_legs,
        "zelda_level2_path_prefix": level2_path_prefix_route_legs,
        "zelda_level2_door_path": level2_door_path_route_legs,
        "zelda_level3_complete": level3_complete_route_legs,
        "zelda_level4_complete": level4_complete_route_legs,
        "zelda_level5_complete": level5_complete_route_legs,
        "zelda_level9_ganon": level9_fixture_route_legs,
    }
    route = _resolve_named_route(route_id)
    loader = table.get(route.route_id)
    if loader is None:
        available = sorted(table)
        raise KeyError(f"no RouteLeg table for {route.route_id!r}. Available: {available}")
    return loader()


def session_for_named_route(route_id: str) -> RouteSession:
    """Bind L1–L5/L9 (plus optional door_graph) factories and plan the route."""
    route = _resolve_named_route(route_id)
    session = RouteSession(route_id=route.route_id)
    session.bind(default_l1_l2_bindings())
    session.bind(default_later_bindings())
    session.bind(optional_door_graph_bindings())
    session.plan(legs_for_named_route(route.route_id))
    return session


def describe_named_route(route_id: str) -> dict[str, Any]:
    """JSON-serializable would-run plan for a published L1/L2 NamedRoute."""
    return session_for_named_route(route_id).describe()


__all__ = [
    "ControllerBinding",
    "ControllerFactory",
    "RouteSession",
    "default_l1_l2_bindings",
    "default_later_bindings",
    "describe_named_route",
    "leg_key",
    "legs_for_named_route",
    "optional_door_graph_bindings",
    "resolve_factory",
    "session_for_named_route",
]
