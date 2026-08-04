"""Route leg tables and plan helpers for Zelda I early-game routes.

Geometry, nodes, and graph builders live in :mod:`zelda_i.overworld`.
"""

from __future__ import annotations

from retro_harness.adventure.graph import RouteGraph, RouteLeg

from zelda_i.overworld import (
    LEVEL1_PATH_SCREENS,
    NODE_LEVEL1_COMPLETE,
    NODE_LEVEL1_DUNGEON,
    NODE_LEVEL1_ENTRANCE,
    NODE_LEVEL1_ENTRY_ROOM,
    NODE_LEVEL1_EXIT_OVERWORLD,
    NODE_LEVEL1_FIRST_KEY,
    NODE_LEVEL1_FIRST_KEY_ROOM,
    NODE_LEVEL1_NORTH_CLEARED,
    NODE_LEVEL1_NORTH_ROOM,
    NODE_LEVEL1_ROOM_53,
    NODE_LEVEL1_ROOM_53_CLEARED,
    NODE_LEVEL1_ROOM_54,
    NODE_LEVEL1_ROOM_54_CLEARED,
    NODE_LEVEL2_PATH_4A,
    NODE_START,
    NODE_SWORD_CAVE,
    build_early_route_graph,
    node_id_for_screen,
)

def sword_cave_route_legs() -> tuple[RouteLeg, ...]:
    """Canonical M3 segment: start overworld → sword cave → back with sword."""
    return (
        RouteLeg(
            leg_id="enter_sword_cave",
            source_id=NODE_START,
            target_id=NODE_SWORD_CAVE,
            acquires=frozenset(),
            goal="enter_wooden_sword_cave",
        ),
        RouteLeg(
            leg_id="take_wooden_sword_and_exit",
            source_id=NODE_SWORD_CAVE,
            target_id=NODE_START,
            acquires=frozenset({"wooden_sword"}),
            goal="wooden_sword_on_start_screen",
        ),
    )


def level1_route_legs() -> tuple[RouteLeg, ...]:
    """Sword cave + overworld path into Level 1 dungeon."""
    hops = list(zip(LEVEL1_PATH_SCREENS, LEVEL1_PATH_SCREENS[1:]))
    ow_legs = tuple(
        RouteLeg(
            leg_id=f"ow_{src:02x}_to_{dst:02x}",
            source_id=node_id_for_screen(src),
            target_id=node_id_for_screen(dst),
            requires=frozenset({"wooden_sword"}),
            goal=f"reach_screen_{dst:02X}",
        )
        for src, dst in hops
    )
    return (
        *sword_cave_route_legs(),
        *ow_legs,
        RouteLeg(
            leg_id="enter_level1_dungeon",
            source_id=NODE_LEVEL1_ENTRANCE,
            target_id=NODE_LEVEL1_DUNGEON,
            requires=frozenset({"wooden_sword"}),
            goal="inside_level1",
        ),
    )


def level1_first_key_route_legs() -> tuple[RouteLeg, ...]:
    """Power-on route suffix through the first Level 1 key."""
    return (
        *level1_route_legs(),
        RouteLeg(
            leg_id="settle_level1_entrance",
            source_id=NODE_LEVEL1_DUNGEON,
            target_id=NODE_LEVEL1_ENTRY_ROOM,
            requires=frozenset({"wooden_sword"}),
            goal="level1_room_73_ready",
        ),
        RouteLeg(
            leg_id="enter_level1_first_key_room",
            source_id=NODE_LEVEL1_ENTRY_ROOM,
            target_id=NODE_LEVEL1_FIRST_KEY_ROOM,
            requires=frozenset({"wooden_sword"}),
            goal="reach_level1_room_74",
        ),
        RouteLeg(
            leg_id="collect_level1_first_key",
            source_id=NODE_LEVEL1_FIRST_KEY_ROOM,
            target_id=NODE_LEVEL1_FIRST_KEY,
            requires=frozenset({"wooden_sword"}),
            acquires=frozenset({"keys"}),
            goal="level1_keys_at_least_1",
        ),
    )


def level1_north_route_legs() -> tuple[RouteLeg, ...]:
    """Power-on route suffix through the north door into room 0x63."""
    return (
        *level1_first_key_route_legs(),
        RouteLeg(
            leg_id="resume_after_level1_first_key",
            source_id=NODE_LEVEL1_FIRST_KEY,
            target_id=NODE_LEVEL1_FIRST_KEY_ROOM,
            requires=frozenset({"wooden_sword", "keys"}),
            goal="level1_first_key_room_with_key",
        ),
        RouteLeg(
            leg_id="return_to_level1_entrance",
            source_id=NODE_LEVEL1_FIRST_KEY_ROOM,
            target_id=NODE_LEVEL1_ENTRY_ROOM,
            requires=frozenset({"wooden_sword", "keys"}),
            goal="return_level1_room_73",
        ),
        RouteLeg(
            leg_id="unlock_level1_north",
            source_id=NODE_LEVEL1_ENTRY_ROOM,
            target_id=NODE_LEVEL1_NORTH_ROOM,
            requires=frozenset({"wooden_sword", "keys"}),
            goal="level1_room_63_ready",
            constraints=("consumes_one_key",),
        ),
    )


def level1_clear63_route_legs() -> tuple[RouteLeg, ...]:
    """Power-on route suffix that clears the three Stalfos in room 0x63."""
    return (
        *level1_north_route_legs(),
        RouteLeg(
            leg_id="clear_level1_room_63",
            source_id=NODE_LEVEL1_NORTH_ROOM,
            target_id=NODE_LEVEL1_NORTH_CLEARED,
            requires=frozenset({"wooden_sword"}),
            goal="level1_room_63_cleared",
        ),
    )


def level1_clear53_route_legs() -> tuple[RouteLeg, ...]:
    """Power-on route suffix through the room 0x53 clear and fixed key."""
    return (
        *level1_clear63_route_legs(),
        RouteLeg(
            leg_id="enter_level1_room_53",
            source_id=NODE_LEVEL1_NORTH_CLEARED,
            target_id=NODE_LEVEL1_ROOM_53,
            requires=frozenset({"wooden_sword"}),
            goal="reach_level1_room_53",
        ),
        RouteLeg(
            leg_id="clear_level1_room_53",
            source_id=NODE_LEVEL1_ROOM_53,
            target_id=NODE_LEVEL1_ROOM_53_CLEARED,
            requires=frozenset({"wooden_sword"}),
            acquires=frozenset({"keys"}),
            goal="level1_room_53_cleared_and_key_collected",
        ),
    )


def level1_clear54_route_legs() -> tuple[RouteLeg, ...]:
    """Power-on route suffix through the east branch and room 0x54 clear."""
    return (
        *level1_clear53_route_legs(),
        RouteLeg(
            leg_id="enter_level1_room_54",
            source_id=NODE_LEVEL1_ROOM_53_CLEARED,
            target_id=NODE_LEVEL1_ROOM_54,
            requires=frozenset({"wooden_sword"}),
            goal="reach_level1_room_54",
        ),
        RouteLeg(
            leg_id="clear_level1_room_54",
            source_id=NODE_LEVEL1_ROOM_54,
            target_id=NODE_LEVEL1_ROOM_54_CLEARED,
            requires=frozenset({"wooden_sword"}),
            goal="level1_room_54_cleared",
        ),
    )


def level1_complete_route_legs() -> tuple[RouteLeg, ...]:
    """Power-on route through Aquamentus and Triforce shard 1."""
    return (
        *level1_clear53_route_legs(),
        RouteLeg(
            leg_id="complete_level1_eagle",
            source_id=NODE_LEVEL1_ROOM_53_CLEARED,
            target_id=NODE_LEVEL1_COMPLETE,
            requires=frozenset({"wooden_sword"}),
            acquires=frozenset({"triforce_shard_1"}),
            goal="level1_triforce_bit_0_set",
        ),
    )


def early_route_plan(graph: RouteGraph | None = None):
    """Plan the sword-cave legs on the early graph (empty initial inventory)."""
    g = graph or build_early_route_graph()
    return g.plan_legs(sword_cave_route_legs(), initial_capabilities=frozenset())


def level1_route_plan(graph: RouteGraph | None = None):
    """Plan sword + overworld path into Level 1."""
    g = graph or build_early_route_graph()
    return g.plan_legs(level1_route_legs(), initial_capabilities=frozenset())


def level1_first_key_route_plan(graph: RouteGraph | None = None):
    """Plan sword + overworld + Level 1 first-key segment."""
    g = graph or build_early_route_graph()
    return g.plan_legs(
        level1_first_key_route_legs(),
        initial_capabilities=frozenset(),
    )


def level1_north_route_plan(graph: RouteGraph | None = None):
    """Plan power-on through the first locked Level 1 door."""
    g = graph or build_early_route_graph()
    return g.plan_legs(
        level1_north_route_legs(),
        initial_capabilities=frozenset(),
    )


def level1_clear63_route_plan(graph: RouteGraph | None = None):
    """Plan power-on through the room 0x63 Stalfos clear."""
    g = graph or build_early_route_graph()
    return g.plan_legs(
        level1_clear63_route_legs(),
        initial_capabilities=frozenset(),
    )


def level1_clear53_route_plan(graph: RouteGraph | None = None):
    """Plan power-on through the room 0x53 Stalfos clear and room key."""
    g = graph or build_early_route_graph()
    return g.plan_legs(
        level1_clear53_route_legs(),
        initial_capabilities=frozenset(),
    )


def level1_clear54_route_plan(graph: RouteGraph | None = None):
    """Plan power-on through the observed east branch room 0x54 clear."""
    g = graph or build_early_route_graph()
    return g.plan_legs(
        level1_clear54_route_legs(),
        initial_capabilities=frozenset(),
    )


def level1_complete_route_plan(graph: RouteGraph | None = None):
    """Plan power-on through the verified Level 1 completion route."""
    g = graph or build_early_route_graph()
    return g.plan_legs(
        level1_complete_route_legs(),
        initial_capabilities=frozenset(),
    )


def level2_path_prefix_route_legs() -> tuple[RouteLeg, ...]:
    """Legs from Level 1 complete through the verified walk prefix to 0x4A."""
    return (
        *level1_complete_route_legs(),
        RouteLeg(
            leg_id="settle_post_triforce_overworld",
            source_id=NODE_LEVEL1_COMPLETE,
            target_id=NODE_LEVEL1_EXIT_OVERWORLD,
            requires=frozenset({"wooden_sword", "triforce_shard_1"}),
            goal="overworld_0x37_after_triforce",
        ),
        RouteLeg(
            leg_id="walk_level2_path_prefix",
            source_id=NODE_LEVEL1_EXIT_OVERWORLD,
            target_id=NODE_LEVEL2_PATH_4A,
            requires=frozenset({"wooden_sword", "triforce_shard_1"}),
            goal="reach_screen_4A_post_triforce",
        ),
    )


def level2_path_prefix_route_plan(graph: RouteGraph | None = None):
    """Plan power-on through Level 1 and the verified Level 2 walk prefix."""
    g = graph or build_early_route_graph()
    return g.plan_legs(
        level2_path_prefix_route_legs(),
        initial_capabilities=frozenset(),
    )
