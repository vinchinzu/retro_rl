"""Super Metroid navigation system.

Public API for map data loading, inter-room pathfinding, and
automatic waypoint generation for the hill climbing optimizer.

Usage:
    from super_metroid_rl.navigation import load_world, WorldGraph, generate_segment_waypoints

    world = load_world()
    graph = WorldGraph(world)
    path = graph.find_path(0x91F8, 0x9804)  # Landing Site → Bomb Torizo
"""

from super_metroid_rl.navigation.map_data import (
    WorldData,
    RoomData,
    DoorInfo,
    NavEdge,
    NavNode,
    load_world,
    load_room,
    load_nav_graph,
    DEFAULT_EXPORT_DIR,
)
from super_metroid_rl.navigation.world_graph import WorldGraph, PathStep
from super_metroid_rl.navigation.room_navigator import RoomNavigator
from super_metroid_rl.navigation.waypoint_gen import (
    generate_segment_waypoints,
    generate_all_segment_waypoints,
    needs_waypoints,
)
from super_metroid_rl.navigation.route import (
    RouteStep,
    SPEEDRUN_ROUTE,
    get_route_step,
    route_summary,
)
from super_metroid_rl.navigation.seed_synth import (
    synthesize_seed,
    preview_seed,
    SEGMENT_HINTS,
)
from super_metroid_rl.navigation.trace_renderer import (
    render_trace_on_map,
    detect_area,
)

__all__ = [
    # Data types
    "WorldData",
    "RoomData",
    "DoorInfo",
    "NavEdge",
    "NavNode",
    "PathStep",
    "RouteStep",
    # Loaders
    "load_world",
    "load_room",
    "load_nav_graph",
    "DEFAULT_EXPORT_DIR",
    # Graph / navigation
    "WorldGraph",
    "RoomNavigator",
    # Waypoint generation
    "generate_segment_waypoints",
    "generate_all_segment_waypoints",
    "needs_waypoints",
    # Route
    "SPEEDRUN_ROUTE",
    "get_route_step",
    "route_summary",
    # Seed synthesis
    "synthesize_seed",
    "preview_seed",
    "SEGMENT_HINTS",
    # Trace rendering
    "render_trace_on_map",
    "detect_area",
]
