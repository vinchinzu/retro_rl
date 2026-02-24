"""Auto-generate waypoints for speedrun segments from map data.

Combines the world graph (inter-room path) with room navigators
(intra-room screen-level BFS) to produce waypoint lists compatible
with platformer_common's WaypointTracker.
"""

from __future__ import annotations

from super_metroid_rl.navigation.map_data import WorldData
from super_metroid_rl.navigation.room_navigator import RoomNavigator
from super_metroid_rl.navigation.route import RouteStep, get_route_step, SPEEDRUN_ROUTE


def _to_float_tuples(
    waypoints: list[tuple[int, int] | tuple[float, float]],
) -> list[tuple[float, float]]:
    """Ensure all waypoints are (float, float) tuples."""
    return [(float(x), float(y)) for x, y in waypoints]


def generate_segment_waypoints(
    world: WorldData,
    segment_id: str,
    entry_room_id: int,
    entry_pixel: tuple[int, int],
    exit_room_id: int,
    exit_pixel: tuple[int, int] | None = None,
) -> list[tuple[float, float]]:
    """Generate (x, y) waypoints for a speedrun segment.

    For single-room segments, generates intra-room waypoints from entry
    to exit door. For multi-room segments (rare), concatenates waypoints
    through each room.

    Args:
        world: Loaded world data with rooms and nav graph
        segment_id: Segment identifier (for logging)
        entry_room_id: Room ID where Samus starts
        entry_pixel: Approximate (x, y) pixel position where Samus spawns
        exit_room_id: Room ID Samus must reach (0 = stay in room)
        exit_pixel: Override for exit position (used when door isn't in room data)

    Returns:
        List of (x, y) float waypoint tuples for WaypointTracker
    """
    if exit_room_id == 0:
        # Item collection segment — no exit door to navigate to
        room = world.rooms.get(entry_room_id)
        if room:
            target = (room.width_blocks * 16 - 32, entry_pixel[1])
            return _to_float_tuples([entry_pixel, target])
        return _to_float_tuples([entry_pixel, (entry_pixel[0] + 256, entry_pixel[1])])

    room = world.rooms.get(entry_room_id)
    if not room:
        return _to_float_tuples([entry_pixel, entry_pixel])

    nav = RoomNavigator(room)

    # Find exit position: explicit override > room door data > direction estimate
    target: tuple[int, int] | None = exit_pixel
    if target is None:
        door_pos = nav.find_door_position(exit_room_id)
        if door_pos is not None:
            target = door_pos

    if target is None:
        # Estimate exit position from the route direction
        override = EXIT_OVERRIDES.get(segment_id)
        if override:
            target = override
        else:
            target = _estimate_exit_position(room, segment_id)

    # Generate screen-level waypoints
    raw_waypoints = nav.screen_path(entry_pixel, target)
    return _to_float_tuples(raw_waypoints)


def _estimate_exit_position(room, segment_id: str) -> tuple[int, int]:
    """Estimate exit position when door data is missing."""
    room_w = room.width_blocks * 16
    room_h = room.height_blocks * 16
    # Default: bottom-center for descent, top-center for ascent
    if "descent" in segment_id or "down" in segment_id:
        return (room_w // 2, room_h - 48)
    elif "return" in segment_id or "up" in segment_id:
        return (room_w // 2, 48)
    elif "right" in segment_id or "flyway" in segment_id:
        return (room_w - 48, room_h // 2)
    return (room_w // 2, room_h // 2)


def _estimate_entry_pixel(
    world: WorldData,
    room_id: int,
    from_direction: str | None,
) -> tuple[int, int]:
    """Estimate where Samus spawns when entering a room."""
    room = world.rooms.get(room_id)
    if not room:
        return (128, 128)

    room_w = room.width_blocks * 16
    room_h = room.height_blocks * 16

    if from_direction == "Left":
        return (room_w - 48, room_h // 2)
    elif from_direction == "Right":
        return (48, room_h // 2)
    elif from_direction == "Up":
        return (room_w // 2, room_h - 48)
    elif from_direction == "Down":
        return (room_w // 2, 48)

    return (room_w // 2, room_h // 2)


# Hard-coded entry positions for segments where the auto-estimate is wrong.
ENTRY_OVERRIDES: dict[str, tuple[int, int]] = {
    # Landing Site: Samus spawns mid-room near ship (bottom half, screen row ~3)
    "sm_landing_site": (1100, 900),
    # Parlor: entering from Landing Site (right door), top-right area
    "sm_parlor_descent": (1260, 128),
    # Parlor return: entering from Climb (bottom door), bottom-left
    "sm_parlor_to_flyway": (384, 1260),
    # Morph ball collect: entering from elevator room, top area
    "sm_morph_ball_collect": (1400, 32),
}

# Hard-coded exit positions for segments where the door isn't in room data
# (patched world graph edges, morph tunnels, fall-throughs, etc.)
EXIT_OVERRIDES: dict[str, tuple[int, int]] = {
    # Climb → Pit Room: door blocks at col 31, rows 134-137 (bottom-right interior)
    "sm_climb_descent": (504, 2176),
    # Pit Room → Climb: return through top-left area
    "sm_pit_room_return": (8, 128),
    # Climb return → Parlor: up through top door
    "sm_climb_return": (384, 8),
    # Parlor → Flyway: morph tunnel exit on right side, mid-height
    "sm_parlor_to_flyway": (1260, 900),
}


def generate_all_segment_waypoints(
    world: WorldData,
) -> dict[str, list[tuple[float, float]]]:
    """Generate waypoints for all segments in the speedrun route.

    Returns a dict mapping segment_id to waypoint list.
    """
    result: dict[str, list[tuple[float, float]]] = {}

    for step in SPEEDRUN_ROUTE:
        if step.segment_id in ENTRY_OVERRIDES:
            entry = ENTRY_OVERRIDES[step.segment_id]
        else:
            # Estimate entry position from route context
            idx = SPEEDRUN_ROUTE.index(step)
            if idx > 0:
                prev = SPEEDRUN_ROUTE[idx - 1]
                room = world.rooms.get(step.entry_room_id)
                if room:
                    # Find the door back to previous room to estimate entry point
                    for door in room.doors:
                        if door.dest_room_id == prev.entry_room_id:
                            entry = (door.pixel_x, door.pixel_y)
                            break
                    else:
                        entry = _estimate_entry_pixel(world, step.entry_room_id, None)
                else:
                    entry = _estimate_entry_pixel(world, step.entry_room_id, None)
            else:
                entry = _estimate_entry_pixel(world, step.entry_room_id, None)

        waypoints = generate_segment_waypoints(
            world, step.segment_id, step.entry_room_id, entry, step.exit_room_id,
        )
        result[step.segment_id] = waypoints

    return result


def needs_waypoints(segment_id: str, world: WorldData) -> bool:
    """Check if a segment would benefit from waypoint tracking.

    Simple single-direction corridors (1-screen wide rooms, elevators)
    work fine with monotonic axis tracking. Multi-screen rooms with
    direction changes need waypoints.
    """
    step = get_route_step(segment_id)
    if not step or step.exit_room_id == 0:
        return False

    room = world.rooms.get(step.entry_room_id)
    if not room:
        return False

    # Simple rooms: 1 screen in any dimension, or very narrow
    if room.width_screens <= 1 or room.height_screens <= 1:
        return False

    # Multi-screen rooms that are taller and wider than 2 screens
    # likely have non-linear paths
    if room.width_screens >= 3 and room.height_screens >= 3:
        return True

    return False
