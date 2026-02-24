"""Allow running as: python -m super_metroid_rl

Extends the platformer_common CLI with Super Metroid navigation commands:
  nav-path       Find inter-room path between two rooms
  nav-room       Show intra-room screen path
  nav-waypoints  Generate waypoints for a speedrun segment
  nav-info       Show room information (doors, dimensions, etc.)
"""

import sys


def _is_nav_command() -> bool:
    """Check if the first positional arg is a nav-* command."""
    for arg in sys.argv[1:]:
        if arg.startswith("-"):
            continue
        return arg.startswith("nav-")
    return False


def _parse_room_id(s: str) -> int:
    """Parse a room ID from hex string like '0x91F8' or '91F8' or decimal."""
    s = s.strip()
    if s.startswith("0x") or s.startswith("0X"):
        return int(s, 16)
    try:
        return int(s, 16)
    except ValueError:
        return int(s)


def _run_nav() -> None:
    """Handle nav-* subcommands."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Super Metroid Navigation Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # nav-path
    p_path = sub.add_parser("nav-path", help="Find inter-room path")
    p_path.add_argument("--from", dest="from_room", required=True, help="Start room ID (hex)")
    p_path.add_argument("--to", dest="to_room", required=True, help="End room ID (hex)")
    p_path.add_argument("--abilities", nargs="*", default=None,
                        help="Available abilities (e.g. morph_ball missile)")
    p_path.add_argument("--data-dir", default="/tmp/sm_export", help="Export data directory")

    # nav-room
    p_room = sub.add_parser("nav-room", help="Show intra-room screen path")
    p_room.add_argument("--room", required=True, help="Room ID (hex)")
    p_room.add_argument("--entry", required=True, help="Entry direction: left/right/up/down or x,y pixel")
    p_room.add_argument("--exit", required=True, help="Exit direction or dest room ID (hex) or x,y pixel")
    p_room.add_argument("--data-dir", default="/tmp/sm_export")

    # nav-waypoints
    p_wp = sub.add_parser("nav-waypoints", help="Generate waypoints for a segment")
    p_wp.add_argument("--segment", required=True, help="Segment ID (e.g. parlor_descent)")
    p_wp.add_argument("--data-dir", default="/tmp/sm_export")

    # nav-info
    p_info = sub.add_parser("nav-info", help="Show room information")
    p_info.add_argument("--room", required=True, help="Room ID (hex)")
    p_info.add_argument("--data-dir", default="/tmp/sm_export")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    from super_metroid_rl.navigation import (
        load_world,
        WorldGraph,
        RoomNavigator,
        generate_segment_waypoints,
        generate_all_segment_waypoints,
        route_summary,
        SPEEDRUN_ROUTE,
        get_route_step,
    )

    data_dir = Path(args.data_dir)
    world = load_world(data_dir)

    if args.command == "nav-path":
        _cmd_nav_path(args, world)
    elif args.command == "nav-room":
        _cmd_nav_room(args, world)
    elif args.command == "nav-waypoints":
        _cmd_nav_waypoints(args, world)
    elif args.command == "nav-info":
        _cmd_nav_info(args, world)


def _cmd_nav_path(args, world) -> None:
    from super_metroid_rl.navigation import WorldGraph

    graph = WorldGraph(world)
    from_id = _parse_room_id(args.from_room)
    to_id = _parse_room_id(args.to_room)
    abilities = set(args.abilities) if args.abilities else None

    abilities_str = ", ".join(sorted(abilities)) if abilities else "all (no filter)"
    print(f"Path: 0x{from_id:04X} ({graph.room_name(from_id)}) → "
          f"0x{to_id:04X} ({graph.room_name(to_id)})")
    print(f"Abilities: {abilities_str}")
    print()

    path = graph.find_path(from_id, to_id, abilities=abilities)
    if path is None:
        print("No path found!")
        return

    print(f"Route ({len(path)} rooms):")
    for i, step in enumerate(path):
        dir_str = f" → {step.direction}" if step.direction else ""
        elev = " [ELEVATOR]" if step.is_elevator else ""
        print(f"  {i+1}. 0x{step.room_id:04X} {step.room_name}{dir_str}{elev}")


def _cmd_nav_room(args, world) -> None:
    from super_metroid_rl.navigation import RoomNavigator

    room_id = _parse_room_id(args.room)
    room = world.rooms.get(room_id)
    if not room:
        print(f"Room 0x{room_id:04X} not found in room data")
        return

    nav = RoomNavigator(room)
    print(f"Room: 0x{room_id:04X} {room.name} ({room.width_screens}x{room.height_screens} screens)")
    print(f"Doors: {len(room.doors)}")
    for d in room.doors:
        print(f"  → 0x{d.dest_room_id:04X} ({d.dest_room_handle}) "
              f"dir={d.direction} at ({d.pixel_x},{d.pixel_y})")

    # Parse entry/exit positions
    entry = _parse_position(args.entry, room, nav, is_entry=True)
    exit_pos = _parse_position(args.exit, room, nav, is_entry=False)

    print(f"\nEntry: ({entry[0]}, {entry[1]}) screen={nav.pixel_to_screen(*entry)}")
    print(f"Exit:  ({exit_pos[0]}, {exit_pos[1]}) screen={nav.pixel_to_screen(*exit_pos)}")

    waypoints = nav.screen_path(entry, exit_pos)
    print(f"\nWaypoints ({len(waypoints)}):")
    for i, (x, y) in enumerate(waypoints):
        print(f"  {i}: ({x}, {y}) screen={nav.pixel_to_screen(x, y)}")

    print(f"\nScreen adjacency:")
    print(nav.screen_adjacency_str())


def _parse_position(s: str, room, nav, is_entry: bool) -> tuple[int, int]:
    """Parse a position string: 'left', 'right', 'up', 'down', 'x,y', or hex room ID."""
    s = s.strip().lower()
    room_w = room.width_blocks * 16
    room_h = room.height_blocks * 16

    if s == "left":
        return (48, room_h // 2)
    elif s == "right":
        return (room_w - 48, room_h // 2)
    elif s in ("up", "top"):
        return (room_w // 2, 48)
    elif s in ("down", "bottom"):
        return (room_w // 2, room_h - 48)
    elif "," in s:
        parts = s.split(",")
        return (int(parts[0]), int(parts[1]))
    else:
        # Try as room ID for door lookup
        try:
            dest_id = _parse_room_id(s)
            pos = nav.find_door_position(dest_id)
            if pos:
                return pos
        except ValueError:
            pass
        return (room_w // 2, room_h // 2)


def _cmd_nav_waypoints(args, world) -> None:
    from super_metroid_rl.navigation import (
        generate_all_segment_waypoints,
        get_route_step,
        needs_waypoints,
        SPEEDRUN_ROUTE,
        route_summary,
    )

    segment = args.segment
    # Allow shorthand: "parlor_descent" → "sm_parlor_descent"
    if not segment.startswith("sm_"):
        segment = "sm_" + segment

    step = get_route_step(segment)
    if step is None:
        print(f"Unknown segment: {segment}")
        print(f"\nAvailable segments:")
        for s in SPEEDRUN_ROUTE:
            print(f"  {s.segment_id}")
        return

    all_wps = generate_all_segment_waypoints(world)
    waypoints = all_wps.get(segment, [])

    needs = needs_waypoints(segment, world)
    room = world.rooms.get(step.entry_room_id)
    room_info = f"{room.width_screens}x{room.height_screens}" if room else "?"

    print(f"Segment: {segment}")
    print(f"Room: 0x{step.entry_room_id:04X} ({room_info} screens)")
    print(f"Exit: 0x{step.exit_room_id:04X}" if step.exit_room_id else "Exit: item collect")
    print(f"Needs waypoints: {needs}")
    print(f"\nWaypoints ({len(waypoints)}):")
    for i, (x, y) in enumerate(waypoints):
        print(f"  {i}: ({x:.0f}, {y:.0f})")

    # Print in config-ready format
    if waypoints:
        print(f"\nConfig format:")
        wp_str = ", ".join(f"({x:.0f}, {y:.0f})" for x, y in waypoints)
        print(f"  waypoints=[{wp_str}]")


def _cmd_nav_info(args, world) -> None:
    from super_metroid_rl.navigation import RoomNavigator

    room_id = _parse_room_id(args.room)

    # Show node info
    node = world.nodes.get(room_id)
    if node:
        print(f"Room: 0x{room_id:04X} {node.name}")
        print(f"Area: {node.area_name} ({node.area})")
        print(f"Map position: ({node.map_x}, {node.map_y})")
        print(f"Size: {node.width_screens}x{node.height_screens} screens")
    else:
        print(f"Room 0x{room_id:04X} not found in nav graph")

    # Show room data
    room = world.rooms.get(room_id)
    if room:
        print(f"\nCollision grid: {room.width_blocks}x{room.height_blocks} blocks")
        print(f"Doors ({len(room.doors)}):")
        for d in room.doors:
            ability_str = f" [{d.required_ability}]" if d.required_ability else ""
            cap_str = f" ({d.door_cap_color})" if d.door_cap_color else ""
            print(f"  → 0x{d.dest_room_id:04X} ({d.dest_room_handle}) "
                  f"dir={d.direction} at ({d.pixel_x},{d.pixel_y}){ability_str}{cap_str}")

        print(f"Items: {len(room.items)}")
        print(f"Enemies: {len(room.enemies)}")

        nav = RoomNavigator(room)
        print(f"\nScreen adjacency:")
        print(nav.screen_adjacency_str())
    else:
        print(f"No room collision data available")

    # Show edges from nav graph
    from_edges = [e for e in world.edges if e.from_room_id == room_id]
    to_edges = [e for e in world.edges if e.to_room_id == room_id]
    if from_edges:
        print(f"\nOutgoing edges:")
        for e in from_edges:
            name = world.nodes.get(e.to_room_id)
            name_str = name.name if name else f"0x{e.to_room_id:04X}"
            ability_str = f" [{e.required_ability}]" if e.required_ability else ""
            print(f"  → {name_str} dir={e.direction}{ability_str}")
    if to_edges:
        print(f"\nIncoming edges:")
        for e in to_edges:
            name = world.nodes.get(e.from_room_id)
            name_str = name.name if name else f"0x{e.from_room_id:04X}"
            ability_str = f" [{e.required_ability}]" if e.required_ability else ""
            print(f"  ← {name_str} dir={e.direction}{ability_str}")


if _is_nav_command():
    _run_nav()
else:
    import platformer_common.levels.super_metroid  # noqa: F401 - trigger registration
    from platformer_common.runner import main
    main(default_level="sm_landing_site")
