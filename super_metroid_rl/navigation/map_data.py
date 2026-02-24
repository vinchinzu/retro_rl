"""Dataclasses and JSON parser for SMEDIT-exported map data.

Loads room collision grids, door connectivity, and the navigation graph
from the JSON files exported by the Super Metroid editor CLI.

Default data directory: /tmp/sm_export/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# SM uses 16x16 pixel blocks, 16x16 block screens
BLOCK_SIZE = 16
SCREEN_WIDTH_BLOCKS = 16
SCREEN_HEIGHT_BLOCKS = 16
SCREEN_WIDTH_PX = SCREEN_WIDTH_BLOCKS * BLOCK_SIZE   # 256
SCREEN_HEIGHT_PX = SCREEN_HEIGHT_BLOCKS * BLOCK_SIZE  # 256

# Collision tile types
TILE_AIR = 0x0
TILE_SLOPE = 0x1
TILE_SOLID = 0x8
TILE_DOOR = 0x9
TILE_SPIKE = 0xA
TILE_CRUMBLE = 0xB
TILE_BOMB = 0xF

# Passable tiles (Samus can move through these)
PASSABLE_TILES = {TILE_AIR, TILE_SLOPE, TILE_DOOR, TILE_CRUMBLE, TILE_SPIKE}

DEFAULT_EXPORT_DIR = Path("/tmp/sm_export")


@dataclass(frozen=True)
class DoorInfo:
    """A door connecting this room to another."""
    dest_room_id: int
    dest_room_handle: str
    direction: str              # "Left", "Right", "Up", "Down"
    required_ability: str | None
    door_cap_color: str | None
    is_elevator: bool
    pixel_x: int                # Center of door block cluster (pixel coords)
    pixel_y: int                # Center of door block cluster (pixel coords)


@dataclass(frozen=True)
class RoomData:
    """Parsed room data from a room JSON file."""
    room_id: int
    handle: str
    name: str
    area: int
    area_name: str
    width_screens: int
    height_screens: int
    width_blocks: int
    height_blocks: int
    collision: list[list[int]]  # [row][col], row=Y top-down, col=X
    bts: list[list[int]]
    doors: list[DoorInfo]
    items: list[dict]
    enemies: list[dict]


@dataclass(frozen=True)
class NavEdge:
    """A directed edge in the inter-room navigation graph."""
    from_room_id: int
    to_room_id: int
    direction: str              # "Left", "Right", "Up", "Down"
    is_elevator: bool
    required_ability: str | None
    door_cap_color: str | None


@dataclass(frozen=True)
class NavNode:
    """A node in the navigation graph (room metadata without collision)."""
    room_id: int
    handle: str
    name: str
    area: int
    area_name: str
    map_x: int
    map_y: int
    width_screens: int
    height_screens: int


@dataclass
class WorldData:
    """All loaded map data: rooms, nodes, and edges."""
    rooms: dict[int, RoomData] = field(default_factory=dict)
    nodes: dict[int, NavNode] = field(default_factory=dict)
    edges: list[NavEdge] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Door position detection from collision grid
# ---------------------------------------------------------------------------

def _find_door_clusters(collision: list[list[int]], width_blocks: int, height_blocks: int) -> list[dict]:
    """Find contiguous clusters of 0x9 (door) blocks and determine their side.

    Returns list of {side, blocks, center_x, center_y, bts_values} dicts.
    Side is one of "Left", "Right", "Up", "Down", or "Interior".
    """
    # Collect all door block positions
    door_positions: set[tuple[int, int]] = set()
    for r in range(height_blocks):
        for c in range(width_blocks):
            if collision[r][c] == TILE_DOOR:
                door_positions.add((r, c))

    if not door_positions:
        return []

    # Flood-fill to group contiguous clusters
    clusters: list[list[tuple[int, int]]] = []
    remaining = set(door_positions)

    while remaining:
        seed = next(iter(remaining))
        cluster: list[tuple[int, int]] = []
        stack = [seed]
        while stack:
            pos = stack.pop()
            if pos not in remaining:
                continue
            remaining.discard(pos)
            cluster.append(pos)
            r, c = pos
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in remaining:
                    stack.append((nr, nc))
        clusters.append(cluster)

    # Determine side for each cluster
    results = []
    for cluster in clusters:
        rows = [r for r, c in cluster]
        cols = [c for r, c in cluster]
        min_r, max_r = min(rows), max(rows)
        min_c, max_c = min(cols), max(cols)

        # Center pixel position
        center_x = int((min_c + max_c) / 2.0 * BLOCK_SIZE + BLOCK_SIZE / 2)
        center_y = int((min_r + max_r) / 2.0 * BLOCK_SIZE + BLOCK_SIZE / 2)

        # Determine which room edge this cluster touches
        on_left = min_c == 0
        on_right = max_c == width_blocks - 1
        on_top = min_r == 0
        on_bottom = max_r == height_blocks - 1

        if on_left and not on_right:
            side = "Left"
        elif on_right and not on_left:
            side = "Right"
        elif on_top and not on_bottom:
            side = "Up"
        elif on_bottom and not on_top:
            side = "Down"
        else:
            side = "Interior"

        results.append({
            "side": side,
            "blocks": cluster,
            "center_x": center_x,
            "center_y": center_y,
        })

    return results


def _match_doors_to_clusters(
    door_entries: list[dict],
    clusters: list[dict],
    bts: list[list[int]],
) -> list[DoorInfo]:
    """Match JSON door entries to detected door clusters.

    Strategy:
    1. Match by direction (Left door entry → cluster on Left edge)
    2. When multiple clusters share a side, use BTS values to disambiguate
       (each door has a unique BTS index)
    """
    doors: list[DoorInfo] = []

    # Group clusters by side
    from collections import defaultdict
    side_clusters: dict[str, list[dict]] = defaultdict(list)
    for cluster in clusters:
        side_clusters[cluster["side"]].append(cluster)
    # Interior clusters can match any direction (e.g. elevator shafts)
    interior = side_clusters.pop("Interior", [])

    for entry in door_entries:
        direction = entry["direction"]
        candidates = list(side_clusters.get(direction, []))

        # Also consider interior clusters (for elevators, morph tunnels)
        if entry.get("isElevator", False) or not candidates:
            candidates.extend(interior)

        if not candidates:
            # No matching cluster found - skip this door
            continue

        if len(candidates) == 1:
            matched = candidates[0]
        else:
            # Disambiguate by BTS: pick the cluster whose BTS values
            # are closest to the door's index in the door list
            # BTS for door blocks encodes the door index
            door_idx = door_entries.index(entry)
            best_cluster = candidates[0]
            best_score = float("inf")
            for cand in candidates:
                bts_vals = set()
                for r, c in cand["blocks"]:
                    bts_vals.add(bts[r][c])
                # BTS values often encode door index directly
                score = min(abs(bv - door_idx) for bv in bts_vals) if bts_vals else float("inf")
                if score < best_score:
                    best_score = score
                    best_cluster = cand
            matched = best_cluster

        doors.append(DoorInfo(
            dest_room_id=entry["destRoomId"],
            dest_room_handle=entry.get("destRoomHandle", ""),
            direction=direction,
            required_ability=entry.get("requiredAbility"),
            door_cap_color=entry.get("doorCapColor"),
            is_elevator=entry.get("isElevator", False),
            pixel_x=matched["center_x"],
            pixel_y=matched["center_y"],
        ))

    return doors


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_room(path: Path) -> RoomData:
    """Load a single room JSON file."""
    with open(path) as f:
        data = json.load(f)

    collision = data["collision"]
    bts_data = data["bts"]
    width_blocks = data["widthBlocks"]
    height_blocks = data["heightBlocks"]

    # Detect door positions from collision grid
    clusters = _find_door_clusters(collision, width_blocks, height_blocks)
    doors = _match_doors_to_clusters(data.get("doors", []), clusters, bts_data)

    return RoomData(
        room_id=data["roomId"],
        handle=data.get("handle", ""),
        name=data.get("name", ""),
        area=data.get("area", 0),
        area_name=data.get("areaName", ""),
        width_screens=data["widthScreens"],
        height_screens=data["heightScreens"],
        width_blocks=width_blocks,
        height_blocks=height_blocks,
        collision=collision,
        bts=bts_data,
        doors=doors,
        items=data.get("items", []),
        enemies=data.get("enemies", []),
    )


def load_nav_graph(path: Path) -> tuple[list[NavNode], list[NavEdge]]:
    """Load the navigation graph JSON."""
    with open(path) as f:
        data = json.load(f)

    nodes = []
    for n in data["nodes"]:
        nodes.append(NavNode(
            room_id=n["roomId"],
            handle=n.get("handle", ""),
            name=n.get("name", ""),
            area=n.get("area", 0),
            area_name=n.get("areaName", ""),
            map_x=n.get("mapX", 0),
            map_y=n.get("mapY", 0),
            width_screens=n.get("widthScreens", 1),
            height_screens=n.get("heightScreens", 1),
        ))

    edges = []
    for e in data["edges"]:
        edges.append(NavEdge(
            from_room_id=e["fromRoomId"],
            to_room_id=e["toRoomId"],
            direction=e.get("direction", ""),
            is_elevator=e.get("isElevator", False),
            required_ability=e.get("requiredAbility"),
            door_cap_color=e.get("doorCapColor"),
        ))

    return nodes, edges


def load_world(export_dir: Path | str = DEFAULT_EXPORT_DIR) -> WorldData:
    """Load all map data from the export directory.

    Loads:
    - nav_graph.json for nodes and edges
    - rooms/*.json for collision grids and door positions
    """
    export_dir = Path(export_dir)
    world = WorldData()

    # Load nav graph
    nav_path = export_dir / "nav_graph.json"
    if nav_path.exists():
        nodes, edges = load_nav_graph(nav_path)
        for node in nodes:
            world.nodes[node.room_id] = node
        world.edges = edges

    # Load room files
    rooms_dir = export_dir / "rooms"
    if rooms_dir.exists():
        for room_file in sorted(rooms_dir.glob("room_*.json")):
            room = load_room(room_file)
            world.rooms[room.room_id] = room

    return world


def iter_room_files(export_dir: Path | str = DEFAULT_EXPORT_DIR) -> Iterator[Path]:
    """Iterate over room JSON files in the export directory."""
    rooms_dir = Path(export_dir) / "rooms"
    if rooms_dir.exists():
        yield from sorted(rooms_dir.glob("room_*.json"))
