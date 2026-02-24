"""Inter-room BFS pathfinding through the SM world graph.

Uses nav_graph.json edges to find shortest paths between rooms,
respecting ability gates (door cap colors, required abilities).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from super_metroid_rl.navigation.map_data import NavEdge, WorldData


@dataclass(frozen=True)
class PathStep:
    """One step in an inter-room path."""
    room_id: int
    room_name: str
    direction: str  # Direction taken to reach next room ("" for final step)
    is_elevator: bool


# Manual patches for edges missing from the exported nav_graph.
# These are connections through morph tunnels, fall-throughs, or other
# non-standard transitions that the SMEDIT exporter doesn't capture.
ROUTE_PATCHES: list[NavEdge] = [
    # Parlor → Flyway: internal morph tunnel (not a standard door)
    NavEdge(0x92FD, 0x9879, "Right", False, "morph_ball", None),
    # Flyway → Parlor: return through morph tunnel
    NavEdge(0x9879, 0x92FD, "Left", False, "morph_ball", None),
    # Climb → Pit Room: bottom-right door (not in nav_graph, but exists in game)
    NavEdge(0x96BA, 0x975C, "Down", False, None, None),
    # Pit Room → Climb: return path (nav_graph has this with boss_event, add free version)
    NavEdge(0x975C, 0x96BA, "Up", False, None, None),
]


class WorldGraph:
    """Directed graph over rooms with BFS pathfinding.

    Edges come from nav_graph.json + manual patches. Each edge may
    require an ability (e.g., "morph_ball", "power_bomb") to traverse.
    """

    def __init__(self, world: WorldData) -> None:
        self._world = world
        # Build adjacency list: room_id -> [(dest_room_id, edge)]
        self._adj: dict[int, list[tuple[int, NavEdge]]] = defaultdict(list)
        for edge in world.edges:
            self._adj[edge.from_room_id].append((edge.to_room_id, edge))
        # Apply manual patches
        for edge in ROUTE_PATCHES:
            self._adj[edge.from_room_id].append((edge.to_room_id, edge))

    @property
    def world(self) -> WorldData:
        return self._world

    def room_name(self, room_id: int) -> str:
        """Get the display name for a room."""
        node = self._world.nodes.get(room_id)
        if node:
            return node.name
        room = self._world.rooms.get(room_id)
        if room:
            return room.name
        return f"0x{room_id:04X}"

    def neighbors(self, room_id: int, abilities: set[str] | None = None) -> list[tuple[int, NavEdge]]:
        """Get reachable neighbors from a room, filtered by abilities."""
        result = []
        for dest, edge in self._adj.get(room_id, []):
            if edge.required_ability and abilities is not None:
                if edge.required_ability not in abilities:
                    continue
            result.append((dest, edge))
        return result

    def find_path(
        self,
        from_room: int,
        to_room: int,
        abilities: set[str] | None = None,
    ) -> list[PathStep] | None:
        """BFS shortest path from one room to another.

        Args:
            from_room: Starting room ID
            to_room: Destination room ID
            abilities: Set of abilities Samus has. Edges requiring an ability
                not in this set are excluded. None means all edges are passable.

        Returns:
            List of PathStep objects from source to destination (inclusive),
            or None if no path exists.
        """
        if from_room == to_room:
            return [PathStep(from_room, self.room_name(from_room), "", False)]

        # BFS
        visited: set[int] = {from_room}
        # parent[room_id] = (prev_room_id, edge_used)
        parent: dict[int, tuple[int, NavEdge]] = {}
        queue: deque[int] = deque([from_room])

        while queue:
            current = queue.popleft()
            for dest, edge in self.neighbors(current, abilities):
                if dest in visited:
                    continue
                visited.add(dest)
                parent[dest] = (current, edge)
                if dest == to_room:
                    # Reconstruct path
                    path: list[PathStep] = []
                    node = to_room
                    while node in parent:
                        prev, edge_used = parent[node]
                        path.append(PathStep(
                            room_id=node,
                            room_name=self.room_name(node),
                            direction="",  # filled below
                            is_elevator=edge_used.is_elevator,
                        ))
                        node = prev
                    path.append(PathStep(from_room, self.room_name(from_room), "", False))
                    path.reverse()

                    # Fill direction fields (direction taken to reach next step)
                    for i in range(len(path) - 1):
                        step = path[i]
                        next_step = path[i + 1]
                        # Find the edge connecting these rooms
                        for dest2, edge2 in self._adj.get(step.room_id, []):
                            if dest2 == next_step.room_id:
                                path[i] = PathStep(
                                    step.room_id,
                                    step.room_name,
                                    edge2.direction,
                                    edge2.is_elevator,
                                )
                                break
                    return path
                queue.append(dest)

        return None  # No path found

    def find_all_paths(
        self,
        from_room: int,
        to_room: int,
        abilities: set[str] | None = None,
        max_depth: int = 20,
    ) -> list[list[PathStep]]:
        """Find all simple paths up to max_depth (for debugging)."""
        results: list[list[PathStep]] = []

        def _dfs(current: int, visited: set[int], path: list[int]) -> None:
            if len(path) > max_depth:
                return
            if current == to_room:
                # Convert to PathSteps
                steps = []
                for i, room_id in enumerate(path):
                    direction = ""
                    is_elev = False
                    if i < len(path) - 1:
                        for dest, edge in self._adj.get(room_id, []):
                            if dest == path[i + 1]:
                                direction = edge.direction
                                is_elev = edge.is_elevator
                                break
                    steps.append(PathStep(room_id, self.room_name(room_id), direction, is_elev))
                results.append(steps)
                return
            for dest, edge in self.neighbors(current, abilities):
                if dest not in visited:
                    visited.add(dest)
                    path.append(dest)
                    _dfs(dest, visited, path)
                    path.pop()
                    visited.discard(dest)

        _dfs(from_room, {from_room}, [from_room])
        return results
