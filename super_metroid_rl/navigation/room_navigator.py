"""Intra-room screen-level waypoint generation.

Divides a room into screen tiles, builds a screen adjacency graph based
on passable boundaries, and BFS to generate coarse waypoints from entry
to exit door.

These waypoints guide the hill climber through non-linear rooms without
needing full physics simulation — the optimizer handles actual platforming
within each screen.
"""

from __future__ import annotations

from collections import deque

from super_metroid_rl.navigation.map_data import (
    BLOCK_SIZE,
    PASSABLE_TILES,
    SCREEN_HEIGHT_BLOCKS,
    SCREEN_WIDTH_BLOCKS,
    SCREEN_HEIGHT_PX,
    SCREEN_WIDTH_PX,
    RoomData,
)


class RoomNavigator:
    """Navigate within a single room using screen-level BFS.

    Screens are 16x16 blocks (256x256 px). Two screens connect if their
    shared boundary has at least one passable block pair.
    """

    def __init__(self, room: RoomData) -> None:
        self._room = room
        self._w = room.width_screens
        self._h = room.height_screens
        # Build screen adjacency once
        self._adj = self._build_screen_adjacency()

    @property
    def room(self) -> RoomData:
        return self._room

    def _build_screen_adjacency(self) -> dict[tuple[int, int], list[tuple[int, int]]]:
        """Build adjacency graph between screens.

        Two horizontally adjacent screens connect if any block pair on their
        shared vertical boundary is passable on both sides. Similarly for
        vertical adjacency.
        """
        adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for sy in range(self._h):
            for sx in range(self._w):
                neighbors: list[tuple[int, int]] = []

                # Check right neighbor
                if sx + 1 < self._w:
                    if self._screens_connected_h(sx, sy, sx + 1, sy):
                        neighbors.append((sx + 1, sy))

                # Check left neighbor
                if sx - 1 >= 0:
                    if self._screens_connected_h(sx - 1, sy, sx, sy):
                        neighbors.append((sx - 1, sy))

                # Check down neighbor
                if sy + 1 < self._h:
                    if self._screens_connected_v(sx, sy, sx, sy + 1):
                        neighbors.append((sx, sy + 1))

                # Check up neighbor
                if sy - 1 >= 0:
                    if self._screens_connected_v(sx, sy - 1, sx, sy):
                        neighbors.append((sx, sy - 1))

                adj[(sx, sy)] = neighbors

        return adj

    def _screens_connected_h(self, left_sx: int, left_sy: int, right_sx: int, right_sy: int) -> bool:
        """Check if two horizontally adjacent screens are connected.

        Tests the rightmost column of the left screen against the leftmost
        column of the right screen.
        """
        collision = self._room.collision
        # Right edge of left screen
        col_r = left_sx * SCREEN_WIDTH_BLOCKS + SCREEN_WIDTH_BLOCKS - 1
        # Left edge of right screen
        col_l = right_sx * SCREEN_WIDTH_BLOCKS
        # Row range for this screen row
        row_start = left_sy * SCREEN_HEIGHT_BLOCKS
        row_end = min(row_start + SCREEN_HEIGHT_BLOCKS, self._room.height_blocks)

        for row in range(row_start, row_end):
            if col_r < self._room.width_blocks and col_l < self._room.width_blocks:
                if collision[row][col_r] in PASSABLE_TILES and collision[row][col_l] in PASSABLE_TILES:
                    return True
        return False

    def _screens_connected_v(self, top_sx: int, top_sy: int, bot_sx: int, bot_sy: int) -> bool:
        """Check if two vertically adjacent screens are connected.

        Tests the bottom row of the top screen against the top row of the
        bottom screen.
        """
        collision = self._room.collision
        # Bottom edge of top screen
        row_b = top_sy * SCREEN_HEIGHT_BLOCKS + SCREEN_HEIGHT_BLOCKS - 1
        # Top edge of bottom screen
        row_t = bot_sy * SCREEN_HEIGHT_BLOCKS
        # Column range for this screen column
        col_start = top_sx * SCREEN_WIDTH_BLOCKS
        col_end = min(col_start + SCREEN_WIDTH_BLOCKS, self._room.width_blocks)

        for col in range(col_start, col_end):
            if row_b < self._room.height_blocks and row_t < self._room.height_blocks:
                if collision[row_b][col] in PASSABLE_TILES and collision[row_t][col] in PASSABLE_TILES:
                    return True
        return False

    def pixel_to_screen(self, px: int, py: int) -> tuple[int, int]:
        """Convert pixel coordinates to screen coordinates."""
        sx = max(0, min(px // SCREEN_WIDTH_PX, self._w - 1))
        sy = max(0, min(py // SCREEN_HEIGHT_PX, self._h - 1))
        return sx, sy

    def screen_center(self, sx: int, sy: int) -> tuple[int, int]:
        """Get the pixel center of a screen."""
        cx = sx * SCREEN_WIDTH_PX + SCREEN_WIDTH_PX // 2
        cy = sy * SCREEN_HEIGHT_PX + SCREEN_HEIGHT_PX // 2
        return cx, cy

    def find_door_position(self, dest_room_id: int) -> tuple[int, int] | None:
        """Find the pixel position of the door leading to dest_room_id."""
        for door in self._room.doors:
            if door.dest_room_id == dest_room_id:
                return door.pixel_x, door.pixel_y
        return None

    def screen_path(
        self,
        start_px: tuple[int, int],
        target_px: tuple[int, int],
    ) -> list[tuple[int, int]]:
        """BFS from start pixel to target pixel, returning screen-center waypoints.

        Returns a list of (pixel_x, pixel_y) waypoints at screen centers
        along the shortest path. The first waypoint is the start position
        and the last is the target position.
        """
        start_screen = self.pixel_to_screen(*start_px)
        target_screen = self.pixel_to_screen(*target_px)

        if start_screen == target_screen:
            # Same screen: just start → target
            return [start_px, target_px]

        # BFS through screen graph
        visited = {start_screen}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        queue: deque[tuple[int, int]] = deque([start_screen])
        found = False

        while queue and not found:
            current = queue.popleft()
            for neighbor in self._adj.get(current, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                parent[neighbor] = current
                if neighbor == target_screen:
                    found = True
                    break
                queue.append(neighbor)

        if target_screen not in parent and start_screen != target_screen:
            # No path found — fall back to direct line
            return [start_px, target_px]

        # Reconstruct screen path
        screen_path: list[tuple[int, int]] = []
        node = target_screen
        while node in parent:
            screen_path.append(node)
            node = parent[node]
        screen_path.append(start_screen)
        screen_path.reverse()

        # Convert to pixel waypoints
        waypoints: list[tuple[int, int]] = [start_px]
        for screen in screen_path[1:-1]:
            waypoints.append(self.screen_center(*screen))
        waypoints.append(target_px)

        return waypoints

    def screen_adjacency_str(self) -> str:
        """Debug string showing screen adjacency."""
        lines = []
        for sy in range(self._h):
            for sx in range(self._w):
                neighbors = self._adj.get((sx, sy), [])
                dirs = []
                for nx, ny in neighbors:
                    if nx > sx:
                        dirs.append("R")
                    elif nx < sx:
                        dirs.append("L")
                    elif ny > sy:
                        dirs.append("D")
                    elif ny < sy:
                        dirs.append("U")
                lines.append(f"  ({sx},{sy}): {','.join(dirs) or 'none'}")
        return "\n".join(lines)
