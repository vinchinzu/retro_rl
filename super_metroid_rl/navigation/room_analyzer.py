#!/usr/bin/env python3
"""Analyze room collision grids to find optimal traversal paths.

Uses exported SM room data to:
1. Render collision maps with recording paths overlaid
2. Find the clearest vertical/horizontal paths through rooms
3. Generate waypoints for optimal traversal

Usage:
    uv run python -m super_metroid_rl.navigation.room_analyzer 0x92FD --fall
    uv run python -m super_metroid_rl.navigation.room_analyzer 0x92FD --overlay recording.json
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

SM_EXPORT = Path("/tmp/sm_export")

# Block type constants
AIR = 0x0
SLOPE = 0x1
SOLID = 0x8
DOOR = 0x9
SPIKE = 0xA
CRUMBLE = 0xB
BOMB = 0xF

BLOCKING = {SOLID, DOOR, SPIKE, BOMB}  # Samus can't pass through these
CHARS = {AIR: " ", SLOPE: "/", SOLID: "#", DOOR: "D", SPIKE: "^", CRUMBLE: ".", BOMB: "B"}

# Samus's hitbox in blocks (approximate)
SAMUS_WIDTH_BLOCKS = 2  # ~32 pixels wide
SAMUS_HEIGHT_BLOCKS = 3  # ~48 pixels tall


@dataclass
class RoomCollision:
    """Parsed room collision data."""

    room_id: int
    name: str
    width_screens: int
    height_screens: int
    width_blocks: int
    height_blocks: int
    grid: list[list[int]]  # [row][col], row=Y top-down

    @classmethod
    def load(cls, room_id: int) -> RoomCollision:
        path = SM_EXPORT / "rooms" / f"room_{room_id:04X}.json"
        with open(path) as f:
            data = json.load(f)
        return cls(
            room_id=room_id,
            name=data["name"],
            width_screens=data["widthScreens"],
            height_screens=data["heightScreens"],
            width_blocks=data.get("widthBlocks", data["widthScreens"] * 16),
            height_blocks=data.get("heightBlocks", data["heightScreens"] * 16),
            grid=data["collision"],
        )

    def is_solid(self, row: int, col: int) -> bool:
        """Check if a block is solid (blocks Samus)."""
        if row < 0 or row >= self.height_blocks or col < 0 or col >= self.width_blocks:
            return True  # out of bounds = solid
        return self.grid[row][col] in BLOCKING

    def count_solids_in_column(self, col: int, row_start: int = 0, row_end: int | None = None) -> int:
        """Count solid blocks in a vertical column range."""
        if row_end is None:
            row_end = self.height_blocks
        return sum(1 for r in range(row_start, row_end) if self.is_solid(r, col))

    def find_clear_fall_columns(
        self, row_start: int, row_end: int, col_start: int = 0, col_end: int | None = None,
    ) -> list[tuple[int, int, list[int]]]:
        """Find columns with fewest obstructions for vertical falling.

        Returns: list of (col, solid_count, solid_rows) sorted by solid_count.
        """
        if col_end is None:
            col_end = self.width_blocks
        results = []
        for col in range(col_start, col_end):
            solid_rows = [r for r in range(row_start, row_end) if self.is_solid(r, col)]
            results.append((col, len(solid_rows), solid_rows))
        results.sort(key=lambda x: x[1])
        return results

    def find_optimal_fall_path(
        self, col_start: int, col_end: int, row_start: int, row_end: int,
    ) -> list[tuple[int, int]]:
        """Find path through room minimizing time on platforms.

        Uses BFS on a (col, row) grid where movement is:
        - Fall down (free, gravity)
        - Walk left/right on solid ground
        - Each platform landing costs extra time

        Returns: list of (col, row) waypoints.
        """
        from collections import deque

        # State: (col, row), cost = frames approximation
        # Falling: 1 frame per block (simplified)
        # Walking: 2 frames per block (slower than falling)
        # Landing penalty: ~30 frames (deceleration + re-jump)

        INF = float("inf")
        best = {}  # (col, row) -> min cost
        parent = {}  # (col, row) -> (prev_col, prev_row)
        queue = deque()

        # Start from all columns in the start range
        for col in range(col_start, col_end):
            if not self.is_solid(row_start, col):
                queue.append((col, row_start, 0))
                best[(col, row_start)] = 0

        while queue:
            col, row, cost = queue.popleft()
            if cost > best.get((col, row), INF):
                continue

            # Reached bottom?
            if row >= row_end - 1:
                continue

            # Try falling down
            next_row = row + 1
            if not self.is_solid(next_row, col):
                new_cost = cost + 1  # 1 frame to fall 1 block
                if new_cost < best.get((col, next_row), INF):
                    best[(col, next_row)] = new_cost
                    parent[(col, next_row)] = (col, row)
                    queue.append((col, next_row, new_cost))
            else:
                # Landed on platform - try walking left/right + jumping
                for dc in [-1, 1]:
                    for walk_dist in range(1, 6):  # walk up to 5 blocks
                        nc = col + dc * walk_dist
                        if nc < col_start or nc >= col_end:
                            break
                        if self.is_solid(row, nc):
                            break
                        # Can we fall from this new column?
                        if not self.is_solid(next_row, nc):
                            # Landing penalty + walk cost
                            new_cost = cost + 30 + walk_dist * 2
                            if new_cost < best.get((nc, next_row), INF):
                                best[(nc, next_row)] = new_cost
                                parent[(nc, next_row)] = (col, row)
                                queue.append((nc, next_row, new_cost))

        # Find best endpoint
        best_end = None
        best_cost = INF
        for col in range(col_start, col_end):
            c = best.get((col, row_end - 1), INF)
            if c < best_cost:
                best_cost = c
                best_end = (col, row_end - 1)

        if best_end is None:
            return []

        # Trace path
        path = [best_end]
        pos = best_end
        while pos in parent:
            pos = parent[pos]
            path.append(pos)
        path.reverse()

        # Convert to pixel waypoints, keeping only direction changes
        waypoints = []
        for col, row in path:
            px = col * 16 + 8
            py = row * 16 + 8
            if not waypoints or (waypoints[-1][0] != px):
                waypoints.append((px, py))
            else:
                waypoints[-1] = (px, py)  # update y for same column

        return waypoints

    def render_ascii(
        self,
        col_start: int = 0,
        col_end: int | None = None,
        row_start: int = 0,
        row_end: int | None = None,
        path_pixels: list[tuple[int, int]] | None = None,
    ) -> str:
        """Render collision grid as ASCII art."""
        if col_end is None:
            col_end = self.width_blocks
        if row_end is None:
            row_end = self.height_blocks

        # Convert pixel path to block coords
        path_blocks: set[tuple[int, int]] = set()
        if path_pixels:
            for px, py in path_pixels:
                path_blocks.add((int(py) // 16, int(px) // 16))

        lines = []
        lines.append(f"Room: {self.name} (0x{self.room_id:04X})")
        lines.append(f"Size: {self.width_screens}x{self.height_screens} screens, "
                      f"{self.width_blocks}x{self.height_blocks} blocks")
        lines.append(f"Showing cols {col_start}-{col_end-1} (x={col_start*16}-{col_end*16})")
        lines.append("")

        header = "Row  Y_px  |"
        for c in range(col_start, col_end):
            header += str(c % 10)
        header += "|"
        lines.append(header)

        for row in range(row_start, row_end):
            y_px = row * 16
            line = f"{row:3d}  {y_px:4d}  |"
            for col in range(col_start, col_end):
                val = self.grid[row][col]
                if (row, col) in path_blocks:
                    line += "*"
                else:
                    line += CHARS.get(val, "?")
            line += "|"
            lines.append(line)

        return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze SM room collision data")
    parser.add_argument("room_id", help="Room ID in hex (e.g., 0x92FD)")
    parser.add_argument("--fall", action="store_true", help="Analyze vertical fall paths")
    parser.add_argument("--cols", help="Column range (e.g., 18-32)")
    parser.add_argument("--rows", help="Row range (e.g., 0-80)")
    parser.add_argument("--path", action="store_true", help="Find optimal fall path")
    parser.add_argument("--overlay", help="JSON recording to overlay path from")
    args = parser.parse_args()

    room_id = int(args.room_id, 16) if args.room_id.startswith("0x") else int(args.room_id)
    room = RoomCollision.load(room_id)

    col_start = 0
    col_end = room.width_blocks
    row_start = 0
    row_end = room.height_blocks

    if args.cols:
        parts = args.cols.split("-")
        col_start, col_end = int(parts[0]), int(parts[1]) + 1
    if args.rows:
        parts = args.rows.split("-")
        row_start, row_end = int(parts[0]), int(parts[1]) + 1

    if args.fall:
        results = room.find_clear_fall_columns(row_start, row_end, col_start, col_end)
        print(f"\nFall column analysis for {room.name}:")
        print(f"{'Col':>4s}  {'X_px':>5s}  {'Solids':>6s}  Solid rows")
        for col, count, rows in results[:15]:
            print(f"{col:4d}  {col*16+8:5d}  {count:6d}  {rows}")

    if args.path:
        waypoints = room.find_optimal_fall_path(col_start, col_end, row_start, row_end)
        print(f"\nOptimal fall path ({len(waypoints)} waypoints):")
        for px, py in waypoints:
            print(f"  ({px}, {py})")

    path_pixels = None
    if args.overlay:
        # Load recording and extract player positions
        # (would need to replay through emulator - for now just show the grid)
        pass

    print(room.render_ascii(col_start, col_end, row_start, row_end, path_pixels))


if __name__ == "__main__":
    main()
