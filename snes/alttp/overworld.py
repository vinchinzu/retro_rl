"""Light-world overworld screen graph and castle route helpers."""

from __future__ import annotations

from collections import deque

from alttp.ram import (
    HYRULE_CASTLE_SCREEN,
    LINKS_HOUSE_ROOM,
    LINKS_HOUSE_SCREEN,
    AlttpSnapshot,
)

SCREEN_LABELS = {
    0x13: "sanctuary",
    0x1B: "hyrule_castle",
    0x1C: "castle_approach",
    0x24: "north_field",
    0x2C: "links_house",
}

# Escape Link's House porch before heading north.
LINKS_HOUSE_ESCAPE_CLEAR_Y = 2936
LINKS_HOUSE_ESCAPE_X_MIN = 2386
LINKS_HOUSE_ESCAPE_X_MAX = 2402


def screen_to_grid(screen_id: int) -> tuple[int, int]:
    return divmod(int(screen_id) & 0x3F, 8)


def _neighbor_screens(screen_id: int) -> dict[str, int | None]:
    local = int(screen_id) & 0x3F
    base = int(screen_id) & ~0x3F
    row, col = divmod(local, 8)
    return {
        "north": (base + local - 8) if row > 0 else None,
        "south": (base + local + 8) if row < 7 else None,
        "west": (base + local - 1) if col > 0 else None,
        "east": (base + local + 1) if col < 7 else None,
    }


def shortest_screen_path(start_screen: int, target_screen: int) -> list[int]:
    """BFS on the 8×8 light/dark overworld grid (same world only)."""
    start = int(start_screen)
    target = int(target_screen)
    if (start & ~0x3F) != (target & ~0x3F):
        raise ValueError("Cross-world paths are not supported")
    if start == target:
        return [start]

    queue: deque[int] = deque([start])
    parents: dict[int, int | None] = {start: None}
    while queue:
        current = queue.popleft()
        for neighbor in _neighbor_screens(current).values():
            if neighbor is None or neighbor in parents:
                continue
            parents[neighbor] = current
            if neighbor == target:
                path = [target]
                node = target
                while parents[node] is not None:
                    node = int(parents[node])
                    path.append(node)
                path.reverse()
                return path
            queue.append(neighbor)
    raise ValueError(f"No path from {start:#04x} to {target:#04x}")


def next_screen_in_path(current_screen: int, target_screen: int) -> int:
    path = shortest_screen_path(current_screen, target_screen)
    return path[0] if len(path) < 2 else path[1]


def direction_to_screen(current_screen: int, target_screen: int) -> str | None:
    current_row, current_col = screen_to_grid(current_screen)
    target_row, target_col = screen_to_grid(target_screen)
    if current_row > target_row:
        return "UP"
    if current_row < target_row:
        return "DOWN"
    if current_col > target_col:
        return "LEFT"
    if current_col < target_col:
        return "RIGHT"
    return None


def is_links_house_room(snapshot: AlttpSnapshot) -> bool:
    return snapshot.room_base_id == LINKS_HOUSE_ROOM


def next_direction_to_hyrule_castle(snapshot: AlttpSnapshot) -> str | None:
    """Return one cardinal direction toward castle grounds, or None."""
    if snapshot.indoors and is_links_house_room(snapshot):
        return "DOWN"
    if snapshot.indoors or snapshot.dark_world:
        return None
    if snapshot.screen_id == LINKS_HOUSE_SCREEN:
        if snapshot.link_x < LINKS_HOUSE_ESCAPE_X_MIN:
            if snapshot.link_y < LINKS_HOUSE_ESCAPE_CLEAR_Y:
                return "DOWN"
            return "RIGHT"
        if snapshot.link_x > LINKS_HOUSE_ESCAPE_X_MAX:
            return "LEFT"
        return "UP"
    next_screen = next_screen_in_path(snapshot.screen_id, HYRULE_CASTLE_SCREEN)
    return direction_to_screen(snapshot.screen_id, next_screen)


def on_hyrule_castle_screen(snapshot: AlttpSnapshot) -> bool:
    return (
        (not snapshot.indoors)
        and (not snapshot.dark_world)
        and snapshot.screen_id == HYRULE_CASTLE_SCREEN
    )
