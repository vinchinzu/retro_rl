"""Grid pathfinding and capability-aware room-graph search."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable, Mapping, Sequence

from super_metroid.rooms.capabilities import normalize_ability


def _nearest_open(
    collision: Sequence[Sequence[int]],
    point: tuple[int, int],
) -> tuple[int, int] | None:
    if not collision or not collision[0]:
        return None
    width = len(collision[0])
    height = len(collision)
    start = (
        min(max(point[0], 0), width - 1),
        min(max(point[1], 0), height - 1),
    )
    queue = deque([start])
    seen = {start}
    while queue:
        x, y = queue.popleft()
        if int(collision[y][x]) == 0:
            return x, y
        for nxt in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            nx, ny = nxt
            if 0 <= nx < width and 0 <= ny < height and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return None


def _grid_path(
    collision: Sequence[Sequence[int]],
    start: tuple[int, int],
    target: tuple[int, int],
) -> list[tuple[int, int]] | None:
    source = _nearest_open(collision, start)
    goal = _nearest_open(collision, target)
    if source is None or goal is None:
        return None
    queue = deque([source])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {source: None}
    width = len(collision[0])
    height = len(collision)
    while queue and goal not in parent:
        x, y = queue.popleft()
        for nxt in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            nx, ny = nxt
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if nxt in parent or int(collision[ny][nx]) != 0:
                continue
            parent[nxt] = (x, y)
            queue.append(nxt)
    if goal not in parent:
        return None
    path = []
    cursor: tuple[int, int] | None = goal
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    return list(reversed(path))


def _compress_path(path: Sequence[tuple[int, int]]) -> list[list[int]]:
    if len(path) <= 2:
        return [list(point) for point in path]
    result = [path[0]]
    previous_direction = (
        path[1][0] - path[0][0],
        path[1][1] - path[0][1],
    )
    for index in range(1, len(path) - 1):
        direction = (
            path[index + 1][0] - path[index][0],
            path[index + 1][1] - path[index][1],
        )
        if direction != previous_direction:
            result.append(path[index])
        previous_direction = direction
    result.append(path[-1])
    return [list(point) for point in result]


def _capability_path(
    edges: Sequence[Mapping[str, object]],
    source: int,
    target: int,
    capabilities: set[str],
) -> list[Mapping[str, object]] | None:
    if source == target:
        return []
    outgoing: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for edge in edges:
        outgoing[int(edge["source"]["roomId"])].append(edge)
    queue = deque([source])
    parent: dict[int, tuple[int, Mapping[str, object]]] = {}
    seen = {source}
    while queue:
        room_id = queue.popleft()
        for edge in outgoing.get(room_id, []):
            if edge.get("impossible"):
                continue
            if not set(edge.get("requires", [])).issubset(capabilities):
                continue
            next_room = int(edge["target"]["roomId"])
            if next_room in seen:
                continue
            seen.add(next_room)
            parent[next_room] = (room_id, edge)
            if next_room == target:
                path: list[Mapping[str, object]] = []
                cursor = target
                while cursor != source:
                    previous, used = parent[cursor]
                    path.append(used)
                    cursor = previous
                return list(reversed(path))
            queue.append(next_room)
    return None


def shortest_room_path(
    graph: Mapping[str, object],
    source_room_id: int,
    target_room_id: int,
    capabilities: Iterable[str] = (),
) -> list[Mapping[str, object]] | None:
    normalized = {normalize_ability(value) for value in capabilities}
    return _capability_path(
        graph["edges"],
        source_room_id,
        target_room_id,
        normalized,
    )
