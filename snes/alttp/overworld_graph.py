"""Light- and dark-world screen graph plus overworld entrances.

Screen nodes are map ``screenId`` values: light ``0x00``–``0x3F``, dark
``0x40``–``0x7F`` (same split as ``$008A`` on the swamp exit probe, screen
123). Cardinal edges stay inside one world. There is no wrap.

This is candidate screen topology only. It does not inspect terrain, tile
collisions, loaded 2×2 screen areas, stair transitions, or the player's exact
position. A path is a possible screen sequence, not evidence that Link can
navigate it.

Entrance edges are z3 connections whose endpoints touch a LightWorld or
DarkWorld room. A whole-world region name (the Light World room, a dark
region room) is not a teleporter: walking inside that region is the screen
grid. A name is tied to a screen only from an outdoor map (``z3Label`` +
outdoor ``screenId``) or from ``ram`` screen constants when no map exists
yet. Approach xy stays in the map files.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from alttp.paths import MAPS_DIR
from alttp.ram import LINKS_HOUSE_SCREEN, SANCTUARY_SCREEN
from alttp.z3_json_data import Z3JsonData

_OW_TYPES = frozenset({"LightWorld", "DarkWorld"})
# room_0e Ice door was a light-world warp of the pin, not the dark screen.
_WARPED_NOTE_MARKERS = ("not dw", "module_predungeon", "lw-warped")
# No outdoor sanctuary or Link's house map yet. Sanctuary 0x13 is the
# project constant, not a new live scan.
_CONSTANT_SCREENS: dict[str, int] = {
    "Sanctuary": SANCTUARY_SCREEN,
    "Sanctuary Exit": SANCTUARY_SCREEN,
    "Links House": LINKS_HOUSE_SCREEN,
    "Links House Exit": LINKS_HOUSE_SCREEN,
}


@dataclass(frozen=True)
class Entrance:
    """One z3 connection that touches the overworld."""

    connection_type: str
    origin: str
    destination: str


def screen_token(screen_id: int) -> str:
    return f"screen:{int(screen_id) & 0xFF:02X}"


def parse_token(node: str | int) -> str:
    if isinstance(node, int):
        return screen_token(node)
    text = str(node)
    if text.lower().startswith("screen:"):
        return screen_token(int(text.split(":", 1)[1], 16))
    return text


def screen_neighbors(screen_id: int) -> tuple[int, ...]:
    """Cardinal neighbors inside the same 8×8 world. No wrap, no cross-world."""
    screen = int(screen_id)
    if not 0 <= screen <= 0x7F:
        raise ValueError(f"screen {screen_id:#x} is outside light/dark 0x00-0x7F")
    local = screen & 0x3F
    base = screen & ~0x3F
    row, col = divmod(local, 8)
    out: list[int] = []
    if row > 0:
        out.append(base + local - 8)
    if row < 7:
        out.append(base + local + 8)
    if col > 0:
        out.append(base + local - 1)
    if col < 7:
        out.append(base + local + 1)
    return tuple(out)


def _warped_note(notes: object) -> bool:
    text = str(notes or "").lower()
    return any(marker in text for marker in _WARPED_NOTE_MARKERS)


def _add_screen(found: dict[str, set[int]], label: object, screen: object) -> None:
    if not isinstance(label, str) or not label:
        return
    if isinstance(screen, bool) or not isinstance(screen, int):
        return
    found.setdefault(label, set()).add(int(screen))
    found.setdefault(f"{label} Exit", set()).add(int(screen))


def measured_screen_bindings(maps_dir: Path | None = None) -> dict[str, int]:
    """Bind z3 labels to a screen only when every outdoor map agrees."""
    root = MAPS_DIR if maps_dir is None else maps_dir
    found: dict[str, set[int]] = {}
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        label = data.get("z3Label")
        if data.get("outdoors") is True:
            _add_screen(found, label, data.get("screenId"))
        for door in data.get("doors") or []:
            if not isinstance(door, dict) or door.get("outdoors") is not True:
                continue
            if _warped_note(door.get("notes")):
                continue
            _add_screen(found, label, door.get("screenId"))
    bindings: dict[str, int] = {}
    for name, screens in found.items():
        if len(screens) == 1:
            bindings[name] = next(iter(screens))
    return bindings


def _name_kinds(data: Z3JsonData) -> dict[str, set[str]]:
    kinds: dict[str, set[str]] = {}
    for room in data.rooms:
        kinds.setdefault(room.name, set()).add(room.room_type)
        for node in room.nodes:
            kinds.setdefault(node.name, set()).add(room.room_type)
    return kinds


def _link(adj: dict[str, set[str]], left: str, right: str) -> None:
    if left == right:
        adj.setdefault(left, set())
        return
    adj.setdefault(left, set()).add(right)
    adj.setdefault(right, set()).add(left)


class OverworldGraph:
    """Candidate screen topology, with z3 entrance labels attached to screens.

    The graph does not establish tile-level reachability or account for which
    2×2 screen area is currently loaded by the game. Use map geometry and a
    real predecessor run to establish navigation.
    """

    def __init__(
        self,
        neighbors: dict[int, tuple[int, ...]],
        bindings: dict[str, int],
        entrances: tuple[Entrance, ...],
        adj: dict[str, set[str]],
    ) -> None:
        self.neighbors = neighbors
        self.bindings = bindings
        self.entrances = entrances
        self._adj = adj

    @classmethod
    def load(cls, *, data: Z3JsonData | None = None) -> OverworldGraph:
        loaded = data if data is not None else Z3JsonData.load()
        kinds = _name_kinds(loaded)
        # Overworld region rooms are areas, not single tiles. Linking every
        # exit to that room name would teleport across the world.
        region_rooms = {
            room.name
            for room in loaded.rooms
            if room.room_type in _OW_TYPES
        }
        bindings = dict(_CONSTANT_SCREENS)
        bindings.update(measured_screen_bindings())
        neighbors = {screen: screen_neighbors(screen) for screen in range(0x80)}
        adj: dict[str, set[str]] = {screen_token(screen): set() for screen in neighbors}
        for screen, nbs in neighbors.items():
            for other in nbs:
                _link(adj, screen_token(screen), screen_token(other))
        entrances: list[Entrance] = []
        for conn in loaded.connections:
            origin_ow = bool(kinds.get(conn.origin, set()) & _OW_TYPES)
            dest_ow = bool(kinds.get(conn.destination, set()) & _OW_TYPES)
            if not origin_ow and not dest_ow:
                continue
            entrances.append(
                Entrance(conn.connection_type, conn.origin, conn.destination)
            )
            if conn.origin in region_rooms or conn.destination in region_rooms:
                continue
            _link(adj, conn.origin, conn.destination)
        for name, screen in bindings.items():
            if name in kinds:
                _link(adj, name, screen_token(screen))
        return cls(neighbors, bindings, tuple(entrances), adj)

    def screen_of(self, name: str) -> int | None:
        return self.bindings.get(name)

    def path(self, start: str | int, goal: str | int) -> list[str]:
        """Return a shortest candidate topological path, not a walk plan.

        Screen ids are tokens ``screen:XX``. Cardinal screen adjacency ignores
        terrain and the game's loaded 2×2 screen area, so callers must not
        treat this result as tile-level reachability evidence.
        """
        src = parse_token(start)
        dst = parse_token(goal)
        if src == dst:
            return [src]
        if src not in self._adj or dst not in self._adj:
            raise ValueError(f"no path from {src} to {dst}")
        queue: deque[str] = deque([src])
        previous: dict[str, str | None] = {src: None}
        while queue:
            current = queue.popleft()
            for nxt in self._adj[current]:
                if nxt in previous:
                    continue
                previous[nxt] = current
                if nxt == dst:
                    out = [dst]
                    parent = previous[dst]
                    while parent is not None:
                        out.append(parent)
                        parent = previous[parent]
                    out.reverse()
                    return out
                queue.append(nxt)
        raise ValueError(f"no path from {src} to {dst}")
