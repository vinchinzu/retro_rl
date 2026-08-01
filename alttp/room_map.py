"""Measured room geometry: JSON schema + load/save (single authority).

Geometry lives in ``alttp/maps/room_XX.json``. Runtime sensing/overlays stay
in :mod:`alttp.room_sense`; clear/path/door play stays in
:mod:`alttp.opening_route.room_engine`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from alttp.paths import MAPS_DIR

__all__ = [
    "MAPS_DIR",
    "ClearPolicy",
    "KnownDoor",
    "RoomMap",
    "RoomMapPoint",
    "list_room_maps",
    "load_room_map",
    "room_map_path",
    "save_room_map",
]


@dataclass(frozen=True)
class RoomMapPoint:
    """Named measured point inside a room (for pathing / overlays)."""

    label: str
    x: int
    y: int
    role: str = "waypoint"  # waypoint | approach | edge | spawn
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "xy": [self.x, self.y],
            "role": self.role,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoomMapPoint:
        xy = data.get("xy") or [data.get("x", 0), data.get("y", 0)]
        return cls(
            label=str(data["label"]),
            x=int(xy[0]),
            y=int(xy[1]),
            role=str(data.get("role") or "waypoint"),
            notes=str(data.get("notes") or ""),
        )


@dataclass(frozen=True)
class KnownDoor:
    """Static measured door / exit for a room map (geometry authority)."""

    label: str
    direction: str  # LEFT | RIGHT | UP | DOWN
    to_room: int | None = None
    approach_xy: tuple[int, int] = (0, 0)
    landing_xy: tuple[int, int] | None = None
    outdoors: bool = False
    screen_id: int | None = None
    role: str = "primary"  # zelda_path | alternate | backtrack | primary
    path: tuple[str, ...] = ()  # ordered point labels → approach
    # Optional fallback after the primary path has physically wedged.  These
    # labels are still map geometry; room_engine only decides when to use them.
    recovery_path: tuple[str, ...] = ()
    path_tolerances: Mapping[str, int] = field(default_factory=dict)
    push_frames: int = 300
    notes: str = ""

    def tolerance_for(self, point_label: str, default: int = 12) -> int:
        if point_label in self.path_tolerances:
            return int(self.path_tolerances[point_label])
        if "default" in self.path_tolerances:
            return int(self.path_tolerances["default"])
        return default

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "direction": self.direction,
            "toRoom": self.to_room,
            "toRoomHex": (
                f"0x{self.to_room & 0xFF:02X}" if self.to_room is not None else None
            ),
            "approachXy": list(self.approach_xy),
            "landingXy": list(self.landing_xy) if self.landing_xy else None,
            "outdoors": self.outdoors,
            "screenId": self.screen_id,
            "role": self.role,
            "path": list(self.path),
            "recoveryPath": list(self.recovery_path),
            "pathTolerances": dict(self.path_tolerances),
            "pushFrames": self.push_frames,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> KnownDoor:
        approach = data.get("approachXy") or data.get("approach_xy") or [0, 0]
        landing_raw = data.get("landingXy", data.get("landing_xy"))
        landing = None
        if landing_raw is not None:
            landing = (int(landing_raw[0]), int(landing_raw[1]))
        to_room = data.get("toRoom", data.get("to_room"))
        path = data.get("path") or ()
        recovery_path = data.get("recoveryPath", data.get("recovery_path")) or ()
        tols = data.get("pathTolerances") or data.get("path_tolerances") or {}
        return cls(
            label=str(data["label"]),
            direction=str(data["direction"]).upper(),
            to_room=None if to_room is None else int(to_room),
            approach_xy=(int(approach[0]), int(approach[1])),
            landing_xy=landing,
            outdoors=bool(data.get("outdoors", False)),
            screen_id=(
                None
                if data.get("screenId", data.get("screen_id")) is None
                else int(data.get("screenId", data.get("screen_id")))
            ),
            role=str(data.get("role") or "primary"),
            path=tuple(str(p) for p in path),
            recovery_path=tuple(str(p) for p in recovery_path),
            path_tolerances={str(k): int(v) for k, v in dict(tols).items()},
            push_frames=int(data.get("pushFrames", data.get("push_frames", 300))),
            notes=str(data.get("notes") or ""),
        )


@dataclass(frozen=True)
class ClearPolicy:
    """Combat bounds for room clear + corridor skirmish."""

    max_distance: int = 180
    attack_distance: int = 50
    max_cycles: int = 350
    skirmish_max_distance: int = 90
    skirmish_pad: int = 12
    skirmish_max_cycles: int = 80

    def to_dict(self) -> dict[str, Any]:
        return {
            "maxDistance": self.max_distance,
            "attackDistance": self.attack_distance,
            "maxCycles": self.max_cycles,
            "skirmishMaxDistance": self.skirmish_max_distance,
            "skirmishPad": self.skirmish_pad,
            "skirmishMaxCycles": self.skirmish_max_cycles,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ClearPolicy:
        if not data:
            return cls()
        return cls(
            max_distance=int(data.get("maxDistance", data.get("max_distance", 180))),
            attack_distance=int(
                data.get("attackDistance", data.get("attack_distance", 50))
            ),
            max_cycles=int(data.get("maxCycles", data.get("max_cycles", 350))),
            skirmish_max_distance=int(
                data.get("skirmishMaxDistance", data.get("skirmish_max_distance", 90))
            ),
            skirmish_pad=int(data.get("skirmishPad", data.get("skirmish_pad", 12))),
            skirmish_max_cycles=int(
                data.get("skirmishMaxCycles", data.get("skirmish_max_cycles", 80))
            ),
        )


@dataclass
class RoomMap:
    """Static measured layout for one dungeon room base id.

    Loaded from ``alttp/maps/room_XX.json`` — single geometry authority.
    """

    room_base_id: int
    name: str
    points: tuple[RoomMapPoint, ...] = ()
    doors: tuple[KnownDoor, ...] = ()
    clear_policy: ClearPolicy = field(default_factory=ClearPolicy)
    notes: tuple[str, ...] = ()
    walk_bbox: tuple[int, int, int, int] | None = None  # x0,y0,x1,y1
    source_state: str = ""
    measured: str = ""
    hostiles_at_entry: tuple[dict[str, Any], ...] = ()
    map_id: str = ""  # e.g. room_61

    def point(self, label: str) -> RoomMapPoint | None:
        for p in self.points:
            if p.label == label:
                return p
        return None

    def door(self, label: str) -> KnownDoor | None:
        for d in self.doors:
            if d.label == label:
                return d
        return None

    def door_by_role(self, role: str) -> KnownDoor | None:
        for d in self.doors:
            if d.role == role:
                return d
        return None

    def waypoints_for_door(
        self, door: KnownDoor | str
    ) -> tuple[tuple[int, int, str, int], ...]:
        """Ordered (x, y, label, tolerance) for door path, ending at approach."""
        d = self.door(door) if isinstance(door, str) else door
        if d is None:
            return ()
        out: list[tuple[int, int, str, int]] = []
        for label in d.path or ():
            pt = self.point(label)
            if pt is None:
                continue
            out.append((pt.x, pt.y, label, d.tolerance_for(label)))
        # Ensure door approach coords are last (geometry authority: door.approach_xy).
        ax, ay = d.approach_xy
        if not out or (out[-1][0], out[-1][1]) != (ax, ay):
            out.append((ax, ay, f"{d.label}_approach", d.tolerance_for("default")))
        return tuple(out)

    def recovery_waypoints_for_door(
        self, door: KnownDoor | str
    ) -> tuple[tuple[int, int, str, int], ...]:
        """Fallback path for a measured door wedge, ending at its approach."""
        d = self.door(door) if isinstance(door, str) else door
        if d is None or not d.recovery_path:
            return ()
        out: list[tuple[int, int, str, int]] = []
        for label in d.recovery_path:
            pt = self.point(label)
            if pt is None:
                continue
            out.append((pt.x, pt.y, label, d.tolerance_for(label)))
        ax, ay = d.approach_xy
        if not out or (out[-1][0], out[-1][1]) != (ax, ay):
            out.append((ax, ay, f"{d.label}_approach", d.tolerance_for("default")))
        return tuple(out)

    def compact_summary(self) -> dict[str, Any]:
        """Small dict for agent context (avoid dumping full segment code)."""
        return {
            "mapId": self.map_id or f"room_{self.room_base_id:02x}",
            "roomHex": f"0x{self.room_base_id & 0xFF:02X}",
            "name": self.name,
            "sourceState": self.source_state,
            "doors": [
                {
                    "label": d.label,
                    "dir": d.direction,
                    "to": (
                        f"0x{d.to_room & 0xFF:02X}"
                        if d.to_room is not None
                        else ("outdoors" if d.outdoors else None)
                    ),
                    "role": d.role,
                    "approach": list(d.approach_xy),
                    "path": list(d.path),
                    "recoveryPath": list(d.recovery_path),
                }
                for d in self.doors
            ],
            "points": {p.label: [p.x, p.y] for p in self.points},
            "clear": self.clear_policy.to_dict(),
            "notes": list(self.notes)[:4],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "roomBaseId": self.room_base_id,
            "roomHex": f"0x{self.room_base_id & 0xFF:02X}",
            "name": self.name,
            "measured": self.measured,
            "sourceState": self.source_state,
            "points": [p.to_dict() for p in self.points],
            "doors": [d.to_dict() for d in self.doors],
            "clearPolicy": self.clear_policy.to_dict(),
            "walkBbox": list(self.walk_bbox) if self.walk_bbox else None,
            "hostilesAtEntry": list(self.hostiles_at_entry),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, map_id: str = "") -> RoomMap:
        points = tuple(RoomMapPoint.from_dict(p) for p in (data.get("points") or ()))
        raw_doors = data.get("doors") or data.get("knownEdges") or ()
        doors = tuple(KnownDoor.from_dict(d) for d in raw_doors)
        walk = data.get("walkBbox") or data.get("walk_bbox")
        walk_bbox = None
        if walk is not None:
            walk_bbox = (int(walk[0]), int(walk[1]), int(walk[2]), int(walk[3]))
        room_id = int(data.get("roomBaseId", data.get("room_base_id", 0)))
        return cls(
            room_base_id=room_id,
            name=str(data.get("name") or f"room 0x{room_id:02X}"),
            points=points,
            doors=doors,
            clear_policy=ClearPolicy.from_dict(data.get("clearPolicy")),
            notes=tuple(str(n) for n in (data.get("notes") or ())),
            walk_bbox=walk_bbox,
            source_state=str(data.get("sourceState") or ""),
            measured=str(data.get("measured") or ""),
            hostiles_at_entry=tuple(data.get("hostilesAtEntry") or ()),
            map_id=map_id or str(data.get("mapId") or f"room_{room_id:02x}"),
        )


def room_map_path(map_id: str, *, maps_dir: Path | None = None) -> Path:
    """Resolve ``room_61`` / ``0x61`` / ``61`` → maps/room_61.json."""
    raw = map_id.strip().lower().replace(".json", "")
    if raw.startswith("0x"):
        raw = f"room_{int(raw, 16):02x}"
    elif raw.isdigit():
        raw = f"room_{int(raw):02x}"
    elif not raw.startswith("room_"):
        raw = f"room_{raw}"
    root = maps_dir if maps_dir is not None else MAPS_DIR
    return root / f"{raw}.json"


@lru_cache(maxsize=32)
def load_room_map(map_id: str) -> RoomMap:
    """Load measured room map JSON (geometry authority). Cached by map_id."""
    return _load_room_map_uncached(map_id, maps_dir=None)


def _load_room_map_uncached(map_id: str, *, maps_dir: Path | None = None) -> RoomMap:
    path = room_map_path(map_id, maps_dir=maps_dir)
    if maps_dir is not None and not path.is_file():
        cand = Path(maps_dir) / f"{map_id}.json"
        if cand.is_file():
            path = cand
    if not path.is_file():
        raise FileNotFoundError(f"room map not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"room map must be an object: {path}")
    return RoomMap.from_dict(data, map_id=path.stem)


def list_room_maps(*, maps_dir: Path | None = None) -> list[str]:
    root = maps_dir or MAPS_DIR
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("room_*.json"))


def save_room_map(room_map: RoomMap, path: Path | None = None) -> Path:
    """Write room map JSON (emit from measured session; prefer hand-edit maps/)."""
    out = path or room_map_path(room_map.map_id or f"room_{room_map.room_base_id:02x}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(room_map.to_dict(), indent=2) + "\n", encoding="utf-8")
    load_room_map.cache_clear()
    return out
