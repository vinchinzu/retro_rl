"""Shared doorway-natural segment contract for isolated room practice.

Single model for bootstrap provenance, scaffold policies, and run reports.
RNG re-rolls re-bootstrap via the same door boundary (doorPtr + orientation).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from super_metroid.door_kinematics import DoorKinematicsRequirement
from super_metroid.paths import GAME_DIR
from super_metroid.rooms.topology import (
    PhysicalConnection,
    _load_connections,
    _load_reference_rooms,
)

_ORIENT_TO_DIR = {
    "left": "LEFT",
    "right": "RIGHT",
    "up": "UP",
    "down": "DOWN",
}
_DIR_OPPOSITE = {
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
    "UP": "DOWN",
    "DOWN": "UP",
}

DEFAULT_REFERENCE_ROOT = GAME_DIR / "refs" / "sm-json-data"

# Cached connection load for resolve fallback (catalogs without baked doorPtr).
_CONNECTIONS_CACHE: tuple[PhysicalConnection, ...] | None = None

_DEFAULT_RNG_NOTE = (
    "Re-roll enemy/door RNG by re-running doorway bootstrap with a "
    "different boot state or boot_idle_frames; door boundary stays fixed."
)


def orientation_to_direction(orientation: str | None) -> str | None:
    if not orientation:
        return None
    return _ORIENT_TO_DIR.get(str(orientation).lower())


def opposite_direction(direction: str | None) -> str | None:
    if not direction:
        return None
    return _DIR_OPPOSITE.get(direction)


def entry_door_orientation(problem: Mapping[str, Any]) -> str | None:
    entry = problem.get("entry")
    if not isinstance(entry, dict):
        return None
    endpoint = entry.get("endpoint")
    if not isinstance(endpoint, dict):
        return None
    orient = str(endpoint.get("orientation") or "").lower()
    return orient or None


def exit_travel_direction(problem: Mapping[str, Any]) -> str | None:
    exit_data = problem.get("exit")
    if not isinstance(exit_data, dict):
        return None
    endpoint = exit_data.get("endpoint")
    if not isinstance(endpoint, dict):
        return None
    return orientation_to_direction(str(endpoint.get("orientation") or ""))


def _load_connections_cached(
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
) -> tuple[PhysicalConnection, ...]:
    global _CONNECTIONS_CACHE
    if _CONNECTIONS_CACHE is not None:
        return _CONNECTIONS_CACHE
    if not reference_root.is_dir():
        return ()
    rooms = _load_reference_rooms(reference_root)
    _CONNECTIONS_CACHE = _load_connections(reference_root, rooms)
    return _CONNECTIONS_CACHE


def resolve_entry_door_ptr(
    problem: Mapping[str, Any],
    *,
    connections: Sequence[PhysicalConnection] | None = None,
    reference_root: Path = DEFAULT_REFERENCE_ROOT,
) -> int | None:
    """Door-warp pointer that enters the problem room from its catalog source.

    Prefer baked ``entry.doorPtr`` / ``doorPtrHex`` from catalog export; fall
    back to the peer endpoint of the matching reference connection.
    """
    entry = problem.get("entry")
    if not isinstance(entry, dict):
        return None
    if entry.get("doorPtr") is not None:
        return int(entry["doorPtr"]) & 0xFFFF
    raw_hex = entry.get("doorPtrHex")
    if raw_hex:
        return int(str(raw_hex), 0) & 0xFFFF

    source = entry.get("sourceRoomId")
    if source is None:
        return None
    target = int(problem["roomId"])
    source_id = int(source)
    local_node = None
    endpoint = entry.get("endpoint")
    if isinstance(endpoint, dict) and endpoint.get("nodeId") is not None:
        local_node = int(endpoint["nodeId"])

    conns = (
        connections
        if connections is not None
        else _load_connections_cached(reference_root)
    )
    for connection in conns:
        pairs = [(connection.first, connection.second)]
        if connection.direction == "Bidirectional":
            pairs.append((connection.second, connection.first))
        for peer, local in pairs:
            # peer is source-room door; local is entry door in problem room
            if peer.room_id != source_id or local.room_id != target:
                continue
            if local_node is not None and local.node_id != local_node:
                continue
            if peer.door_ptr is not None:
                return peer.door_ptr
    return None


@dataclass(frozen=True)
class EntryContract:
    """Doorway-natural start boundary for an isolated room segment.

    Optional ``entry_kinematics`` declares expected spawn speed/position after
    a *natural* door hop (not door-warp practice fixtures, which zero motion).
    Use for continuous/pure segment handoffs that depend on run-in speed.
    """

    kind: str = "doorway_natural"
    schema_version: int = 1
    door_ptr: int | None = None
    entry_source_room_id: int | None = None
    door_orientation: str | None = None
    face: str | None = None
    spawn_x: int | None = None
    spawn_y: int | None = None
    spawn_pose: int | None = None
    inset_px: int | None = None
    door_block: list[int] | None = None
    warp_sample: dict[str, int] | None = None
    exit_travel_direction: str | None = None
    same_door_return: bool = False
    boot_idle_frames: int | None = None
    rng_note: str = _DEFAULT_RNG_NOTE
    # Expected natural-entry kinematics (continuous/pure). Practice bootstrap
    # fixtures intentionally clear momentum — leave this None for them.
    entry_kinematics: DoorKinematicsRequirement | None = None
    # Expected leave kinematics *into* this room's entry door (source room).
    leave_kinematics: DoorKinematicsRequirement | None = None

    @property
    def door_ptr_hex(self) -> str | None:
        if self.door_ptr is None:
            return None
        return f"0x{self.door_ptr:04X}"

    @property
    def entry_source_room_id_hex(self) -> str | None:
        if self.entry_source_room_id is None:
            return None
        return f"0x{self.entry_source_room_id:04X}"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schemaVersion": self.schema_version,
            "kind": self.kind,
            "doorPtrHex": self.door_ptr_hex,
            "entrySourceRoomIdHex": self.entry_source_room_id_hex,
            "doorOrientation": self.door_orientation,
            "face": self.face,
            "exitTravelDirection": self.exit_travel_direction,
            "sameDoorReturn": self.same_door_return,
            "rngNote": self.rng_note,
        }
        if self.spawn_x is not None and self.spawn_y is not None:
            payload["spawn"] = {
                "x": self.spawn_x,
                "y": self.spawn_y,
                "pose": self.spawn_pose,
            }
        if self.inset_px is not None:
            payload["insetPx"] = self.inset_px
        if self.door_block is not None:
            payload["doorBlock"] = self.door_block
        if self.warp_sample is not None:
            payload["warpSample"] = self.warp_sample
        if self.boot_idle_frames is not None:
            payload["bootIdleFrames"] = self.boot_idle_frames
        if self.door_ptr is not None:
            payload["doorPtr"] = self.door_ptr
        if self.entry_source_room_id is not None:
            payload["entrySourceRoomId"] = self.entry_source_room_id
        if self.entry_kinematics is not None:
            payload["entryKinematics"] = self.entry_kinematics.to_dict()
        if self.leave_kinematics is not None:
            payload["leaveKinematics"] = self.leave_kinematics.to_dict()
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> EntryContract | None:
        if not isinstance(raw, dict):
            return None
        door_ptr = raw.get("doorPtr")
        if door_ptr is None and raw.get("doorPtrHex"):
            door_ptr = int(str(raw["doorPtrHex"]), 0)
        elif door_ptr is not None:
            door_ptr = int(door_ptr)
        source = raw.get("entrySourceRoomId")
        if source is None and raw.get("entrySourceRoomIdHex"):
            source = int(str(raw["entrySourceRoomIdHex"]), 0)
        elif source is not None:
            source = int(source)
        spawn = raw.get("spawn") if isinstance(raw.get("spawn"), dict) else {}
        block = raw.get("doorBlock")
        door_block = [int(v) for v in block] if isinstance(block, list) else None
        warp = raw.get("warpSample")
        warp_sample = (
            {"x": int(warp["x"]), "y": int(warp["y"])}
            if isinstance(warp, dict) and "x" in warp and "y" in warp
            else None
        )
        return cls(
            kind=str(raw.get("kind") or "doorway_natural"),
            schema_version=int(raw.get("schemaVersion") or 1),
            door_ptr=door_ptr,
            entry_source_room_id=source,
            door_orientation=(
                str(raw["doorOrientation"]).lower()
                if raw.get("doorOrientation")
                else None
            ),
            face=str(raw["face"]) if raw.get("face") else None,
            spawn_x=int(spawn["x"]) if spawn.get("x") is not None else None,
            spawn_y=int(spawn["y"]) if spawn.get("y") is not None else None,
            spawn_pose=int(spawn["pose"]) if spawn.get("pose") is not None else None,
            inset_px=int(raw["insetPx"]) if raw.get("insetPx") is not None else None,
            door_block=door_block,
            warp_sample=warp_sample,
            exit_travel_direction=(
                str(raw["exitTravelDirection"])
                if raw.get("exitTravelDirection")
                else None
            ),
            same_door_return=bool(raw.get("sameDoorReturn", False)),
            boot_idle_frames=(
                int(raw["bootIdleFrames"])
                if raw.get("bootIdleFrames") is not None
                else None
            ),
            rng_note=str(raw.get("rngNote") or _DEFAULT_RNG_NOTE),
            entry_kinematics=DoorKinematicsRequirement.from_dict(
                raw.get("entryKinematics") or raw.get("entry_kinematics")
            ),
            leave_kinematics=DoorKinematicsRequirement.from_dict(
                raw.get("leaveKinematics") or raw.get("leave_kinematics")
            ),
        )

    @classmethod
    def from_problem(
        cls,
        problem: Mapping[str, Any],
        *,
        door_ptr: int | None = None,
        spawn: Mapping[str, Any] | None = None,
        boot_idle_frames: int | None = None,
    ) -> EntryContract:
        """Build contract from catalog problem (+ optional bootstrap spawn)."""
        entry = problem.get("entry") if isinstance(problem.get("entry"), dict) else {}
        exit_data = problem.get("exit") if isinstance(problem.get("exit"), dict) else {}
        objective = str(problem.get("objective", ""))
        # Return-style objectives are same-door only when the catalog exit
        # targets the entry source room. Through-stations (e.g. Draygon Save,
        # Nutella Refill) keep visit_station naming but exit the far door.
        objective_return = objective in {
            "visit_station_and_return",
            "enter_objective_and_return",
            "collect_and_return",
        }
        source = entry.get("sourceRoomId") if entry else None
        exit_target = exit_data.get("targetRoomId") if exit_data else None
        topology_same = (
            source is not None
            and exit_target is not None
            and int(source) == int(exit_target)
        )
        same_door = objective_return and topology_same
        ptr = door_ptr if door_ptr is not None else resolve_entry_door_ptr(problem)
        orient = entry_door_orientation(problem)
        spawn_data = dict(spawn or {})
        block = None
        endpoint = entry.get("endpoint") if entry else None
        if isinstance(endpoint, dict) and endpoint.get("block"):
            block = [int(endpoint["block"][0]), int(endpoint["block"][1])]
        elif spawn_data.get("doorBlock"):
            block = [
                int(spawn_data["doorBlock"][0]),
                int(spawn_data["doorBlock"][1]),
            ]
        warp_sample = None
        if isinstance(spawn_data.get("warpSample"), dict):
            warp_sample = {
                "x": int(spawn_data["warpSample"]["x"]),
                "y": int(spawn_data["warpSample"]["y"]),
            }
        return cls(
            door_ptr=ptr,
            entry_source_room_id=int(source) if source is not None else None,
            door_orientation=orient,
            face=str(spawn_data["face"]) if spawn_data.get("face") else None,
            spawn_x=int(spawn_data["x"]) if spawn_data.get("x") is not None else None,
            spawn_y=int(spawn_data["y"]) if spawn_data.get("y") is not None else None,
            spawn_pose=(
                int(spawn_data["pose"]) if spawn_data.get("pose") is not None else None
            ),
            inset_px=(
                int(spawn_data["insetPx"])
                if spawn_data.get("insetPx") is not None
                else None
            ),
            door_block=block,
            warp_sample=warp_sample,
            exit_travel_direction=exit_travel_direction(problem),
            same_door_return=same_door,
            boot_idle_frames=boot_idle_frames,
        )


def segment_boundary_dict() -> dict[str, str]:
    return {
        "start": "doorway_natural_entry",
        "end": "natural_exit_room_id",
        "rngFuture": (
            "Re-bootstrap entry with alternate boot_idle_frames / boot state "
            "to re-roll room RNG without changing the door boundary."
        ),
    }
