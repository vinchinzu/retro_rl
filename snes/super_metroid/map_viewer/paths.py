"""Load dense Samus CoG paths and export *area-local* segmented polylines.

Segment rule (critical for readable maps):
  * never connect across room changes
  * never connect if pixel step > ``max_step_px`` (default 48)
  * never connect if area changes

Sparse continuous reports export **markers only** (no long door-to-door lines).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from super_metroid.map_viewer.coords import (
    AreaBounds,
    MapPoint,
    RoomPlacement,
    area_bounds,
    area_slug,
    load_room_index,
    point_from_sample,
)
from super_metroid.paths import GAME_DIR, RECORDINGS_DIR

SCHEMA = "super_metroid_area_path_v2"
# Keep reading old exports if needed.
LEGACY_SCHEMA = "super_metroid_world_path_v1"

DEFAULT_COLORS = [
    "#00e5ff",
    "#ff4081",
    "#69f0ae",
    "#ffd740",
    "#b388ff",
    "#ff6e40",
    "#40c4ff",
    "#eeff41",
]

# Same-room continuity: larger than a normal frame of movement, smaller than a
# door warp / teleport across the map image.
DEFAULT_MAX_STEP_PX = 48.0


@dataclass
class PathSegment:
    """One continuous polyline (same room, short steps only)."""

    area: str
    room_id: int
    points: list[MapPoint] = field(default_factory=list)

    def to_dict(self, *, compact: bool = True) -> dict[str, Any]:
        return {
            "area": self.area,
            "area_slug": area_slug(self.area),
            "room_id": self.room_id,
            "room_id_hex": f"0x{self.room_id:04X}",
            "points": [p.to_dict(compact=compact) for p in self.points],
        }


@dataclass
class WorldPath:
    """Named CoG trail as safe segments (+ optional sparse markers)."""

    id: str
    label: str
    source: str
    kind: str  # tas_series | human_trace | continuous_sparse | generic
    points: list[MapPoint] = field(default_factory=list)
    segments: list[PathSegment] = field(default_factory=list)
    markers: list[MapPoint] = field(default_factory=list)
    color: str = "#00e5ff"
    meta: dict[str, Any] = field(default_factory=dict)
    primary_area: str = ""

    def to_dict(self, *, compact: bool = True) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "id": self.id,
            "label": self.label,
            "source": self.source,
            "kind": self.kind,
            "color": self.color,
            "primary_area": self.primary_area,
            "primary_area_slug": area_slug(self.primary_area) if self.primary_area else "",
            "point_count": len(self.points),
            "segment_count": len(self.segments),
            "segments": [s.to_dict(compact=compact) for s in self.segments],
            "markers": [m.to_dict(compact=compact) for m in self.markers],
            "meta": self.meta,
        }


def segment_points(
    points: Sequence[MapPoint],
    *,
    max_step_px: float = DEFAULT_MAX_STEP_PX,
) -> list[PathSegment]:
    """Split samples into safe polylines (same room, short steps)."""
    if not points:
        return []
    segs: list[PathSegment] = []
    cur = PathSegment(area=points[0].area, room_id=points[0].room_id, points=[points[0]])
    for pt in points[1:]:
        prev = cur.points[-1]
        dist = math.hypot(pt.ax - prev.ax, pt.ay - prev.ay)
        same_room = pt.room_id == cur.room_id and pt.area == cur.area
        if same_room and dist <= max_step_px:
            cur.points.append(pt)
            continue
        if len(cur.points) >= 2:
            segs.append(cur)
        elif len(cur.points) == 1:
            # Singleton kept only as potential marker later; drop as segment.
            pass
        cur = PathSegment(area=pt.area, room_id=pt.room_id, points=[pt])
    if len(cur.points) >= 2:
        segs.append(cur)
    return segs


def primary_area_for(points: Sequence[MapPoint]) -> str:
    if not points:
        return ""
    counts: dict[str, int] = {}
    for p in points:
        counts[p.area] = counts.get(p.area, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _sample_fields(raw: Mapping[str, Any]) -> dict[str, Any]:
    room_id = raw.get("room_id", raw.get("roomId", raw.get("room")))
    x = raw.get("x", raw.get("samus_x", raw.get("samusX")))
    y = raw.get("y", raw.get("samus_y", raw.get("samusY")))
    frame = raw.get("frame", raw.get("f", 0))
    return {
        "room_id": int(room_id) if room_id is not None else 0,
        "x": int(x) if x is not None else 0,
        "y": int(y) if y is not None else 0,
        "frame": int(frame) if frame is not None else 0,
        "x_sub": raw.get("x_sub", raw.get("samus_x_sub")),
        "y_sub": raw.get("y_sub", raw.get("samus_y_sub")),
        "pose": raw.get("pose"),
        "phase": raw.get("phase"),
    }


def points_from_samples(
    rooms: Mapping[int, RoomPlacement],
    bounds: Mapping[str, AreaBounds],
    samples: Iterable[Mapping[str, Any]],
    *,
    stride: int = 1,
    max_points: int | None = None,
) -> list[MapPoint]:
    out: list[MapPoint] = []
    stride = max(1, int(stride))
    for i, raw in enumerate(samples):
        if i % stride != 0:
            continue
        fields = _sample_fields(raw)
        pt = point_from_sample(
            rooms,
            bounds,
            room_id=fields["room_id"],
            x=fields["x"],
            y=fields["y"],
            frame=fields["frame"],
            x_sub=int(fields["x_sub"]) if fields["x_sub"] is not None else None,
            y_sub=int(fields["y_sub"]) if fields["y_sub"] is not None else None,
            pose=int(fields["pose"]) if fields["pose"] is not None else None,
            phase=str(fields["phase"]) if fields["phase"] is not None else None,
        )
        if pt is None:
            continue
        if out and out[-1].ax == pt.ax and out[-1].ay == pt.ay and out[-1].room_id == pt.room_id:
            out[-1] = MapPoint(
                frame=pt.frame,
                room_id=pt.room_id,
                area=pt.area,
                x=pt.x,
                y=pt.y,
                ax=pt.ax,
                ay=pt.ay,
                pose=pt.pose,
                phase=pt.phase,
            )
            continue
        out.append(pt)
        if max_points is not None and len(out) >= max_points:
            break
    return out


def _finish_path(
    *,
    path_id: str,
    label: str,
    source: str,
    kind: str,
    points: list[MapPoint],
    color: str,
    meta: dict[str, Any],
    max_step_px: float,
    markers_only: bool = False,
) -> WorldPath:
    if markers_only:
        segs: list[PathSegment] = []
        markers = list(points)
    else:
        segs = segment_points(points, max_step_px=max_step_px)
        markers = []
    primary = primary_area_for(points)
    meta = {
        **meta,
        "max_step_px": max_step_px,
        "areas": sorted({p.area for p in points}),
    }
    return WorldPath(
        id=path_id,
        label=label,
        source=source,
        kind=kind,
        points=points,
        segments=segs,
        markers=markers,
        color=color,
        meta=meta,
        primary_area=primary,
    )


def load_series_jsonl(
    path: Path,
    rooms: Mapping[int, RoomPlacement],
    bounds: Mapping[str, AreaBounds] | None = None,
    *,
    stride: int = 1,
    max_points: int | None = None,
    path_id: str | None = None,
    label: str | None = None,
    color: str = DEFAULT_COLORS[0],
    max_step_px: float = DEFAULT_MAX_STEP_PX,
) -> WorldPath:
    path = Path(path)
    bounds = bounds if bounds is not None else area_bounds(rooms)
    points = points_from_samples(
        rooms, bounds, _iter_jsonl(path), stride=stride, max_points=max_points
    )
    return _finish_path(
        path_id=path_id or path.parent.name or path.stem,
        label=label or path.parent.name or path.stem,
        source=str(path),
        kind="tas_series",
        points=points,
        color=color,
        meta={"stride": stride, "raw_path": str(path)},
        max_step_px=max_step_px,
    )


def load_human_task(
    path: Path,
    rooms: Mapping[int, RoomPlacement],
    bounds: Mapping[str, AreaBounds] | None = None,
    *,
    stride: int = 1,
    max_points: int | None = None,
    path_id: str | None = None,
    label: str | None = None,
    color: str = DEFAULT_COLORS[1],
    max_step_px: float = DEFAULT_MAX_STEP_PX,
) -> WorldPath:
    path = Path(path)
    bounds = bounds if bounds is not None else area_bounds(rooms)
    data = json.loads(path.read_text(encoding="utf-8"))
    points = points_from_samples(
        rooms, bounds, data.get("trace") or [], stride=stride, max_points=max_points
    )
    return _finish_path(
        path_id=path_id or path.stem,
        label=label or data.get("name") or path.stem,
        source=str(path),
        kind="human_trace",
        points=points,
        color=color,
        meta={
            "stride": stride,
            "frame_count": data.get("frame_count") or len(data.get("frames") or []),
            "start_state": data.get("start_state"),
        },
        max_step_px=max_step_px,
    )


def load_continuous_report(
    path: Path,
    rooms: Mapping[int, RoomPlacement],
    bounds: Mapping[str, AreaBounds] | None = None,
    *,
    path_id: str | None = None,
    label: str | None = None,
    color: str = DEFAULT_COLORS[2],
    max_step_px: float = DEFAULT_MAX_STEP_PX,
) -> WorldPath:
    """Sparse door samples → **markers only** (no cross-map lines)."""
    path = Path(path)
    bounds = bounds if bounds is not None else area_bounds(rooms)
    data = json.loads(path.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = []
    for tr in data.get("transitions") or []:
        for key in ("leave_kinematics", "entry_kinematics"):
            kin = tr.get(key) or {}
            if kin.get("room_id") is None or "samus_x" not in kin:
                continue
            samples.append(
                {
                    "frame": kin.get("frame", tr.get("frame", 0)),
                    "room_id": kin["room_id"],
                    "x": kin.get("samus_x", 0),
                    "y": kin.get("samus_y", 0),
                    "x_sub": kin.get("samus_x_sub"),
                    "y_sub": kin.get("samus_y_sub"),
                    "pose": kin.get("pose"),
                    "phase": kin.get("phase"),
                }
            )
    final = data.get("final_state") or {}
    if final.get("room_id") is not None and "samus_x" in final:
        samples.append(
            {
                "frame": final.get("frame", data.get("total_frames", 0)),
                "room_id": final["room_id"],
                "x": final.get("samus_x", 0),
                "y": final.get("samus_y", 0),
                "x_sub": final.get("samus_x_sub"),
                "y_sub": final.get("samus_y_sub"),
                "pose": final.get("pose"),
                "phase": final.get("phase"),
            }
        )
    points = points_from_samples(rooms, bounds, samples, stride=1)
    return _finish_path(
        path_id=path_id or path.stem,
        label=label or path.stem,
        source=str(path),
        kind="continuous_sparse",
        points=points,
        color=color,
        meta={
            "total_frames": data.get("total_frames"),
            "success": data.get("success"),
            "note": "markers only — continuous reports lack dense CoG",
        },
        max_step_px=max_step_px,
        markers_only=True,
    )


def load_generic_trace_json(
    path: Path,
    rooms: Mapping[int, RoomPlacement],
    bounds: Mapping[str, AreaBounds] | None = None,
    *,
    stride: int = 1,
    max_points: int | None = None,
    path_id: str | None = None,
    label: str | None = None,
    color: str = DEFAULT_COLORS[3],
    max_step_px: float = DEFAULT_MAX_STEP_PX,
) -> WorldPath:
    path = Path(path)
    bounds = bounds if bounds is not None else area_bounds(rooms)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") == SCHEMA:
        # Rehydrate segments/markers.
        segs: list[PathSegment] = []
        for raw_seg in data.get("segments") or []:
            pts = [
                MapPoint(
                    frame=int(p.get("f", 0)),
                    room_id=int(p.get("r", raw_seg.get("room_id", 0))),
                    area=str(raw_seg.get("area") or data.get("primary_area") or ""),
                    x=int(p.get("x", 0)),
                    y=int(p.get("y", 0)),
                    ax=float(p["ax"]),
                    ay=float(p["ay"]),
                )
                for p in raw_seg.get("points") or []
                if "ax" in p and "ay" in p
            ]
            if len(pts) >= 2:
                segs.append(
                    PathSegment(
                        area=str(raw_seg.get("area") or pts[0].area),
                        room_id=int(raw_seg.get("room_id") or pts[0].room_id),
                        points=pts,
                    )
                )
        markers = [
            MapPoint(
                frame=int(p.get("f", 0)),
                room_id=int(p.get("r", 0)),
                area=str(data.get("primary_area") or ""),
                x=0,
                y=0,
                ax=float(p["ax"]),
                ay=float(p["ay"]),
            )
            for p in data.get("markers") or []
            if "ax" in p
        ]
        flat = [p for s in segs for p in s.points] + markers
        return WorldPath(
            id=path_id or data.get("id") or path.stem,
            label=label or data.get("label") or path.stem,
            source=str(path),
            kind=str(data.get("kind") or "generic"),
            points=flat,
            segments=segs,
            markers=markers,
            color=color or data.get("color") or DEFAULT_COLORS[3],
            meta=dict(data.get("meta") or {}),
            primary_area=str(data.get("primary_area") or primary_area_for(flat)),
        )
    trace = data.get("trace") or data.get("points") or data.get("series") or []
    points = points_from_samples(
        rooms, bounds, trace, stride=stride, max_points=max_points
    )
    return _finish_path(
        path_id=path_id or path.stem,
        label=label or data.get("level") or data.get("name") or path.stem,
        source=str(path),
        kind="generic",
        points=points,
        color=color,
        meta={"stride": stride},
        max_step_px=max_step_px,
    )


def detect_source_kind(path: Path) -> str:
    path = Path(path)
    if path.suffix == ".jsonl" or path.name == "series.jsonl":
        return "tas_series"
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if path.suffix == ".json" and size <= 8_000_000:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            if data.get("schema") in (SCHEMA, LEGACY_SCHEMA):
                return "world_path" if data.get("schema") == LEGACY_SCHEMA else "area_path"
            transitions = data.get("transitions")
            if isinstance(transitions, list) and transitions:
                sample = transitions[0] if transitions else {}
                if isinstance(sample, dict) and (
                    "leave_kinematics" in sample or "entry_kinematics" in sample
                ):
                    return "continuous_sparse"
            trace = data.get("trace")
            if isinstance(trace, list) and trace:
                if data.get("frames") is not None and data.get("name") is not None:
                    return "human_trace"
                return "generic"
            return "generic"
    text = path.read_text(encoding="utf-8", errors="replace")[:64_000]
    if f'"schema": "{SCHEMA}"' in text or f'"schema":"{SCHEMA}"' in text:
        return "area_path"
    if '"leave_kinematics"' in text and '"transitions"' in text:
        return "continuous_sparse"
    if '"trace"' in text and ('"room"' in text or '"room_id"' in text):
        if '"frames"' in text and '"name"' in text:
            return "human_trace"
        return "generic"
    return "generic"


def load_path_source(
    path: Path | str,
    rooms: Mapping[int, RoomPlacement] | None = None,
    bounds: Mapping[str, AreaBounds] | None = None,
    *,
    stride: int = 1,
    max_points: int | None = None,
    path_id: str | None = None,
    label: str | None = None,
    color: str | None = None,
    kind: str | None = None,
    max_step_px: float = DEFAULT_MAX_STEP_PX,
) -> WorldPath:
    path = Path(path)
    if path.is_dir():
        series = path / "series.jsonl"
        if series.is_file():
            path = series
        else:
            raise FileNotFoundError(f"No series.jsonl in {path}")

    rooms = rooms if rooms is not None else load_room_index()
    bounds = bounds if bounds is not None else area_bounds(rooms)
    kind = kind or detect_source_kind(path)
    color = color or DEFAULT_COLORS[0]

    if kind == "tas_series":
        return load_series_jsonl(
            path, rooms, bounds, stride=stride, max_points=max_points,
            path_id=path_id, label=label, color=color, max_step_px=max_step_px,
        )
    if kind == "human_trace":
        return load_human_task(
            path, rooms, bounds, stride=stride, max_points=max_points,
            path_id=path_id, label=label, color=color, max_step_px=max_step_px,
        )
    if kind == "continuous_sparse":
        return load_continuous_report(
            path, rooms, bounds, path_id=path_id, label=label, color=color,
            max_step_px=max_step_px,
        )
    if kind in ("area_path", "world_path", "generic"):
        return load_generic_trace_json(
            path, rooms, bounds, stride=stride, max_points=max_points,
            path_id=path_id, label=label, color=color, max_step_px=max_step_px,
        )
    return load_generic_trace_json(
        path, rooms, bounds, stride=stride, max_points=max_points,
        path_id=path_id, label=label, color=color, max_step_px=max_step_px,
    )


def export_path(path: WorldPath, out: Path | str, *, compact: bool = True) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(path.to_dict(compact=compact), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return out


def export_catalog(
    paths: Sequence[WorldPath],
    out_dir: Path | str,
    *,
    compact: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, wp in enumerate(paths):
        if not wp.color:
            wp.color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        dest = out_dir / f"{wp.id}.json"
        export_path(wp, dest, compact=compact)
        entries.append(
            {
                "id": wp.id,
                "label": wp.label,
                "kind": wp.kind,
                "color": wp.color,
                "point_count": len(wp.points),
                "segment_count": len(wp.segments),
                "marker_count": len(wp.markers),
                "primary_area": wp.primary_area,
                "primary_area_slug": area_slug(wp.primary_area) if wp.primary_area else "",
                "file": dest.name,
                "source": wp.source,
                "meta": wp.meta,
            }
        )
    index_path = out_dir / "index.json"
    index_path.write_text(
        json.dumps({"schema": "super_metroid_path_catalog_v2", "paths": entries}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return index_path


def discover_default_sources() -> list[Path]:
    """Prefer one dense Crateria human path + optional TAS resync (same area)."""
    found: list[Path] = []
    # Best single demo: dense CoG in one room graph area.
    for name in (
        "parlor_left_human.json",
        "parlor_left_human2.json",
        "charge_human.json",
    ):
        p = GAME_DIR / "tasks" / name
        if p.is_file():
            found.append(p)
            break
    # Dense TAS that mostly sits in Crateria
    for name in ("resync_zebes_rooms", "resync_landing_zebes"):
        series = RECORDINGS_DIR / "tas_import" / name / "series.jsonl"
        if series.is_file() and series.stat().st_size > 1000:
            found.append(series)
            break
    return found
