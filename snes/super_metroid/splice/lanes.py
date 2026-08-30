"""Ten item-seam lane inventories from one route manifest.

Filter edges by the s23–s32 / late-tape seam ranges. Tiny or synthetic
manifests (no seam segments) assign by hop-order or labels. Unique
artifact directories per lane. Does not write bank.json.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from super_metroid.splice.schema import RouteEdge, RouteManifest, rel_path

# Plan § Phase 3 item-seam ranges. s28 is Plasma→GT only when not superseded.
_SEAM_RE = re.compile(r"(?:^|/)s(\d{1,2})(?:/|$)", re.IGNORECASE)
_LATE_SEG_MIN = 33
_LATE_TAPES = ("g4_tourian_human", "g4_tourian_human_bb", "g4_tourian_human_mb")


def _safe(token: str) -> str:
    return str(token).replace(":", "_").replace("/", "_").replace("\\", "_")


def lane_artifact_dir(lane_id: str) -> str:
    return f"snes/super_metroid/recordings/splice/lanes/{_safe(lane_id)}/"


def lane_owner_package(lane_id: str) -> str:
    return f"snes/super_metroid/routes/kpdr/seams/{_safe(lane_id)}"


@dataclass(frozen=True)
class LaneSpec:
    """Static item-seam range. Inventory fills task_ids from a manifest."""

    lane_id: str
    name: str
    segments: tuple[str, ...]
    superseded_segments: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()

    @property
    def artifact_dir(self) -> str:
        return lane_artifact_dir(self.lane_id)

    @property
    def owner_package(self) -> str:
        return lane_owner_package(self.lane_id)


ITEM_SEAM_LANES: tuple[LaneSpec, ...] = (
    LaneSpec(
        "attic_gravity",
        "Attic → Gravity",
        ("s23",),
        labels=("attic", "bowling", "homing geemer", "pancakes"),
    ),
    LaneSpec(
        "gravity_grapple",
        "Gravity → Grapple",
        ("s24",),
        labels=("grapple",),
    ),
    LaneSpec(
        "grapple_main_street",
        "Grapple → Main Street",
        ("s25",),
        labels=("main street",),
    ),
    LaneSpec(
        "main_street_space_jump",
        "Main Street → Space Jump",
        ("s26",),
        labels=("space jump", "botwoon", "draygon"),
    ),
    LaneSpec(
        "space_jump_plasma",
        "Space Jump → Plasma",
        ("s27",),
        labels=("plasma",),
    ),
    LaneSpec(
        "plasma_golden_torizo",
        "Plasma → Golden Torizo",
        ("s29",),
        superseded_segments=("s28",),
        labels=("golden torizo", "golden-torizo"),
    ),
    LaneSpec(
        "golden_torizo_screw",
        "Golden Torizo → Screw Attack",
        ("s30",),
        labels=("screw attack",),
    ),
    LaneSpec(
        "screw_metal_pirates",
        "Screw Attack → Metal Pirates",
        ("s31",),
        labels=("metal pirates", "metal pirate"),
    ),
    LaneSpec(
        "metal_pirates_ridley",
        "Metal Pirates → post-Ridley",
        ("s32",),
        labels=("post-ridley", "post ridley"),
    ),
    LaneSpec(
        "ridley_credits",
        "Ridley → G4 → Tourian → Mother Brain → escape → ship/credits",
        (),
        labels=("g4", "tourian", "mother brain", "escape", "credits", "ship"),
    ),
)


@dataclass(frozen=True)
class Lane:
    """One item-seam work range with unique owner and artifact directory."""

    lane_id: str
    name: str
    segments: tuple[str, ...]
    order: int
    task_ids: tuple[str, ...]
    hop_keys: tuple[str, ...]
    artifact_dir: str
    owner_package: str
    superseded_segments: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_manifest(manifest: RouteManifest | Mapping[str, Any]) -> RouteManifest:
    if isinstance(manifest, RouteManifest):
        return manifest
    payload = json.loads(json.dumps(dict(manifest)))
    return RouteManifest.from_dict(payload)


def _segment_id(edge: RouteEdge) -> str | None:
    for raw in (edge.segment, edge.tape_path):
        if raw is None or str(raw).strip() == "":
            continue
        text = str(raw).strip().replace("\\", "/")
        if re.fullmatch(r"s\d{1,2}", text, re.I):
            return f"s{int(text[1:])}"
        match = _SEAM_RE.search(text)
        if match:
            return f"s{int(match.group(1))}"
    return None


def _is_superseded(edge: RouteEdge) -> bool:
    blob = " ".join((edge.task_id, *(edge.source_notes or ()))).lower()
    return "superseded" in blob


def _label_blob(edge: RouteEdge) -> str:
    parts = [
        edge.task_id,
        edge.hop_key,
        edge.goal or "",
        edge.segment or "",
        edge.tape_path or "",
        *edge.source_notes,
    ]
    return " ".join(str(p) for p in parts if p).lower().replace("_", " ").replace("-", " ")


def _late_tape(edge: RouteEdge) -> bool:
    blob = " ".join(
        str(p) for p in (edge.tape_path, edge.segment, *edge.source_notes) if p
    ).lower()
    return any(name in blob for name in _LATE_TAPES)


def _spec_for_segment(seg: str) -> LaneSpec | None:
    if seg in {f"s{n}" for n in range(_LATE_SEG_MIN, 100)}:
        return ITEM_SEAM_LANES[-1]
    for spec in ITEM_SEAM_LANES:
        if seg in spec.segments or seg in spec.superseded_segments:
            return spec
    return None


def _filter_mode(edges: Sequence[RouteEdge]) -> bool:
    for edge in edges:
        seg = _segment_id(edge)
        if seg is not None and _spec_for_segment(seg) is not None:
            return True
        if _late_tape(edge):
            return True
    return False


def _label_spec(edge: RouteEdge) -> LaneSpec | None:
    blob = _label_blob(edge)
    # Reverse so late-route labels (credits, g4) beat earlier item names.
    for spec in reversed(ITEM_SEAM_LANES):
        tokens = (spec.lane_id.replace("_", " "), spec.name.lower(), *spec.labels)
        if any(token and token in blob for token in tokens):
            return spec
    return None


def _empty_buckets() -> list[list[RouteEdge]]:
    return [[] for _ in ITEM_SEAM_LANES]


def _assign_filter(edges: Sequence[RouteEdge]) -> list[list[RouteEdge]]:
    buckets = _empty_buckets()
    index = {spec.lane_id: i for i, spec in enumerate(ITEM_SEAM_LANES)}
    for edge in edges:
        seg = _segment_id(edge)
        spec: LaneSpec | None = None
        if seg is not None:
            spec = _spec_for_segment(seg)
            if (
                spec is not None
                and seg in spec.superseded_segments
                and _is_superseded(edge)
            ):
                continue
        if spec is None and _late_tape(edge):
            spec = ITEM_SEAM_LANES[-1]
        if spec is None:
            continue
        buckets[index[spec.lane_id]].append(edge)
    return buckets


def _assign_synthetic(edges: Sequence[RouteEdge]) -> list[list[RouteEdge]]:
    buckets = _empty_buckets()
    index = {spec.lane_id: i for i, spec in enumerate(ITEM_SEAM_LANES)}
    leftover: list[RouteEdge] = []
    for edge in edges:
        spec = _label_spec(edge)
        if spec is None:
            leftover.append(edge)
            continue
        buckets[index[spec.lane_id]].append(edge)
    empty = [i for i, rows in enumerate(buckets) if not rows]
    for i, edge in enumerate(leftover):
        if i < len(empty):
            buckets[empty[i]].append(edge)
        else:
            buckets[-1].append(edge)
    return buckets


def inventory_from_manifest(
    manifest: RouteManifest | Mapping[str, Any],
) -> tuple[Lane, ...]:
    """Always ten named lanes; edges filtered or hop-order assigned."""
    route = _as_manifest(manifest)
    if _filter_mode(route.edges):
        buckets = _assign_filter(route.edges)
    else:
        buckets = _assign_synthetic(route.edges)
    lanes: list[Lane] = []
    for order, (spec, rows) in enumerate(zip(ITEM_SEAM_LANES, buckets)):
        lanes.append(
            Lane(
                lane_id=spec.lane_id,
                name=spec.name,
                segments=spec.segments,
                order=order,
                task_ids=tuple(e.task_id for e in rows),
                hop_keys=tuple(e.hop_key for e in rows),
                artifact_dir=rel_path(spec.artifact_dir) or spec.artifact_dir,
                owner_package=rel_path(spec.owner_package) or spec.owner_package,
                superseded_segments=spec.superseded_segments,
            )
        )
    return tuple(lanes)


def lane_by_id(
    lanes: Sequence[Lane],
    lane_id: str,
) -> Lane | None:
    for lane in lanes:
        if lane.lane_id == lane_id:
            return lane
    return None
