"""Room-hop inventory + skills/graph extraction board from TAS annotate.

Offline (no emulator). Reads ``trace.json`` / ``pins.json`` / ``summary.json``
under ``recordings/tas_import/<run_id>/`` and emits:

* hop inventory (from_room → to_room, frames, items, pose tech)
* skill candidates (Layer 1) mapped to ``routes/skills/`` modules
* continuous progression graph edge status (propose only — pure owns hop)
* practice ``room_graph`` topology note (not continuous evidence)

```bash
uv run python -m super_metroid.tas.extract_hops \\
  snes/super_metroid/recordings/tas_import/sniq_100_full \\
  --out snes/super_metroid/recordings/tas_import/sniq_100_full/extraction_board.json
```

Does **not** STATUS-promote or auto-wire continuous tip.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from super_metroid.paths import GAME_DIR, RECORDINGS_DIR
from super_metroid.routes.kpdr import room_ids as rid
from super_metroid.tas.stages import STAGE_CATALOG, export_room_body_spec

# ---------------------------------------------------------------------------
# Room name registry (product room_ids + common Ceres)
# ---------------------------------------------------------------------------

_ROOM_NAME_OVERRIDES: dict[int, str] = {
    rid.ROOM_CERES_ELEVATOR: "Ceres Elevator",
    rid.ROOM_CERES_FALLING: "Ceres Falling Tile",
    rid.ROOM_CERES_MAGNET: "Ceres Magnet Stairs",
    rid.ROOM_CERES_SCIENTIST: "Ceres Dead Scientists",
    rid.ROOM_CERES_FLAT: "Ceres Flat Room",
    rid.ROOM_CERES_RIDLEY: "Ceres Ridley",
    rid.ROOM_LANDING_SITE: "Landing Site",
    rid.ROOM_PARLOR: "Parlor",
    rid.ROOM_CLIMB: "Climb",
    rid.ROOM_PIT: "Pit Room",
    rid.ROOM_BLUE_BRINSTAR_ELEVATOR: "Blue Brinstar Elevator",
    rid.ROOM_MORPH: "Morph Ball Room",
    rid.ROOM_ICE_GATE: "Ice Beam Gate Room",
    rid.ROOM_ICE_ACID: "Ice Beam Acid Room",
    rid.ROOM_ICE_SNAKE: "Ice Beam Snake Room",
    rid.ROOM_ICE: "Ice Beam Room",
    rid.ROOM_WAVE: "Wave Beam Room",
    rid.ROOM_SPEED: "Speed Booster Room",
    rid.ROOM_BUBBLE: "Bubble Mountain",
}


def _build_room_names() -> dict[int, str]:
    names = dict(_ROOM_NAME_OVERRIDES)
    for attr in dir(rid):
        if not attr.startswith("ROOM_"):
            continue
        val = getattr(rid, attr)
        if isinstance(val, int) and val not in names:
            # ROOM_FOO_BAR → "Foo Bar"
            label = attr[5:].replace("_", " ").title()
            names[val] = label
    return names


ROOM_NAMES = _build_room_names()


def room_name(room_id: int) -> str:
    return ROOM_NAMES.get(int(room_id), f"Unknown 0x{int(room_id):04X}")


def room_hex(room_id: int) -> str:
    return f"0x{int(room_id):04X}"


# Pose-cluster → Layer-1 skill module (routes/skills/)
_TECH_TO_SKILL: dict[str, str] = {
    "morph": "morph_bomb",
    "shinespark": "shinespark",
    "knockback": "knockback",
    "walljump": "walljump",
    "spinjump": "walljump",  # spin often feeds WJ chains
    "speed_echo": "runway",
    "shine_arm": "shinespark",
}

# Product pure / continuous tip ownership (not TAS STATUS).
# Values: pure_green | pure_open | continuous_green | unknown | n_a
_PRODUCT_EDGE_STATUS: dict[tuple[int, int], str] = {
    (rid.ROOM_CERES_ELEVATOR, rid.ROOM_CERES_FALLING): "continuous_green",
    (rid.ROOM_LANDING_SITE, rid.ROOM_PARLOR): "continuous_green",
    (rid.ROOM_PARLOR, rid.ROOM_CLIMB): "continuous_green",
    (rid.ROOM_CLIMB, rid.ROOM_PIT): "continuous_green",
    (rid.ROOM_PIT, rid.ROOM_BLUE_BRINSTAR_ELEVATOR): "continuous_green",
    (rid.ROOM_BLUE_BRINSTAR_ELEVATOR, rid.ROOM_MORPH): "continuous_green",
    (rid.ROOM_ICE_GATE, rid.ROOM_ICE_ACID): "pure_green",  # rr-9t4 dual
    (rid.ROOM_ICE_ACID, rid.ROOM_ICE_SNAKE): "pure_green",  # rr-5cf dual
}

# High-leverage tech tags for skill extraction priority.
_HIGH_LEVERAGE_TECH = frozenset(
    {"shinespark", "morph", "walljump", "speed_echo", "shine_arm", "knockback"}
)


@dataclass
class RoomHop:
    """One settled room_enter → next room_enter (or end)."""

    hop_id: str
    index: int
    from_room: int
    to_room: int | None
    enter_frame: int
    leave_frame: int | None
    frames: int | None
    enter_pose: int = 0
    enter_x: int = 0
    enter_y: int = 0
    items_gained: list[str] = field(default_factory=list)
    beams_gained: list[str] = field(default_factory=list)
    capacity_gains: list[str] = field(default_factory=list)
    pose_clusters: list[str] = field(default_factory=list)
    tech_tags: list[str] = field(default_factory=list)
    desync_in_hop: bool = False
    death_in_hop: bool = False
    usable: bool = True  # False if desync/death dominated
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["from_room_hex"] = room_hex(self.from_room)
        d["from_room_name"] = room_name(self.from_room)
        d["to_room_hex"] = room_hex(self.to_room) if self.to_room else None
        d["to_room_name"] = room_name(self.to_room) if self.to_room else None
        return d


@dataclass
class ExtractionCandidate:
    """One row on the skills/graph extraction board."""

    hop_id: str
    from_room: int
    to_room: int | None
    frames: int | None
    tech: list[str]
    skill_modules: list[str]
    pure_status: str
    graph_edge_status: str
    practice_topology: str
    tas_body: dict[str, Any] | None
    residual_next_knob: str
    priority: int  # lower = sooner
    bead_hint: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["from_room_hex"] = room_hex(self.from_room)
        d["from_room_name"] = room_name(self.from_room)
        d["to_room_hex"] = room_hex(self.to_room) if self.to_room else None
        d["to_room_name"] = room_name(self.to_room) if self.to_room else None
        return d


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_annotate_dir(run_dir: Path | str) -> dict[str, Any]:
    """Load trace/summary/pins from an annotate output directory."""
    root = Path(run_dir)
    out: dict[str, Any] = {"dir": str(root)}
    for name in ("summary", "trace", "pins"):
        p = root / f"{name}.json"
        if p.is_file():
            out[name] = _load_json(p)
    if "trace" not in out and "summary" not in out:
        raise FileNotFoundError(f"no trace.json or summary.json under {root}")
    return out


def _events(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "trace" in data and isinstance(data["trace"], Mapping):
        return list(data["trace"].get("events") or [])
    if "pins" in data:
        return list(data["pins"] or [])
    return []


def build_hops(
    events: Sequence[Mapping[str, Any]],
    *,
    run_id: str = "run",
) -> list[RoomHop]:
    """Build ordered room hops from annotate events."""
    enters = [e for e in events if e.get("kind") == "room_enter"]
    if not enters:
        return []

    # Index secondary events by frame for hop windows.
    by_kind: dict[str, list[Mapping[str, Any]]] = {
        "item_gain": [],
        "beam_gain": [],
        "capacity_gain": [],
        "pose_cluster": [],
        "speed_echo": [],
        "shine_arm": [],
        "desync_suspect": [],
        "death": [],
        "room_leave": [],
    }
    for e in events:
        k = str(e.get("kind") or "")
        if k in by_kind:
            by_kind[k].append(e)

    hops: list[RoomHop] = []
    for i, ent in enumerate(enters):
        start = int(ent["frame"])
        from_room = int(ent.get("room_id") or 0)
        if i + 1 < len(enters):
            nxt = enters[i + 1]
            end = int(nxt["frame"])
            to_room: int | None = int(nxt.get("room_id") or 0)
            frames: int | None = end - start
            leave_frame: int | None = end
        else:
            end = 10**12
            to_room = None
            frames = None
            leave_frame = None

        def _in_window(ev: Mapping[str, Any]) -> bool:
            f = int(ev.get("frame") or 0)
            return start <= f < end

        items = [str(e.get("detail") or "") for e in by_kind["item_gain"] if _in_window(e)]
        beams = [str(e.get("detail") or "") for e in by_kind["beam_gain"] if _in_window(e)]
        caps = [
            str(e.get("detail") or "") for e in by_kind["capacity_gain"] if _in_window(e)
        ]
        poses = sorted(
            {
                str(e.get("detail") or "")
                for e in by_kind["pose_cluster"]
                if _in_window(e) and e.get("detail")
            }
        )
        tech = list(poses)
        if any(_in_window(e) for e in by_kind["speed_echo"]):
            tech.append("speed_echo")
        if any(_in_window(e) for e in by_kind["shine_arm"]):
            tech.append("shine_arm")
        # de-dupe preserve order
        seen: set[str] = set()
        tech_tags: list[str] = []
        for t in tech:
            if t and t not in seen:
                seen.add(t)
                tech_tags.append(t)

        desync = any(_in_window(e) for e in by_kind["desync_suspect"])
        death = any(_in_window(e) for e in by_kind["death"])
        usable = not death and (not desync or frames is not None and frames < 500)

        hop_id = f"{run_id}_h{i:04d}_{room_hex(from_room)}"
        if to_room is not None:
            hop_id += f"_to_{room_hex(to_room)}"

        hops.append(
            RoomHop(
                hop_id=hop_id,
                index=i,
                from_room=from_room,
                to_room=to_room,
                enter_frame=start,
                leave_frame=leave_frame,
                frames=frames,
                enter_pose=int(ent.get("pose") or 0),
                enter_x=int(ent.get("x") or 0),
                enter_y=int(ent.get("y") or 0),
                items_gained=items,
                beams_gained=beams,
                capacity_gains=caps,
                pose_clusters=poses,
                tech_tags=tech_tags,
                desync_in_hop=desync,
                death_in_hop=death,
                usable=usable,
                notes="",
            )
        )
    return hops


def _lookup_progression_edge(from_room: int, to_room: int | None) -> str:
    if to_room is None:
        return "n_a"
    pair = (int(from_room), int(to_room))
    if pair in _PRODUCT_EDGE_STATUS:
        st = _PRODUCT_EDGE_STATUS[pair]
        if st == "continuous_green":
            return "edge_owned_continuous"
        if st == "pure_green":
            return "edge_owned_pure"
        if st == "pure_open":
            return "edge_candidate_product_p0"
        return st
    # Try live graph tables (optional — import can be heavy / cycle-safe).
    try:
        from super_metroid.progression import SPEED_GRAPH

        edge = SPEED_GRAPH.edge_for(int(from_room), int(to_room))
        if edge is not None:
            return f"graph_{edge.verification}:{edge.edge_id}"
    except Exception:
        pass
    return "no_graph_edge"


def _skill_modules(tech: Sequence[str]) -> list[str]:
    mods: list[str] = []
    seen: set[str] = set()
    for t in tech:
        m = _TECH_TO_SKILL.get(t)
        if m and m not in seen:
            seen.add(m)
            mods.append(m)
    if not mods:
        mods.append("door")  # default door transit skill
    return mods


def _pure_status(from_room: int, to_room: int | None, hop: RoomHop) -> str:
    if not hop.usable:
        return "tas_desync_unusable"
    if to_room is None:
        return "open_ended"
    pair = (int(from_room), int(to_room))
    if pair in _PRODUCT_EDGE_STATUS:
        return _PRODUCT_EDGE_STATUS[pair]
    if hop.items_gained or hop.beams_gained:
        return "tas_item_window"  # reference only
    if hop.tech_tags and set(hop.tech_tags) & _HIGH_LEVERAGE_TECH:
        return "skill_extract_candidate"
    return "tas_reference_only"


def _priority(hop: RoomHop, pure: str, graph: str) -> int:
    """Lower = work sooner."""
    if pure == "pure_open" or "product_p0" in graph:
        return 0
    if pure == "skill_extract_candidate" and hop.usable:
        return 1
    if pure == "continuous_green":
        return 5  # already owned — low extract urgency
    if pure == "tas_desync_unusable":
        return 9
    if hop.items_gained or hop.beams_gained:
        return 2
    if hop.usable and hop.frames is not None and hop.frames < 2000:
        return 3
    return 6


def _residual_knob(hop: RoomHop, pure: str) -> str:
    if pure == "pure_open":
        return "pure_probe_2wj_bands"  # Ice Snake→PLM style (rr-5if)
    if pure == "skill_extract_candidate":
        tech = hop.tech_tags[0] if hop.tech_tags else "door"
        return f"pure_probe_skill:{tech}"
    if pure == "tas_desync_unusable":
        return "re_anchor_control_state"
    if pure == "tas_item_window":
        return "state_pin_at_item_gain"
    if pure.startswith("continuous") or pure == "pure_green":
        return "none_owned"
    return "annotate_reanchor"


def _bead_hint(hop: RoomHop, pure: str) -> str:
    if pure == "pure_open":
        # Product open Ice residual after Acid→Snake dual GREEN.
        if (
            hop.from_room == rid.ROOM_ICE_SNAKE
            and hop.to_room == rid.ROOM_ICE
        ):
            return "rr-5if"
        return "pure_open"
    if pure == "skill_extract_candidate":
        return "discovered-from:rr-wpy skill pure probe"
    if pure == "tas_desync_unusable":
        return "re-anchor only; no pure card"
    return ""


def build_extraction_board(
    hops: Sequence[RoomHop],
    *,
    pins: Sequence[Mapping[str, Any]] | None = None,
    run_id: str = "run",
    source: str = "",
) -> dict[str, Any]:
    """Assemble machine-readable extraction board + hop inventory."""
    pin_list = list(pins or [])
    candidates: list[ExtractionCandidate] = []
    stage_exports: list[dict[str, Any]] = []

    # Map stage catalog onto hops when rooms match.
    stage_by_pair: dict[tuple[int, int], str] = {}
    for sid, st in STAGE_CATALOG.items():
        if st.goal_room_id is not None:
            stage_by_pair[(st.room_id, st.goal_room_id)] = sid

    for hop in hops:
        pure = _pure_status(hop.from_room, hop.to_room, hop)
        graph = _lookup_progression_edge(hop.from_room, hop.to_room)
        skills = _skill_modules(hop.tech_tags)
        tas_body = None
        pair = (
            (hop.from_room, hop.to_room)
            if hop.to_room is not None
            else None
        )
        if pair and pair in stage_by_pair:
            st = STAGE_CATALOG[stage_by_pair[pair]]
            tas_body = export_room_body_spec(
                st, pin_list, after_frame=max(0, hop.enter_frame - 1)
            )
            stage_exports.append(tas_body)
        elif hop.usable and hop.to_room is not None and hop.frames:
            tas_body = {
                "schema": "sm_tas_room_body_v1",
                "stage_id": None,
                "track": "tas_import",
                "control": {
                    "room_id": hop.from_room,
                    "room_id_hex": room_hex(hop.from_room),
                    "settle": "gs==8 && door_transition==0 && room_id!=0",
                },
                "goal": {
                    "kind": "enter_room",
                    "room_id": hop.to_room,
                    "room_id_hex": room_hex(hop.to_room),
                },
                "movie_start": hop.enter_frame,
                "body_frames": hop.frames,
                "tech": hop.tech_tags,
                "status": "plan_only",
            }

        candidates.append(
            ExtractionCandidate(
                hop_id=hop.hop_id,
                from_room=hop.from_room,
                to_room=hop.to_room,
                frames=hop.frames,
                tech=list(hop.tech_tags),
                skill_modules=skills,
                pure_status=pure,
                graph_edge_status=graph,
                practice_topology=(
                    "practice_room_graph_may_list_pair; not continuous evidence"
                ),
                tas_body=tas_body,
                residual_next_knob=_residual_knob(hop, pure),
                priority=_priority(hop, pure, graph),
                bead_hint=_bead_hint(hop, pure),
                notes=hop.notes,
            )
        )

    candidates.sort(key=lambda c: (c.priority, c.hop_id))

    # Unique directed edges with hop counts.
    edge_counts: Counter[tuple[int, int]] = Counter()
    for h in hops:
        if h.to_room is not None:
            edge_counts[(h.from_room, h.to_room)] += 1

    unique_edges = [
        {
            "from_room": a,
            "to_room": b,
            "from_room_hex": room_hex(a),
            "to_room_hex": room_hex(b),
            "from_room_name": room_name(a),
            "to_room_name": room_name(b),
            "count": n,
            "product_status": _PRODUCT_EDGE_STATUS.get((a, b), "unknown"),
            "graph_edge_status": _lookup_progression_edge(a, b),
        }
        for (a, b), n in edge_counts.most_common()
    ]

    # Top skill candidates: usable + high-leverage tech, de-duped by edge.
    top_skills: list[dict[str, Any]] = []
    seen_edges: set[tuple[int, int | None]] = set()
    for c in candidates:
        if c.priority > 3:
            continue
        key = (c.from_room, c.to_room)
        if key in seen_edges:
            continue
        if c.pure_status in ("tas_desync_unusable", "continuous_green", "pure_green"):
            if c.pure_status != "pure_open":
                continue
        seen_edges.add(key)
        top_skills.append(c.to_dict())
        if len(top_skills) >= 12:
            break

    # Always surface product Ice P0 (Snake→PLM) even if not in this run's hops.
    # Acid→Snake is pure dual GREEN (rr-5cf); next is Snake→Ice prefer 2WJ.
    ice_pair = (rid.ROOM_ICE_SNAKE, rid.ROOM_ICE)
    if not any(
        c.from_room == ice_pair[0] and c.to_room == ice_pair[1] for c in candidates
    ):
        top_skills.insert(
            0,
            ExtractionCandidate(
                hop_id=f"{run_id}_product_ice_snake_to_ice",
                from_room=ice_pair[0],
                to_room=ice_pair[1],
                frames=None,
                tech=["walljump"],
                skill_modules=["walljump"],
                pure_status="pure_open",
                graph_edge_status="edge_candidate_product_p0",
                practice_topology="not continuous evidence",
                tas_body=None,
                residual_next_knob="pure_probe_2wj_bands",
                priority=0,
                bead_hint="rr-5if",
                notes="Product P0 (always listed). Prefer 2WJ; Acid→Snake dual GREEN rr-5cf.",
            ).to_dict(),
        )

    desync_frames = [
        int(e["frame"])
        for e in (pins or [])
        if e.get("kind") == "desync_suspect"
    ]
    if not desync_frames:
        desync_frames = [
            int(e["frame"])
            for e in []  # filled from hops
        ]
    desync_hops = [h.to_dict() for h in hops if h.desync_in_hop]

    board = {
        "schema": "sm_tas_extraction_board_v1",
        "run_id": run_id,
        "source": source,
        "rules": [
            "TAS is reference — pure-first before graph edge / STATUS",
            "practice room_graph ≠ continuous evidence",
            "never sanitize L+R; assist off on TAS replay",
            "do not auto-wire continuous tip from this board",
        ],
        "summary": {
            "hop_count": len(hops),
            "usable_hops": sum(1 for h in hops if h.usable),
            "unique_edges": len(unique_edges),
            "item_hops": sum(1 for h in hops if h.items_gained or h.beams_gained),
            "desync_hops": sum(1 for h in hops if h.desync_in_hop),
            "death_hops": sum(1 for h in hops if h.death_in_hop),
            "unique_rooms": len({h.from_room for h in hops} | {h.to_room for h in hops if h.to_room}),
        },
        "hops": [h.to_dict() for h in hops],
        "unique_edges": unique_edges,
        "candidates": [c.to_dict() for c in candidates],
        "top_skill_room_candidates": top_skills,
        "stage_body_exports": stage_exports,
        "desync_hops": desync_hops,
        "stage_catalog_ids": sorted(STAGE_CATALOG.keys()),
    }
    return board


def extract_run(run_dir: Path | str) -> dict[str, Any]:
    """Full pipeline: load annotate dir → hops → board."""
    root = Path(run_dir)
    data = load_annotate_dir(root)
    events = _events(data)
    if not events and "pins" in data:
        events = list(data["pins"])
    summary = data.get("summary") or {}
    source = str(summary.get("source") or data.get("trace", {}).get("source") or root.name)
    run_id = root.name
    hops = build_hops(events, run_id=run_id)
    pins = data.get("pins") or [
        e
        for e in events
        if e.get("kind")
        in (
            "room_enter",
            "control",
            "item_gain",
            "beam_gain",
            "desync_suspect",
            "death",
        )
    ]
    board = build_extraction_board(
        hops, pins=pins, run_id=run_id, source=source
    )
    # Enrich summary from annotate summary if present.
    if summary:
        board["annotate_summary"] = {
            k: summary.get(k)
            for k in (
                "num_frames",
                "frames_played",
                "event_count",
                "room_enter_count",
                "unique_rooms",
                "final",
            )
            if k in summary
        }
        ann = summary.get("annotate") or {}
        board["annotate_summary"]["by_kind"] = ann.get("by_kind")
        board["annotate_summary"]["first_control_frame"] = ann.get(
            "first_control_frame"
        )
    return board


def write_board(board: Mapping[str, Any], out_path: Path | str) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
    return path


def write_hop_csv(hops: Sequence[Mapping[str, Any]], out_path: Path | str) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "index",
        "hop_id",
        "enter_frame",
        "frames",
        "from_room_hex",
        "to_room_hex",
        "from_room_name",
        "to_room_name",
        "tech_tags",
        "items_gained",
        "beams_gained",
        "usable",
        "desync_in_hop",
    ]
    lines = [",".join(cols)]
    for h in hops:
        row = []
        for c in cols:
            v = h.get(c, "")
            if isinstance(v, list):
                v = ";".join(str(x) for x in v)
            row.append(str(v).replace(",", ";"))
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Annotate output dir (trace.json / pins.json)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write extraction_board.json path (default: <run_dir>/extraction_board.json)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional hop inventory CSV path",
    )
    p.add_argument(
        "--top",
        type=int,
        default=8,
        help="Print top N skill/room candidates to stdout",
    )
    p.add_argument(
        "--list-stages",
        action="store_true",
        help="Print STAGE_CATALOG and exit",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_stages:
        for sid, st in sorted(STAGE_CATALOG.items()):
            print(
                f"{sid:28s}  {st.room_hex()} → {st.goal_room_hex() or '-':8s}  "
                f"track={st.track:8s}  tech={','.join(st.tech) or '-'}"
            )
        return 0
    if args.run_dir is None:
        print("error: run_dir required (or --list-stages)", file=sys.stderr)
        return 2
    board = extract_run(args.run_dir)
    out = args.out or (Path(args.run_dir) / "extraction_board.json")
    write_board(board, out)
    csv_path = args.csv or (Path(args.run_dir) / "hop_inventory.csv")
    write_hop_csv(board["hops"], csv_path)

    s = board["summary"]
    print(
        f"extraction board: hops={s['hop_count']} usable={s['usable_hops']} "
        f"edges={s['unique_edges']} items={s['item_hops']} "
        f"desync_hops={s['desync_hops']} → {out}"
    )
    print(f"hop csv → {csv_path}")
    print("top skill/room candidates:")
    for i, c in enumerate(board["top_skill_room_candidates"][: args.top], 1):
        print(
            f"  {i}. p{c['priority']} {c['from_room_hex']}→{c.get('to_room_hex')} "
            f"pure={c['pure_status']} tech={c['tech']} skills={c['skill_modules']} "
            f"knob={c['residual_next_knob']} bead={c.get('bead_hint') or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
