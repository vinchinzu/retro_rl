"""Room-hop inventory, skill groups, hop slice resolve, tape extract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_metroid.human_tape.anchors import (
    load_anchors_index,
    match_anchor,
    parse_items_value,
    parse_room_id,
    verify_end_against_trace,
)

# super_metroid/ game root (this file lives in human_tape/)
_GAME_DIR = Path(__file__).resolve().parent.parent

# Route-specific multi-hop skill spans (Grapple / Maridia path markers).
# (start_room, end_room, skill_id, note, prefer_last_start). No-ops when absent.
KNOWN_SKILL_SPANS: list[tuple[int, int, str, str, bool]] = [
    (0xA322, 0xA253, "caterpillar_to_red", "Hellway + Red Tower", False),
    (0xA253, 0xCEFB, "red_to_glass", "Bat→Below→West→Glass tube", False),
    (0xCEFB, 0xA7DE, "glass_to_business", "East→Warehouse→Business", False),
    (0xA7DE, 0xA98D, "business_to_crocomire", "Gate→Crumble→Speedway→Croc", False),
    (0xA98D, 0xAC2B, "croc_to_grapple", "Post-Croc path → Grapple Beam", False),
    (0xAC2B, 0xAB64, "grapple_tutorials_return", "Tutorial 1→2→3 return", True),
    (0xAB64, 0xA7DE, "grapple_return_business", "Croc Escape → Business", True),
    (0xA7DE, 0xCFC9, "business_to_main_street", "elev→Glass→Main Street", True),
]


def load_task_json(task_path: Path | str) -> dict[str, Any]:
    path = Path(task_path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_room_hops(
    trace: Sequence[Mapping[str, Any]],
    *,
    room_names: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Collapse per-frame trace into door-aligned room hops.

    Uses **trace array indices** for dwell (not ``row['frame']`` alone).
    PlaySession checkpoint reloads can renumber ``frame`` non-monotonically
    while the trace list keeps growing — index-based bounds stay correct.

    Starts on the first frame where ``room_id`` changes (often still mid
    ``door_transition``). For bank/timing clocks aligned with live
    ``room_enter`` / RoomTimer settled ordinary entry, pass the result through
    :func:`settle_room_hops`.
    """
    names = room_names or {}
    hops: list[dict[str, Any]] = []
    prev_room: int | None = None
    for i, row in enumerate(trace):
        room = int(row.get("room") or 0)
        if room == prev_room:
            continue
        hops.append(
            {
                "index": len(hops),
                "start_index": i,
                "frame": int(row.get("frame", i)),
                "room": f"0x{room:04X}",
                "room_id": room,
                "name": names.get(room, "?"),
                "xy": [int(row.get("x", 0)), int(row.get("y", 0))],
                "pose": int(row.get("pose", 0)),
                "energy": row.get("energy"),
                "missiles": row.get("missiles"),
                "items": (
                    f"0x{int(row['items']):04X}"
                    if isinstance(row.get("items"), int)
                    else row.get("items")
                ),
            }
        )
        prev_room = room

    n_trace = len(trace)
    for i, hop in enumerate(hops):
        end_i = (
            hops[i + 1]["start_index"] - 1
            if i + 1 < len(hops)
            else max(0, n_trace - 1)
        )
        last = trace[end_i] if end_i < n_trace else trace[-1]
        hop["end_index"] = end_i
        hop["end_frame"] = int(last.get("frame", end_i))
        hop["end_xy"] = [int(last.get("x", 0)), int(last.get("y", 0))]
        hop["end_pose"] = int(last.get("pose", 0))
        # Dwell in recorded samples (stable under checkpoint renumbering).
        hop["dwell"] = int(end_i) - int(hop["start_index"]) + 1
        if last.get("items") is not None:
            items = last["items"]
            hop["end_items"] = (
                f"0x{int(items):04X}" if isinstance(items, int) else items
            )
    return hops


def _row_is_settled_ordinary(row: Mapping[str, Any]) -> bool:
    """True when a trace row matches live room_enter / RoomTimer settle."""
    if int(row.get("door_transition") or 0) != 0:
        return False
    phase = row.get("phase")
    if phase is None:
        # Older traces omit phase; door_transition==0 is the settle signal.
        return True
    label = str(
        getattr(phase, "name", None)
        or getattr(phase, "value", None)
        or phase
    ).lower()
    return "ordinary" in label


def _copy_hop(hop: Mapping[str, Any]) -> dict[str, Any]:
    """Shallow hop copy with list fields duplicated."""
    out = dict(hop)
    for key in ("xy", "end_xy"):
        val = out.get(key)
        if isinstance(val, list):
            out[key] = list(val)
    return out


def _format_items(items: Any) -> Any:
    if isinstance(items, int):
        return f"0x{items:04X}"
    return items


def hop_items_int(hop: Mapping[str, Any], *, key: str | None = None) -> int | None:
    """Parse ``items`` / ``end_items`` from a hop dict (int or ``0xHHHH`` hex).

    Tries ``items`` then ``end_items`` unless ``key`` is set.
    """
    keys: tuple[str, ...] = (key,) if key is not None else ("items", "end_items")
    for k in keys:
        val = hop.get(k)
        if val is None:
            continue
        # Bool is not a valid items mask in hop dicts (legacy skip).
        if isinstance(val, bool):
            continue
        parsed = parse_items_value(val)
        if parsed is not None:
            return parsed
    return None


def settle_room_hops(
    hops: Sequence[Mapping[str, Any]],
    trace: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Align hop starts to first settled ordinary frame (RoomTimer clock).

    ``build_room_hops`` starts on the room-id change edge, which is often still
    ``door_transition != 0`` / room_transition phase. Live ``room_enter`` anchors
    and RoomTimer use the first settled ordinary sample. This helper deep-copies
    hops and moves ``start_index`` forward within each hop's ``[start, end]``
    range when a settled row exists — raw hops and replay slice bounds stay
    available from the original list.

    Added / updated fields on each returned hop:
      - ``raw_start_index`` — original room-change edge index
      - ``settled_entry`` — True when a settled ordinary row was found
      - ``transition_frames`` — ``start_index - raw_start_index`` (leading load)
      - ``start_index`` / ``frame`` / ``dwell`` / ``xy`` / ``pose`` / ``items``
        recomputed from the settled start row when the edge was mid-transition
    """
    n_trace = len(trace)
    out: list[dict[str, Any]] = []
    for hop in hops:
        h = _copy_hop(hop)
        raw_start = int(h["start_index"])
        end_i = int(h.get("end_index", raw_start))
        h["raw_start_index"] = raw_start

        settled_i: int | None = None
        if n_trace:
            lo = max(0, raw_start)
            hi = min(end_i, n_trace - 1)
            for i in range(lo, hi + 1):
                if _row_is_settled_ordinary(trace[i]):
                    settled_i = i
                    break

        if settled_i is None:
            h["settled_entry"] = False
            h["transition_frames"] = 0
            out.append(h)
            continue

        h["settled_entry"] = True
        h["transition_frames"] = int(settled_i) - raw_start
        if settled_i != raw_start:
            row = trace[settled_i]
            h["start_index"] = settled_i
            h["frame"] = int(row.get("frame", settled_i))
            h["dwell"] = int(end_i) - int(settled_i) + 1
            h["xy"] = [int(row.get("x", 0)), int(row.get("y", 0))]
            h["pose"] = int(row.get("pose", 0))
            if row.get("energy") is not None:
                h["energy"] = row.get("energy")
            if row.get("missiles") is not None:
                h["missiles"] = row.get("missiles")
            if row.get("items") is not None:
                h["items"] = _format_items(row["items"])
        out.append(h)
    return out


def default_skill_groups(hops: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Heuristic multi-hop skill bundles.

    Generic fallback: one skill per room hop. Named groups fire when known
    room ids appear in order.

    ``KNOWN_SKILL_SPANS`` is **route-specific** (Grapple / Maridia path
    markers) — convenience for those tapes, not a general skill ontology.
    Spans no-op when those rooms are absent from the hop list.
    """
    if not hops:
        return []
    by_room = {int(h["room_id"]): h for h in hops}
    groups: list[dict[str, Any]] = []

    def _span(
        start_room: int,
        end_room: int,
        skill_id: str,
        note: str,
        *,
        prefer_last_start: bool = False,
    ) -> None:
        # Ordered (start_i, end_i) pairs with start before end; take first or last.
        pairs: list[tuple[int, int]] = []
        for i, h in enumerate(hops):
            if int(h["room_id"]) != start_room:
                continue
            for j in range(i + 1, len(hops)):
                if int(hops[j]["room_id"]) == end_room:
                    pairs.append((i, j))
                    break
        if not pairs:
            return
        start_i, end_i = pairs[-1] if prefer_last_start else pairs[0]
        h0, h1 = hops[start_i], hops[end_i]
        groups.append(
            {
                "id": skill_id,
                "frames": [h0["frame"], h1["end_frame"]],
                "rooms": [hops[i]["room"] for i in range(start_i, end_i + 1)],
                "note": note,
            }
        )

    # prefer_last_start=True for return legs that re-visit early rooms.
    seen_ids: set[str] = set()
    for a, b, sid, note, last_start in KNOWN_SKILL_SPANS:
        if a in by_room and b in by_room and sid not in seen_ids:
            _span(a, b, sid, note, prefer_last_start=last_start)
            seen_ids.add(sid)

    if not groups:
        for h in hops:
            groups.append(
                {
                    "id": f"hop_{h['index']:02d}_{h['room']}",
                    "frames": [h["frame"], h["end_frame"]],
                    "rooms": [h["room"]],
                    "note": h.get("name") or h["room"],
                }
            )
    return groups


def load_room_names(graph_path: Path | None = None) -> dict[int, str]:
    """room-id → Map Rando / sm-json-data name.

    Prefers ``maps/maprando_room_names.json`` (canonical Map Rando names).
    Falls back to ``full_room_graph.json`` when the compact index is absent.
    Pass ``graph_path`` to force a specific graph/catalog JSON.
    """
    if graph_path is None:
        try:
            from super_metroid.rooms.canonical_names import load_canonical_names

            names = load_canonical_names()
            if names:
                return names
        except (OSError, ImportError, ValueError, TypeError, KeyError):
            pass
        graph_path = _GAME_DIR / "maps" / "full_room_graph.json"
    if not graph_path.is_file():
        return {}
    try:
        data = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    rooms = data.get("rooms") or []
    out: dict[int, str] = {}
    for r in rooms:
        rid = r.get("roomId") if "roomId" in r else r.get("room_id")
        if rid is None and r.get("roomIdHex"):
            try:
                rid = int(str(r["roomIdHex"]), 0)
            except ValueError:
                rid = None
        name = r.get("name")
        if rid is not None and name:
            out[int(rid)] = str(name)
    return out


def extract_tape(
    task_path: Path | str,
    *,
    room_names: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Offline hop inventory + skill groups from a guided_human task JSON."""
    path = Path(task_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    trace = list(data.get("trace") or [])
    meta = dict(data.get("metadata") or {})
    names = dict(room_names) if room_names is not None else load_room_names()
    hops = build_room_hops(trace, room_names=names)
    skills = default_skill_groups(hops)
    end_fp = meta.get("end_fingerprint")
    verify = None
    if end_fp and trace:
        verify = verify_end_against_trace(end_fp, trace)
    anchors_path = path.with_name(path.stem + "_anchors.json")
    anchors = None
    if anchors_path.is_file():
        try:
            anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            anchors = {"error": f"unreadable {anchors_path}"}
    return {
        "task": str(path),
        "name": data.get("name") or path.stem,
        "frame_count": int(data.get("frame_count") or len(data.get("frames") or [])),
        "start_state": data.get("start_state"),
        "recorded_at": data.get("recorded_at"),
        "assist": (meta.get("assist") or {}),
        "end_fingerprint": end_fp,
        "end_verify": verify,
        "room_hops": hops,
        "skill_groups": skills,
        "anchors_index": str(anchors_path) if anchors_path.is_file() else None,
        "anchors": anchors,
        "transitions_meta": meta.get("transitions"),
    }


@dataclass(frozen=True)
class HopSlice:
    """Resolved hop frame window + optional live anchor for open-loop replay."""

    task: str
    name: Any
    frame_count: int
    start_index: int
    end_index: int
    replay_start: int
    start_room: int | None
    leave_room: int | None
    start_xy: list[int] | None
    end_xy: list[int] | None
    anchor_path: str | None
    anchor_frame: int | None
    hop_index: int | None
    hop: dict[str, Any] | None
    anchor: dict[str, Any] | None
    n_hops: int
    steps: int
    start_room_hex: str | None = None
    leave_room_hex: str | None = None
    anchor_warning: str | None = None
    anchor_mismatch_risk: bool = False
    anchor_room_mismatch: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_room_hops(
    task_path: Path | str | None = None,
    *,
    task_data: Mapping[str, Any] | None = None,
    room_names: Mapping[int, str] | None = None,
    settle: bool = True,
    trace: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Canonical hop inventory for replay / bodies / bank.

    Default ``settle=True`` aligns starts with live ``room_enter`` / RoomTimer
    so hop-replay bounds match materialize body export.
    """
    if trace is None:
        if task_data is not None:
            data = dict(task_data)
        elif task_path is not None:
            data = load_task_json(task_path)
        else:
            raise ValueError("load_room_hops requires task_path, task_data, or trace")
        trace = list(data.get("trace") or [])
    names = dict(room_names) if room_names is not None else load_room_names()
    raw = build_room_hops(trace, room_names=names) if trace else []
    if settle and raw:
        return settle_room_hops(raw, trace)
    return list(raw)


def resolve_hop_slice(
    task_path: Path | str,
    *,
    hop_index: int | None = None,
    from_frame: int | None = None,
    to_frame: int | None = None,
    to_room: int | str | None = None,
    room: int | str | None = None,
    frames_count: int | None = None,
    leave_extra: int = 1,
    room_names: Mapping[int, str] | None = None,
    anchors_index: Mapping[str, Any] | None = None,
    task_data: Mapping[str, Any] | None = None,
    settle: bool = True,
) -> dict[str, Any]:
    """Resolve a hop frame window + optional live anchor for open-loop replay.

    Selection modes (first match wins):
      - ``hop_index``: Nth room hop (settled by default)
      - ``room`` (+ optional ``to_room``): first hop with start room, optional leave
      - ``from_frame`` / ``to_frame`` / ``frames_count``: explicit index window

    Default ``settle=True`` matches materialize hop bodies / bank dwell.

    Returns a dict (``HopSlice.to_dict()``) with keys: start_index, end_index,
    start_room, leave_room, start_xy, end_xy, anchor_path, anchor_frame,
    replay_start, hop, …
    """
    return resolve_hop_slice_typed(
        task_path,
        hop_index=hop_index,
        from_frame=from_frame,
        to_frame=to_frame,
        to_room=to_room,
        room=room,
        frames_count=frames_count,
        leave_extra=leave_extra,
        room_names=room_names,
        anchors_index=anchors_index,
        task_data=task_data,
        settle=settle,
    ).to_dict()


def resolve_hop_slice_typed(
    task_path: Path | str,
    *,
    hop_index: int | None = None,
    from_frame: int | None = None,
    to_frame: int | None = None,
    to_room: int | str | None = None,
    room: int | str | None = None,
    frames_count: int | None = None,
    leave_extra: int = 1,
    room_names: Mapping[int, str] | None = None,
    anchors_index: Mapping[str, Any] | None = None,
    task_data: Mapping[str, Any] | None = None,
    settle: bool = True,
) -> HopSlice:
    """Typed hop-slice resolve; see ``resolve_hop_slice`` for selection modes."""
    path = Path(task_path)
    data = dict(task_data) if task_data is not None else load_task_json(path)
    trace = list(data.get("trace") or [])
    n_frames = int(data.get("frame_count") or len(data.get("frames") or []) or len(trace))
    names = dict(room_names) if room_names is not None else load_room_names()
    hops = load_room_hops(
        task_data=data,
        room_names=names,
        settle=settle,
        trace=trace,
    )

    if anchors_index is None:
        anchors_index = load_anchors_index(path)

    hop: dict[str, Any] | None = None
    start_index: int
    end_index: int
    start_room: int | None = None
    leave_room: int | None = None
    start_xy: list[int] | None = None
    end_xy: list[int] | None = None

    if hop_index is not None:
        if hop_index < 0 or hop_index >= len(hops):
            raise IndexError(
                f"hop_index {hop_index} out of range (0..{max(0, len(hops) - 1)}; "
                f"{len(hops)} hops)"
            )
        hop = dict(hops[hop_index])
    elif room is not None:
        rid = parse_room_id(room)
        if rid is None:
            raise ValueError(f"invalid room {room!r}")
        to_rid = parse_room_id(to_room) if to_room is not None else None
        found = None
        for h in hops:
            if int(h["room_id"]) != rid:
                continue
            if to_rid is not None:
                nxt = hops[h["index"] + 1] if h["index"] + 1 < len(hops) else None
                if nxt is None or int(nxt["room_id"]) != to_rid:
                    continue
            found = h
            break
        if found is None:
            raise ValueError(
                f"no hop for room 0x{rid:04X}"
                + (f" → 0x{to_rid:04X}" if to_rid is not None else "")
            )
        hop = dict(found)

    if hop is not None:
        start_index = int(hop["start_index"])
        # Inclusive last frame still in room; +leave_extra to observe leave.
        end_index = int(hop["end_index"]) + max(0, int(leave_extra))
        start_room = int(hop["room_id"])
        start_xy = list(hop.get("xy") or [0, 0])
        end_xy = list(hop.get("end_xy") or start_xy)
        nxt_i = int(hop["index"]) + 1
        if nxt_i < len(hops):
            leave_room = int(hops[nxt_i]["room_id"])
        else:
            leave_room = None
        if to_room is not None:
            tr = parse_room_id(to_room)
            if tr is not None:
                leave_room = tr
    else:
        if from_frame is None:
            raise ValueError(
                "resolve_hop_slice requires hop_index, room, or from_frame"
            )
        start_index = int(from_frame)
        explicit_end = to_frame is not None or frames_count is not None
        if to_frame is not None:
            end_index = int(to_frame)
        elif frames_count is not None:
            end_index = start_index + int(frames_count) - 1
        else:
            end_index = start_index
        if trace and 0 <= start_index < len(trace):
            row = trace[start_index]
            start_room = int(row.get("room") or 0)
            start_xy = [int(row.get("x", 0)), int(row.get("y", 0))]
        if trace and 0 <= min(end_index, len(trace) - 1) < len(trace):
            row = trace[min(end_index, len(trace) - 1)]
            end_xy = [int(row.get("x", 0)), int(row.get("y", 0))]
            end_room = int(row.get("room") or 0)
            leave_room = end_room
            # Only auto-extend to leave when the caller did not pin an end window
            if (
                not explicit_end
                and start_room is not None
                and end_room == start_room
            ):
                for j in range(end_index, len(trace)):
                    r = int(trace[j].get("room") or 0)
                    if r != start_room:
                        leave_room = r
                        end_xy = [int(trace[j].get("x", 0)), int(trace[j].get("y", 0))]
                        end_index = max(end_index, j)
                        break
        if to_room is not None:
            tr = parse_room_id(to_room)
            if tr is not None:
                leave_room = tr
        if room is not None and start_room is None:
            start_room = parse_room_id(room)

    # Explicit overrides after hop selection
    if from_frame is not None and hop is not None:
        start_index = int(from_frame)
    if to_frame is not None and hop is not None:
        end_index = int(to_frame)
    elif frames_count is not None and hop is not None and from_frame is not None:
        end_index = int(from_frame) + int(frames_count) - 1

    end_index = min(end_index, max(0, n_frames - 1))
    start_index = max(0, min(start_index, end_index))

    anchor = match_anchor(
        anchors_index,
        start_index,
        start_room,
        task_path=path,
    )
    anchor_path = Path(anchor["path"]) if anchor and anchor.get("path") else None
    anchor_frame = int(anchor["frame"]) if anchor and anchor.get("frame") is not None else None

    # replay_start: after loading dump taken post-step at frame F, next input is F+1.
    # No silent magic clamp ladder — flag risks instead.
    anchor_warning: str | None = None
    anchor_mismatch_risk = False
    anchor_room_mismatch = False

    if anchor_frame is not None:
        replay_start = min(max(int(anchor_frame) + 1, 0), end_index)
    else:
        replay_start = start_index

    if anchor is not None and start_room is not None:
        ar = anchor.get("room_id")
        if ar is None:
            ar = parse_room_id(anchor.get("room"))
        if ar is not None and int(ar) != int(start_room):
            anchor_room_mismatch = True
            anchor_mismatch_risk = True
            anchor_warning = (
                f"anchor room 0x{int(ar):04X} != hop start room "
                f"0x{int(start_room):04X} (frame={anchor_frame}, "
                f"kind={anchor.get('kind')})"
            )

    # Far-before boot (or other pin): replaying the huge prefix is rarely right.
    # Prefer a room_enter near hop start. If only a far pin exists, clamp to hop
    # start when rooms match and flag risk — never silent wrong-room boot.
    if (
        hop is not None
        and anchor is not None
        and anchor_frame is not None
        and not anchor_room_mismatch
        and int(anchor_frame) < int(hop["start_index"])
    ):
        hop_start = int(hop["start_index"])
        gap = hop_start - int(anchor_frame)
        kind = str(anchor.get("kind") or "pin")
        # Clamp when pin is strictly before hop (boot class especially).
        if gap > 0 and kind in ("boot", "end", "pin"):
            replay_start = hop_start
            anchor_mismatch_risk = True
            anchor_warning = (
                f"{kind} anchor at frame {anchor_frame} is {gap} frames before "
                f"hop start {hop_start}; clamped replay_start to hop start "
                f"(prefer a room_enter / mid_lockstep pin)"
            )

    steps = max(0, int(end_index) - int(replay_start) + 1)
    return HopSlice(
        task=str(path),
        name=data.get("name") or path.stem,
        frame_count=n_frames,
        start_index=start_index,
        end_index=end_index,
        replay_start=int(replay_start),
        start_room=start_room,
        leave_room=leave_room,
        start_xy=start_xy,
        end_xy=end_xy,
        anchor_path=str(anchor_path) if anchor_path else None,
        anchor_frame=anchor_frame,
        hop_index=int(hop["index"]) if hop else hop_index,
        hop=hop,
        anchor=anchor,
        n_hops=len(hops),
        steps=steps,
        start_room_hex=f"0x{start_room:04X}" if start_room is not None else None,
        leave_room_hex=f"0x{leave_room:04X}" if leave_room is not None else None,
        anchor_warning=anchor_warning,
        anchor_mismatch_risk=anchor_mismatch_risk,
        anchor_room_mismatch=anchor_room_mismatch,
    )
