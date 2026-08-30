"""Route-manifest load/save and product-chain board adapter.

Hop order comes from the board (or a loaded manifest). Optional TipSpec hop
ids label edges by index; they do not reorder. Does not write bank.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_metroid.hop_id import make_hop_key, parse_items_value, parse_room_id
from super_metroid.leave_specs import LeaveSpec
from super_metroid.splice.errors import SchemaError
from super_metroid.splice.preflight import file_digest
from super_metroid.splice.schema import (
    CANDIDATE_KINDS,
    Capacities,
    EntryContract,
    EntryFingerprint,
    LeaveSpecRef,
    RouteEdge,
    RouteManifest,
    rel_path,
)

DEFAULT_OWNER = "snes/super_metroid/routes/kpdr"
_WIDE = (0, 10_000)


def load_manifest(path: Path | str) -> RouteManifest:
    dest = Path(path)
    try:
        raw = json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(
            f"cannot read manifest {dest}",
            code="schema.manifest",
            details={"path": rel_path(dest)},
        ) from exc
    if not isinstance(raw, dict):
        raise SchemaError("manifest must be an object", code="schema.type")
    return RouteManifest.from_dict(raw)


def dest_leave_spec(*, hop: str, room_id: int) -> LeaveSpec:
    """Wide dest-room glance when no named LeaveSpec is bound yet."""
    return LeaveSpec(hop=hop, room=int(room_id), x=_WIDE, y=_WIDE, pose_class="any")


def _board_hops(board: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    hops = board.get("hops")
    if hops is None:
        raise SchemaError("board hops required", code="schema.missing")
    if not isinstance(hops, (list, tuple)):
        raise SchemaError("board hops must be a sequence", code="schema.type")
    out: list[Mapping[str, Any]] = []
    for row in hops:
        if not isinstance(row, Mapping):
            raise SchemaError("each board hop must be an object", code="schema.type")
        out.append(row)
    return out


def _kinds_for(hop: Mapping[str, Any]) -> tuple[str, ...]:
    kinds: list[str] = ["tape", "controller"]
    if hop.get("policy_id"):
        kinds.append("reactive_policy")
    if str(hop.get("mode") or "") == "combat":
        kinds.append("boss")
    return tuple(k for k in CANDIDATE_KINDS if k in set(kinds))


def _selected_for(hop: Mapping[str, Any], allowed: Sequence[str]) -> dict[str, str]:
    allowed_set = set(allowed)
    selected: dict[str, str] = {}
    policy = str(hop.get("policy_id") or "").strip()
    if policy and "reactive_policy" in allowed_set:
        selected["survival"] = f"reactive_policy:{policy}"
    tape_kind = "tape" if "tape" in allowed_set else (allowed[0] if allowed else "")
    if tape_kind:
        selected.setdefault("scaffold", f"{tape_kind}:board")
        selected.setdefault("clean", f"{tape_kind}:board")
        selected.setdefault("survival", f"{tape_kind}:board")
    return selected


def _task_id(base: str, used: set[str], index: int) -> str:
    text = str(base).strip()
    if not text:
        raise SchemaError("empty task_id", code="schema.task_id")
    if text not in used:
        return text
    return f"{text}#{index}"


def manifest_from_board(
    board: Mapping[str, Any],
    *,
    hop_ids: Sequence[str] | None = None,
    route_id: str | None = None,
    variant: str = "kpdr",
    owner_package: str = DEFAULT_OWNER,
) -> RouteManifest:
    """Build a RouteManifest from a product-chain-like hop list.

    ``hop_ids`` labels edges in board order (TipSpec hop ids). It never
    reorders hops.
    """
    hops = _board_hops(board)
    labels = [str(x).strip() for x in (hop_ids or ())]
    used: set[str] = set()
    edges: list[RouteEdge] = []
    for i, hop in enumerate(hops):
        room = parse_room_id(hop.get("room_id", hop.get("room")))
        if room is None:
            raise SchemaError(
                f"board hop {i} missing room_id",
                code="schema.room",
                details={"index": i},
            )
        pred = parse_room_id(hop.get("from_room_id", hop.get("from_room")))
        nxt = parse_room_id(hop.get("to_room_id", hop.get("to_room")))
        items = parse_items_value(hop.get("items", hop.get("items_hex")))
        goal = str(hop["goal"]).strip() if hop.get("goal") else None
        hop_key = make_hop_key(
            int(room),
            from_room_id=pred,
            to_room_id=nxt,
            items=items,
            goal=goal,
        )
        label = labels[i] if i < len(labels) and labels[i] else hop_key
        task_id = _task_id(label, used, i)
        used.add(task_id)
        leave_room = int(nxt) if nxt is not None else int(room)
        leave = dest_leave_spec(hop=hop_key, room_id=leave_room)
        pin = rel_path(hop.get("anchor_path"))
        tape = rel_path(hop.get("tape"))
        dwell = hop.get("dwell")
        try:
            max_frames = max(int(dwell), 1) if dwell not in (None, "") else 10_000
        except (TypeError, ValueError) as exc:
            raise SchemaError(
                f"board hop {i} dwell is not int",
                code="schema.budget",
            ) from exc
        start = hop.get("start_index", hop.get("frame"))
        try:
            frame_start = int(start) if start not in (None, "") else None
        except (TypeError, ValueError):
            frame_start = None
        frame_end = None if frame_start is None else frame_start + max_frames - 1
        allowed = _kinds_for(hop)
        notes = hop.get("notes") or ()
        if isinstance(notes, str):
            notes = (notes,)
        entry = EntryContract(
            fingerprint=EntryFingerprint(
                room_id=int(room),
                items=items,
                prior_room_id=pred,
            ),
            state_path=pin,
            state_digest=file_digest(hop.get("anchor_path")) if hop.get("anchor_path") else None,
        )
        edges.append(
            RouteEdge.from_dict(
                {
                    "task_id": task_id,
                    "hop_key": hop_key,
                    "room_id": int(room),
                    "predecessor_room_id": pred,
                    "next_room_id": nxt,
                    "goal": goal,
                    "required_items": items,
                    "entry": entry.to_dict(),
                    "successor_leave": LeaveSpecRef.from_leave_spec(leave).to_dict(),
                    "allowed_kinds": list(allowed),
                    "selected": _selected_for(hop, allowed),
                    "owner_package": owner_package,
                    "integration_order": i,
                    "max_frames": max_frames,
                    "max_no_progress": max(1, min(600, max_frames)),
                    "route_variant": variant,
                    "segment": hop.get("segment"),
                    "hop_index": hop.get("hop_index", i),
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "tape_path": tape,
                    "tape_digest": file_digest(hop.get("tape")) if hop.get("tape") else None,
                    "source_notes": [str(n) for n in notes],
                    "capacities": Capacities().to_dict(),
                }
            )
        )
    rid = route_id or str(board.get("task") or board.get("kind") or "product_chain")
    return RouteManifest.from_dict(
        {
            "route_id": rid,
            "variant": variant,
            "edges": [e.to_dict() for e in edges],
        }
    )


def manifest_from_product_chain(
    task_path: Path | str | None = None,
    *,
    hop_ids: Sequence[str] | None = None,
    include_live: bool = True,
) -> RouteManifest:
    """Adapter over ``build_product_chain_board`` (migration oracle)."""
    from super_metroid.human_tape.product_chain import (
        DEFAULT_TASK,
        build_product_chain_board,
    )

    path = Path(task_path) if task_path is not None else DEFAULT_TASK
    board = build_product_chain_board(path, include_live=include_live)
    return manifest_from_board(board, hop_ids=hop_ids, route_id=rel_path(path) or "product_chain")
