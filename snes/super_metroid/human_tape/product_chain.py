"""full_start_v1 product-chain hop inventory (work list for rr-4nli).

Each row is one settled room hop on the seam-deduped RTA chain. This is the
autopilot work list: hop-replay dual-green from a live pin is the *seed*;
RoomAutopilot + ``room_adapter.search_live_adapter`` then join from the exact
live emulator state (subpixels, door speed, enemy phase). Not tape concat.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_metroid.hop_id import make_hop_key
from super_metroid.human_tape.anchors import (
    load_anchors_index,
    match_anchor,
    parse_items_value,
    parse_room_id,
    resolve_anchor_path,
)
from super_metroid.human_tape.hops import hop_items_int, load_room_names
from super_metroid.human_tape.rta_clock import product_chain_segments
from super_metroid.paths import GAME_DIR
from super_metroid.human_tape.trim import COMBAT_ROOM_IDS

DEFAULT_TASK = GAME_DIR / "tasks" / "full_start_v1.json"
DEFAULT_BOARD = GAME_DIR / "tasks" / "PRODUCT_CHAIN_HOP_BOARD.json"
DEFAULT_POLICY_DIR = GAME_DIR / "policies" / "reactive_rooms"
DEFAULT_BANK = GAME_DIR / "recordings" / "skill_bank" / "bank.json"

# How AP is supposed to absorb door / subpixel / enemy mismatch.
AP_JOIN = (
    "RoomAutopilot attaches mid-room; room_adapter.search_live_adapter "
    "starts from exact live RAM (subpixels, vx/mom, pose, enemy phase) "
    "and pulse-searches onto the policy trajectory. Door kinematics live "
    "in door_kinematics.DoorKinematics — do not assume the recorded pin "
    "speed. Enemy RNG is not cut out of tapes; AP must rejoin live phase."
)


@dataclass(frozen=True)
class PolicyIndexRow:
    policy_id: str
    room_id: int
    from_room_id: int | None
    exit_room_id: int | None
    status: str
    path: str


@dataclass
class ProductChainHop:
    """One settled hop on the product RTA chain."""

    segment: str
    hop_index: int
    hop_key: str
    room_id: int
    room: str
    name: str
    from_room_id: int | None
    to_room_id: int | None
    items: int | None
    items_hex: str
    dwell: int
    mode: str
    tape: str
    has_anchor: bool
    anchor_path: str | None
    anchor_kind: str | None
    policy_id: str | None
    policy_status: str | None
    bank_dual_green: bool
    bank_frames: int | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.from_room_id is not None:
            d["from_room"] = f"0x{self.from_room_id:04X}"
        if self.to_room_id is not None:
            d["to_room"] = f"0x{self.to_room_id:04X}"
        return d


def _rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(GAME_DIR.resolve()))
    except ValueError:
        return str(p)


def _safe_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_policy_index(
    policy_dir: Path | str = DEFAULT_POLICY_DIR,
) -> tuple[PolicyIndexRow, ...]:
    """Scan compiled reactive-room JSON (no emulator)."""
    root = Path(policy_dir)
    rows: list[PolicyIndexRow] = []
    if not root.is_dir():
        return ()
    for cand in sorted(root.glob("*.json")):
        raw = _safe_json(cand)
        if not raw or raw.get("kind") != "super_metroid_reactive_room_policy":
            continue
        room = parse_room_id(raw.get("roomId") if raw.get("roomId") is not None else raw.get("room_id"))
        if room is None:
            continue
        frm = parse_room_id(
            raw.get("fromRoomId") if raw.get("fromRoomId") is not None else raw.get("from_room_id")
        )
        ext = parse_room_id(
            raw.get("exitRoomId") if raw.get("exitRoomId") is not None else raw.get("exit_room_id")
        )
        rows.append(
            PolicyIndexRow(
                policy_id=str(raw.get("policyId") or raw.get("policy_id") or cand.stem),
                room_id=int(room),
                from_room_id=int(frm) if frm is not None else None,
                exit_room_id=int(ext) if ext is not None else None,
                status=str(raw.get("status") or ""),
                path=str(cand),
            )
        )
    return tuple(rows)


def match_policy(
    index: Sequence[PolicyIndexRow],
    *,
    room_id: int,
    from_room_id: int | None,
    to_room_id: int | None,
) -> PolicyIndexRow | None:
    """Prefer from-room + exit match, then room-only."""

    def score(row: PolicyIndexRow) -> tuple[int, int, str]:
        from_hit = int(
            from_room_id is not None and row.from_room_id == int(from_room_id)
        )
        exit_hit = int(to_room_id is not None and row.exit_room_id == int(to_room_id))
        return (from_hit, exit_hit, row.policy_id)

    cands = [r for r in index if r.room_id == int(room_id)]
    if not cands:
        return None
    return max(cands, key=score)


def _load_bank_best(bank_path: Path) -> dict[str, tuple[bool, int]]:
    """hop_key → (dual_green, frames) for the bank's best row."""
    raw = _safe_json(bank_path)
    if not raw:
        return {}
    try:
        from super_metroid.skill_bank import SkillBank

        bank = SkillBank.from_dict(raw)
    except (ImportError, TypeError, ValueError, KeyError):
        bank = None
    out: dict[str, tuple[bool, int]] = {}
    if bank is not None:
        for key in bank.records:
            rec = bank.best(key)
            if rec is not None:
                out[key] = (bool(rec.dual_green), int(rec.frames))
        return out
    # Fallback: raw JSON {hop_key: [records]} or {records: ...}
    records = raw.get("records") if isinstance(raw.get("records"), dict) else raw
    if not isinstance(records, dict):
        return {}
    for key, rows in records.items():
        if not isinstance(rows, list) or not rows:
            continue
        best = min(
            (r for r in rows if isinstance(r, dict)),
            key=lambda r: (not bool(r.get("dual_green")), int(r.get("frames") or 10**9)),
            default=None,
        )
        if best is not None:
            out[str(key)] = (bool(best.get("dual_green")), int(best.get("frames") or 0))
    return out


def _hops_from_extract(extract: Mapping[str, Any]) -> list[dict[str, Any]]:
    hops = extract.get("room_hops") or extract.get("hops_settled") or extract.get("hops")
    if isinstance(hops, list):
        return [dict(h) for h in hops if isinstance(h, Mapping)]
    return []


def _leave_room(hops: Sequence[Mapping[str, Any]], i: int) -> int | None:
    if i + 1 < len(hops):
        return parse_room_id(hops[i + 1].get("room_id") or hops[i + 1].get("room"))
    return None


def _from_room(hops: Sequence[Mapping[str, Any]], i: int) -> int | None:
    if i > 0:
        return parse_room_id(hops[i - 1].get("room_id") or hops[i - 1].get("room"))
    return None


def _segment_tape(row: Mapping[str, Any], task_path: Path) -> Path | None:
    source = str(row.get("source") or "")
    if source.startswith("s") and source[1:].isdigit():
        tape = task_path.with_name(task_path.stem + "_segments") / source / "tape.json"
        if tape.is_file():
            return tape
    return None


def build_product_chain_board(
    task_path: Path | str = DEFAULT_TASK,
    *,
    include_live: bool = True,
    policy_dir: Path | str = DEFAULT_POLICY_DIR,
    bank_path: Path | str = DEFAULT_BANK,
) -> dict[str, Any]:
    """Inventory every product-chain hop (offline, no emulator)."""
    path = Path(task_path)
    names = load_room_names()
    policies = load_policy_index(policy_dir)
    bank_best = _load_bank_best(Path(bank_path))
    chain, notes = product_chain_segments(path)
    hops_out: list[ProductChainHop] = []
    missing_anchor = 0
    missing_policy = 0
    combat_n = 0
    dual_n = 0

    def ingest(seg_label: str, tape: Path, extract: Mapping[str, Any] | None) -> None:
        nonlocal missing_anchor, missing_policy, combat_n, dual_n
        hops = _hops_from_extract(extract or {})
        if not hops:
            notes.append(f"{seg_label}: no extract room_hops")
            return
        anchors = load_anchors_index(tape)
        for i, hop in enumerate(hops):
            room = parse_room_id(hop.get("room_id") or hop.get("room")) or 0
            items = hop_items_int(hop)
            frm = _from_room(hops, i)
            to = _leave_room(hops, i)
            key = make_hop_key(room, from_room_id=frm, to_room_id=to, items=items)
            start_i = int(hop.get("start_index") or hop.get("frame") or 0)
            hit = match_anchor(anchors, start_i, room, task_path=tape) if anchors else None
            apath = None
            akind = None
            if hit is not None:
                resolved = resolve_anchor_path(hit, anchors_index=anchors, task_path=tape)
                apath = str(resolved) if resolved is not None else hit.get("path")
                akind = str(hit.get("kind") or "")
            pol = match_policy(policies, room_id=room, from_room_id=frm, to_room_id=to)
            bank = bank_best.get(key)
            mode = "combat" if room in COMBAT_ROOM_IDS else "traversal"
            row_notes: list[str] = []
            if not apath:
                missing_anchor += 1
                row_notes.append("no live enter/boot pin")
            if pol is None:
                missing_policy += 1
                row_notes.append("no reactive policy — AP cannot join")
            if mode == "combat":
                combat_n += 1
                row_notes.append("combat: enemy phase from live RAM, not tape splice")
            if bank and bank[0]:
                dual_n += 1
            hops_out.append(
                ProductChainHop(
                    segment=seg_label,
                    hop_index=int(hop.get("index", i)),
                    hop_key=key,
                    room_id=int(room),
                    room=f"0x{int(room):04X}",
                    name=str(hop.get("name") or names.get(int(room), "?")),
                    from_room_id=frm,
                    to_room_id=to,
                    items=items,
                    items_hex=f"0x{int(items):04X}" if items is not None else "any",
                    dwell=int(hop.get("dwell") or 0),
                    mode=mode,
                    tape=_rel(tape) or str(tape),
                    has_anchor=bool(apath),
                    anchor_path=_rel(apath) if apath else None,
                    anchor_kind=akind,
                    policy_id=pol.policy_id if pol else None,
                    policy_status=pol.status if pol else None,
                    bank_dual_green=bool(bank[0]) if bank else False,
                    bank_frames=int(bank[1]) if bank else None,
                    notes=row_notes,
                )
            )

    for row in chain:
        tape = _segment_tape(row, path)
        if tape is None:
            notes.append(f"s{row['sid']}: missing tape.json")
            continue
        extract = _safe_json(tape.with_name("extract.json"))
        ingest(f"s{int(row['sid'])}", tape, extract)

    if include_live and path.is_file():
        live_extract = _safe_json(path.with_name(path.stem + "_extract.json"))
        ingest("live", path, live_extract)

    ready = [
        h
        for h in hops_out
        if h.has_anchor
        and h.mode == "traversal"
        and h.policy_id is None
        and not h.bank_dual_green
        and h.dwell >= 60
    ]
    ready.sort(key=lambda h: (h.dwell, h.segment, h.hop_index))
    next_hop = ready[0] if ready else None

    return {
        "kind": "super_metroid_product_chain_hop_board",
        "schemaVersion": 1,
        "task": _rel(path) or str(path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ap_join": AP_JOIN,
        "epic": "rr-4nli",
        "counts": {
            "hops": len(hops_out),
            "segments": len(chain),
            "with_anchor": sum(1 for h in hops_out if h.has_anchor),
            "missing_anchor": missing_anchor,
            "with_policy": sum(1 for h in hops_out if h.policy_id),
            "missing_policy": missing_policy,
            "combat": combat_n,
            "bank_dual_green": dual_n,
            "ready_short": len(ready),
        },
        "next_hop": next_hop.to_dict() if next_hop else None,
        "hops": [h.to_dict() for h in hops_out],
        "notes": notes,
    }


def format_board_summary(board: Mapping[str, Any]) -> str:
    c = board.get("counts") or {}
    lines = [
        f"product-chain hops={c.get('hops')}  segments={c.get('segments')}",
        (
            f"  anchors {c.get('with_anchor')}/{c.get('hops')}  "
            f"policies {c.get('with_policy')}/{c.get('hops')}  "
            f"bank dual-green {c.get('bank_dual_green')}  "
            f"combat {c.get('combat')}"
        ),
        f"  AP join: {board.get('ap_join')}",
    ]
    nxt = board.get("next_hop")
    if isinstance(nxt, Mapping):
        lines.append(
            f"  next: {nxt.get('segment')} hop {nxt.get('hop_index')} "
            f"{nxt.get('name')} {nxt.get('hop_key')} dwell={nxt.get('dwell')}f"
        )
    else:
        lines.append("  next: none (all short traversal hops have policy or dual-green)")
    return "\n".join(lines)


def write_product_chain_board(
    task_path: Path | str = DEFAULT_TASK,
    *,
    out: Path | str = DEFAULT_BOARD,
    include_live: bool = True,
) -> dict[str, Any]:
    board = build_product_chain_board(task_path, include_live=include_live)
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(board, indent=2) + "\n", encoding="utf-8")
    board["written"] = str(dest)
    return board
