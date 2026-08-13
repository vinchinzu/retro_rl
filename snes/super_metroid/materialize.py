"""Post-record materialize: settled room leaves + run_timing + skill-bank delta.

Single spine for guided_human / extract CLI. Offline only (no emulator).

Pipeline::

    task JSON + anchors
      → raw hops (build_room_hops)
      → settle_room_hops  (RoomTimer-aligned entry)
      → room_splits + events_from_task_payload
      → RunTimingReport
      → export hop bodies under <stem>_hops/
      → records_from_hops_and_anchors (dual_green=False until hop-replay)
      → optional merge into recordings/skill_bank/bank.json

Open-loop verification is hop-replay / compose (separate CLIs), not this module.

See ``docs/RUN_TIMING_AND_SKILL_BANK.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from super_metroid.human_tape.hops import (
    build_room_hops,
    default_skill_groups,
    extract_tape,
    load_room_names,
    load_task_json,
    settle_room_hops,
)
from super_metroid.human_tape.anchors import load_anchors_index
from super_metroid.run_splits import (
    build_run_timing,
    events_from_task_payload,
    room_splits_from_hops,
)
from super_metroid.skill_bank import (
    DEFAULT_BANK_PATH,
    HopSkillRecord,
    SkillBank,
    records_from_hops_and_anchors,
)
from super_metroid.human_tape.bodies import export_hop_bodies


@dataclass
class MaterializeResult:
    """Artifacts from one take materialize."""

    task: Path
    name: str
    hops_raw: list[dict[str, Any]]
    hops_settled: list[dict[str, Any]]
    run_timing: dict[str, Any]
    bank_records: list[HopSkillRecord]
    extract_path: Path | None = None
    run_timing_path: Path | None = None
    bank_path: Path | None = None
    hop_body_paths: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "super_metroid_materialize",
            "schemaVersion": 1,
            "task": str(self.task),
            "name": self.name,
            "room_hops_raw": len(self.hops_raw),
            "room_hops_settled": len(self.hops_settled),
            "run_timing_summary": (self.run_timing or {}).get("summary"),
            "bank_records": len(self.bank_records),
            "extract_path": str(self.extract_path) if self.extract_path else None,
            "run_timing_path": (
                str(self.run_timing_path) if self.run_timing_path else None
            ),
            "bank_path": str(self.bank_path) if self.bank_path else None,
            "hop_body_paths": list(self.hop_body_paths),
            "notes": list(self.notes),
            "hop_keys": [r.hop_key for r in self.bank_records],
        }


def materialize_take(
    task_path: Path | str,
    *,
    write: bool = True,
    write_extract: bool = True,
    write_run_timing: bool = True,
    write_hop_bodies: bool = True,
    merge_bank: bool = False,
    bank_path: Path | str | None = None,
    assist: bool | None = None,
    settle: bool = True,
    timeline: str = "frame",
    stitch: bool = False,
    stitch_print_table: bool = False,
) -> MaterializeResult:
    """Build settled hops, run_timing, hop bodies, and bank records.

    Parameters
    ----------
    write:
        When True, write sidecars next to the task (and bank if ``merge_bank``).
    write_extract / write_run_timing:
        Control individual sidecars: ``<stem>_extract.json``, ``<stem>_run_timing.json``.
    write_hop_bodies:
        When True (default with write), export per-hop SNES-12 bodies under
        ``<stem>_hops/`` and attach paths on bank records for hill-climb / compose.
    merge_bank:
        When True, append records into ``bank_path`` (default
        ``recordings/skill_bank/bank.json``). Records stay ``dual_green=False``
        until hop-replay verifies.
    settle:
        Align hop starts to first ordinary non-door frame (shared leaf clock).
    timeline:
        Passed to ``room_splits_from_hops``. Default ``"frame"`` so leaf bounds
        align with anchor / trace event frames used in folds. Bank **frames**
        still come from hop ``dwell`` (index span after settle).
    stitch:
        When True, also write multi-session RTA timing fold (``*_stitched.json``).
        Off by default — use ``./play --pb`` for PB table, not every F5.
    stitch_print_table:
        Print PB table when *stitch* is True.
    """
    path = Path(task_path)
    data = load_task_json(path)
    trace = list(data.get("trace") or [])
    frames = list(data.get("frames") or [])
    meta = dict(data.get("metadata") or {})
    name = str(data.get("name") or path.stem)
    names = load_room_names()

    anchors = load_anchors_index(path)
    if anchors is None:
        # extract_tape may still surface a path; keep None for event merge
        pass

    hops_raw = build_room_hops(trace, room_names=names)
    hops_settled = settle_room_hops(hops_raw, trace) if settle else list(hops_raw)

    rooms = room_splits_from_hops(hops_settled, names=names, timeline=timeline)
    events = events_from_task_payload(trace=trace, anchors=anchors, rooms=rooms)
    report = build_run_timing(
        rooms,
        events,
        source=name,
        total_frames=int(data.get("frame_count") or len(data.get("frames") or []) or 0)
        or None,
    )
    timing_dict = report.to_dict()

    if assist is None:
        assist_meta = meta.get("assist") or {}
        if isinstance(assist_meta, Mapping):
            # guided_human stores unlimited_energy; treat any truthy assist block as ON
            if "unlimited_energy" in assist_meta:
                assist = bool(assist_meta.get("unlimited_energy"))
            elif assist_meta:
                assist = True

    records = records_from_hops_and_anchors(
        hops_settled,
        anchors=anchors,
        source=name,
        run_id=name,
        dual_green=False,
        assist=assist,
        names=names,
    )
    # Parent tape path on every record for compose_route_plan.
    for rec in records:
        rec.meta.setdefault("source_task", str(path.resolve()))
        rec.meta.setdefault("task", str(path.resolve()))

    notes: list[str] = []
    if not anchors:
        notes.append("no live anchors index — entry_anchor empty; re-record with anchors ON")
    settled_n = sum(1 for h in hops_settled if h.get("settled_entry"))
    notes.append(f"settled_entry hops: {settled_n}/{len(hops_settled)}")
    notes.append(
        "bank records dual_green=False until hop-replay; "
        "theoretical Frankenstein only after dual-green compose"
    )
    if not report.items:
        notes.append("no item_delta folds (inventory unchanged or no events)")
    if not report.bosses:
        notes.append("no boss folds (no boss rooms / events in take)")

    extract_path: Path | None = None
    run_timing_path: Path | None = None
    out_bank: Path | None = None
    hop_body_paths: list[str] = []

    if write:
        if write_hop_bodies and frames and hops_settled:
            body_paths = export_hop_bodies(
                path,
                hops_settled,
                frames=frames,
                hop_keys=[r.hop_key for r in records],
                entry_anchors=[r.entry_anchor for r in records],
            )
            hop_body_paths = [str(p) for p in body_paths]
            for rec, bp in zip(records, body_paths):
                rec.body_path = str(bp)
                rec.meta["body_path"] = str(bp)
            notes.append(f"hop bodies: {len(body_paths)} under {path.stem}_hops/")

        if write_extract:
            board = extract_tape(path, room_names=names)
            # Prefer settled hops for timing-aware consumers; keep raw too.
            board["room_hops"] = hops_settled
            board["room_hops_raw"] = hops_raw
            board["skill_groups"] = default_skill_groups(hops_settled)
            board["settled"] = bool(settle)
            board["hop_bodies"] = hop_body_paths
            extract_path = path.with_name(path.stem + "_extract.json")
            extract_path.parent.mkdir(parents=True, exist_ok=True)
            # Slim anchors body like extract CLI
            slim = dict(board)
            anc = slim.get("anchors")
            if isinstance(anc, dict) and isinstance(anc.get("anchors"), list):
                slim["anchors"] = {
                    "task": anc.get("task"),
                    "anchors_dir": anc.get("anchors_dir"),
                    "count": anc.get("count"),
                    "index_path": slim.get("anchors_index"),
                    "anchors": anc.get("anchors"),
                }
            extract_path.write_text(
                json.dumps(slim, indent=2) + "\n", encoding="utf-8"
            )

        if write_run_timing:
            run_timing_path = path.with_name(path.stem + "_run_timing.json")
            run_timing_path.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(timing_dict)
            payload["notes"] = notes
            payload["materialize"] = {
                "settle": settle,
                "timeline": timeline,
                "anchors": bool(anchors),
                "hop_bodies": len(hop_body_paths),
            }
            run_timing_path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )

        if merge_bank:
            bp = Path(bank_path) if bank_path is not None else DEFAULT_BANK_PATH
            if bp.is_file():
                bank = SkillBank.load(bp)
            else:
                bank = SkillBank()
            for rec in records:
                bank.add(rec)
            out_bank = bank.save(bp)

    # Optional multi-session RTA timing fold (not button compose).
    if stitch:
        try:
            from super_metroid.human_tape.stitch import materialize_stitch

            stitch_rep = materialize_stitch(
                path,
                write=write,
                print_table=stitch_print_table,
                max_rooms=None,
            )
            notes.append(
                f"stitched total={stitch_rep.total_frames}f "
                f"prefix={stitch_rep.prefix_events} take={stitch_rep.take_events}"
            )
            if write:
                notes.append(
                    f"stitched → {path.with_name(path.stem + '_stitched.json')}"
                )
        except Exception as exc:  # pragma: no cover - timing only
            notes.append(f"stitch skipped: {exc}")

    return MaterializeResult(
        task=path,
        name=name,
        hops_raw=list(hops_raw),
        hops_settled=list(hops_settled),
        run_timing=timing_dict,
        bank_records=records,
        extract_path=extract_path,
        run_timing_path=run_timing_path,
        bank_path=out_bank,
        hop_body_paths=hop_body_paths,
        notes=notes,
    )
