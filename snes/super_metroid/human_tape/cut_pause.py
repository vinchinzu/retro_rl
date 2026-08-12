"""Cut freeze-time (pause menu / trailing idle) from a guided_human tape.

Super Metroid **pause freezes the world** (enemies/RNG do not advance), so
removing ``pause_or_inventory`` spans is open-loop-safe for hop-replay — unlike
mid-traversal idle cuts that skip live enemy ticks.

Typical improve loop::

    record → pause mid-take → F5
    cut_pause_tape(task)          # drop menu freeze + trailing stand
    durable pin + materialize     # clean RTA + ./play supers seam

Also supports optional pure trailing idle (all-zero buttons after last input).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_metroid.human_tape.anchors import load_anchors_index
from super_metroid.human_tape.hops import load_task_json
from super_metroid.human_tape.rta_clock import fmt_time
from super_metroid.human_tape.segment_archive import archive_existing_take
from super_metroid.human_tape.trim import is_idle_frame


PHASE_PAUSE = "pause_or_inventory"


@dataclass
class CutSpan:
    start: int  # inclusive
    end: int  # inclusive
    frames: int
    reason: str
    room: str | None = None
    phase: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CutPauseReport:
    task: str
    frames_before: int
    frames_after: int
    cut_frames: int
    cut_time: str
    spans: list[CutSpan] = field(default_factory=list)
    kept_ranges: list[tuple[int, int]] = field(default_factory=list)  # [lo, hi)
    anchors_remapped: int = 0
    anchors_dropped: int = 0
    out_path: str | None = None
    backup_path: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "super_metroid_cut_pause",
            "schemaVersion": 1,
            "task": self.task,
            "frames_before": self.frames_before,
            "frames_after": self.frames_after,
            "cut_frames": self.cut_frames,
            "cut_time": self.cut_time,
            "spans": [s.to_dict() for s in self.spans],
            "kept_ranges": [[a, b] for a, b in self.kept_ranges],
            "anchors_remapped": self.anchors_remapped,
            "anchors_dropped": self.anchors_dropped,
            "out_path": self.out_path,
            "backup_path": self.backup_path,
            "notes": list(self.notes),
        }


def find_phase_runs(
    trace: Sequence[Mapping[str, Any]],
    *,
    phase: str = PHASE_PAUSE,
    min_frames: int = 30,
) -> list[CutSpan]:
    """Half-closed style spans converted to inclusive CutSpan for phase runs."""
    spans: list[CutSpan] = []
    start: int | None = None
    for i, row in enumerate(trace):
        ph = str(row.get("phase") or "")
        if ph == phase:
            if start is None:
                start = i
        else:
            if start is not None:
                n = i - start
                if n >= min_frames:
                    r0 = trace[start]
                    spans.append(
                        CutSpan(
                            start=start,
                            end=i - 1,
                            frames=n,
                            reason=f"phase:{phase}",
                            room=str(r0.get("room_hex") or r0.get("room") or ""),
                            phase=phase,
                        )
                    )
                start = None
    if start is not None:
        n = len(trace) - start
        if n >= min_frames:
            r0 = trace[start]
            spans.append(
                CutSpan(
                    start=start,
                    end=len(trace) - 1,
                    frames=n,
                    reason=f"phase:{phase}",
                    room=str(r0.get("room_hex") or r0.get("room") or ""),
                    phase=phase,
                )
            )
    return spans


def find_trailing_idle(
    frames: Sequence[Sequence[int]],
    trace: Sequence[Mapping[str, Any]] | None = None,
    *,
    min_frames: int = 30,
    keep_tail: int = 0,
) -> CutSpan | None:
    """Cut pure button-idle after last non-idle input (standing at F5)."""
    n = len(frames)
    if n == 0:
        return None
    last_btn = None
    for i in range(n - 1, -1, -1):
        if not is_idle_frame(frames[i]):
            last_btn = i
            break
    if last_btn is None:
        return None
    # start of trailing idle
    start = last_btn + 1
    end = n - 1 - max(0, int(keep_tail))
    if end < start:
        return None
    cut_n = end - start + 1
    if cut_n < min_frames:
        return None
    room = ""
    if trace and start < len(trace):
        room = str(trace[start].get("room_hex") or trace[start].get("room") or "")
    return CutSpan(
        start=start,
        end=end,
        frames=cut_n,
        reason="trailing_idle",
        room=room or None,
        phase="ordinary_gameplay",
    )


def _merge_spans(spans: Sequence[CutSpan]) -> list[CutSpan]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    merged: list[CutSpan] = [ordered[0]]
    for s in ordered[1:]:
        cur = merged[-1]
        if s.start <= cur.end + 1:
            end = max(cur.end, s.end)
            merged[-1] = CutSpan(
                start=cur.start,
                end=end,
                frames=end - cur.start + 1,
                reason=f"{cur.reason}+{s.reason}",
                room=cur.room or s.room,
                phase=cur.phase or s.phase,
            )
        else:
            merged.append(s)
    return merged


def spans_to_kept_ranges(n: int, cut_spans: Sequence[CutSpan]) -> list[tuple[int, int]]:
    """Return kept [lo, hi) ranges covering frames not in any cut span."""
    if n <= 0:
        return []
    cut = [False] * n
    for s in cut_spans:
        lo = max(0, int(s.start))
        hi = min(n - 1, int(s.end))
        for i in range(lo, hi + 1):
            cut[i] = True
    ranges: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if cut[i]:
            i += 1
            continue
        j = i
        while j < n and not cut[j]:
            j += 1
        ranges.append((i, j))
        i = j
    return ranges


def remap_frame(
    old: int,
    kept_ranges: Sequence[tuple[int, int]],
    *,
    clamp: bool = False,
) -> int | None:
    """Map original frame index → new index.

    When *clamp* is False (default), frames inside a cut return None.
    When *clamp* is True, snap to the nearest kept frame (prefer previous).
    """
    old = int(old)
    new = 0
    prev_new_end: int | None = None
    for lo, hi in kept_ranges:
        if old < lo:
            # In a cut gap (or before first kept)
            if clamp and prev_new_end is not None:
                return prev_new_end
            if clamp and kept_ranges:
                # Before first kept → first kept frame 0
                return 0
            return None
        if old < hi:
            return new + (old - lo)
        prev_new_end = new + (hi - lo) - 1
        new += hi - lo
    # Past end
    if clamp and prev_new_end is not None:
        return prev_new_end
    return None


def apply_kept_ranges(
    frames: Sequence[Sequence[int]],
    trace: Sequence[Mapping[str, Any]],
    kept_ranges: Sequence[tuple[int, int]],
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    new_frames: list[list[int]] = []
    new_trace: list[dict[str, Any]] = []
    for lo, hi in kept_ranges:
        for i in range(lo, hi):
            new_i = len(new_frames)
            fr = frames[i] if i < len(frames) else [0] * 12
            new_frames.append([int(x) for x in fr[:12]] + [0] * max(0, 12 - len(fr)))
            if i < len(trace):
                row = dict(trace[i])
                row["frame"] = new_i
                new_trace.append(row)
            else:
                new_trace.append({"frame": new_i})
    return new_frames, new_trace


def remap_anchors_index(
    index: Mapping[str, Any] | None,
    kept_ranges: Sequence[tuple[int, int]],
) -> tuple[dict[str, Any] | None, int, int]:
    """Return updated anchors index, remapped count, dropped count."""
    if not isinstance(index, Mapping):
        return None, 0, 0
    out = dict(index)
    rows_in = index.get("anchors")
    if not isinstance(rows_in, list):
        return out, 0, 0
    rows_out: list[dict[str, Any]] = []
    remapped = 0
    dropped = 0
    # Kinds that should survive a cut by snapping to the nearest kept frame
    # (end pin / item pickup still meaningful after pause strip).
    clamp_kinds = frozenset({"end", "item_delta", "manual", "boot"})
    for row in rows_in:
        if not isinstance(row, Mapping):
            continue
        old_f = int(row.get("frame") or 0)
        kind = str(row.get("kind") or "")
        use_clamp = kind in clamp_kinds
        new_f = remap_frame(old_f, kept_ranges, clamp=use_clamp)
        if new_f is None:
            dropped += 1
            continue
        r = dict(row)
        r["frame"] = new_f
        rows_out.append(r)
        remapped += 1
    out["anchors"] = rows_out
    out["count"] = len(rows_out)
    return out, remapped, dropped


def cut_pause_tape(
    task_path: Path | str,
    *,
    write: bool = True,
    in_place: bool = True,
    out_path: Path | str | None = None,
    archive_first: bool = True,
    cut_pause_phase: bool = True,
    cut_trailing_idle: bool = True,
    min_pause_frames: int = 30,
    min_trailing_idle: int = 30,
    keep_trailing: int = 0,
    materialize: bool = True,
    merge_bank: bool = False,
) -> CutPauseReport:
    """Cut pause-menu freeze (+ optional trailing idle) and rewrite the take.

    Archives the pre-cut tape under ``*_segments/sN/`` when *archive_first*.
    """
    path = Path(task_path)
    data = load_task_json(path)
    frames = list(data.get("frames") or [])
    trace = list(data.get("trace") or [])
    n = max(len(frames), len(trace))
    # Align lengths
    while len(frames) < n:
        frames.append([0] * 12)
    while len(trace) < n:
        trace.append({"frame": len(trace)})

    spans: list[CutSpan] = []
    if cut_pause_phase:
        spans.extend(
            find_phase_runs(trace, phase=PHASE_PAUSE, min_frames=min_pause_frames)
        )
    if cut_trailing_idle:
        trail = find_trailing_idle(
            frames,
            trace,
            min_frames=min_trailing_idle,
            keep_tail=keep_trailing,
        )
        if trail is not None:
            spans.append(trail)

    spans = _merge_spans(spans)
    report = CutPauseReport(
        task=str(path),
        frames_before=n,
        frames_after=n,
        cut_frames=0,
        cut_time=fmt_time(0),
        spans=spans,
    )
    if not spans:
        report.notes.append("no pause / trailing-idle spans to cut")
        report.kept_ranges = [(0, n)] if n else []
        return report

    kept = spans_to_kept_ranges(n, spans)
    cut_n = n - sum(hi - lo for lo, hi in kept)
    report.kept_ranges = kept
    report.cut_frames = cut_n
    report.cut_time = fmt_time(cut_n)
    report.frames_after = n - cut_n
    report.notes.append(
        f"cut {len(spans)} span(s) = {cut_n}f ({report.cut_time}); "
        f"kept {report.frames_after}f ({fmt_time(report.frames_after)})"
    )

    new_frames, new_trace = apply_kept_ranges(frames, trace, kept)

    # End fingerprint from last kept row
    end_row = new_trace[-1] if new_trace else {}
    end_fp = {
        "kind": "end",
        "frame": max(0, len(new_frames) - 1),
        "room": end_row.get("room_hex") or (
            f"0x{int(end_row['room']):04X}" if end_row.get("room") is not None else None
        ),
        "room_id": int(end_row["room"]) if end_row.get("room") is not None else None,
        "xy": [int(end_row.get("x") or 0), int(end_row.get("y") or 0)],
        "pose": int(end_row["pose"]) if end_row.get("pose") is not None else None,
        "items": (
            f"0x{int(end_row['items']):04X}"
            if end_row.get("items") is not None
            else None
        ),
        "energy": end_row.get("energy"),
        "missiles": end_row.get("missiles"),
        "supers": end_row.get("supers"),
        "pbs": end_row.get("pbs"),
        "cut_pause": True,
    }

    meta = dict(data.get("metadata") or {})
    meta["end_fingerprint"] = end_fp
    meta["cut_pause"] = {
        "cut_frames": cut_n,
        "cut_time": report.cut_time,
        "spans": [s.to_dict() for s in spans],
        "kept_ranges": [[a, b] for a, b in kept],
        "cut_at": datetime.now(timezone.utc).isoformat(),
    }
    # Recompute lightweight duration
    meta["frame_count"] = len(new_frames)
    meta["duration_seconds"] = len(new_frames) / 60.0

    out_data = dict(data)
    out_data["frames"] = new_frames
    out_data["trace"] = new_trace
    out_data["frame_count"] = len(new_frames)
    out_data["metadata"] = meta

    dest = Path(out_path) if out_path is not None else path
    if not in_place and out_path is None:
        dest = path.with_name(path.stem + "_cut.json")

    backup_path: Path | None = None
    if write:
        # Prefer a non-chain backup of the fat take. Using archive_existing_take
        # alone would leave a pause-bloated segment in the RTA chain and then
        # double-count when the cut live tape is archived on the next ./play.
        pre_cut_path = path.with_name(path.stem + "_pre_cut.json")
        if archive_first and in_place and path.is_file():
            try:
                pre_cut_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
                backup_path = pre_cut_path
                report.notes.append(f"pre-cut backup → {pre_cut_path}")
                # Optional segment slot: mark rta_exclude so resolve_rta_clock skips it
                try:
                    archived = archive_existing_take(path)
                    if archived is not None:
                        join_path = archived / "join.json"
                        join = {}
                        if join_path.is_file():
                            join = json.loads(join_path.read_text(encoding="utf-8"))
                        join["rta_exclude"] = True
                        join["reason"] = "pre_cut_pause_backup"
                        join_path.write_text(
                            json.dumps(join, indent=2) + "\n", encoding="utf-8"
                        )
                        # Keep fat tape under tape_pre_cut.json; tape.json stays for
                        # forensics but RTA skips this segment via rta_exclude.
                        fat = archived / "tape.json"
                        if fat.is_file():
                            fat.rename(archived / "tape_pre_cut.json")
                        report.notes.append(
                            f"segment pre-cut (rta_exclude) → {archived}"
                        )
                except Exception as exc:  # noqa: BLE001
                    report.notes.append(f"segment archive note failed: {exc}")
            except Exception as exc:  # noqa: BLE001
                report.notes.append(f"pre-cut backup failed: {exc}")

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out_data, indent=2) + "\n", encoding="utf-8")
        report.out_path = str(dest)
        report.backup_path = str(backup_path) if backup_path else None

        # Remap anchors index frames (state files keep old names — paths still valid)
        anchors_idx_path = path.with_name(path.stem + "_anchors.json")
        idx = load_anchors_index(path)
        new_idx, remapped, dropped = remap_anchors_index(idx, kept)
        report.anchors_remapped = remapped
        report.anchors_dropped = dropped
        if new_idx is not None and in_place:
            new_idx["frame_count"] = len(new_frames)
            new_idx["end_fingerprint"] = end_fp
            new_idx["cut_pause"] = meta["cut_pause"]
            anchors_idx_path.write_text(
                json.dumps(new_idx, indent=2) + "\n", encoding="utf-8"
            )
            report.notes.append(
                f"anchors remapped={remapped} dropped={dropped} → {anchors_idx_path.name}"
            )

        cut_report_path = dest.with_name(dest.stem + "_cut_pause.json")
        cut_report_path.write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        report.notes.append(f"cut report → {cut_report_path}")

        if materialize:
            try:
                from super_metroid.materialize import materialize_take

                mat = materialize_take(
                    dest,
                    write=True,
                    write_extract=True,
                    write_run_timing=True,
                    merge_bank=merge_bank,
                    stitch=True,
                    stitch_print_table=False,
                )
                report.notes.append(
                    f"materialize rooms={len(mat.hops_settled)} "
                    f"bodies={len(mat.hop_body_paths)}"
                )
            except Exception as exc:  # noqa: BLE001
                report.notes.append(f"materialize failed: {exc}")

    return report


def promote_end_durable_pin(
    task_path: Path | str,
    *,
    stem: str,
    integration_scratch: Path | str | None = None,
    tasks_dir: Path | str | None = None,
) -> list[str]:
    """Copy task end.state → durable scratch/tasks pins (e.g. full_start_v1_supers)."""
    path = Path(task_path)
    written: list[str] = []
    end_state = path.with_name(path.stem + "_end.state")
    if not end_state.is_file():
        return [f"missing end state {end_state}"]
    blob = end_state.read_bytes()
    targets: list[Path] = []
    if tasks_dir is not None:
        targets.append(Path(tasks_dir) / f"{stem}.state")
    else:
        targets.append(path.with_name(f"{stem}.state"))
    if integration_scratch is not None:
        targets.append(Path(integration_scratch) / f"{stem}.state")
    for dest in targets:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        written.append(str(dest))
    return written
