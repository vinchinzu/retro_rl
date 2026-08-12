"""Multi-hop compose: pin → body → leave pin (natural-entry chain).

Product path for "stitch best segments" and full-run seam replay:

  for each hop:
    boot live entry_anchor
    open-loop hop body only
    verify leave room / end_xy
    (optional) dump leave pin for next hop if kinematics changed

This is **not** frame-append of multi-session tapes and not multi-minute
power-on open-loop. Each hop re-pins from its live gzip anchor so enemy RNG
and subpixel error do not compound across rooms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_metroid.human_tape.hops import load_room_hops, load_room_names, load_task_json
from super_metroid.human_tape.replay import run_hop_replay

@dataclass
class ComposeHopResult:
    """One hop in a compose run."""

    hop_index: int
    green: bool
    room: str | None = None
    leave_room: str | None = None
    anchor_path: str | None = None
    steps: int | None = None
    reason: str | None = None
    report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class ComposeReport:
    """Result of composing one or more hops from a single tape."""

    task: str
    hops_planned: int
    hops_run: int
    hops_green: int
    green: bool
    results: list[ComposeHopResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "super_metroid_hop_compose",
            "schemaVersion": 1,
            "task": self.task,
            "hops_planned": self.hops_planned,
            "hops_run": self.hops_run,
            "hops_green": self.hops_green,
            "green": self.green,
            "results": [r.to_dict() for r in self.results],
            "notes": list(self.notes),
        }


def compose_hops(
    task_path: Path | str,
    hop_indices: Sequence[int] | None = None,
    *,
    dual: bool = False,
    xy_tol: int = 24,
    leave_extra: int = 1,
    boot_settle: int = 0,
    assist: bool = True,
    stop_on_red: bool = True,
    require_anchor: bool = True,
) -> ComposeReport:
    """Replay selected hops each from their live pin (re-pin per hop).

    Parameters
    ----------
    hop_indices:
        Room hop indices to run. Default: all hops that resolve an anchor
        (when *require_anchor*) or all hops.
    stop_on_red:
        Stop chain on first RED hop.
    require_anchor:
        Skip hops without a live anchor when selecting default indices;
        when an explicit index lacks an anchor, mark RED.
    """
    path = Path(task_path)
    data = load_task_json(path)
    names = load_room_names()
    hops = load_room_hops(task_data=data, room_names=names, settle=True)
    notes: list[str] = [
        "compose unit = hop from live entry_anchor (anti-desync)",
        "settled hop bounds (match materialize bodies)",
        "not multi-minute full-tape open-loop; not seam frame-append",
        "each hop re-boots its pre-recorded pin (dual-verify, not natural-entry leave seed)",
    ]

    if hop_indices is None:
        if require_anchor:
            from super_metroid.human_tape.hops import resolve_hop_slice

            selected: list[int] = []
            for h in hops:
                info = resolve_hop_slice(
                    path,
                    hop_index=int(h["index"]),
                    leave_extra=leave_extra,
                    task_data=data,
                )
                if info.get("anchor_path"):
                    selected.append(int(h["index"]))
                else:
                    notes.append(f"skip hop {h['index']}: no anchor")
            hop_indices = selected
        else:
            hop_indices = [int(h["index"]) for h in hops]

    planned = list(hop_indices)
    results: list[ComposeHopResult] = []
    green_n = 0

    for idx in planned:
        try:
            report = run_hop_replay(
                path,
                hop_index=int(idx),
                dual=dual,
                xy_tol=xy_tol,
                leave_extra=leave_extra,
                boot_settle=boot_settle,
                assist=assist,
            )
        except Exception as exc:  # noqa: BLE001 — surface per-hop
            results.append(
                ComposeHopResult(
                    hop_index=int(idx),
                    green=False,
                    reason=f"error: {exc}",
                )
            )
            if stop_on_red:
                notes.append(f"stopped at hop {idx} after error")
                break
            continue

        ok = bool(report.get("green"))
        if ok:
            green_n += 1
        sl = report.get("slice") or {}
        # Keep full hop-replay report so --promote-bank can match bank hop_key.
        slim = {
            "green": ok,
            "ok": ok,
            "dual": report.get("dual"),
            "check": report.get("check"),
            "replay_start": report.get("replay_start"),
            "replay_end": report.get("replay_end"),
            "anchor_path": report.get("anchor_path"),
            "assist": report.get("assist"),
            "slice": sl,
            "reason": report.get("reason"),
        }
        results.append(
            ComposeHopResult(
                hop_index=int(idx),
                green=ok,
                room=sl.get("start_room_hex"),
                leave_room=sl.get("leave_room_hex"),
                anchor_path=str(report.get("anchor_path") or sl.get("anchor_path") or ""),
                steps=sl.get("steps"),
                reason=report.get("reason") if not ok else None,
                report=slim,
            )
        )
        if not ok and stop_on_red:
            notes.append(f"stopped at hop {idx} RED: {report.get('reason')}")
            break

    all_green = bool(planned) and green_n == len(results) and all(
        r.green for r in results
    )
    return ComposeReport(
        task=str(data.get("name") or path.stem),
        hops_planned=len(planned),
        hops_run=len(results),
        hops_green=green_n,
        green=all_green,
        results=results,
        notes=notes,
    )


def compose_route_plan(
    plan: Mapping[str, Any],
    *,
    dual: bool = True,
    xy_tol: int = 24,
    assist: bool = True,
    stop_on_red: bool = True,
) -> dict[str, Any]:
    """Execute a ``compose_plan`` steps list via hop-replay.

    Each step needs ``entry_anchor`` + either ``body_path`` (task hop slice
    stored as seed) or hop metadata pointing at a task. Today bank bodies
    point at hop JSON under ``*_hops/``; when only ``entry_anchor`` +
    source task hop index are known, callers should use :func:`compose_hops`.

    This helper runs steps that include ``task`` + ``hop_index`` in the step
    dict (set by materialize bank meta); pure body_path-only steps without a
    parent tape are deferred (status ``skipped_no_task``).
    """
    steps = list(plan.get("steps") or [])
    out_steps: list[dict[str, Any]] = []
    missing = list(plan.get("missing") or [])
    all_ok = True

    for step in steps:
        status = step.get("status")
        if status == "missing":
            out_steps.append({**dict(step), "compose": "missing"})
            all_ok = False
            continue
        task = step.get("task") or step.get("source_task")
        hop_index = step.get("hop_index")
        if hop_index is None and isinstance(step.get("meta"), Mapping):
            hop_index = step["meta"].get("hop_index")
        if not task or hop_index is None:
            out_steps.append(
                {
                    **dict(step),
                    "compose": "skipped_no_task",
                    "note": "need task + hop_index for hop-replay compose",
                }
            )
            all_ok = False
            continue
        report = run_hop_replay(
            Path(str(task)),
            hop_index=int(hop_index),
            dual=dual,
            xy_tol=xy_tol,
            assist=assist,
            anchor_path=step.get("entry_anchor"),
        )
        ok = bool(report.get("green"))
        if not ok:
            all_ok = False
        out_steps.append(
            {
                **dict(step),
                "compose": "green" if ok else "red",
                "reason": report.get("reason"),
            }
        )
        if not ok and stop_on_red:
            break

    return {
        "kind": "super_metroid_route_compose",
        "schemaVersion": 1,
        "green": all_ok and not missing and bool(out_steps),
        "steps": out_steps,
        "missing": missing,
        "note": (
            "Hop-compose from live pins. Dual-green per hop before STATUS "
            "promote; theoretical Frankenstein remains labeled until then."
        ),
    }
