#!/usr/bin/env python3
"""Parse SMB human recordings into control-relative skills / jump chunks.

Pulls hillclimb-ready fragments from ``record_human`` JSON:

- **stage** skills — per world-level (4-1, 4-2, …) from natural control → exit
- **jump** skills — each grounded→air→land arc (chunk windows for local search)
- **run** skills — long RIGHT+B holds (optional)

Each skill is control-relative: rebased frame indices + entry fingerprint so a
reactive runner can attach it without absolute-frame stitch.

```bash
uv run python -m smb.scripts.parse_human_recording \\
  nes/smb/recordings/human/late_v1.json --export-skills

# Only human-sourced frames (ignore bot prefix in --from auto)
uv run python -m smb.scripts.parse_human_recording PATH --human-only --export-skills

# Dump jump table to stdout
uv run python -m smb.scripts.parse_human_recording PATH --list-jumps
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from smb.paths import GAME_DIR, MODELS_DIR, RECORDINGS_DIR
from smb.policy import compress_nes9_rle

HUMAN_DIR = RECORDINGS_DIR / "human"
SKILLS_DIR = MODELS_DIR / "human_skills"


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") not in ("smb_human_nes9", "nes9_rle"):
        # Still accept if frames+trace present.
        if "frames" not in data and "segments" not in data:
            raise ValueError(f"unsupported recording format: {data.get('format')!r}")
    return data


def _frames_and_trace(
    data: dict[str, Any], *, human_only: bool
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    frames = data.get("frames") or []
    trace = data.get("trace") or []
    if not frames and data.get("segments"):
        from smb.policy import expand_nes9_rle

        frames = expand_nes9_rle(data)
    if human_only and trace:
        kept_f: list[list[int]] = []
        kept_t: list[dict[str, Any]] = []
        for fr, row in zip(frames, trace):
            if row.get("source", "human") == "human":
                kept_f.append(fr)
                kept_t.append(dict(row, frame=len(kept_f) - 1))
        return kept_f, kept_t
    # Normalize frame indices.
    norm_trace = []
    for i, row in enumerate(trace[: len(frames)]):
        r = dict(row)
        r["frame"] = i
        norm_trace.append(r)
    if len(norm_trace) < len(frames):
        # Pad minimal rows if trace missing.
        for i in range(len(norm_trace), len(frames)):
            norm_trace.append({"frame": i, "x": 0, "y": 0, "in_air": False, "stage": "?"})
    return [list(map(int, f[:9])) for f in frames], norm_trace


def _fingerprint_from_row(row: dict[str, Any]) -> dict[str, int]:
    return {
        "world": int(row.get("world", 0)),
        "level": int(row.get("level", 0)),
        "player_x": int(row.get("x", 0)),
        "player_y": int(row.get("y", 0)),
        "x_speed": int(row.get("xs", 0)),
        "y_speed": int(row.get("ys", 0)),
        "player_state": int(row.get("player_state", 0)),
        "timer": int(row.get("timer", 0)),
        "oper_mode": int(row.get("oper_mode", 1)),
        "lives": int(row.get("lives", 2)),
        "area_pointer": int(row.get("area_pointer", 0)),
    }


def extract_stage_skills(
    frames: list[list[int]],
    trace: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One skill per stage visit: first control-ish frame → stage exit."""
    if not trace:
        return []
    skills: list[dict[str, Any]] = []
    stage0 = str(trace[0].get("stage", "?"))
    start = 0
    prev_stage = stage0

    def _flush(end_excl: int, stage: str) -> None:
        if end_excl <= start:
            return
        # Prefer first near-control state (player_state 7/8) as skill entry.
        entry_i = start
        for i in range(start, min(start + 300, end_excl)):
            ps = int(trace[i].get("player_state", 0))
            if ps in (7, 8) and int(trace[i].get("oper_mode", 1)) == 1:
                entry_i = i
                break
        body = frames[entry_i:end_excl]
        if not body:
            return
        entry_row = trace[entry_i]
        exit_row = trace[end_excl - 1]
        skills.append(
            {
                "skill_id": f"stage_{stage}_{entry_i}_{end_excl}",
                "kind": "stage",
                "stage": stage,
                "src_start": entry_i,
                "src_end": end_excl,
                "frames": len(body),
                "entry": _fingerprint_from_row(entry_row),
                "exit": _fingerprint_from_row(exit_row),
                "dx": int(exit_row.get("x", 0)) - int(entry_row.get("x", 0)),
                "segments": compress_nes9_rle(body),
                "raw_buttons": body,
            }
        )

    for i, row in enumerate(trace):
        stage = str(row.get("stage", "?"))
        if stage != prev_stage:
            _flush(i, prev_stage)
            start = i
            prev_stage = stage
    _flush(len(trace), prev_stage)
    return skills


def extract_jump_skills(
    frames: list[list[int]],
    trace: list[dict[str, Any]],
    *,
    min_air: int = 4,
    pad_pre: int = 4,
    pad_post: int = 4,
) -> list[dict[str, Any]]:
    """Chunk each jump arc for hillclimb windows.

    Window = [takeoff - pad_pre, land + pad_post), control-relative to takeoff.
    """
    skills: list[dict[str, Any]] = []
    n = len(trace)
    i = 0
    jump_idx = 0
    while i < n:
        in_air = bool(trace[i].get("in_air"))
        # A pressed on this frame (index 8)
        a_now = i < len(frames) and len(frames[i]) > 8 and int(frames[i][8]) != 0
        a_prev = i > 0 and len(frames[i - 1]) > 8 and int(frames[i - 1][8]) != 0
        rising_a = a_now and not a_prev
        if not (rising_a or (in_air and i > 0 and not bool(trace[i - 1].get("in_air")))):
            i += 1
            continue
        takeoff = i
        # Find landing: first grounded after min_air airborne frames.
        j = takeoff
        air = 0
        land = None
        while j < n:
            if bool(trace[j].get("in_air")):
                air += 1
            elif air >= min_air:
                land = j
                break
            elif air > 0 and not bool(trace[j].get("in_air")):
                # short hop
                land = j
                break
            j += 1
        if land is None:
            i = takeoff + 1
            continue
        if air < min_air and (land - takeoff) < min_air:
            i = takeoff + 1
            continue
        start = max(0, takeoff - pad_pre)
        end = min(n, land + pad_post)
        body = frames[start:end]
        entry_row = trace[takeoff]
        peak_y = min(int(trace[k].get("y", 999)) for k in range(takeoff, land + 1))
        skills.append(
            {
                "skill_id": f"jump_{jump_idx:03d}_{takeoff}_{land}",
                "kind": "jump",
                "stage": entry_row.get("stage"),
                "src_start": start,
                "src_end": end,
                "takeoff": takeoff,
                "land": land,
                "air_frames": air,
                "frames": len(body),
                "entry": _fingerprint_from_row(entry_row),
                "exit": _fingerprint_from_row(trace[min(land, n - 1)]),
                "peak_y": peak_y,
                "dx": int(trace[land].get("x", 0)) - int(entry_row.get("x", 0)),
                "x0": int(entry_row.get("x", 0)),
                "segments": compress_nes9_rle(body),
                "raw_buttons": body,
                # Hillclimb window relative to skill start (takeoff-centric).
                "hillclimb_window": {
                    "start": takeoff - start,
                    "end": land - start,
                    "label": f"jump_{jump_idx:03d}",
                },
            }
        )
        jump_idx += 1
        i = land + 1
    return skills


def extract_run_skills(
    frames: list[list[int]],
    trace: list[dict[str, Any]],
    *,
    min_len: int = 40,
) -> list[dict[str, Any]]:
    """Long RIGHT+B (or RIGHT) holds — useful run-line seeds."""
    skills: list[dict[str, Any]] = []
    start = None
    run_idx = 0
    for i, fr in enumerate(frames):
        right = len(fr) > 7 and int(fr[7]) != 0
        b = len(fr) > 0 and int(fr[0]) != 0
        is_run = right and b
        if is_run:
            if start is None:
                start = i
        elif start is not None:
            if i - start >= min_len:
                body = frames[start:i]
                skills.append(
                    {
                        "skill_id": f"run_{run_idx:03d}_{start}_{i}",
                        "kind": "run",
                        "stage": trace[start].get("stage") if start < len(trace) else "?",
                        "src_start": start,
                        "src_end": i,
                        "frames": len(body),
                        "entry": _fingerprint_from_row(trace[start]),
                        "exit": _fingerprint_from_row(trace[i - 1]),
                        "dx": int(trace[i - 1].get("x", 0)) - int(trace[start].get("x", 0)),
                        "segments": compress_nes9_rle(body),
                        "raw_buttons": body,
                    }
                )
                run_idx += 1
            start = None
    if start is not None and len(frames) - start >= min_len:
        body = frames[start:]
        skills.append(
            {
                "skill_id": f"run_{run_idx:03d}_{start}_{len(frames)}",
                "kind": "run",
                "stage": trace[start].get("stage"),
                "src_start": start,
                "src_end": len(frames),
                "frames": len(body),
                "entry": _fingerprint_from_row(trace[start]),
                "exit": _fingerprint_from_row(trace[-1]),
                "dx": int(trace[-1].get("x", 0)) - int(trace[start].get("x", 0)),
                "segments": compress_nes9_rle(body),
                "raw_buttons": body,
            }
        )
    return skills


def parse_recording(
    path: Path,
    *,
    human_only: bool = False,
    min_air: int = 4,
) -> dict[str, Any]:
    data = _load(path)
    frames, trace = _frames_and_trace(data, human_only=human_only)
    stages = extract_stage_skills(frames, trace)
    jumps = extract_jump_skills(frames, trace, min_air=min_air)
    runs = extract_run_skills(frames, trace)
    report = {
        "source": str(path),
        "name": data.get("name") or path.stem,
        "handoff": data.get("handoff"),
        "handoff_entry": data.get("handoff_entry"),
        "human_only": human_only,
        "total_frames": len(frames),
        "counts": {
            "stage": len(stages),
            "jump": len(jumps),
            "run": len(runs),
        },
        "stages": [
            {k: v for k, v in s.items() if k not in ("raw_buttons", "segments")}
            for s in stages
        ],
        "jumps": [
            {k: v for k, v in s.items() if k not in ("raw_buttons", "segments")}
            for s in jumps
        ],
        "runs": [
            {k: v for k, v in s.items() if k not in ("raw_buttons", "segments")}
            for s in runs
        ],
        "_skills_full": stages + jumps + runs,
    }
    return report


def export_skills(
    report: dict[str, Any],
    *,
    out_dir: Path | None = None,
    kinds: set[str] | None = None,
) -> list[Path]:
    """Write each skill as nes9_rle + raw_buttons for hillclimb."""
    out = out_dir or (SKILLS_DIR / str(report.get("name") or "unnamed"))
    out.mkdir(parents=True, exist_ok=True)
    kinds = kinds or {"stage", "jump", "run"}
    written: list[Path] = []
    index: list[dict[str, Any]] = []
    for skill in report.get("_skills_full") or []:
        if skill.get("kind") not in kinds:
            continue
        body = skill.get("raw_buttons") or []
        if not body:
            continue
        sid = skill["skill_id"]
        seed = {
            "format": "nes9_rle",
            "skill_id": sid,
            "kind": skill["kind"],
            "stage": skill.get("stage"),
            "entry": skill.get("entry"),
            "exit": skill.get("exit"),
            "src_start": skill.get("src_start"),
            "src_end": skill.get("src_end"),
            "hillclimb_window": skill.get("hillclimb_window"),
            "source_recording": report.get("source"),
            "num_frames": len(body),
            "segments": skill.get("segments") or compress_nes9_rle(body),
        }
        path = out / f"{sid}.json"
        path.write_text(json.dumps(seed, indent=2) + "\n", encoding="utf-8")
        # Raw buttons sidecar (platformer hillclimb likes raw_buttons keys).
        raw_path = out / f"{sid}_raw.json"
        raw_path.write_text(
            json.dumps(
                {
                    "raw_buttons": body,
                    "metadata": {
                        "skill_id": sid,
                        "kind": skill["kind"],
                        "entry": skill.get("entry"),
                        "hillclimb_window": skill.get("hillclimb_window"),
                        "source": report.get("source"),
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        written.append(path)

        def _rel(p: Path) -> str:
            try:
                return str(p.resolve().relative_to(GAME_DIR.resolve()))
            except ValueError:
                return str(p)

        index.append(
            {
                "skill_id": sid,
                "kind": skill["kind"],
                "stage": skill.get("stage"),
                "frames": len(body),
                "path": _rel(path),
                "raw_path": _rel(raw_path),
                "entry": skill.get("entry"),
                "hillclimb_window": skill.get("hillclimb_window"),
                "dx": skill.get("dx"),
            }
        )
    index_path = out / "skills_index.json"
    index_path.write_text(
        json.dumps(
            {
                "name": report.get("name"),
                "source": report.get("source"),
                "counts": report.get("counts"),
                "skills": index,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    written.append(index_path)
    return written


def parse_and_export(
    path: Path,
    *,
    do_export: bool = True,
    human_only: bool = False,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    report = parse_recording(path, human_only=human_only)
    report_path = path.with_name(path.stem + "_skills_report.json")
    slim = {k: v for k, v in report.items() if k != "_skills_full"}
    report_path.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    print(
        f"[PARSE] {path.name}: stages={report['counts']['stage']} "
        f"jumps={report['counts']['jump']} runs={report['counts']['run']} "
        f"→ {report_path}"
    )
    if do_export:
        paths = export_skills(report, out_dir=out_dir)
        print(f"[PARSE] exported {max(0, len(paths) - 1)} skills under {paths[-1].parent}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("recording", type=Path, help="Path to smb_human_nes9 JSON")
    parser.add_argument(
        "--export-skills",
        action="store_true",
        help="Write per-skill nes9_rle + raw_buttons under models/human_skills/",
    )
    parser.add_argument(
        "--human-only",
        action="store_true",
        help="Drop bot-sourced frames (--from auto recordings)",
    )
    parser.add_argument(
        "--list-jumps",
        action="store_true",
        help="Print jump table and exit",
    )
    parser.add_argument(
        "--kinds",
        default="stage,jump,run",
        help="Comma kinds to export (default: stage,jump,run)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--min-air", type=int, default=4)
    args = parser.parse_args()
    path = args.recording
    if not path.is_file():
        # Allow stem under recordings/human/
        alt = HUMAN_DIR / f"{path}.json"
        if alt.is_file():
            path = alt
        else:
            raise SystemExit(f"recording not found: {args.recording}")

    report = parse_recording(path, human_only=args.human_only, min_air=args.min_air)
    if args.list_jumps:
        print(f"{'id':32s} {'stage':6s} {'takeoff':>7} {'land':>6} {'air':>4} {'dx':>5} x0")
        for j in report["jumps"]:
            print(
                f"{j['skill_id']:32s} {str(j.get('stage')):6s} "
                f"{j.get('takeoff', 0):7d} {j.get('land', 0):6d} "
                f"{j.get('air_frames', 0):4d} {j.get('dx', 0):5d} {j.get('x0', 0)}"
            )
        print(f"\n{len(report['jumps'])} jumps · {report['total_frames']} frames")
        return

    slim = {k: v for k, v in report.items() if k != "_skills_full"}
    report_path = path.with_name(path.stem + "_skills_report.json")
    report_path.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    print(
        f"[PARSE] stages={report['counts']['stage']} "
        f"jumps={report['counts']['jump']} runs={report['counts']['run']}"
    )
    print(f"[PARSE] report → {report_path}")

    if args.export_skills:
        kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
        paths = export_skills(report, out_dir=args.out_dir, kinds=kinds)
        print(f"[PARSE] wrote {len(paths)} files under {paths[-1].parent}")
        # Show top jump windows for hillclimb
        jumps = [s for s in (report.get("_skills_full") or []) if s.get("kind") == "jump"]
        jumps_sorted = sorted(jumps, key=lambda s: (-abs(int(s.get("dx", 0))), s["frames"]))
        if jumps_sorted:
            print("\nBest jump chunks by |dx| (hillclimb windows):")
            for s in jumps_sorted[:8]:
                w = s.get("hillclimb_window") or {}
                print(
                    f"  {s['skill_id']}  stage={s.get('stage')}  "
                    f"dx={s.get('dx')}  air={s.get('air_frames')}  "
                    f"win=[{w.get('start')},{w.get('end')})  "
                    f"x0={s.get('x0')}"
                )


if __name__ == "__main__":
    main()
