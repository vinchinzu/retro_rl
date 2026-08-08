"""Compare fceumm HappyLee 8-2 control-relative chain to FCEUX oracle fingerprints.

Reaches real 8-3 control via the existing HL chain (1-1→…→8-2 body), then plays
HappyLee FM2 body candidates and samples mid-level fingerprints. Finds the
**first meaningful divergence** vs FCEUX oracle checkpoints (not entry/death only).

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.tas.oracle.compare_fceumm_chain
```

Writes ``compare_evidence.json`` and optional minimal correction candidate under
``oracle_happylee_8_3/`` (never overwrites shared seeds).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from smb.policy import compress_nes9_rle, expand_nes9_rle, load_nes9_rle_seed
from smb.ram import (
    PLAYER_STATE_DYING,
    rich_handoff_fingerprint,
    read_snapshot,
)
from smb.tas.chain import reach_stage_control
from smb.tas.fm2 import parse_fm2
from smb.tas.oracle.extract_fceux_checkpoints import (
    DEFAULT_FM2,
    ORACLE_EVIDENCE_DIR,
    load_jsonl,
)
from smb.tas.replay import IDLE, make_level1_env, to_action9
from smb.tas.stages import (
    HL_8_2_FM2_START,
    is_8_3_control,
    is_8_4_control,
)

# Compare fields that define "meaningful" phase / pose divergence.
COMPARE_KEYS = (
    "world",
    "level",
    "oper_mode",
    "player_state",
    "player_x",
    "player_y",
    "x_speed",
    "y_speed",
    "grounded",
    "timer",
    "timer_mod21",
    "lives",
    "screen_x",
    "x_frac",
    "y_frac",
    "frame_counter",
)


@dataclass
class DivPoint:
    """First (or a later) fingerprint mismatch between oracle and fceumm."""

    name: str
    oracle_movie_frame: int | None
    body_offset: int | None
    field: str
    oracle_value: Any
    fceumm_value: Any
    oracle_fp: dict[str, Any] = field(default_factory=dict)
    fceumm_fp: dict[str, Any] = field(default_factory=dict)


def _fp_core(fp: dict[str, Any]) -> dict[str, Any]:
    return {k: fp.get(k) for k in COMPARE_KEYS}


def _diff_fps(
    name: str,
    oracle_fp: dict[str, Any],
    fceumm_fp: dict[str, Any],
    *,
    body_offset: int | None = None,
    x_tol: int = 0,
) -> list[DivPoint]:
    diffs: list[DivPoint] = []
    for k in COMPARE_KEYS:
        ov, fv = oracle_fp.get(k), fceumm_fp.get(k)
        if k == "player_x" and isinstance(ov, int) and isinstance(fv, int):
            if abs(ov - fv) <= x_tol:
                continue
        if k in ("x_frac", "y_frac", "frame_counter") and ov is None:
            continue
        if ov != fv:
            diffs.append(
                DivPoint(
                    name=name,
                    oracle_movie_frame=oracle_fp.get("movie_frame")
                    or oracle_fp.get("frame"),
                    body_offset=body_offset,
                    field=k,
                    oracle_value=ov,
                    fceumm_value=fv,
                    oracle_fp=_fp_core(oracle_fp),
                    fceumm_fp=_fp_core(fceumm_fp),
                )
            )
    return diffs


def load_oracle_checkpoints(
    path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    path = path or (ORACLE_EVIDENCE_DIR / "fceux_checkpoints.json")
    if not path.is_file():
        # fall back to named jsonl
        named = load_jsonl(ORACLE_EVIDENCE_DIR / "fceux_named_checkpoints.jsonl")
        return {r["name"]: r for r in named if r.get("name") and not str(r["name"]).startswith("_")}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("checkpoints") or []:
        name = row.get("name")
        if name:
            snap = row.get("snapshot") or row
            out[name] = snap
    return out


def _index_trace_by_frame(trace: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(r["movie_frame"]): r for r in trace if "movie_frame" in r}


def reach_8_3_control_fceumm() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Level1_1 HL chain → real 8-3 control. Returns env, gate dict, rich fp."""
    env = make_level1_env()
    gate = reach_stage_control(env, "8-3")
    if not gate.get("success"):
        return env, gate, {}
    ram = env.get_ram()
    fp = rich_handoff_fingerprint(ram, frame=0)
    return env, gate, fp


def play_and_sample(
    env: Any,
    frames: list[list[int]],
    *,
    start: int,
    sample_offsets: list[int],
    max_play: int,
    start_lives: int,
) -> dict[str, Any]:
    """Play FM2 body from *start*; capture rich FPs at body offsets + leave/death."""
    samples: dict[int, dict[str, Any]] = {}
    leave: int | None = None
    death: int | None = None
    max_x = 0
    reached_84 = False
    control_84_offset: int | None = None
    want = set(sample_offsets)

    # Offset 0 = state at body start (before first FM2 body frame).
    if 0 in want:
        samples[0] = rich_handoff_fingerprint(env.get_ram(), frame=0)

    for i in range(max_play):
        idx = start + i
        if idx >= len(frames):
            break
        env.step(to_action9(frames[idx]))
        ram = env.get_ram()
        snap = read_snapshot(ram, frame=i + 1)
        px = int(snap.player_x)
        if 0 < px < 20000:
            max_x = max(max_x, px)
        if death is None and (
            int(snap.lives) < start_lives or int(snap.player_state) == PLAYER_STATE_DYING
        ):
            death = i + 1
        if leave is None and int(snap.world) == 7 and int(snap.level) == 3:
            leave = i + 1
        if not reached_84 and is_8_4_control(snap):
            reached_84 = True
            control_84_offset = i + 1
        if (i + 1) in want:
            samples[i + 1] = rich_handoff_fingerprint(ram, frame=i + 1)

    # idle a bit if we left but not yet 8-4 control
    if leave is not None and not reached_84:
        for j in range(400):
            env.step(IDLE)
            snap = read_snapshot(env.get_ram(), 0)
            if is_8_4_control(snap):
                reached_84 = True
                control_84_offset = (leave or 0) + j + 1
                break

    return {
        "start": start,
        "max_x": max_x,
        "leave": leave,
        "death": death,
        "reached_8_4_control": reached_84,
        "control_8_4_offset": control_84_offset,
        "samples": samples,
    }


def oracle_body_offsets(
    oracle: dict[str, dict[str, Any]],
    control_name: str = "control_8_3",
) -> dict[str, int]:
    """Map checkpoint name → frames after 8-3 control on the oracle movie."""
    if control_name not in oracle:
        return {}
    c0 = int(oracle[control_name]["movie_frame"])
    out: dict[str, int] = {}
    for name, row in oracle.items():
        if name == control_name:
            out[name] = 0
            continue
        mf = row.get("movie_frame")
        if mf is None:
            continue
        off = int(mf) - c0
        if off >= 0:
            out[name] = off
    return out


def find_first_divergence(
    oracle: dict[str, dict[str, Any]],
    fceumm_samples: dict[int, dict[str, Any]],
    offsets: dict[str, int],
    *,
    order: list[str] | None = None,
    x_tol: int = 2,
) -> list[DivPoint]:
    """Walk ordered mid-level landmarks; collect field diffs (first is primary)."""
    if order is None:
        order = [
            "control_8_3",
            "early_8_3_after_first_obstacle",
            "mid_8_3_x900",
            "mid_8_3_x1600",
            "hammer_bro_nearby_8_3",
            "flag_approach_8_3",
            "flagpole_grab_8_3",
            "leave_8_3_to_8_4",
            "control_8_4",
        ]
    all_diffs: list[DivPoint] = []
    for name in order:
        if name not in oracle or name not in offsets:
            continue
        off = offsets[name]
        o_fp = dict(oracle[name])
        o_fp.setdefault("movie_frame", o_fp.get("movie_frame"))
        f_fp = fceumm_samples.get(off) or fceumm_samples.get(off + 1) or fceumm_samples.get(off - 1)
        if f_fp is None:
            all_diffs.append(
                DivPoint(
                    name=name,
                    oracle_movie_frame=o_fp.get("movie_frame"),
                    body_offset=off,
                    field="__missing_sample__",
                    oracle_value="present",
                    fceumm_value=None,
                    oracle_fp=_fp_core(o_fp),
                )
            )
            continue
        diffs = _diff_fps(name, o_fp, f_fp, body_offset=off, x_tol=x_tol)
        # Also check enemy type multiset (meaningful spawn phase)
        o_types = sorted(e.get("type", -1) for e in (o_fp.get("enemies") or []))
        f_types = sorted(e.get("type", -1) for e in (f_fp.get("enemies") or []))
        if o_types != f_types:
            diffs.append(
                DivPoint(
                    name=name,
                    oracle_movie_frame=o_fp.get("movie_frame"),
                    body_offset=off,
                    field="enemy_types",
                    oracle_value=o_types,
                    fceumm_value=f_types,
                    oracle_fp=_fp_core(o_fp),
                    fceumm_fp=_fp_core(f_fp),
                )
            )
        all_diffs.extend(diffs)
        # stop at first landmark that has any non-frac-only mismatch
        meaningful = [
            d
            for d in diffs
            if d.field
            not in (
                "x_frac",
                "y_frac",
                "frame_counter",
                "screen_x",
            )
        ]
        if meaningful:
            # keep collecting but caller uses first meaningful
            pass
    return all_diffs


def try_minimal_phase_correction(
    env_factory,
    fm2_frames: list[list[int]],
    *,
    oracle_control_frame: int,
    start_candidates: list[int],
    lead_idles: list[int],
    sample_offsets: list[int],
    max_play: int = 2800,
) -> dict[str, Any]:
    """Search small (fm2_start, lead_idle) space for cleaner mid-level match / 8-4.

    Lead idle is a state-gated pad at 8-3 control before the FM2 body — not a
    natural_82 splice and not a long absolute tape rewrite.
    """
    best: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    for si in start_candidates:
        for lead in lead_idles:
            env = env_factory()
            try:
                gate = reach_stage_control(env, "8-3")
                if not gate.get("success"):
                    trials.append({"si": si, "lead": lead, "ok": False, "stage": "gate"})
                    continue
                lives = int(gate["control_snap"].lives)
                for _ in range(lead):
                    env.step(IDLE)
                result = play_and_sample(
                    env,
                    fm2_frames,
                    start=si,
                    sample_offsets=sample_offsets,
                    max_play=max_play,
                    start_lives=lives,
                )
                score = (
                    1 if result["reached_8_4_control"] else 0,
                    1 if result["leave"] else 0,
                    result["max_x"],
                    -(result["death"] or 99999),
                    -lead,
                )
                row = {
                    "si": si,
                    "lead": lead,
                    "score": list(score),
                    "max_x": result["max_x"],
                    "leave": result["leave"],
                    "death": result["death"],
                    "reached_8_4_control": result["reached_8_4_control"],
                    "control_8_4_offset": result["control_8_4_offset"],
                }
                trials.append(row)
                if best is None or score > tuple(best["score"]):
                    best = {**row, "samples": result["samples"]}
            finally:
                env.close()
    return {"best": best, "trials": trials, "n": len(trials)}


def export_correction_candidate(
    *,
    fm2_frames: list[list[int]],
    start: int,
    lead: int,
    n_frames: int,
    path: Path,
) -> Path:
    """Write a distinct oracle correction candidate (never overwrites shared seeds)."""
    body: list[list[int]] = []
    idle = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for _ in range(lead):
        body.append(list(idle))
    for i in range(n_frames):
        idx = start + i
        if idx >= len(fm2_frames):
            break
        body.append([int(x) for x in fm2_frames[idx][:9]])
    payload = {
        "format": "nes9_rle",
        "route_id": "smb_8_3_oracle_phase_correction_candidate",
        "game_name": "SuperMarioBros-Nes",
        "num_frames": len(body),
        "source": (
            f"FCEUX-oracle-informed fceumm correction: lead_idle={lead} + "
            f"HappyLee FM2[{start}:{start + n_frames}] (preserve L+R; not natural_82)"
        ),
        "oracle_meta": {
            "fm2_start": start,
            "lead_idle": lead,
            "body_frames": n_frames,
            "kind": "state_gated_lead_plus_fm2_body",
        },
        "segments": compress_nes9_rle(body),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # Never silently overwrite — write sibling if present.
        path = path.with_name(path.stem + "_v2" + path.suffix)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def compare_chain_to_oracle(
    *,
    fm2_path: Path = DEFAULT_FM2,
    oracle_path: Path | None = None,
    search_correction: bool = True,
) -> dict[str, Any]:
    """Full compare pipeline → evidence dict."""
    ORACLE_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    oracle = load_oracle_checkpoints(oracle_path)
    if "control_8_3" not in oracle:
        return {
            "success": False,
            "error": "missing oracle control_8_3 — run extract_fceux_checkpoints first",
            "oracle_keys": sorted(oracle.keys()),
        }

    offsets = oracle_body_offsets(oracle)
    sample_offs = sorted(set(offsets.values()) | {0, 60, 120, 200, 400, 600, 900, 1200, 1600, 2000})
    # always include landmark offsets
    for v in offsets.values():
        sample_offs.append(v)
    sample_offs = sorted(set(max(0, o) for o in sample_offs))

    fm2 = parse_fm2(fm2_path).frames
    oracle_c83 = int(oracle["control_8_3"]["movie_frame"])

    # Primary trial: FM2 absolute control frame as body start (phase-blocked risk)
    env, gate, ctrl_fp = reach_8_3_control_fceumm()
    report: dict[str, Any] = {
        "schema": "smb.fceux_oracle_vs_fceumm.v1",
        "gate": {
            k: gate.get(k)
            for k in (
                "success",
                "ctrl_wait_8_3",
                "leave_8_1",
                "leave_8_2",
                "ctrl_wait_8_2",
                "stage",
            )
        },
        "fceumm_control_8_3_fp": ctrl_fp,
        "oracle_control_8_3": _fp_core(oracle["control_8_3"]),
        "oracle_control_movie_frame": oracle_c83,
        "body_offsets": offsets,
    }
    if not gate.get("success"):
        env.close()
        report["success"] = False
        report["error"] = "fceumm failed to reach 8-3 control via HL chain"
        return report

    lives = int(gate["control_snap"].lives)
    # Control-gate comparison (entry fingerprint)
    entry_diffs = _diff_fps(
        "control_8_3",
        oracle["control_8_3"],
        {**ctrl_fp, "movie_frame": oracle_c83},
        body_offset=0,
        x_tol=0,
    )
    report["entry_diffs"] = [asdict(d) for d in entry_diffs]

    primary = play_and_sample(
        env,
        fm2,
        start=oracle_c83,
        sample_offsets=sample_offs,
        max_play=2800,
        start_lives=lives,
    )
    env.close()
    report["primary_body"] = {
        k: primary[k]
        for k in (
            "start",
            "max_x",
            "leave",
            "death",
            "reached_8_4_control",
            "control_8_4_offset",
        )
    }

    all_divs = find_first_divergence(
        oracle, primary["samples"], offsets, x_tol=2
    )
    meaningful = [
        d
        for d in all_divs
        if d.field
        not in ("x_frac", "y_frac", "frame_counter", "screen_x", "__missing_sample__")
    ]
    # Prefer first landmark with meaningful field mismatch
    first: DivPoint | None = meaningful[0] if meaningful else (all_divs[0] if all_divs else None)
    report["divergences"] = [asdict(d) for d in all_divs[:40]]
    report["first_meaningful_divergence"] = asdict(first) if first else None
    report["primary_reached_8_4"] = primary["reached_8_4_control"]

    # Dense mid-level x-progress compare using oracle trace if present
    trace_path = ORACLE_EVIDENCE_DIR / "fceux_ram_trace.jsonl"
    mid_compare: list[dict[str, Any]] = []
    if trace_path.is_file() and primary["samples"]:
        trace_idx = _index_trace_by_frame(load_jsonl(trace_path))
        for off, f_fp in sorted(primary["samples"].items()):
            mf = oracle_c83 + off
            o = trace_idx.get(mf)
            if not o:
                continue
            dx = int(f_fp.get("player_x") or 0) - int(o.get("player_x") or 0)
            same_level = (
                int(f_fp.get("world") or -1) == int(o.get("world") or -2)
                and int(f_fp.get("level") or -1) == int(o.get("level") or -2)
            )
            mid_compare.append(
                {
                    "body_offset": off,
                    "oracle_movie_frame": mf,
                    "oracle_x": o.get("player_x"),
                    "fceumm_x": f_fp.get("player_x"),
                    "dx": dx,
                    "oracle_timer": o.get("timer"),
                    "fceumm_timer": f_fp.get("timer"),
                    "oracle_ps": o.get("player_state"),
                    "fceumm_ps": f_fp.get("player_state"),
                    "same_level": same_level,
                    "oracle_dead": int(o.get("player_state") or 0) == PLAYER_STATE_DYING,
                    "fceumm_dead": int(f_fp.get("player_state") or 0) == PLAYER_STATE_DYING,
                }
            )
        # first mid sample where |dx|>8 or death/level mismatch
        first_mid = next(
            (
                m
                for m in mid_compare
                if (not m["same_level"])
                or m["fceumm_dead"]
                or abs(int(m["dx"])) > 8
                or (m["oracle_timer"] != m["fceumm_timer"] and m["body_offset"] > 0)
            ),
            None,
        )
        report["mid_level_compare_head"] = mid_compare[:25]
        report["first_mid_level_break"] = first_mid
        if first_mid and (
            first is None
            or (first_mid["body_offset"] or 0) < (first.body_offset or 10**9)
        ):
            report["first_meaningful_divergence"] = {
                "name": "mid_level_trace",
                "oracle_movie_frame": first_mid["oracle_movie_frame"],
                "body_offset": first_mid["body_offset"],
                "field": "player_x" if abs(int(first_mid["dx"])) > 8 else "timer_or_level",
                "oracle_value": {
                    "x": first_mid["oracle_x"],
                    "timer": first_mid["oracle_timer"],
                    "ps": first_mid["oracle_ps"],
                },
                "fceumm_value": {
                    "x": first_mid["fceumm_x"],
                    "timer": first_mid["fceumm_timer"],
                    "ps": first_mid["fceumm_ps"],
                },
                "detail": first_mid,
            }

    correction: dict[str, Any] | None = None
    if search_correction and not primary["reached_8_4_control"]:
        # Fan small neighborhood around oracle control frame + known HL search band
        cands = sorted(
            set(
                [oracle_c83]
                + list(range(oracle_c83 - 40, oracle_c83 + 41, 2))
                + list(range(13080, 13420, 5))
            )
        )
        leads = [0, 1, 2, 3, 4, 5, 8, 10, 12, 16, 21]
        # landmark sample offsets only (faster)
        land_offs = sorted(set(offsets.values()) | {0, 200, 600, 1200})

        def factory():
            return make_level1_env()

        correction = try_minimal_phase_correction(
            factory,
            fm2,
            oracle_control_frame=oracle_c83,
            start_candidates=cands,
            lead_idles=leads,
            sample_offsets=land_offs,
            max_play=2600,
        )
        report["correction_search"] = {
            "n": correction["n"],
            "best": {
                k: correction["best"].get(k)
                for k in (
                    "si",
                    "lead",
                    "max_x",
                    "leave",
                    "death",
                    "reached_8_4_control",
                    "control_8_4_offset",
                    "score",
                )
            }
            if correction.get("best")
            else None,
            "top": sorted(
                correction["trials"],
                key=lambda r: tuple(r.get("score") or [0]),
                reverse=True,
            )[:12],
        }
        best = correction.get("best")
        if best and best.get("reached_8_4_control"):
            # Export candidate body: lead + FM2 until leave+pad
            n_body = int(best.get("control_8_4_offset") or best.get("leave") or 2400) + 30
            cand_path = ORACLE_EVIDENCE_DIR / "smb_8_3_oracle_phase_correction_candidate.json"
            written = export_correction_candidate(
                fm2_frames=fm2,
                start=int(best["si"]),
                lead=int(best["lead"]),
                n_frames=n_body,
                path=cand_path,
            )
            report["correction_export"] = str(written)
            report["correction_reached_8_4"] = True
        elif best:
            report["correction_reached_8_4"] = False
            report["correction_best_max_x"] = best.get("max_x")

    report["success"] = True
    report["reached_8_4_control"] = bool(
        primary["reached_8_4_control"]
        or (correction or {}).get("best", {}).get("reached_8_4_control")
    )

    out = ORACLE_EVIDENCE_DIR / "compare_evidence.json"
    # strip bulky sample blobs
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    report["output"] = str(out)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fm2", type=Path, default=DEFAULT_FM2)
    ap.add_argument("--no-search", action="store_true", help="Skip correction search")
    ap.add_argument("--oracle", type=Path, default=None)
    args = ap.parse_args(argv)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from smb.paths import REPO_ROOT

    os.chdir(REPO_ROOT.resolve())
    report = compare_chain_to_oracle(
        fm2_path=args.fm2,
        oracle_path=args.oracle,
        search_correction=not args.no_search,
    )
    summary = {
        "success": report.get("success"),
        "gate": report.get("gate"),
        "oracle_control_movie_frame": report.get("oracle_control_movie_frame"),
        "entry_diff_fields": [d["field"] for d in report.get("entry_diffs") or []],
        "first_meaningful_divergence": report.get("first_meaningful_divergence"),
        "first_mid_level_break": report.get("first_mid_level_break"),
        "primary_body": report.get("primary_body"),
        "reached_8_4_control": report.get("reached_8_4_control"),
        "correction_search_best": (report.get("correction_search") or {}).get("best"),
        "correction_export": report.get("correction_export"),
        "output": report.get("output"),
        "error": report.get("error"),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    sys.exit(main())
