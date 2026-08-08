"""Focused fceumm probe for early HappyLee 8-3 jump/physics divergence.

Root cause (pinned by FCEUX 2.6.6 oracle vs stable-retro/fceumm):

- 8-3 **entry** matches exactly (``entry_diffs == []``), including timer /
  framerule phase and ``frame_counter``.
- Body offsets **0–100** match oracle pose (jumps 1–2 OK).
- First ``y`` / ``y_speed`` break is at body offset **101** (movie 13222):
  same ``x=248 y=152``, oracle ``ys=-3`` vs fceumm ``ys=-5``.
- Landmark **114** (first obstacle, movie 13235): oracle ``y=135 ys=-1`` vs
  fceumm baseline ``y=109 ys=-3`` (same ``x=280``, same timer/framerule).

This is **not** an 8-2→8-3 transition bug. Repair target is local jump-3
A-hold / input timing on FM2 ``13121–13235``. L+R is preserved; no
``natural_82`` mid-splice.

```bash
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
  uv run python -m smb.tas.oracle.probe_early_8_3
# optional local A-edge search + candidate export:
  uv run python -m smb.tas.oracle.probe_early_8_3 --search-repair --export
```

Writes under ``recordings/tas_import/oracle_happylee_8_3/`` (distinct names;
never overwrites shared seeds).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smb.policy import compress_nes9_rle
from smb.ram import PLAYER_STATE_DYING, read_snapshot, rich_handoff_fingerprint
from smb.tas.chain import reach_stage_control
from smb.tas.fm2 import parse_fm2
from smb.tas.oracle.compare_fceumm_chain import (
    load_oracle_checkpoints,
    oracle_body_offsets,
)
from smb.tas.oracle.extract_fceux_checkpoints import (
    DEFAULT_FM2,
    ORACLE_EVIDENCE_DIR,
    load_jsonl,
)
from smb.tas.replay import get_state, make_level1_env, set_state, to_action9
from smb.tas.stages import is_8_4_control

# NES-9 slot for A (stable-retro layout; L+R at 6/7 — never sanitize).
NES_A = 8
NES_LEFT = 6
NES_RIGHT = 7

# Oracle pins (HappyLee #1715M under FCEUX 2.6.6).
ORACLE_CONTROL_FRAME = 13121
ORACLE_FIRST_OBSTACLE_FRAME = 13235
ORACLE_FIRST_DIVERGENCE_OFFSET = 101  # movie 13222
ORACLE_FIRST_OBSTACLE_OFFSET = 114

# Landmark gate order (exact pose compare; max_x alone is not success).
GATE_ORDER = (
    "early_8_3_after_first_obstacle",
    "mid_8_3_x900",
    "mid_8_3_x1600",
    "hammer_bro_nearby_8_3",
    "flag_approach_8_3",
    "flagpole_grab_8_3",
    "leave_8_3_to_8_4",
    "control_8_4",
)

POSE_KEYS = (
    "player_x",
    "player_y",
    "y_speed",
    "x_speed",
    "timer",
    "timer_mod21",
    "grounded",
)


@dataclass
class DenseRow:
    body_offset: int
    movie_frame: int
    oracle: dict[str, Any]
    fceumm: dict[str, Any]
    buttons: str
    y_div: bool
    any_div: bool


@dataclass
class ProbeReport:
    schema: str = "smb.oracle_early83_dense_probe.v1"
    success: bool = False
    entry_match: bool = False
    entry_diffs: list[dict[str, Any]] = field(default_factory=list)
    first_y_vy_divergence: dict[str, Any] | None = None
    first_any_divergence: dict[str, Any] | None = None
    landmark_114: dict[str, Any] = field(default_factory=dict)
    baseline_body: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    repair: dict[str, Any] | None = None
    diagnosis: dict[str, Any] = field(default_factory=dict)
    dense_head: list[dict[str, Any]] = field(default_factory=list)
    output: str | None = None
    candidate: str | None = None


def _btn_fmt(frame: list[int]) -> str:
    names = ("B", "Sel", "St", "U", "D", "L", "R", "A")
    idxs = (0, 2, 3, 4, 5, 6, 7, 8)
    bits = [n for n, i in zip(names, idxs) if i < len(frame) and frame[i]]
    return "".join(bits) or "."


def _pose(fp: dict[str, Any]) -> dict[str, Any]:
    return {k: fp.get(k) for k in POSE_KEYS}


def _diff_pose(
    oracle_fp: dict[str, Any], fceumm_fp: dict[str, Any], *, x_tol: int = 0
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for k in POSE_KEYS:
        ov, fv = oracle_fp.get(k), fceumm_fp.get(k)
        if k == "player_x" and isinstance(ov, int) and isinstance(fv, int):
            if abs(ov - fv) <= x_tol:
                continue
        if ov != fv:
            out.append({"field": k, "oracle": ov, "fceumm": fv})
    return out


def clone_fm2_body(
    fm2_frames: list[list[int]],
    *,
    control_frame: int = ORACLE_CONTROL_FRAME,
    n: int = 2500,
) -> list[list[int]]:
    return [list(fm2_frames[control_frame + i][:9]) for i in range(n)]


def apply_a_release(body: list[list[int]], release_from: int, release_to: int = 120) -> list[list[int]]:
    """Zero A on body indices [release_from, release_to). Preserves L+R."""
    out = [list(fr) for fr in body]
    for i in range(release_from, min(release_to, len(out))):
        out[i][NES_A] = 0
    return out


def count_lr(body: list[list[int]]) -> int:
    return sum(1 for fr in body if fr[NES_LEFT] and fr[NES_RIGHT])


def play_body(
    env: Any,
    body: list[list[int]],
    *,
    lives: int,
    sample_offsets: set[int],
    max_play: int | None = None,
    dense_until: int = 0,
) -> dict[str, Any]:
    """Play body from current env state; sample rich FPs; track leave/death."""
    samples: dict[int, dict[str, Any]] = {}
    dense: dict[int, dict[str, Any]] = {}
    max_x = 0
    death: int | None = None
    leave: int | None = None
    reached_84 = False
    control_84: int | None = None
    samples[0] = rich_handoff_fingerprint(env.get_ram(), frame=0)
    if dense_until >= 0:
        dense[0] = samples[0]
    limit = len(body) if max_play is None else min(len(body), max_play)
    for i in range(limit):
        env.step(to_action9(body[i]))
        off = i + 1
        ram = env.get_ram()
        snap = read_snapshot(ram, frame=off)
        px = int(snap.player_x)
        if 0 < px < 20000:
            max_x = max(max_x, px)
        if death is None and (
            int(snap.lives) < lives or int(snap.player_state) == PLAYER_STATE_DYING
        ):
            death = off
        if leave is None and int(snap.world) == 7 and int(snap.level) == 3:
            leave = off
        if not reached_84 and is_8_4_control(snap):
            reached_84 = True
            control_84 = off
        if off in sample_offsets:
            samples[off] = rich_handoff_fingerprint(ram, frame=off)
        if off <= dense_until:
            dense[off] = rich_handoff_fingerprint(ram, frame=off)
        if death is not None and off > death + 8:
            break
        if leave is not None and off > leave + 80:
            break
    return {
        "samples": samples,
        "dense": dense,
        "max_x": max_x,
        "death": death,
        "leave": leave,
        "reached_8_4_control": reached_84,
        "control_8_4_offset": control_84,
    }


def dense_compare_to_oracle(
    dense: dict[int, dict[str, Any]],
    oracle_trace: dict[int, dict[str, Any]],
    body: list[list[int]],
    *,
    control_frame: int = ORACLE_CONTROL_FRAME,
    until: int = 120,
) -> list[DenseRow]:
    rows: list[DenseRow] = []
    for off in range(0, until + 1):
        mf = control_frame + off
        o = oracle_trace.get(mf) or {}
        f = dense.get(off) or {}
        if not o and not f:
            continue
        y_div = bool(o) and bool(f) and (
            o.get("player_y") != f.get("player_y") or o.get("y_speed") != f.get("y_speed")
        )
        any_div = bool(o) and bool(f) and bool(
            _diff_pose(o, f, x_tol=0)
        )
        btn = "."
        if off > 0 and off - 1 < len(body):
            btn = _btn_fmt(body[off - 1])
        rows.append(
            DenseRow(
                body_offset=off,
                movie_frame=mf,
                oracle=_pose(o) if o else {},
                fceumm=_pose(f) if f else {},
                buttons=btn,
                y_div=y_div,
                any_div=any_div,
            )
        )
    return rows


def gate_landmarks(
    samples: dict[int, dict[str, Any]],
    oracle: dict[str, dict[str, Any]],
    offsets: dict[str, int],
    *,
    x_tol: int = 0,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in GATE_ORDER:
        if name not in oracle or name not in offsets:
            continue
        off = offsets[name]
        o = oracle[name]
        f = samples.get(off) or {}
        diffs = _diff_pose(o, f, x_tol=x_tol) if f else [
            {"field": k, "oracle": o.get(k), "fceumm": None} for k in POSE_KEYS
        ]
        out[name] = {
            "body_offset": off,
            "match": len(diffs) == 0 and bool(f),
            "diffs": diffs,
            "fceumm": _pose(f) if f else {},
            "oracle": _pose(o),
        }
    return out


def gate_progress(gates: dict[str, Any], *, xy_tol: int = 1) -> dict[str, bool]:
    """Ordered success flags. max_x alone is never treated as a pass."""
    early = gates.get("early_8_3_after_first_obstacle") or {}
    f = early.get("fceumm") or {}
    o = early.get("oracle") or {}
    xy_ok = (
        f.get("player_x") == o.get("player_x")
        and isinstance(f.get("player_y"), int)
        and isinstance(o.get("player_y"), int)
        and abs(int(f["player_y"]) - int(o["player_y"])) <= xy_tol
    )
    x900 = gates.get("mid_8_3_x900") or {}
    x1600 = gates.get("mid_8_3_x1600") or {}
    leave = gates.get("leave_8_3_to_8_4") or {}
    c84 = gates.get("control_8_4") or {}

    def _x_near(g: dict[str, Any], target: int, tol: int = 8) -> bool:
        fx = (g.get("fceumm") or {}).get("player_x")
        return isinstance(fx, int) and abs(fx - target) <= tol

    return {
        "first_obstacle_exact": bool(early.get("match")),
        "first_obstacle_xy": bool(xy_ok),
        "x900": _x_near(x900, 900),
        "x1600": _x_near(x1600, 1600),
        "flag_or_leave": bool(leave.get("match"))
        or bool((leave.get("fceumm") or {}).get("player_x") and
                abs(int((leave.get("fceumm") or {}).get("player_x") or 0) - 3554) <= 20),
        "control_8_4": bool(c84.get("match")),
    }


def search_a_release_repairs(
    env: Any,
    ctrl_state: Any,
    base_body: list[list[int]],
    *,
    lives: int,
    oracle: dict[str, dict[str, Any]],
    offsets: dict[str, int],
    sample_offsets: set[int],
    release_from_range: range = range(96, 116),
) -> dict[str, Any]:
    """Local state-gated A-release search on jump-3 window only."""
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    o114 = oracle["early_8_3_after_first_obstacle"]

    def rank(row: dict[str, Any]) -> tuple:
        """Ordered gates: first obstacle → x900 → x1600 → leave/8-4.

        max_x is a tie-break only — never sole success.
        """
        g = row["gate_progress"]
        return (
            1 if g.get("first_obstacle_exact") else 0,
            1 if g.get("first_obstacle_xy") else 0,
            1 if g.get("x900") else 0,
            1 if g.get("x1600") else 0,
            1 if g.get("flag_or_leave") else 0,
            1 if g.get("control_8_4") else 0,
            -abs(row.get("dy114") or 99),
            -abs(row.get("dys114") or 99),
            row.get("max_x") or 0,
            -(row.get("death") or 10**9),
        )

    # baseline
    set_state(env, ctrl_state)
    base_res = play_body(
        env, base_body, lives=lives, sample_offsets=sample_offsets, max_play=2200
    )
    base_gates = gate_landmarks(base_res["samples"], oracle, offsets)
    s114 = base_res["samples"].get(ORACLE_FIRST_OBSTACLE_OFFSET) or {}
    base_row = {
        "name": "baseline_fm2",
        "release_from": None,
        "max_x": base_res["max_x"],
        "death": base_res["death"],
        "leave": base_res["leave"],
        "reached_8_4_control": base_res["reached_8_4_control"],
        "s114": _pose(s114),
        "dy114": int(s114.get("player_y") or 0) - int(o114["player_y"]),
        "dys114": int(s114.get("y_speed") or 0) - int(o114["y_speed"]),
        "gate_progress": gate_progress(base_gates),
        "lr_frames": count_lr(base_body),
    }
    trials.append(base_row)
    best = {**base_row, "body": base_body, "gates": base_gates}

    for rel in release_from_range:
        body = apply_a_release(base_body, rel)
        set_state(env, ctrl_state)
        res = play_body(
            env, body, lives=lives, sample_offsets=sample_offsets, max_play=2200
        )
        gates = gate_landmarks(res["samples"], oracle, offsets)
        s114 = res["samples"].get(ORACLE_FIRST_OBSTACLE_OFFSET) or {}
        row = {
            "name": f"A_release_from_{rel}",
            "release_from": rel,
            "max_x": res["max_x"],
            "death": res["death"],
            "leave": res["leave"],
            "reached_8_4_control": res["reached_8_4_control"],
            "s114": _pose(s114),
            "dy114": int(s114.get("player_y") or 0) - int(o114["player_y"]),
            "dys114": int(s114.get("y_speed") or 0) - int(o114["y_speed"]),
            "gate_progress": gate_progress(gates),
            "lr_frames": count_lr(body),
        }
        trials.append(row)
        if best is None or rank(row) > rank(best):
            best = {**row, "body": body, "gates": gates}

    trials_sorted = sorted(trials, key=rank, reverse=True)
    return {
        "n": len(trials),
        "baseline": base_row,
        "best": {k: best[k] for k in best if k not in ("body", "gates")} if best else None,
        "best_gates": (best or {}).get("gates"),
        "best_body": (best or {}).get("body"),
        "top": trials_sorted[:12],
        "rank_key": "gates then |dy114| then max_x (max_x never sole success)",
    }


def export_candidate(
    body: list[list[int]],
    *,
    path: Path,
    meta: dict[str, Any],
    route_id: str = "smb_8_3_oracle_early_jump_repair_candidate",
) -> Path:
    """Write a distinct candidate; never overwrite shared seeds under models/."""
    path = Path(path)
    if path.exists():
        path = path.with_name(path.stem + "_v2" + path.suffix)
    # drop trailing empty pad if death known
    n = int(meta.get("export_frames") or len(body))
    body = body[:n]
    payload = {
        "format": "nes9_rle",
        "route_id": route_id,
        "game_name": "SuperMarioBros-Nes",
        "num_frames": len(body),
        "source": (
            "Oracle-informed early-8-3 jump-3 A-hold repair from HappyLee FM2 "
            f"@ control {ORACLE_CONTROL_FRAME}; L+R preserved; not natural_82"
        ),
        "oracle_meta": {
            "kind": "state_gated_local_A_edge_repair",
            "fm2_control_frame": ORACLE_CONTROL_FRAME,
            "preserve_lr": True,
            "no_natural_82_splice": True,
            **meta,
        },
        "segments": compress_nes9_rle(body),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run_probe(
    *,
    fm2_path: Path = DEFAULT_FM2,
    search_repair: bool = True,
    export: bool = True,
    dense_until: int = 120,
) -> ProbeReport:
    """Reach real 8-3 control, dense-compare early body, optional A-edge repair."""
    ORACLE_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    report = ProbeReport()
    oracle = load_oracle_checkpoints()
    if "control_8_3" not in oracle or "early_8_3_after_first_obstacle" not in oracle:
        report.diagnosis = {"error": "missing oracle checkpoints — run extract first"}
        return report

    offsets = oracle_body_offsets(oracle)
    sample_offs = set(offsets.values()) | {
        0,
        ORACLE_FIRST_DIVERGENCE_OFFSET,
        ORACLE_FIRST_OBSTACLE_OFFSET,
        200,
        400,
        600,
        900,
        1200,
        1600,
        2000,
    }

    fm2 = parse_fm2(fm2_path).frames
    base_body = clone_fm2_body(fm2)
    # Oracle dense trace for frame-level compare
    trace_rows = load_jsonl(ORACLE_EVIDENCE_DIR / "fceux_ram_trace.jsonl")
    oracle_trace = {
        int(r["movie_frame"]): r for r in trace_rows if "movie_frame" in r
    }

    env = make_level1_env()
    try:
        gate = reach_stage_control(env, "8-3")
        if not gate.get("success"):
            report.diagnosis = {"error": "failed to reach 8-3 control", "gate": gate}
            return report
        lives = int(gate["control_snap"].lives)
        ctrl_fp = rich_handoff_fingerprint(env.get_ram(), frame=0)
        entry_diffs = _diff_pose(oracle["control_8_3"], ctrl_fp, x_tol=0)
        # also frame_counter / subpixel if present
        for k in ("frame_counter", "x_frac", "y_frac"):
            if oracle["control_8_3"].get(k) != ctrl_fp.get(k):
                entry_diffs.append(
                    {
                        "field": k,
                        "oracle": oracle["control_8_3"].get(k),
                        "fceumm": ctrl_fp.get(k),
                    }
                )
        report.entry_diffs = entry_diffs
        report.entry_match = len(entry_diffs) == 0
        ctrl_state = get_state(env)

        # Baseline dense + body
        set_state(env, ctrl_state)
        base_res = play_body(
            env,
            base_body,
            lives=lives,
            sample_offsets=sample_offs,
            max_play=2200,
            dense_until=dense_until,
        )
        rows = dense_compare_to_oracle(
            base_res["dense"], oracle_trace, base_body, until=dense_until
        )
        first_y = next((r for r in rows if r.body_offset > 0 and r.y_div), None)
        first_any = next((r for r in rows if r.body_offset > 0 and r.any_div), None)
        report.first_y_vy_divergence = asdict(first_y) if first_y else None
        report.first_any_divergence = asdict(first_any) if first_any else None
        report.dense_head = [
            asdict(r)
            for r in rows
            if r.body_offset in (0, 1, 3, 26, 97, 100, 101, 102, 114, 116)
            or r.y_div
            or r.body_offset % 20 == 0
        ][:40]

        s114 = base_res["samples"].get(ORACLE_FIRST_OBSTACLE_OFFSET) or {}
        o114 = oracle["early_8_3_after_first_obstacle"]
        report.landmark_114 = {
            "oracle": _pose(o114),
            "fceumm": _pose(s114),
            "dy": int(s114.get("player_y") or 0) - int(o114["player_y"]),
            "dys": int(s114.get("y_speed") or 0) - int(o114["y_speed"]),
            "match": not _diff_pose(o114, s114),
        }
        base_gates = gate_landmarks(base_res["samples"], oracle, offsets)
        report.baseline_body = {
            "max_x": base_res["max_x"],
            "death": base_res["death"],
            "leave": base_res["leave"],
            "reached_8_4_control": base_res["reached_8_4_control"],
            "gate_progress": gate_progress(base_gates),
            "lr_frames_in_body_prefix": count_lr(base_body[:200]),
        }
        report.gates = {"baseline": base_gates}

        repair_info: dict[str, Any] | None = None
        if search_repair:
            repair_info = search_a_release_repairs(
                env,
                ctrl_state,
                base_body,
                lives=lives,
                oracle=oracle,
                offsets=offsets,
                sample_offsets=sample_offs,
            )
            # strip non-json body from nested
            repair_public = {
                k: repair_info[k]
                for k in ("n", "baseline", "best", "top", "rank_key")
            }
            repair_public["best_gates"] = repair_info.get("best_gates")
            report.repair = repair_public
            report.gates["best_repair"] = repair_info.get("best_gates")

            if export and repair_info.get("best_body") is not None:
                best = repair_info["best"] or {}
                death = best.get("death")
                export_n = (death + 50) if death else min(len(base_body), 2250)
                cand = export_candidate(
                    repair_info["best_body"],
                    path=ORACLE_EVIDENCE_DIR
                    / "smb_8_3_oracle_early_jump_repair_candidate.json",
                    meta={
                        "variant": best.get("name"),
                        "release_from": best.get("release_from"),
                        "dy114": best.get("dy114"),
                        "dys114": best.get("dys114"),
                        "max_x": best.get("max_x"),
                        "death": best.get("death"),
                        "gate_progress": best.get("gate_progress"),
                        "export_frames": export_n,
                        "first_divergence_body_offset": (
                            first_y.body_offset if first_y else None
                        ),
                    },
                )
                report.candidate = str(cand)

        # Diagnosis summary (honest; no full-port claim)
        report.diagnosis = {
            "entry_proven_correct": report.entry_match,
            "first_y_vy_body_offset": first_y.body_offset if first_y else None,
            "first_y_vy_movie_frame": first_y.movie_frame if first_y else None,
            "landmark_114_baseline": report.landmark_114,
            "cause_class": (
                "jump3_A_hold_input_timing_or_fceumm_vs_fceux_variable_jump"
            ),
            "notes": (
                "Entry FP exact. Jumps 1–2 match. Jump 3: with identical FM2 A-hold, "
                "fceumm keeps ys=-5 at off 101 while FCEUX has ys=-3 (A-release gravity). "
                "Local A-release near body index 102 recovers y≈135 (|dy|≤1) but not "
                "exact ys=-1; x=900 / leave still fail. Not an 8-2→8-3 transition bug. "
                "L+R preserved. No natural_82 splice. max_x alone is not success."
            ),
            "gate_order": list(GATE_ORDER),
            "full_port": False,
        }
        report.success = True
    finally:
        env.close()

    out_path = ORACLE_EVIDENCE_DIR / "early83_dense_probe_evidence.json"
    # JSON-safe dump (no body blobs)
    payload = {
        "schema": report.schema,
        "success": report.success,
        "entry_match": report.entry_match,
        "entry_diffs": report.entry_diffs,
        "first_y_vy_divergence": report.first_y_vy_divergence,
        "first_any_divergence": report.first_any_divergence,
        "landmark_114": report.landmark_114,
        "baseline_body": report.baseline_body,
        "gates_summary": {
            k: {
                name: {
                    "match": g.get("match"),
                    "fceumm": g.get("fceumm"),
                    "diffs": g.get("diffs"),
                }
                for name, g in (v or {}).items()
            }
            if isinstance(v, dict)
            else v
            for k, v in (report.gates or {}).items()
        },
        "repair": report.repair,
        "diagnosis": report.diagnosis,
        "dense_head": report.dense_head,
        "candidate": report.candidate,
        "pins": {
            "oracle_control_frame": ORACLE_CONTROL_FRAME,
            "oracle_first_obstacle_frame": ORACLE_FIRST_OBSTACLE_FRAME,
            "fm2": str(fm2_path),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    report.output = str(out_path)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fm2", type=Path, default=DEFAULT_FM2)
    ap.add_argument(
        "--search-repair",
        action="store_true",
        default=True,
        help="Search local A-release repairs (default on)",
    )
    ap.add_argument("--no-search-repair", action="store_true")
    ap.add_argument("--export", action="store_true", default=True)
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--dense-until", type=int, default=120)
    args = ap.parse_args(argv)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from smb.paths import REPO_ROOT

    os.chdir(REPO_ROOT.resolve())
    report = run_probe(
        fm2_path=args.fm2,
        search_repair=not args.no_search_repair,
        export=not args.no_export,
        dense_until=args.dense_until,
    )
    summary = {
        "success": report.success,
        "entry_match": report.entry_match,
        "first_y_vy": report.first_y_vy_divergence,
        "landmark_114": report.landmark_114,
        "baseline": report.baseline_body,
        "repair_best": (report.repair or {}).get("best"),
        "candidate": report.candidate,
        "output": report.output,
        "full_port": False,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0 if report.success else 2


if __name__ == "__main__":
    sys.exit(main())
