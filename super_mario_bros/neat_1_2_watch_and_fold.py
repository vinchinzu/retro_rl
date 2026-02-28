#!/usr/bin/env python3
"""Watch NEAT jobs for SMB 1-2, then fold best chained candidate safely.

This script is non-destructive:
- waits for provided trainer PIDs to finish
- runs chained-context analysis for smb_1_2 (updates model registry)
- validates the selected smb_1_2 candidate in a full chained any% route
- writes timestamped reports without overwriting canonical best files
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SMB = ROOT / "super_mario_bros"
OPT = SMB / "optimizer"
# Ensure direct-script execution can import sibling packages (platformer_common, retro_harness, etc.).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def wait_for_pids(pids: list[int], poll_seconds: int, timeout_seconds: int) -> dict[str, Any]:
    started = time.time()
    while True:
        alive = [pid for pid in pids if pid_alive(pid)]
        if not alive:
            return {"timed_out": False, "alive": [], "waited_seconds": int(time.time() - started)}
        if timeout_seconds > 0 and (time.time() - started) >= timeout_seconds:
            return {
                "timed_out": True,
                "alive": alive,
                "waited_seconds": int(time.time() - started),
            }
        print(
            f"[watch] {utc_now()} waiting; alive={alive} "
            f"elapsed={int(time.time() - started)}s"
        )
        time.sleep(max(1, poll_seconds))


def _selected_path_from_analyze(report: dict[str, Any]) -> str:
    segments = report.get("improved", {}).get("segments", [])
    for seg in segments:
        if seg.get("config_id") == "smb_1_2":
            selected = seg.get("selected") or {}
            path = selected.get("path", "")
            if path:
                return str(path)
    raise RuntimeError("No selected candidate for smb_1_2 in analyze report")


def _run_hybrid_analyze(
    *,
    registry_path: Path,
    report_path: Path,
    max_candidates: int,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "super_mario_bros.hybrid_pipeline",
        "analyze",
        "--route",
        "smb_any_percent",
        "--segments",
        "smb_1_2",
        "--selection-context",
        "chained",
        "--force-eval",
        "--eval-runs",
        "1",
        "--max-candidates",
        str(max_candidates),
        "--registry",
        str(registry_path),
        "--report",
        str(report_path),
    ]
    print("[analyze] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)
    return json.loads(report_path.read_text())


def _to_chain_json(route_result) -> dict[str, Any]:
    return {
        "completed_count": route_result.completed_count,
        "total_count": len(route_result.route.segments),
        "all_completed": route_result.all_completed,
        "total_frames": route_result.total_frames,
        "total_seconds": round(route_result.total_frames / 60.0, 3),
        "segments": [
            {
                "label": s.segment.label,
                "config_id": s.segment.config_id,
                "status": s.status,
                "frames": s.frames,
                "recording": str(s.recording_path) if s.recording_path else "",
                "error": s.error,
            }
            for s in route_result.segments
        ],
    }


def _validate_full_chain_with_candidate(
    *,
    base_chain_path: Path,
    candidate_1_2: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import platformer_common.levels.smb  # noqa: F401
    from platformer_common.route import RouteConfig, RouteSegment, chain_live

    baseline = json.loads(base_chain_path.read_text())
    base_segments = baseline.get("segments", [])
    route_segments = []
    for seg in base_segments:
        recording = str(seg.get("recording", ""))
        if seg.get("config_id") == "smb_1_2":
            recording = candidate_1_2
        route_segments.append(
            RouteSegment(
                config_id=str(seg["config_id"]),
                label=str(seg.get("label", seg["config_id"])),
                recording=recording,
            )
        )

    route = RouteConfig(
        route_id="smb_any_percent_1_2_candidate_check",
        display_name="SMB Any% (1-2 candidate check)",
        segments=route_segments,
    )
    result = chain_live(route, save_states=False, verbose=False)
    return baseline, _to_chain_json(result)


def main() -> int:
    ts = run_id()
    default_final = OPT / f"neat_1_2_fold_report_{ts}.json"
    default_analyze = OPT / f"hybrid_report_smb_1_2_neat_{ts}.json"
    default_chain = OPT / f"current_chain_eval_with_1_2_neat_{ts}.json"

    parser = argparse.ArgumentParser(description="Watch NEAT jobs then fold best smb_1_2 candidate")
    parser.add_argument(
        "--wait-pids",
        default="",
        help="Comma-separated trainer process PIDs to wait for before folding",
    )
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=30,
        help="Polling interval while waiting for PIDs (default 30)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=0,
        help="Max wait time (0 = no timeout)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=128,
        help="Max smb_1_2 candidates for chained analyze (default 128)",
    )
    parser.add_argument(
        "--base-chain",
        default=str(OPT / "current_chain_eval.json"),
        help="Baseline chain eval JSON",
    )
    parser.add_argument(
        "--registry",
        default=str(OPT / "model_registry.json"),
        help="Model registry JSON path",
    )
    parser.add_argument(
        "--analyze-report",
        default=str(default_analyze),
        help="Per-segment analyze report path",
    )
    parser.add_argument(
        "--candidate-chain-json",
        default=str(default_chain),
        help="Where to write full-chain result using selected smb_1_2 candidate",
    )
    parser.add_argument(
        "--final-report",
        default=str(default_final),
        help="Top-level fold report path",
    )
    args = parser.parse_args()

    pids: list[int] = []
    if args.wait_pids.strip():
        pids = [int(chunk.strip()) for chunk in args.wait_pids.split(",") if chunk.strip()]

    wait_result = {"timed_out": False, "alive": [], "waited_seconds": 0}
    if pids:
        wait_result = wait_for_pids(
            pids=pids,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        if wait_result["timed_out"]:
            report = {
                "generated_at": utc_now(),
                "status": "timeout_waiting_for_pids",
                "wait_pids": pids,
                "wait_result": wait_result,
            }
            out = Path(args.final_report)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2))
            print(f"[done] timeout report: {out}")
            return 2

    analyze_path = Path(args.analyze_report)
    analyze_path.parent.mkdir(parents=True, exist_ok=True)
    analyze_report = _run_hybrid_analyze(
        registry_path=Path(args.registry),
        report_path=analyze_path,
        max_candidates=max(8, args.max_candidates),
    )
    selected_1_2 = _selected_path_from_analyze(analyze_report)
    print(f"[analyze] selected smb_1_2 candidate: {selected_1_2}")

    baseline, candidate_chain = _validate_full_chain_with_candidate(
        base_chain_path=Path(args.base_chain),
        candidate_1_2=selected_1_2,
    )

    candidate_out = Path(args.candidate_chain_json)
    candidate_out.parent.mkdir(parents=True, exist_ok=True)
    candidate_out.write_text(json.dumps(candidate_chain, indent=2))
    print(f"[chain] wrote candidate chain eval: {candidate_out}")

    baseline_ok = bool(baseline.get("all_completed", False))
    candidate_ok = bool(candidate_chain.get("all_completed", False))
    baseline_frames = int(baseline.get("total_frames", 10**9))
    candidate_frames = int(candidate_chain.get("total_frames", 10**9))

    if candidate_ok and (not baseline_ok or candidate_frames < baseline_frames):
        best_source = "candidate_1_2"
        best_frames = candidate_frames
        improvement_frames = baseline_frames - candidate_frames if baseline_ok else None
    else:
        best_source = "baseline"
        best_frames = baseline_frames
        improvement_frames = 0 if (baseline_ok and candidate_ok) else None

    final_report = {
        "generated_at": utc_now(),
        "status": "ok",
        "wait_pids": pids,
        "wait_result": wait_result,
        "paths": {
            "base_chain": str(Path(args.base_chain)),
            "registry": str(Path(args.registry)),
            "analyze_report": str(analyze_path),
            "candidate_chain_json": str(candidate_out),
        },
        "selected_smb_1_2_recording": selected_1_2,
        "baseline": {
            "all_completed": baseline_ok,
            "total_frames": baseline_frames,
            "total_seconds": round(baseline_frames / 60.0, 3),
        },
        "candidate_route": {
            "all_completed": candidate_ok,
            "total_frames": candidate_frames,
            "total_seconds": round(candidate_frames / 60.0, 3),
        },
        "best": {
            "source": best_source,
            "total_frames": best_frames,
            "total_seconds": round(best_frames / 60.0, 3),
            "improvement_vs_baseline_frames": improvement_frames,
        },
    }

    final_path = Path(args.final_report)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(final_report, indent=2))
    print(f"[done] final fold report: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
