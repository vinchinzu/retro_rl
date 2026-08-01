"""Run the natural-entry reactive eight-exit candidate without state splices.

This is a development runner, not a published benchmark: it starts at the
``Level1_1`` practice state, then uses the stairs-improved 1-1 policy and the
state-gated 1-2 controller before continuing from World 4 in the *same*
environment.  It exists to capture the real shifted predecessor signatures
needed to re-solve 8-2/8-3; it must never be used to quietly pad back to the
M8 seed phase.

The default continuation reproduces the known 63-frame-faster candidate,
which currently dies in 8-2.  With ``--retime-8-2``, control-relative 8-3 and
8-4 repairs take over at their natural entry gates. ``--drop-at`` supports
small timing experiments in memory only. ``--write-seed`` materializes a
successful controller for a separate Clean power-on verification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.env import make_env
from smb.full_run import read_state_bytes
from smb.paths import GAME_DIR, GAME_V0, INTEGRATION_V0_DIR, RECORDINGS_DIR
from smb.policy import (
    CONTINUOUS_SETTLE_FRAMES,
    compress_nes9_rle,
    expand_nes9_rle,
    load_nes9_rle_seed,
)
from smb.ram import read_snapshot, reached_ending
from smb.reactive_12 import Reactive12Policy, play_reactive_12
from smb.reactive_late import LateRouteController
from smb.reactive_route import RouteProgressTracker
from smb.reactive_route import level_control_gate
from smb.routes import ROUTE_WARP_ANY_PERCENT
from smb.scripts.run_1_2 import STAIRS_1_1, _play_1_1_until_clear
from snes_oneshot.segment_runner import configure_headless, write_json_report

LEVEL1_1_STATE = INTEGRATION_V0_DIR / "Level1_1.state"
DEFAULT_CONTINUATION = GAME_DIR / "models" / "smb_1_1_to_ending_stairs_try.json"
# The continuation starts immediately after the reactive 1-2 World-4 split.
DEFAULT_CONTINUATION_START = 3_981
DEFAULT_MAX_FRAMES = 25_000
# Verified from the shifted natural 8-2 control fingerprint. It advances the
# 8-2 split to 15,863f; the late controllers begin only after natural control.
KNOWN_82_RETIME_START = 12_898
KNOWN_82_RETIME_COUNT = 5


def _continuation_frames(
    seed_path: Path,
    *,
    start: int,
    drop_at: int | None,
    drop_count: int,
) -> list[list[int]]:
    frames = expand_nes9_rle(load_nes9_rle_seed(seed_path))
    if not 0 <= start < len(frames):
        raise ValueError(f"continuation start {start} outside {len(frames)} frames")
    if drop_at is not None:
        if drop_at < start:
            raise ValueError("drop-at must be inside the continuation")
        if drop_count <= 0:
            raise ValueError("drop-count must be positive with --drop-at")
        end = drop_at + drop_count
        if end > len(frames):
            raise ValueError(f"drop range [{drop_at}, {end}) outside {len(frames)}")
        del frames[drop_at:end]
    return frames[start:]


def run_reactive_warp_candidate(
    *,
    continuation_seed: Path = DEFAULT_CONTINUATION,
    continuation_start: int = DEFAULT_CONTINUATION_START,
    drop_at: int | None = None,
    drop_count: int = 0,
    retime_8_2: bool = False,
    use_late_controllers: bool = True,
    write_seed: Path | None = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    out_dir: Path | None = None,
    tag: str = "reactive_warp_candidate",
) -> dict[str, Any]:
    """Run the real predecessor through a mutable in-memory continuation."""
    configure_headless()
    if not LEVEL1_1_STATE.exists():
        raise FileNotFoundError(f"missing Level1_1 state: {LEVEL1_1_STATE}")
    if not STAIRS_1_1.exists():
        raise FileNotFoundError(f"missing stairs seed: {STAIRS_1_1}")

    out = out_dir or (RECORDINGS_DIR / "reactive_warp")
    out.mkdir(parents=True, exist_ok=True)
    if retime_8_2:
        if drop_at is not None:
            raise ValueError("--retime-8-2 cannot be combined with --drop-at")
        drop_at = KNOWN_82_RETIME_START
        drop_count = KNOWN_82_RETIME_COUNT
    continuation = _continuation_frames(
        continuation_seed,
        start=continuation_start,
        drop_at=drop_at,
        drop_count=drop_count,
    )
    report: dict[str, Any] = {
        "mode": "reactive_candidate",
        "benchmark_eligible": False,
        "intervention": {
            "class": "development natural predecessor",
            "initial_state": str(LEVEL1_1_STATE.relative_to(GAME_DIR.parent)),
            "mid_attempt_state_loads": 0,
            "note": "stairs 1-1 + state-gated 1-2 then same-env continuation",
        },
        "continuation": {
            "seed": str(continuation_seed),
            "start": continuation_start,
            "input_frames": len(continuation),
            "drop_at": drop_at,
            "drop_count": drop_count if drop_at is not None else 0,
            "late_controllers": use_late_controllers,
        },
        "success": False,
        "stages": {},
    }
    env = make_env(GAME_V0, "NONE", GAME_DIR, render_mode="rgb_array")
    try:
        env.reset()
        env.em.set_state(read_state_bytes(LEVEL1_1_STATE))
        idle = np.zeros(int(env.action_space.shape[0]), dtype=np.int8)
        for _ in range(CONTINUOUS_SETTLE_FRAMES):
            env.step(idle)

        stairs = expand_nes9_rle(load_nes9_rle_seed(STAIRS_1_1))
        stage_11 = _play_1_1_until_clear(env, stairs)
        report["stages"]["1-1"] = {
            key: stage_11[key] for key in ("success", "outcome", "frames")
        }
        if not stage_11["success"]:
            report["outcome"] = f"1-1_{stage_11['outcome']}"
            return report

        stage_12 = play_reactive_12(
            env,
            policy=Reactive12Policy(action_size=int(env.action_space.shape[0])),
        )
        report["stages"]["1-2"] = {
            key: value
            for key, value in stage_12.items()
            if key not in ("recorded", "last_obs")
        }
        if not stage_12["success"]:
            report["outcome"] = f"1-2_{stage_12['outcome']}"
            return report

        recorded_policy = [
            [int(button) for button in frame[:9]]
            for frame in (*stage_11["recorded"], *stage_12["recorded"])
        ]

        start_lives = read_snapshot(env.get_ram()).lives
        progress = RouteProgressTracker(
            ROUTE_WARP_ANY_PERCENT,
            start_lives=start_lives,
            start_index=2,
        )
        outcome = "seed_exhausted"
        tail_frame = 0
        source_frames = 0
        late_controller: LateRouteController | None = None
        last = read_snapshot(env.get_ram())
        for tail_frame in range(1, max_frames + 1):
            if late_controller is None:
                if source_frames >= len(continuation):
                    outcome = "seed_exhausted"
                    break
                raw = continuation[source_frames]
                source_frames += 1
            else:
                if late_controller.exhausted:
                    outcome = "late_controller_exhausted"
                    break
                raw = late_controller.next_frame()
            action = list(raw[: int(env.action_space.shape[0])])
            if len(action) < int(env.action_space.shape[0]):
                action.extend([0] * (int(env.action_space.shape[0]) - len(action)))
            env.step(np.asarray(action, dtype=np.int8))
            recorded_policy.append([int(button) for button in action[:9]])
            last = read_snapshot(env.get_ram(), frame=tail_frame)
            progress.observe(last, frame=tail_frame)
            if last.lives < start_lives or last.dying:
                outcome = "death"
                break
            if reached_ending(env.get_ram(), start_lives=start_lives):
                outcome = "ending"
                break
            try:
                if late_controller is not None:
                    late_controller.observe(last)
                elif (
                    use_late_controllers
                    and progress.next_exit is not None
                    and progress.next_exit.exit_id == "8-3"
                    and level_control_gate(progress.next_exit).matches(last)
                ):
                    late_controller = LateRouteController()
                    late_controller.begin(last)
            except RuntimeError as exc:
                outcome = "late_controller_handoff_failed"
                report["late_controller_error"] = str(exc)
                break
        else:
            outcome = "timeout"

        prefix_frames = int(stage_11["frames"]) + int(stage_12["frames"])
        tail_splits = [dict(row, frame=prefix_frames + int(row["frame"])) for row in progress.completed]
        report["stages"]["continuation"] = {
            "outcome": outcome,
            "tail_frames": tail_frame if continuation else 0,
            "source_frames": source_frames,
            "late_controller": late_controller.report() if late_controller else None,
            "final": {
                "world": last.world,
                "level": last.level,
                "player_x": last.player_x,
                "player_y": last.player_y,
                "lives": last.lives,
                "player_state": last.player_state,
                "oper_mode": last.oper_mode,
            },
            "route_progress": progress.report(),
        }
        report["milestones"] = [
            {"exit_id": "1-1", "frame": int(stage_11["frames"])},
            {"exit_id": "1-2", "frame": prefix_frames},
            *tail_splits,
        ]
        report["success"] = outcome == "ending" and progress.complete
        report["outcome"] = outcome
        report["policy_frames"] = len(recorded_policy)
        if write_seed is not None and report["success"]:
            seed_path = write_seed.resolve()
            if GAME_DIR not in seed_path.parents:
                raise ValueError("reactive candidate seed must live under smb/")
            seed_path.parent.mkdir(parents=True, exist_ok=True)
            source_report = out / f"{tag}_report.json"
            try:
                source_report_label = str(source_report.relative_to(GAME_DIR))
            except ValueError:
                source_report_label = str(source_report)
            seed_path.write_text(
                json.dumps(
                    {
                        "format": "nes9_rle",
                        "route_id": ROUTE_WARP_ANY_PERCENT.route_id,
                        "level_id": "smb_1_1_to_ending_reactive_83_84",
                        "start_state": "Level1_1",
                        "settle_frames": CONTINUOUS_SETTLE_FRAMES,
                        "game_name": GAME_V0,
                        "num_frames": len(recorded_policy),
                        "verified_completed": False,
                        "target": "world_8_4_ending",
                        "source": (
                            "stairs 1-1 + reactive 1-2 + drop-5 8-2 retime + "
                            "control-relative 8-3/8-4 repairs"
                        ),
                        "source_report": source_report_label,
                        "segments": compress_nes9_rle(recorded_policy),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            report["written_seed"] = str(seed_path)
        return report
    finally:
        write_json_report(out / f"{tag}_report.json", report)
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-seed", type=Path, default=DEFAULT_CONTINUATION)
    parser.add_argument("--continuation-start", type=int, default=DEFAULT_CONTINUATION_START)
    parser.add_argument("--drop-at", type=int)
    parser.add_argument("--drop-count", type=int, default=0)
    parser.add_argument(
        "--retime-8-2",
        action="store_true",
        help="apply the verified 12,898:12,903 8-2 control retime",
    )
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument(
        "--no-late-controllers",
        action="store_false",
        dest="use_late_controllers",
        help="diagnostic: leave the historical absolute 8-3/8-4 tail in place",
    )
    parser.add_argument(
        "--write-seed",
        type=Path,
        help="write the successful full controller as an unpromoted nes9_rle seed",
    )
    parser.add_argument("--tag", default="reactive_warp_candidate")
    args = parser.parse_args()
    report = run_reactive_warp_candidate(
        continuation_seed=args.continuation_seed,
        continuation_start=args.continuation_start,
        drop_at=args.drop_at,
        drop_count=args.drop_count,
        retime_8_2=args.retime_8_2,
        use_late_controllers=args.use_late_controllers,
        write_seed=args.write_seed,
        max_frames=args.max_frames,
        tag=args.tag,
    )
    print(
        f"reactive candidate outcome={report.get('outcome')} "
        f"success={report['success']} frames={report.get('policy_frames')} "
        f"milestones={[(row['exit_id'], row['frame']) for row in report.get('milestones', [])]}"
    )


if __name__ == "__main__":
    main()
