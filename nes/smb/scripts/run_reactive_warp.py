"""Run the natural-entry reactive eight-exit candidate without state splices.

This is a development runner, not a published benchmark: it starts at the
``Level1_1`` practice state, then uses the stairs-improved 1-1 policy and the
state-gated 1-2 controller before continuing from World 4 in the *same*
environment.  It exists to capture the real shifted predecessor signatures
needed to re-solve later legs; it must never be used to quietly pad back to
the M8 seed phase.

Default behaviour (``--retime-4-1`` / ``--retime-4-2``): idle through the
W4 pipe transition until natural 4-1 control, resume the continuation body,
then at 4-1 exit auto (flag/score/load) **freeze the source** and idle until
natural 4-2 control before resuming the 4-2 body. That absorbs earlier
savings and variable castle-tally length without W4/W4-2 idle-pad.

With ``--retime-8-2``, snap the continuation at natural 8-2 control, play
``KNOWN_82_LEAD_IDLES`` then the control-relative 8-2 body, and let
control-relative 8-3/8-4 repairs take over at their natural entry gates.
``--drop-at`` supports small timing experiments in memory only.
``--write-seed`` materializes a successful controller for a separate Clean
power-on verification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.env import make_env, read_state_bytes
from smb.paths import GAME_DIR, GAME_V0, INTEGRATION_V0_DIR, RECORDINGS_DIR
from smb.policy import (
    CONTINUOUS_SETTLE_FRAMES,
    DEFAULT_STAIRS_1_1 as STAIRS_1_1,
    compress_nes9_rle,
    expand_nes9_rle,
    load_nes9_rle_seed,
    play_1_1_until_clear as _play_1_1_until_clear,
)
from smb.ram import read_snapshot, reached_ending
from smb.reactive_12 import Reactive12Policy, play_reactive_12
from smb.reactive_late import LateRouteController
from smb.reactive_route import (
    DEFAULT_CONTINUATION,
    DEFAULT_CONTINUATION_START,
    KNOWN_41_CONTROL_RESUME,
    KNOWN_42_CONTROL_RESUME,
    KNOWN_82_CONTROL_RESUME,
    KNOWN_82_LEAD_IDLES,
    KNOWN_82_RETIME_COUNT,
    KNOWN_82_RETIME_START,
    RouteProgressTracker,
    continuation_frames as _continuation_frames,
    in_4_1_exit_auto as _in_4_1_exit_auto,
    level_control_gate,
    snapshot_fingerprint,
)
from smb.routes import ROUTE_WARP_ANY_PERCENT
from retro_harness.segment_runner import configure_headless, write_json_report

LEVEL1_1_STATE = INTEGRATION_V0_DIR / "Level1_1.state"
DEFAULT_MAX_FRAMES = 25_000


def run_reactive_warp_candidate(
    *,
    continuation_seed: Path = DEFAULT_CONTINUATION,
    continuation_start: int = DEFAULT_CONTINUATION_START,
    drop_at: int | None = None,
    drop_count: int = 0,
    retime_4_1: bool = True,
    retime_4_2: bool = True,
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
    if retime_8_2 and drop_at is not None:
        raise ValueError("--retime-8-2 cannot be combined with --drop-at")
    continuation = _continuation_frames(
        continuation_seed,
        start=continuation_start,
        drop_at=drop_at,
        drop_count=drop_count,
    )
    if retime_4_1 and KNOWN_41_CONTROL_RESUME >= len(continuation):
        raise ValueError(
            f"4-1 resume index {KNOWN_41_CONTROL_RESUME} outside "
            f"{len(continuation)} continuation frames"
        )
    if retime_4_2 and KNOWN_42_CONTROL_RESUME >= len(continuation):
        raise ValueError(
            f"4-2 resume index {KNOWN_42_CONTROL_RESUME} outside "
            f"{len(continuation)} continuation frames"
        )
    if retime_8_2 and KNOWN_82_CONTROL_RESUME >= len(continuation):
        raise ValueError(
            f"8-2 resume index {KNOWN_82_CONTROL_RESUME} outside "
            f"{len(continuation)} continuation frames"
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
            "retime_4_1": retime_4_1,
            "retime_4_2": retime_4_2,
            "retime_8_2": retime_8_2,
            "control_resume_index": KNOWN_41_CONTROL_RESUME if retime_4_1 else None,
            "control_resume_index_4_2": (
                KNOWN_42_CONTROL_RESUME if retime_4_2 else None
            ),
            "control_resume_index_8_2": (
                KNOWN_82_CONTROL_RESUME if retime_8_2 else None
            ),
            "lead_idles_8_2": KNOWN_82_LEAD_IDLES if retime_8_2 else 0,
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
        idle_buttons = [0] * 9
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
        gate_4_1 = level_control_gate(ROUTE_WARP_ANY_PERCENT.exits[2])
        gate_4_2 = level_control_gate(ROUTE_WARP_ANY_PERCENT.exits[3])
        gate_8_2 = level_control_gate(ROUTE_WARP_ANY_PERCENT.exits[5])
        outcome = "seed_exhausted"
        tail_frame = 0
        source_frames = 0
        aligning_4_1 = retime_4_1
        aligning_4_2 = False
        snapped_4_2 = not retime_4_2
        saw_4_1_control = not retime_4_1
        snapped_8_2 = not retime_8_2
        lead_8_2_remaining = 0
        align_4_1_meta: dict[str, Any] | None = None
        align_4_2_meta: dict[str, Any] | None = None
        align_8_2_meta: dict[str, Any] | None = None
        late_controller: LateRouteController | None = None
        last = read_snapshot(env.get_ram())
        for tail_frame in range(1, max_frames + 1):
            if aligning_4_1:
                if gate_4_1.matches(last):
                    aligning_4_1 = False
                    saw_4_1_control = True
                    source_frames = KNOWN_41_CONTROL_RESUME
                    align_4_1_meta = {
                        "tail_frame": tail_frame,
                        "resume_index": KNOWN_41_CONTROL_RESUME,
                        "entry": snapshot_fingerprint(last),
                    }
                    raw = continuation[source_frames]
                    source_frames += 1
                else:
                    raw = idle_buttons
            elif retime_4_2 and not snapped_4_2 and saw_4_1_control and (
                aligning_4_2 or gate_4_2.matches(last)
            ):
                # Idle through flag/score/load; snap body at natural 4-2 control.
                if gate_4_2.matches(last):
                    aligning_4_2 = False
                    snapped_4_2 = True
                    source_before = source_frames
                    source_frames = KNOWN_42_CONTROL_RESUME
                    align_4_2_meta = {
                        "tail_frame": tail_frame,
                        "resume_index": KNOWN_42_CONTROL_RESUME,
                        "entry": snapshot_fingerprint(last),
                        "source_before_snap": source_before,
                    }
                    raw = continuation[source_frames]
                    source_frames += 1
                else:
                    aligning_4_2 = True
                    raw = idle_buttons
            elif (
                retime_8_2
                and not snapped_8_2
                and late_controller is None
                and gate_8_2.matches(last)
            ):
                # Snap body at natural 8-2 control; optional lead idles then resume.
                snapped_8_2 = True
                source_before = source_frames
                source_frames = KNOWN_82_CONTROL_RESUME
                lead_8_2_remaining = KNOWN_82_LEAD_IDLES
                align_8_2_meta = {
                    "tail_frame": tail_frame,
                    "resume_index": KNOWN_82_CONTROL_RESUME,
                    "lead_idles": KNOWN_82_LEAD_IDLES,
                    "entry": snapshot_fingerprint(last),
                    "source_before_snap": source_before,
                }
                if lead_8_2_remaining > 0:
                    raw = idle_buttons
                    lead_8_2_remaining -= 1
                else:
                    raw = continuation[source_frames]
                    source_frames += 1
            elif late_controller is None and lead_8_2_remaining > 0:
                raw = idle_buttons
                lead_8_2_remaining -= 1
            elif (
                late_controller is None
                and use_late_controllers
                and progress.next_exit is not None
                and progress.next_exit.exit_id == "8-3"
                and level_control_gate(progress.next_exit).matches(last)
            ):
                # Pre-action takeover at natural 8-3 control (no one-frame leak
                # of the absolute continuation into the late body).
                late_controller = LateRouteController()
                late_controller.begin(last)
                raw = late_controller.next_frame()
            elif late_controller is None:
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
            # After 4-1 control: when exit auto begins, freeze source and idle
            # to natural 4-2 control (handles variable castle-tally length).
            if (
                retime_4_2
                and not snapped_4_2
                and not aligning_4_1
                and not aligning_4_2
                and saw_4_1_control
                and _in_4_1_exit_auto(last)
            ):
                aligning_4_2 = True
            try:
                if late_controller is not None:
                    late_controller.observe(last)
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
            "align_4_1": align_4_1_meta,
            "align_4_2": align_4_2_meta,
            "align_8_2": align_8_2_meta,
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
            source_bits = ["stairs 1-1", "reactive 1-2"]
            if retime_4_1:
                source_bits.append("control-relative 4-1 retime")
            if retime_4_2:
                source_bits.append("control-relative 4-2 retime")
            if retime_8_2:
                source_bits.append(
                    f"control-relative 8-2 retime (+{KNOWN_82_LEAD_IDLES} lead)"
                )
            if use_late_controllers:
                source_bits.append("control-relative 8-3/8-4 repairs")
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
                        "source": " + ".join(source_bits),
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
        "--retime-4-1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="idle to natural 4-1 control then resume body (default on)",
    )
    parser.add_argument(
        "--retime-4-2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "idle through 4-1 flag/score/load to natural 4-2 control then "
            "resume body (default on; no pad)"
        ),
    )
    parser.add_argument(
        "--retime-8-2",
        action="store_true",
        help=(
            "snap at natural 8-2 control, play "
            f"+{KNOWN_82_LEAD_IDLES} lead idle then body at cont "
            f"{KNOWN_82_CONTROL_RESUME} (post 1-2 −97f path)"
        ),
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
        retime_4_1=args.retime_4_1,
        retime_4_2=args.retime_4_2,
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
