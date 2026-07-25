"""Run the aggressive Mission 1 policy until the first combat lock clears."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import get_available_states, make_env, save_state
from snes_oneshot.actions import idle_action
from snes_oneshot.segment_runner import (
    SegmentOutcome,
    SegmentTracker,
    configure_headless,
    save_rgb_png,
    snapshot_state,
    write_json_report,
)
from super_double_dragon.paths import (
    GAME,
    GAME_DIR,
    RECORDINGS_DIR,
    STAGE1_FIRST_CLEAR_STATE,
    STAGE1_STATE,
)
from super_double_dragon.policy import Stage1Policy
from super_double_dragon.ram import parse_game_state


def run_stage1_segment(
    *,
    max_frames: int = 9000,
    clear_hold_frames: int = 45,
    state_name: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, object]:
    """Load a development state and save the first sustained combat clear."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    chosen = state_name or (
        STAGE1_STATE if STAGE1_STATE in available else "NONE"
    )
    out = out_dir or RECORDINGS_DIR / "stage1_segment"
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, chosen, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    tracker = SegmentTracker(
        max_frames=max_frames,
        clear_hold_frames=clear_hold_frames,
        camera_unlock_delta=999,
    )
    screenshots: list[str] = []
    outcome = SegmentOutcome.TIMEOUT
    try:
        reset = env.reset()
        obs = reset[0] if isinstance(reset, tuple) else reset
        state = parse_game_state(env.get_ram())
        tracker.begin(state)
        start = snapshot_state(state)
        screenshots.append(
            save_rgb_png(obs, out / "stage1_0000_start.png").name
        )
        for frame in range(1, max_frames + 1):
            tick = policy.tick(state)
            action = tick.action or idle_action()
            if tick.action is not None:
                tracker.note_reason(tick.action.reason)
                action = tick.action.action
            obs, _reward, _term, _trunc, _info = env.step(action)
            state = parse_game_state(env.get_ram(), frame=frame)
            stop = tracker.update(state)
            if stop is not None:
                outcome = stop
                break
        screenshots.append(
            save_rgb_png(obs, out / f"stage1_{tracker.frames:04d}_end.png").name
        )
        saved_states: list[str] = []
        if outcome is SegmentOutcome.SUCCESS:
            path = save_state(env, GAME_DIR, GAME, STAGE1_FIRST_CLEAR_STATE)
            saved_states.append(path.name)
        report = tracker.to_report(
            outcome=outcome,
            final=state,
            screenshots=screenshots,
            start_state=chosen,
            extras={"start": start, "end": snapshot_state(state)},
        )
        report["saved_states"] = saved_states
        report_path = write_json_report(out / "stage1_segment.json", report)
        report["report_path"] = str(report_path)
        return report
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=9000)
    parser.add_argument("--clear-hold-frames", type=int, default=45)
    parser.add_argument("--state", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    report = run_stage1_segment(
        max_frames=args.max_frames,
        clear_hold_frames=args.clear_hold_frames,
        state_name=args.state,
        out_dir=args.out_dir,
    )
    print(
        f"outcome={report['outcome']} frames={report['frames']} "
        f"hp={report['start_health']}->{report['end_health']} "
        f"enemies={report['start_enemy_count']}->{report['end_enemy_count']}"
    )
    print(f"report={report['report_path']}")
    if report.get("saved_states"):
        print("states: " + ", ".join(report["saved_states"]))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
