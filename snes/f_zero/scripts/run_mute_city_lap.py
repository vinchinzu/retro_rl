"""Run the Mute City centerline policy until one lap or a crash-out."""

from __future__ import annotations

import argparse
from pathlib import Path

from f_zero.paths import GAME, GAME_DIR, MUTE_CITY_STATE, RECORDINGS_DIR
from f_zero.policy import CenterlinePolicy
from f_zero.ram import LapWatch, parse_game_state
from retro_harness.actions import idle_action
from retro_harness.env import get_available_states, make_env
from retro_harness.segment_runner import (
    SegmentOutcome,
    configure_headless,
    save_rgb_png,
    snapshot_state,
    write_json_report,
)


def run_mute_city_lap(
    *,
    max_frames: int = 4500,
    state_name: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, object]:
    """Load MuteCity.state and drive until a finish-line HUD edge or crash."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    chosen = state_name or MUTE_CITY_STATE
    if chosen not in available:
        raise FileNotFoundError(f"{chosen}.state is missing; run scripts/boot_probe.py")
    out = out_dir or RECORDINGS_DIR / "mute_city_lap"
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(GAME, chosen, GAME_DIR, render_mode="rgb_array")
    policy = CenterlinePolicy()
    watch = LapWatch()
    reasons: dict[str, int] = {}
    outcome = SegmentOutcome.TIMEOUT
    try:
        reset = env.reset()
        obs = reset[0] if isinstance(reset, tuple) else reset
        state = parse_game_state(env.get_ram())
        start = snapshot_state(state)
        screenshots = [save_rgb_png(obs, out / "lap_0000_start.png").name]
        frame = 0
        for frame in range(1, max_frames + 1):
            tick = policy.tick(state)
            frame_action = tick.action
            reasons[tick.reason] = reasons.get(tick.reason, 0) + 1
            obs, *_rest = env.step(
                frame_action.action if frame_action is not None else idle_action()
            )
            state = parse_game_state(env.get_ram(), frame=frame)
            watch.update(int(state.extras["screen_text"]))
            if watch.laps >= 1:
                outcome = SegmentOutcome.SUCCESS
                break
            if state.player_dead:
                outcome = SegmentOutcome.DEATH
                break
        screenshots.append(
            save_rgb_png(obs, out / f"lap_{frame:04d}_end.png").name
        )
        report: dict[str, object] = {
            "outcome": outcome.name.lower(),
            "success": outcome is SegmentOutcome.SUCCESS,
            "frames": frame,
            "start_state": chosen,
            "laps": watch.laps,
            "crashed": state.player_dead,
            "power": int(state.extras["power"]),
            "screen_text": int(state.extras["screen_text"]),
            "finish_state": int(state.extras["finish_state"]),
            "heading": int(state.extras["heading"]),
            "checkpoint_facing": int(state.extras["checkpoint_facing"]),
            "reason_counts": dict(sorted(reasons.items())),
            "screenshots": screenshots,
            "extras": {"start": start, "end": snapshot_state(state)},
        }
        report_path = write_json_report(out / "mute_city_lap.json", report)
        report["report_path"] = str(report_path)
        return report
    finally:
        env.close()


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=4500)
    parser.add_argument("--state", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--attempts", type=int, default=1)
    args = parser.parse_args()
    successes = 0
    last: dict[str, object] | None = None
    for attempt in range(1, args.attempts + 1):
        last = run_mute_city_lap(
            max_frames=args.max_frames,
            state_name=args.state,
            out_dir=args.out_dir,
        )
        ok = bool(last["success"])
        successes += int(ok)
        print(
            f"attempt={attempt} outcome={last['outcome']} frames={last['frames']} "
            f"laps={last['laps']} power={last['power']} "
            f"crashed={last['crashed']}"
        )
        print(f"report={last['report_path']}")
        if not ok:
            break
    print(f"record={successes}/{args.attempts}")
    return 0 if successes == args.attempts else 1


if __name__ == "__main__":
    raise SystemExit(main())
