"""Headless Stage1 segment: chain Slum waves until clear / boss / fail."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from final_fight.paths import GAME, GAME_DIR, RECORDINGS_DIR, STAGE1_STATE
from final_fight.policy import Stage1Policy
from final_fight.ram import parse_game_state
from retro_harness.env import get_available_states, make_env, save_state
from retro_harness.actions import idle_action
from retro_harness.ram_state import GameMode, GameState
from retro_harness.segment_runner import (
    SegmentOutcome,
    WaveChainTracker,
    configure_headless,
    save_rgb_png,
    snapshot_state,
    write_json_report,
)


def _reset(env: Any) -> tuple[Any, dict[str, Any]]:
    """Normalize gymnasium vs classic retro reset return."""
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, {}


def run_stage1_segment(
    *,
    max_frames: int = 12000,
    clear_hold_frames: int = 30,
    target_waves: int | None = None,
    stop_on_boss: bool = True,
    save_wave_states: bool = True,
    state_name: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Load Stage1.state, chain waves, write report + PNGs + mid states."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    chosen = state_name or (
        STAGE1_STATE if STAGE1_STATE in available else "NONE"
    )
    out = out_dir or RECORDINGS_DIR
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, chosen, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    tracker = WaveChainTracker(
        max_frames=max_frames,
        clear_hold_frames=clear_hold_frames,
        target_waves=target_waves,
        stop_on_boss=stop_on_boss,
    )
    screenshots: list[str] = []
    saved_states: list[str] = []
    outcome = SegmentOutcome.TIMEOUT
    obs: Any = None
    final_state: GameState | None = None
    start_snap: dict[str, Any] = {}
    last_cleared = 0
    unlock_saved = False
    healthy_unlock_saved = False
    start_cam = 0
    try:
        obs, _info = _reset(env)
        state = parse_game_state(env.get_ram(), frame=0)
        tracker.begin(state)
        start_snap = snapshot_state(state)
        start_cam = state.camera_x
        final_state = state
        png = save_rgb_png(obs, out / "stage1_0000_start.png")
        screenshots.append(png.name)

        for frame_i in range(1, max_frames + 1):
            tick = policy.tick(state)
            if tick.action is not None:
                tracker.note_reason(tick.action.reason)
                action = tick.action.action
            else:
                tracker.note_reason(tick.reason or "no_action")
                action = idle_action()
            obs, _reward, _term, _trunc, _info = env.step(action)

            state = parse_game_state(env.get_ram(), frame=frame_i)
            final_state = state
            stop = tracker.update(state)

            if tracker.waves_cleared > last_cleared:
                last_cleared = tracker.waves_cleared
                wave = tracker.waves[-1]
                tag = f"wave{wave.index}_clear"
                png = save_rgb_png(
                    obs, out / f"stage1_{frame_i:04d}_{tag}.png"
                )
                screenshots.append(png.name)
                # Prefer healthy resumes; soft low-HP clears are bad reload
                # points and must not overwrite stronger states.
                if (
                    save_wave_states
                    and state.mode is GameMode.PLAYING
                    and 40 <= state.health <= 128
                    and state.lives > 0
                ):
                    name = (
                        f"Stage1_Clear_w{wave.index}"
                        f"_cam{state.camera_x}"
                    )
                    path = save_state(env, GAME_DIR, GAME, name)
                    saved_states.append(path.name)

            # Alley unlock with both lives. Never overwrite L2 with a
            # softer HP than the best seen this run (HP≥40 also upgrades).
            post_unlock = (
                save_wave_states
                and state.mode is GameMode.PLAYING
                and state.camera_x > start_cam
                and state.camera_x > 1536
                and state.lives >= 2
                and 0 < state.health <= 128
            )
            if (
                post_unlock
                and not unlock_saved
                and state.health >= 30
            ):
                path = save_state(
                    env, GAME_DIR, GAME, "Stage1_PostUnlock_L2"
                )
                saved_states.append(path.name)
                unlock_saved = True
                png = save_rgb_png(
                    obs,
                    out / f"stage1_{frame_i:04d}_unlock_L2.png",
                )
                screenshots.append(png.name)
            if (
                post_unlock
                and not healthy_unlock_saved
                and state.health >= 40
            ):
                healthy = save_state(
                    env,
                    GAME_DIR,
                    GAME,
                    "Stage1_PostUnlock_L2_Healthy",
                )
                saved_states.append(healthy.name)
                healthy_unlock_saved = True
                # Refresh L2 to the healthier snapshot when available.
                path = save_state(
                    env, GAME_DIR, GAME, "Stage1_PostUnlock_L2"
                )
                if path.name not in saved_states:
                    saved_states.append(path.name)
                unlock_saved = True

            if stop is not None:
                outcome = stop
                break
        else:
            outcome = SegmentOutcome.TIMEOUT

        assert final_state is not None
        end_tag = "end"
        if outcome is SegmentOutcome.SUCCESS and tracker.boss_reached:
            end_tag = "boss"
        elif outcome is SegmentOutcome.DEATH:
            end_tag = "death"
        png = save_rgb_png(
            obs, out / f"stage1_{tracker.frames:04d}_{end_tag}.png"
        )
        screenshots.append(png.name)

        if (
            save_wave_states
            and tracker.boss_reached
            and outcome is SegmentOutcome.SUCCESS
        ):
            path = save_state(env, GAME_DIR, GAME, "Boss")
            saved_states.append(path.name)
            before = save_state(
                env, GAME_DIR, GAME, "Stage1_BeforeBoss"
            )
            saved_states.append(before.name)

        report = tracker.to_report(
            outcome=outcome,
            final=final_state,
            screenshots=screenshots,
            start_state=chosen,
            saved_states=saved_states,
            extras={
                "start": start_snap,
                "end": snapshot_state(final_state),
            },
        )
        report_path = write_json_report(
            out / "stage1_segment.json", report
        )
        report["report_path"] = str(report_path)
        return report
    finally:
        env.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=12000)
    parser.add_argument("--clear-hold-frames", type=int, default=30)
    parser.add_argument(
        "--target-waves",
        type=int,
        default=None,
        help="Stop successfully after N wave clears (default: until boss)",
    )
    parser.add_argument(
        "--no-stop-on-boss",
        action="store_true",
        help="Keep fighting after boss appears (until death/timeout)",
    )
    parser.add_argument(
        "--no-save-states",
        action="store_true",
        help="Skip writing mid-stage .state files",
    )
    parser.add_argument("--state", default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override recordings directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the Stage1 multi-wave segment runner."""
    args = _build_parser().parse_args(argv)
    report = run_stage1_segment(
        max_frames=args.max_frames,
        clear_hold_frames=args.clear_hold_frames,
        target_waves=args.target_waves,
        stop_on_boss=not args.no_stop_on_boss,
        save_wave_states=not args.no_save_states,
        state_name=args.state,
        out_dir=args.out_dir,
    )
    end = report.get("extras", {}).get("end", {})
    print(
        f"outcome={report['outcome']} frames={report['frames']} "
        f"waves={report['waves_cleared']} kills={report['kills']} "
        f"damage={report['damage_dealt']} "
        f"hp={report['start_health']}->{report['end_health']} "
        f"lives={end.get('lives', '?')} "
        f"cam={report['start_camera_x']}->{report['end_camera_x']} "
        f"boss={report['boss_reached']} "
        f"enemies={report['start_enemy_count']}->"
        f"{report['end_enemy_count']}"
    )
    print(f"report={report.get('report_path')}")
    if report.get("saved_states"):
        print("states: " + ", ".join(report["saved_states"]))
    top = sorted(
        report.get("reason_counts", {}).items(),
        key=lambda kv: -kv[1],
    )[:8]
    if top:
        print("reasons: " + ", ".join(f"{k}={v}" for k, v in top))
    # Partial progress still useful; exit 0 if any wave cleared or boss.
    ok = bool(report.get("success")) or int(
        report.get("waves_cleared", 0)
    ) > 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
