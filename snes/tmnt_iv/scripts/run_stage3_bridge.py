"""Bridge Stage 2 clear → fight-ready Stage 3 (Prehistoric)."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from retro_harness.env import get_available_states, make_env, reset_obs, save_state
from retro_harness.actions import buttons, idle_action
from retro_harness.ram_state import GameMode, GameState
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    snapshot_state,
    write_json_report,
)
from tmnt_iv.paths import (
    GAME,
    GAME_DIR,
    RECORDINGS_DIR,
    STAGE2_BEFORE_BOSS_STATE,
    STAGE2_CLEAR_STATE,
    STAGE3_STATE,
)
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.ram import parse_game_state


def _with_stage_progress(
    state: GameState, *, start_stage: int
) -> GameState:
    """Mark prior stage complete when ``ADDR_STAGE`` advances."""
    if state.stage > start_stage:
        return replace(state, level_complete=True)
    return state

def _is_fight_ready(state: GameState, *, min_stage: int) -> bool:
    """True once Stage 3 HUD is live with at least one enemy."""
    return (
        state.mode is GameMode.PLAYING
        and state.stage >= min_stage
        and 0 < state.health <= 0xC0
        and 0 < state.player_x < 512
        and len(state.living_enemies) > 0
    )

def run_stage3_bridge(
    *,
    state_name: str | None = None,
    max_frames: int = 12000,
    idle_after_clear: int = 240,
    start_period: int = 40,
    save_stage3: bool = True,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Clear/wait through Stage 2→3 transition; save fight-ready Stage3."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    preferred = (
        STAGE3_STATE,
        STAGE2_CLEAR_STATE,
        "Stage2_Clear_post",
        STAGE2_BEFORE_BOSS_STATE,
    )
    chosen = state_name or next(
        (name for name in preferred if name in available),
        "NONE",
    )
    out = out_dir or (RECORDINGS_DIR / "stage3_bridge")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, chosen, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    screenshots: list[str] = []
    saved_states: list[str] = []
    try:
        obs, _info = reset_obs(env)
        state = parse_game_state(env.get_ram(), frame=0)
        start_stage = state.stage
        # Stage byte 2 = Prehistoric / Stage 3. Confirm ready if already.
        target_stage = 2 if start_stage < 2 else start_stage
        start_snap = snapshot_state(state)
        png = save_rgb_png(obs, out / "bridge_0000_start.png")
        screenshots.append(png.name)

        cleared_at: int | None = 0 if state.stage >= 2 else None
        ready_at: int | None = None
        final_state = state

        for frame_i in range(0, max_frames + 1):
            if frame_i > 0:
                state = parse_game_state(env.get_ram(), frame=frame_i)
            state = _with_stage_progress(state, start_stage=start_stage)
            final_state = state

            if cleared_at is None and state.stage > start_stage:
                cleared_at = frame_i
                png = save_rgb_png(
                    obs, out / f"bridge_{frame_i:04d}_stage_adv.png"
                )
                screenshots.append(png.name)
                if save_stage3:
                    path = save_state(
                        env, GAME_DIR, GAME, STAGE2_CLEAR_STATE
                    )
                    saved_states.append(path.name)

            if _is_fight_ready(state, min_stage=target_stage):
                ready_at = frame_i
                png = save_rgb_png(
                    obs, out / f"bridge_{frame_i:04d}_stage3.png"
                )
                screenshots.append(png.name)
                if save_stage3:
                    path = save_state(env, GAME_DIR, GAME, STAGE3_STATE)
                    saved_states.append(path.name)
                break

            # Pre-clear: fight Metalhead (or idle through mid-transition).
            if cleared_at is None:
                if state.mode is GameMode.CUTSCENE:
                    act = idle_action()
                else:
                    tick = policy.tick(state)
                    act = (
                        tick.action.action
                        if tick.action is not None
                        else idle_action()
                    )
            else:
                dt = frame_i - cleared_at
                if dt < idle_after_clear:
                    act = idle_action()
                elif state.mode is GameMode.CUTSCENE:
                    act = (
                        buttons("START")
                        if dt % start_period == 0
                        else idle_action()
                    )
                elif state.living_enemies:
                    tick = policy.tick(state)
                    act = (
                        tick.action.action
                        if tick.action is not None
                        else idle_action()
                    )
                else:
                    act = buttons("RIGHT")

            if frame_i >= max_frames:
                break
            obs, _reward, _term, _trunc, _info = env.step(act)

        report = {
            "success": ready_at is not None,
            "start_state": chosen,
            "frames": ready_at if ready_at is not None else max_frames,
            "cleared_at": cleared_at,
            "ready_at": ready_at,
            "start_stage": start_stage,
            "end_stage": final_state.stage,
            "screenshots": screenshots,
            "saved_states": saved_states,
            "extras": {
                "start": start_snap,
                "end": snapshot_state(final_state),
            },
        }
        report_path = write_json_report(out / "stage3_bridge.json", report)
        report["report_path"] = str(report_path)
        return report
    finally:
        env.close()

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=None)
    parser.add_argument("--max-frames", type=int, default=12000)
    parser.add_argument("--idle-after-clear", type=int, default=240)
    parser.add_argument("--start-period", type=int, default=40)
    parser.add_argument(
        "--no-save-states",
        action="store_true",
        help="Skip writing Stage2_Clear / Stage3 saves",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser

def main(argv: list[str] | None = None) -> int:
    """CLI entry for the Stage 2→3 bridge."""
    args = _build_parser().parse_args(argv)
    report = run_stage3_bridge(
        state_name=args.state,
        max_frames=args.max_frames,
        idle_after_clear=args.idle_after_clear,
        start_period=args.start_period,
        save_stage3=not args.no_save_states,
        out_dir=args.out_dir,
    )
    end = report.get("extras", {}).get("end", {})
    print(
        f"success={report['success']} frames={report['frames']} "
        f"cleared_at={report['cleared_at']} ready_at={report['ready_at']} "
        f"stage={report['start_stage']}->{report['end_stage']} "
        f"hp={end.get('health')} lives={end.get('lives')} "
        f"enemies={end.get('enemies')} mode={end.get('mode')}"
    )
    print(f"report={report.get('report_path')}")
    if report.get("saved_states"):
        print("states: " + ", ".join(report["saved_states"]))
    return 0 if report.get("success") else 1

if __name__ == "__main__":
    raise SystemExit(main())
