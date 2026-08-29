"""Stage N−1 clear → fight-ready Stage N.

One loop, two specs (Alleycat and Sewer). Do not copy this file for later
stages — add a ``BridgeSpec``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retro_harness.actions import buttons, idle_action
from retro_harness.env import get_available_states, make_env, reset_obs, save_state
from retro_harness.ram_state import GameMode, GameState
from retro_harness.segment_runner import (
    configure_headless,
    snapshot_state,
    write_json_report,
)
from tmnt_iv.observe import policy_input
from tmnt_iv.paths import (
    GAME,
    GAME_DIR,
    RECORDINGS_DIR,
    STAGE1_BEFORE_BOSS_STATE,
    STAGE1_CLEAR_STATE,
    STAGE2_BEFORE_BOSS_STATE,
    STAGE2_CLEAR_STATE,
    STAGE2_STATE,
    STAGE3_STATE,
)
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.ram import parse_game_state
from tmnt_iv.run.segment import maybe_save_png, with_stage_progress


@dataclass(frozen=True)
class BridgeSpec:
    """Names and RAM byte for a clear → fight-ready hop."""

    dest: int
    target_byte: int
    preferred_states: tuple[str, ...]
    clear_save: str
    ready_save: str
    evidence_dir: str
    report_name: str
    ready_png_tag: str


BRIDGE_SPECS: dict[int, BridgeSpec] = {
    2: BridgeSpec(
        dest=2,
        target_byte=1,
        preferred_states=(
            STAGE2_STATE,
            STAGE1_CLEAR_STATE,
            STAGE1_BEFORE_BOSS_STATE,
            "Stage1_Clear_fresh",
        ),
        clear_save=STAGE1_CLEAR_STATE,
        ready_save=STAGE2_STATE,
        evidence_dir="stage2_bridge",
        report_name="stage2_bridge.json",
        ready_png_tag="stage2",
    ),
    3: BridgeSpec(
        dest=3,
        target_byte=2,
        preferred_states=(
            STAGE3_STATE,
            STAGE2_CLEAR_STATE,
            "Stage2_Clear_post",
            STAGE2_BEFORE_BOSS_STATE,
        ),
        clear_save=STAGE2_CLEAR_STATE,
        ready_save=STAGE3_STATE,
        evidence_dir="stage3_bridge",
        report_name="stage3_bridge.json",
        ready_png_tag="stage3",
    ),
}


def is_fight_ready(state: GameState, *, min_stage: int) -> bool:
    """True once the destination HUD is live with at least one enemy."""
    return (
        state.mode is GameMode.PLAYING
        and state.stage >= min_stage
        and 0 < state.health <= 0xC0
        and 0 < state.player_x < 512
        and len(state.living_enemies) > 0
    )


def run_bridge(
    spec: BridgeSpec,
    *,
    state_name: str | None = None,
    max_frames: int = 12000,
    idle_after_clear: int = 240,
    start_period: int = 40,
    save_states: bool = True,
    screenshots: bool = False,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Clear/wait through the stage transition; optionally save fight-ready."""
    configure_headless()
    available = get_available_states(GAME, GAME_DIR)
    chosen = state_name or next(
        (name for name in spec.preferred_states if name in available),
        "NONE",
    )
    out = out_dir or (RECORDINGS_DIR / spec.evidence_dir)
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, chosen, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    shots: list[str] = []
    saved_states: list[str] = []
    try:
        obs, _info = reset_obs(env)
        state = parse_game_state(env.get_ram(), frame=0)
        start_stage = state.stage
        target_stage = (
            spec.target_byte if start_stage < spec.target_byte else start_stage
        )
        start_snap = snapshot_state(state)
        maybe_save_png(
            obs, out / "bridge_0000_start.png", enabled=screenshots, bag=shots
        )

        cleared_at: int | None = 0 if state.stage >= spec.target_byte else None
        ready_at: int | None = None
        final_state = state

        for frame_i in range(0, max_frames + 1):
            if frame_i > 0:
                state = parse_game_state(env.get_ram(), frame=frame_i)
            state = with_stage_progress(state, start_stage=start_stage)
            final_state = state

            if cleared_at is None and state.stage > start_stage:
                cleared_at = frame_i
                maybe_save_png(
                    obs,
                    out / f"bridge_{frame_i:04d}_stage_adv.png",
                    enabled=screenshots,
                    bag=shots,
                )
                if save_states:
                    path = save_state(env, GAME_DIR, GAME, spec.clear_save)
                    saved_states.append(path.name)

            if is_fight_ready(state, min_stage=target_stage):
                ready_at = frame_i
                maybe_save_png(
                    obs,
                    out / f"bridge_{frame_i:04d}_{spec.ready_png_tag}.png",
                    enabled=screenshots,
                    bag=shots,
                )
                if save_states:
                    path = save_state(env, GAME_DIR, GAME, spec.ready_save)
                    saved_states.append(path.name)
                break

            if cleared_at is None:
                if state.mode is GameMode.CUTSCENE:
                    act = idle_action()
                else:
                    act, _reason = policy_input(policy, state)
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
                    act, _reason = policy_input(policy, state)
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
            "screenshots": shots,
            "saved_states": saved_states,
            "extras": {
                "start": start_snap,
                "end": snapshot_state(final_state),
            },
        }
        report_path = write_json_report(out / spec.report_name, report)
        report["report_path"] = str(report_path)
        return report
    finally:
        env.close()


def bridge_main(spec: BridgeSpec, argv: Sequence[str] | None = None) -> int:
    """Parse argv and run one bridge spec."""
    parser = argparse.ArgumentParser(
        description=(
            f"Bridge previous clear → fight-ready Stage {spec.dest}."
        )
    )
    parser.add_argument("--state", default=None)
    parser.add_argument("--max-frames", type=int, default=12000)
    parser.add_argument("--idle-after-clear", type=int, default=240)
    parser.add_argument("--start-period", type=int, default=40)
    parser.add_argument(
        "--no-save-states",
        action="store_true",
        help=f"Skip writing {spec.clear_save} / {spec.ready_save}",
    )
    parser.add_argument(
        "--screenshots",
        action="store_true",
        help="Write PNG dumps (off by default)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    report = run_bridge(
        spec,
        state_name=args.state,
        max_frames=args.max_frames,
        idle_after_clear=args.idle_after_clear,
        start_period=args.start_period,
        save_states=not args.no_save_states,
        screenshots=args.screenshots,
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
