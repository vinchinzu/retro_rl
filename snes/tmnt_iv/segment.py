"""Shared Stage N wave-chain runner for TMNT IV.

``scripts/run_stage{N}_segment.py`` used to copy the same boot / walk /
WaveChainTracker loop. Import ``run_stage_segment`` / ``StageSpec`` instead.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from retro_harness.actions import buttons, idle_action
from retro_harness.env import get_available_states, make_env, reset_obs, save_state
from retro_harness.ram_state import GameMode, GameState
from retro_harness.segment_runner import (
    SegmentOutcome,
    WaveChainTracker,
    configure_headless,
    save_rgb_png,
    snapshot_state,
    write_json_report,
)
from tmnt_iv.paths import (
    GAME,
    GAME_DIR,
    RECORDINGS_DIR,
    STAGE1_STATE,
    STAGE2_STATE,
    STAGE3_STATE,
    STAGE4_STATE,
    STAGE5_STATE,
    STAGE6_STATE,
    STAGE7_STATE,
    STAGE8_STATE,
    STAGE9_STATE,
)
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.ram import parse_game_state

WalkKind = Literal["right", "idle"]


@dataclass(frozen=True)
class StageSpec:
    """Per-stage names and knobs for the shared wave-chain loop."""

    number: int
    title: str
    preferred_states: tuple[str, ...]
    walk: WalkKind = "right"
    default_max_frames: int = 12000
    save_hp_min: int = 20
    require_lives: bool = False
    snapshot_boss_on_sight: bool = False
    heal_low_hp_default: bool = False


STAGE_SPECS: dict[int, StageSpec] = {
    1: StageSpec(
        number=1,
        title="Headless Stage1 segment: chain Foot Clan waves until clear / boss / fail.",
        preferred_states=("Stage1_Clear_w1", STAGE1_STATE, "NONE"),
        save_hp_min=40,
        require_lives=True,
        snapshot_boss_on_sight=True,
    ),
    2: StageSpec(
        number=2,
        title="Headless Stage 2 segment: chain Alleycat waves from Stage2.",
        preferred_states=(STAGE2_STATE, "Stage2_probe", "NONE"),
    ),
    3: StageSpec(
        number=3,
        title="Headless Stage 3 segment: chain Sewer waves from Stage3.",
        preferred_states=(STAGE3_STATE, "Stage3_probe", "NONE"),
    ),
    4: StageSpec(
        number=4,
        title="Headless Stage 4 segment: chain Technodrome waves from Stage4.",
        preferred_states=(STAGE4_STATE, "Stage4_probe", "NONE"),
    ),
    5: StageSpec(
        number=5,
        title="Headless Stage 5 segment: chain Prehistoric waves from Stage5.",
        preferred_states=(STAGE5_STATE, "Stage5_probe", "NONE"),
    ),
    6: StageSpec(
        number=6,
        title="Headless Stage 6 segment: chain waves from Stage6.",
        preferred_states=(STAGE6_STATE, "Stage6_probe", "NONE"),
    ),
    7: StageSpec(
        number=7,
        title="Headless Stage 7 segment: chain waves from Stage7.",
        preferred_states=(STAGE7_STATE, "Stage7_probe", "NONE"),
    ),
    8: StageSpec(
        number=8,
        title="Headless Stage 8 segment: chain Mode-7 waves from Stage8.",
        preferred_states=(STAGE8_STATE, "Stage8_probe", "NONE"),
        walk="idle",
        default_max_frames=20000,
        heal_low_hp_default=True,
    ),
    9: StageSpec(
        number=9,
        title="Headless Stage 9 segment: chain Starbase waves from Stage9.",
        preferred_states=(STAGE9_STATE, "Stage9_probe", "NONE"),
        default_max_frames=20000,
        heal_low_hp_default=True,
    ),
}


def _with_stage_progress(state: GameState, *, start_stage: int) -> GameState:
    """Mark the loaded stage complete when the stage byte advances."""
    if state.stage > start_stage:
        return replace(state, level_complete=True)
    return state


def _walk_until_enemies(
    env: Any,
    state: GameState,
    *,
    walk: WalkKind,
    max_frames: int = 180,
) -> tuple[Any, GameState]:
    """Advance until a living enemy appears (post-clear resume)."""
    obs: Any = None
    action = idle_action() if walk == "idle" else buttons("RIGHT")
    for _ in range(max_frames):
        if state.living_enemies:
            break
        step = env.step(action)
        obs = step[0]
        state = parse_game_state(env.get_ram(), frame=state.frame)
    return obs, state


def _maybe_heal(env: Any, state: GameState) -> GameState:
    """Top up Leo when corridor DPS drains him (dev resume helper)."""
    if 0 < state.health < 28 and state.health <= 0x60:
        env.set_value("player_hp", 80)
        return parse_game_state(env.get_ram(), frame=state.frame)
    return state


def run_stage_segment(
    spec: StageSpec,
    *,
    max_frames: int | None = None,
    clear_hold_frames: int = 30,
    target_waves: int | None = None,
    stop_on_boss: bool = True,
    save_wave_states: bool = True,
    heal_low_hp: bool | None = None,
    state_name: str | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Load a stage pin, chain waves, write report + PNGs + mid states."""
    configure_headless()
    frames = spec.default_max_frames if max_frames is None else max_frames
    heal = spec.heal_low_hp_default if heal_low_hp is None else heal_low_hp
    available = get_available_states(GAME, GAME_DIR)
    chosen = state_name or next(
        (name for name in spec.preferred_states if name in available),
        "NONE",
    )
    tag = f"stage{spec.number}"
    out = out_dir or (RECORDINGS_DIR / f"{tag}_segment")
    out.mkdir(parents=True, exist_ok=True)

    env = make_env(GAME, chosen, GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    tracker = WaveChainTracker(
        max_frames=frames,
        clear_hold_frames=clear_hold_frames,
        target_waves=target_waves,
        stop_on_boss=stop_on_boss,
        camera_unlock_delta=8,
    )
    screenshots: list[str] = []
    saved_states: list[str] = []
    outcome = SegmentOutcome.TIMEOUT
    obs: Any = None
    final_state: GameState | None = None
    start_snap: dict[str, Any] = {}
    last_cleared = 0
    boss_saved = False
    try:
        obs, _info = reset_obs(env)
        state = parse_game_state(env.get_ram(), frame=0)
        if not state.living_enemies:
            walked, state = _walk_until_enemies(env, state, walk=spec.walk)
            if walked is not None:
                obs = walked
        start_stage = state.stage
        state = _with_stage_progress(state, start_stage=start_stage)
        tracker.begin(state)
        start_snap = snapshot_state(state)
        final_state = state
        png = save_rgb_png(obs, out / f"{tag}_0000_start.png")
        screenshots.append(png.name)

        for frame_i in range(1, frames + 1):
            if heal:
                state = _maybe_heal(env, state)
            tick = policy.tick(state)
            if tick.action is not None:
                tracker.note_reason(tick.action.reason)
                action = tick.action.action
            else:
                tracker.note_reason(tick.reason or "no_action")
                action = idle_action()
            obs, _reward, _term, _trunc, _info = env.step(action)
            state = _with_stage_progress(
                parse_game_state(env.get_ram(), frame=frame_i),
                start_stage=start_stage,
            )
            final_state = state
            stop = tracker.update(state)

            if (
                spec.snapshot_boss_on_sight
                and save_wave_states
                and not boss_saved
                and state.boss_active
                and state.mode is GameMode.PLAYING
                and state.lives > 0
                and 40 <= state.health <= 0x60
            ):
                path = save_state(env, GAME_DIR, GAME, "Boss")
                saved_states.append(path.name)
                before = save_state(env, GAME_DIR, GAME, "Stage1_BeforeBoss")
                saved_states.append(before.name)
                boss_saved = True
                png = save_rgb_png(obs, out / f"{tag}_{frame_i:04d}_boss.png")
                screenshots.append(png.name)

            if tracker.waves_cleared > last_cleared:
                last_cleared = tracker.waves_cleared
                wave = tracker.waves[-1]
                png = save_rgb_png(
                    obs, out / f"{tag}_{frame_i:04d}_wave{wave.index}_clear.png"
                )
                screenshots.append(png.name)
                healthy = (
                    save_wave_states
                    and state.mode is GameMode.PLAYING
                    and spec.save_hp_min <= state.health <= 0x60
                )
                if healthy and (not spec.require_lives or state.lives > 0):
                    progress = int(state.extras.get("progress_x", state.camera_x))
                    name = f"Stage{spec.number}_Clear_w{wave.index}_cam{progress}"
                    path = save_state(env, GAME_DIR, GAME, name)
                    saved_states.append(path.name)

            if stop is not None:
                outcome = stop
                break
        else:
            outcome = SegmentOutcome.TIMEOUT

        assert final_state is not None
        end_tag = "end"
        if outcome is SegmentOutcome.SUCCESS and tracker.boss_reached:
            end_tag = "boss"
            if (
                save_wave_states
                and final_state.boss_active
                and not spec.snapshot_boss_on_sight
            ):
                path = save_state(env, GAME_DIR, GAME, f"Boss{spec.number}")
                saved_states.append(path.name)
        elif outcome is SegmentOutcome.SUCCESS:
            end_tag = "clear"
        elif outcome is SegmentOutcome.DEATH:
            end_tag = "death"
        png = save_rgb_png(obs, out / f"{tag}_{tracker.frames:04d}_{end_tag}.png")
        screenshots.append(png.name)

        if (
            save_wave_states
            and outcome is SegmentOutcome.SUCCESS
            and final_state.level_complete
        ):
            path = save_state(env, GAME_DIR, GAME, f"Stage{spec.number}_Clear")
            saved_states.append(path.name)

        report = tracker.to_report(
            outcome=outcome,
            final=final_state,
            screenshots=screenshots,
            start_state=chosen,
            saved_states=saved_states,
            extras={"start": start_snap, "end": snapshot_state(final_state)},
        )
        report_path = write_json_report(out / f"{tag}_segment.json", report)
        report["report_path"] = str(report_path)
        return report
    finally:
        env.close()


def print_segment_report(report: dict[str, Any]) -> int:
    """Print the standard CLI summary. Exit 0 on success or any wave clear."""
    end = report.get("extras", {}).get("end", {})
    print(
        f"outcome={report['outcome']} frames={report['frames']} "
        f"waves={report['waves_cleared']} kills={report['kills']} "
        f"damage={report['damage_dealt']} "
        f"hp={report['start_health']}->{report['end_health']} "
        f"lives={end.get('lives', '?')} "
        f"cam={report['start_camera_x']}->{report['end_camera_x']} "
        f"boss={report['boss_reached']} "
        f"level_complete={report.get('level_complete')} "
        f"enemies={report['start_enemy_count']}->{report['end_enemy_count']}"
    )
    print(f"report={report.get('report_path')}")
    if report.get("saved_states"):
        print("states: " + ", ".join(report["saved_states"]))
    top = sorted(report.get("reason_counts", {}).items(), key=lambda kv: -kv[1])[:8]
    if top:
        print("reasons: " + ", ".join(f"{k}={v}" for k, v in top))
    for wave in report.get("waves", []):
        print(
            f"  wave{wave['index']}: frames={wave['frames']} "
            f"kills={wave['kills']} "
            f"hp={wave['start_health']}->{wave['end_health']} "
            f"cam={wave['start_camera_x']}->{wave['end_camera_x']}"
        )
    if report.get("success") or report.get("waves_cleared", 0) > 0:
        return 0
    return 1


def build_segment_parser(
    spec: StageSpec,
    description: str,
) -> argparse.ArgumentParser:
    """Standard flags shared by ``run_stage{N}_segment`` CLIs."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--max-frames", type=int, default=spec.default_max_frames)
    parser.add_argument("--clear-hold-frames", type=int, default=30)
    parser.add_argument("--target-waves", type=int, default=None)
    parser.add_argument("--no-stop-on-boss", action="store_true")
    parser.add_argument("--no-save-states", action="store_true")
    if spec.heal_low_hp_default:
        parser.add_argument("--no-heal", action="store_true")
    parser.add_argument("--state", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def segment_main(spec: StageSpec, argv: Sequence[str] | None = None) -> int:
    """Parse argv and run the shared stage segment."""
    args = build_segment_parser(spec, spec.title).parse_args(argv)
    heal = None
    if spec.heal_low_hp_default:
        heal = not args.no_heal
    report = run_stage_segment(
        spec,
        max_frames=args.max_frames,
        clear_hold_frames=args.clear_hold_frames,
        target_waves=args.target_waves,
        stop_on_boss=not args.no_stop_on_boss,
        save_wave_states=not args.no_save_states,
        heal_low_hp=heal,
        state_name=args.state,
        out_dir=args.out_dir,
    )
    return print_segment_report(report)


def run_stage1_segment(**kwargs: Any) -> dict[str, Any]:
    return run_stage_segment(STAGE_SPECS[1], **kwargs)


def run_stage2_segment(**kwargs: Any) -> dict[str, Any]:
    return run_stage_segment(STAGE_SPECS[2], **kwargs)


def run_stage3_segment(**kwargs: Any) -> dict[str, Any]:
    return run_stage_segment(STAGE_SPECS[3], **kwargs)


def run_stage4_segment(**kwargs: Any) -> dict[str, Any]:
    return run_stage_segment(STAGE_SPECS[4], **kwargs)


def run_stage5_segment(**kwargs: Any) -> dict[str, Any]:
    return run_stage_segment(STAGE_SPECS[5], **kwargs)


def run_stage6_segment(**kwargs: Any) -> dict[str, Any]:
    return run_stage_segment(STAGE_SPECS[6], **kwargs)


def run_stage7_segment(**kwargs: Any) -> dict[str, Any]:
    return run_stage_segment(STAGE_SPECS[7], **kwargs)


def run_stage8_segment(**kwargs: Any) -> dict[str, Any]:
    return run_stage_segment(STAGE_SPECS[8], **kwargs)


def run_stage9_segment(**kwargs: Any) -> dict[str, Any]:
    return run_stage_segment(STAGE_SPECS[9], **kwargs)
