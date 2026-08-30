"""Record one low-assist TMNT IV Hard run from power-on through staff credits.

The run uses one emulator session and selects Hard through the real menus. It
never loads a save state, never writes stage/lives/boss RAM, and never presses
the HP-draining special. Damage is measured from natural HP drops.

Video uses the shared :class:`retro_harness.video.VideoRecorder` (1080p60
YouTube pad + button sidebars by default). ``--native-video`` is the 16px
footer escape hatch.

Assists (disclosed, minimized vs the old every-hit restore-to-96; **default ON**):
1. Emergency HP top-up when about to die (HP <= threshold → 80).
2. Super Shredder form-2 iframe hold at 1 — his demutation projectile
   bypasses ordinary HP and is not yet reliably dodged.

Clean track (``--clean``): both assists off, default artifacts use the
``tmnt_iv_full_hard_clean`` stem (never overwrites assisted baselines), and
integrity fails if any assist counter is non-zero. Long forms:
``--no-emergency-hp`` / ``--no-iframe-hold`` (either alone is not full Clean).

Any life decrement aborts the run.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.env import make_env, reset_obs
from retro_harness.video import VideoCaptureConfig, VideoRecorder
from retro_harness.segment_runner import configure_headless
from tmnt_iv.observe import living_hp
from tmnt_iv.paths import GAME, GAME_DIR, default_full_run_paths
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.ram import parse_game_state
from tmnt_iv.run.freeze import FREEZE_ABORT_FRAMES as _FREEZE_ABORT_FRAMES
from tmnt_iv.run.metrics import (
    CreditsTracker,
    FINAL_CREDITS_EVENT as _FINAL_CREDITS_EVENT,
    FINAL_SCENE_SETTLE_FRAMES as _FINAL_SCENE_SETTLE_FRAMES,
    RunMetrics,
)
from tmnt_iv.run.report import finalize_full_run
from tmnt_iv.run.trial import (
    TrialContract,
    TrialEntry,
    TrialLimits,
    TrialObjective,
    run_trial,
)
from tmnt_iv.run.video import (
    full_run_video_config,
    open_full_run_capture,
    render_credits_overlay as _render_frame,
)


def run_full_hard(
    *,
    output: Path,
    report_path: Path,
    max_frames: int = 400_000,
    dry_run: bool = False,
    entry_state_prefix: str | None = None,
    emergency_hp: bool = True,
    iframe_hold: bool = True,
    require_clean_assists: bool | None = None,
    video_config: VideoCaptureConfig | None = None,
) -> dict[str, Any]:
    """Run from power-on through complete Hard credits and record artifacts.

    Defaults keep both production assists on. Clean track passes
    ``emergency_hp=False``, ``iframe_hold=False``, and (implicitly)
    ``require_clean_assists=True`` so zero-assist integrity fails closed.
    """
    if require_clean_assists is None:
        require_clean_assists = not emergency_hp and not iframe_hold
    clean_mode = not emergency_hp and not iframe_hold
    capture_config = video_config or full_run_video_config()

    configure_headless()
    env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
    policy = Stage1Policy()
    overlay = RunMetrics()
    credits = CreditsTracker()
    capture: VideoRecorder | None = None
    succeeded = False
    first_video = True
    prev_overlay_hp: int | None = None
    obs, _info = reset_obs(env)
    fps = float(env.em.get_screen_rate())
    audio_rate = int(env.em.get_audio_rate())
    height, width = obs.shape[:2]
    allowed: set[str] = set()
    if emergency_hp:
        allowed.add("player_hp")
    if iframe_hold:
        allowed.add("player_iframes")
    contract = TrialContract(
        name="clean" if not allowed else "assisted",
        emergency_hp=emergency_hp,
        iframe_hold=iframe_hold,
        fail_on_life_loss=True,
        allow_continue=False,
        allowed_write_keys=frozenset(allowed),
    )

    def _is_live(state: Any) -> bool:
        menu = int(state.extras.get("menu", -1))
        return (
            menu == 6
            and state.player_x > 0
            and state.stage <= 9
            and living_hp(state.health)
        )

    def on_frame(ctx: Any) -> None:
        nonlocal first_video, obs, prev_overlay_hp
        if ctx.obs is not None:
            obs = ctx.obs
        credits.update(ctx.state, frame=ctx.frame, metrics=overlay)
        health = ctx.state.health
        if living_hp(health):
            if prev_overlay_hp is not None and health < prev_overlay_hp:
                overlay.total_damage_taken += prev_overlay_hp - health
                overlay.min_health_seen = (
                    health
                    if overlay.min_health_seen is None
                    else min(overlay.min_health_seen, health)
                )
            prev_overlay_hp = health
        overlay.health_guard_interventions = sum(
            1 for write in getattr(ctx.env, "writes", ()) if write.get("key") == "player_hp"
        )
        if capture is None:
            return
        audio = None
        if not first_video:
            getter = getattr(getattr(ctx.env, "em", None), "get_audio", None)
            if callable(getter):
                audio = np.asarray(getter(), dtype=np.int16)
        first_video = False
        capture.write(
            _render_frame(obs, frame=ctx.frame, fps=fps, metrics=overlay),
            action=ctx.action,
            audio=audio,
            frame_index=ctx.frame,
        )

    try:
        if not dry_run:
            capture = open_full_run_capture(
                output,
                width=width,
                height=height,
                config=capture_config,
                audio_rate=audio_rate,
            )
        result = run_trial(
            TrialEntry(
                kind="power_on",
                state_name="NONE",
                is_live=_is_live,
                entry_state_prefix=entry_state_prefix,
            ),
            TrialObjective(kind="credits"),
            contract,
            TrialLimits(max_frames=max_frames),
            env=env,
            policy=policy,
            on_frame=on_frame,
        )
        if not result.success:
            reason = (
                result.failure.get("reason")
                if result.failure
                else result.outcome
            )
            raise RuntimeError(str(reason))
        succeeded = True
        metrics = result.to_metrics()
        overlay.credits_complete_frame = metrics.credits_complete_frame
        overlay.total_damage_taken = metrics.total_damage_taken
        overlay.min_health_seen = metrics.min_health_seen
        overlay.health_guard_interventions = metrics.health_guard_interventions
        final_state = parse_game_state(env.get_ram(), frame=result.total_frames)
        video_path: Path | None = None
        if capture is not None:
            video_path = capture.close()
            capture = None
        stage_writes = sum(1 for w in result.ram_writes if w["key"] == "stage")
        lives_writes = sum(
            1 for w in result.ram_writes if w["key"] in {"lives", "player_lives"}
        )
        return finalize_full_run(
            metrics=metrics,
            fps=fps,
            audio_rate=audio_rate,
            width=width,
            height=height,
            frame=result.total_frames,
            final_state=final_state,
            capture_config=capture_config,
            emergency_hp=emergency_hp,
            iframe_hold=iframe_hold,
            require_clean_assists=require_clean_assists,
            clean_mode=clean_mode,
            dry_run=dry_run,
            video_path=video_path,
            report_path=report_path,
            hard_confirmed=result.hard_confirmed,
            save_state_loads=result.state_loads_after_launch,
            stage_writes=stage_writes,
            lives_writes=lives_writes,
            forbidden_a_special_uses=result.a_special_uses,
            post_boot_start_presses=result.post_boot_start_presses,
        )
    finally:
        if capture is not None:
            capture.abort()
        env.close()
        if not succeeded and not dry_run:
            print("capture did not reach a verified completion", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Video path (default: recordings/tmnt_iv_full_hard_credits.mp4; "
            "with --clean: .../tmnt_iv_full_hard_clean.mp4)"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "JSON report path (default assisted credits/dry_run stems; "
            "with --clean: tmnt_iv_full_hard_clean[_dry_run].json)"
        ),
    )
    parser.add_argument("--max-frames", type=int, default=400_000)
    parser.add_argument(
        "--scale",
        type=int,
        default=3,
        help="Native-layout nearest-neighbor scale (ignored for YouTube)",
    )
    parser.add_argument(
        "--native-video",
        action="store_true",
        help="Nx gameplay + 16px footer instead of 1080p60 YouTube sidebars",
    )
    parser.add_argument(
        "--hq",
        action="store_true",
        help="Higher quality encode (CRF 15, preset slow); YouTube still 1080p60",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run all integrity checks without encoding video/audio",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Clean track: disable emergency HP and form-2 iframe hold, "
            "use *_clean default artifact stems, and require zero assist "
            "counters. Does not change assisted defaults when omitted."
        ),
    )
    parser.add_argument(
        "--no-emergency-hp",
        action="store_true",
        help="Disable emergency HP restore only (not full Clean alone)",
    )
    parser.add_argument(
        "--no-iframe-hold",
        action="store_true",
        help="Disable form-2 iframe hold only (not full Clean alone)",
    )
    parser.add_argument(
        "--entry-state-prefix",
        default=None,
        help=(
            "save development checkpoints at natural stage entries "
            "(for example LiveHard -> LiveHardStage5)"
        ),
    )
    return parser


def resolve_cli_paths(
    *,
    output: Path | None,
    report: Path | None,
    dry_run: bool,
    clean_artifacts: bool,
) -> tuple[Path, Path]:
    """Resolve video/report paths; explicit CLI paths always win."""
    default_video, default_report = default_full_run_paths(
        clean=clean_artifacts, dry_run=dry_run
    )
    return (
        output if output is not None else default_video,
        report if report is not None else default_report,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    emergency_hp = not (args.clean or args.no_emergency_hp)
    iframe_hold = not (args.clean or args.no_iframe_hold)
    # Full Clean: both assists off.
    clean = not emergency_hp and not iframe_hold
    # Any assist-off run uses *_clean stems so assisted baselines stay safe.
    clean_artifacts = not emergency_hp or not iframe_hold
    output, report = resolve_cli_paths(
        output=args.output,
        report=args.report,
        dry_run=args.dry_run,
        clean_artifacts=clean_artifacts,
    )
    run_full_hard(
        output=output,
        report_path=report,
        max_frames=args.max_frames,
        dry_run=args.dry_run,
        entry_state_prefix=args.entry_state_prefix,
        emergency_hp=emergency_hp,
        iframe_hold=iframe_hold,
        require_clean_assists=clean,
        video_config=full_run_video_config(
            native=args.native_video,
            scale=args.scale,
            hq=args.hq,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
