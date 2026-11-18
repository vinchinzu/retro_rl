"""The one TMNT IV run CLI: live play (ROM-lookahead guarded) or tape replay.

Default is the Clean track: no HP or iframe writes, no state loads, no
A-special, any life loss fails. ``GuardedPolicy`` verifies every committed
frame on the ROM before it is played (``tmnt_iv.lookahead``). Every live run
writes a button tape beside its JSON; the published video is that tape
replayed from power-on with no policy and no lookahead.

  # Clean power-on -> Hard credits, live search (no video), tape + report
  uv run python -m tmnt_iv.scripts.run

  # Develop one stage from a natural pin; stop when the stage byte advances
  uv run python -m tmnt_iv.scripts.run --from CleanStage3 --until-stage 4

  # Save natural stage-entry pins along a power-on run (<prefix>Stage<N>)
  uv run python -m tmnt_iv.scripts.run --save-points Clean

  # Render the publishable video from a tape (RAM-CRC checked every 600f)
  uv run python -m tmnt_iv.scripts.run --replay recordings/tmnt_iv_full_hard_clean.tape.npz

  # Old assisted contract (emergency HP + form-2 iframe hold), no lookahead
  uv run python -m tmnt_iv.scripts.run --assisted --no-lookahead
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless
from retro_harness.video import VideoCaptureConfig, VideoRecorder
from tmnt_iv.lookahead import GuardedPolicy, LookaheadConfig
from tmnt_iv.menus import DEFAULT_TURTLE, TURTLES
from tmnt_iv.observe import living_hp
from tmnt_iv.paths import GAME, GAME_DIR, RECORDINGS_DIR, default_full_run_paths
from tmnt_iv.policy import Stage1Policy
from tmnt_iv.ram import parse_game_state
from tmnt_iv.run.metrics import CreditsTracker, RunMetrics
from tmnt_iv.run.report import finalize_full_run
from tmnt_iv.run.trial import (
    ASSISTED_CONTRACT,
    CLEAN_CONTRACT,
    TrialEntry,
    TrialLimits,
    TrialObjective,
    TrialResult,
    run_trial,
)
from tmnt_iv.run.video import (
    full_run_video_config,
    open_full_run_capture,
    render_credits_overlay,
)
from tmnt_iv.tape import Tape, load_tape, tape_path_for


def _is_live(state: Any) -> bool:
    return (
        int(state.extras.get("menu", -1)) == 6
        and state.player_x > 0
        and state.stage <= 9
        and living_hp(state.health)
    )


def _summary(result: TrialResult, wall: float) -> dict[str, Any]:
    return {
        "outcome": result.outcome,
        "success": result.success,
        "frames": result.total_frames,
        "wall_seconds": round(wall, 1),
        "end_stage": result.end_stage,
        "damage": result.damage_taken,
        "damage_by_stage": {str(k): v for k, v in result.damage_by_stage.items()},
        "min_hp": result.min_hp,
        "life_losses": result.life_losses,
        "pizza_heals": result.pizza_heal_count,
        "emergency_hp_writes": result.emergency_hp_writes,
        "iframe_writes": result.iframe_writes,
        "state_loads": result.state_loads_after_launch,
        "ram_write_count": len(result.ram_writes),
        "a_special_uses": result.a_special_uses,
        "post_boot_start_presses": result.post_boot_start_presses,
        "hard_confirmed": result.hard_confirmed,
        "credits_start_frame": result.credits_start_frame,
        "credits_complete_frame": result.credits_complete_frame,
        "rollout_restores": result.rollout_restores,
        "desync_frame": result.desync_frame,
        "crc_checks": result.crc_checks,
        "hits": result.hits[-40:],
        "lookahead": result.lookahead,
        "failure": (result.failure or {}).get("reason"),
    }


def run(
    *,
    entry_state: str = "NONE",
    until_stage: int | None = None,
    turtle: str = DEFAULT_TURTLE,
    clean: bool = True,
    lookahead: bool = True,
    config: LookaheadConfig | None = None,
    tape: Tape | None = None,
    finish_credits: bool = False,
    save_points: str | None = None,
    report_path: Path | None = None,
    video_path: Path | None = None,
    video_config: VideoCaptureConfig | None = None,
    max_frames: int = 400_000,
) -> dict[str, Any]:
    """Play (or replay) one trial and write its report, tape, and video."""
    power_on = entry_state == "NONE"
    credits = until_stage is None
    if credits and not power_on:
        raise SystemExit("--from a pin needs --until-stage (credits runs start at power-on)")
    if finish_credits and (tape is None or not credits or not power_on):
        raise SystemExit("--finish-credits needs a power-on credits tape via --replay")
    contract = CLEAN_CONTRACT if clean else ASSISTED_CONTRACT
    configure_headless()
    env = make_env(GAME, entry_state, GAME_DIR, render_mode="rgb_array")
    obs, _info = reset_obs(env)
    fps = float(env.em.get_screen_rate())
    audio_rate = int(env.em.get_audio_rate())
    height, width = obs.shape[:2]
    base = Stage1Policy()
    policy: Any = (
        GuardedPolicy(base, config, log=lambda m: print(m, flush=True))
        if lookahead and tape is None
        else base
    )
    overlay = RunMetrics()
    tracker = CreditsTracker()
    prev_hp: int | None = None
    capture: VideoRecorder | None = None
    capture_config = video_config or full_run_video_config()
    first_audio = True
    if video_path is not None:
        capture = open_full_run_capture(
            video_path, width=width, height=height, config=capture_config, audio_rate=audio_rate
        )

    def on_frame(ctx: Any) -> None:
        nonlocal prev_hp, first_audio
        tracker.update(ctx.state, frame=ctx.frame, metrics=overlay)
        hp = ctx.state.health
        if living_hp(hp) and overlay.credits_start_frame is None:
            if prev_hp is not None and hp < prev_hp:
                overlay.total_damage_taken += prev_hp - hp
            overlay.min_health_seen = hp if overlay.min_health_seen is None else min(overlay.min_health_seen, hp)
            prev_hp = hp
        if capture is None or ctx.obs is None:
            return
        audio = None
        if not first_audio:
            audio = np.asarray(ctx.env.em.get_audio(), dtype=np.int16)
        first_audio = False
        capture.write(
            render_credits_overlay(ctx.obs, frame=ctx.frame, fps=fps, metrics=overlay),
            action=ctx.action,
            audio=audio,
            frame_index=ctx.frame,
        )

    started = time.time()
    try:
        result = run_trial(
            TrialEntry(
                kind="power_on" if power_on else "state",
                state_name=entry_state,
                is_live=_is_live if power_on else None,
                entry_state_prefix=save_points,
                turtle=turtle,
                tape=tape,
                finish_credits=finish_credits,
            ),
            TrialObjective(kind="credits")
            if credits
            else TrialObjective(
                kind="stage_advance",
                stop_stage_gt=until_stage - 2,
                # The ending (stage 11) never returns control to a turtle.
                strict_advance=until_stage <= 10,
            ),
            contract,
            TrialLimits(max_frames=max_frames),
            env=env,
            policy=policy,
            on_frame=on_frame,
        )
        video = capture.close() if capture is not None else None
        capture = None
        wall = time.time() - started
        summary = _summary(result, wall)
        if tape is not None:
            summary["tape"] = tape.meta.get("source_path")
            summary["source_tape"] = {
                "path": tape.meta.get("source_path"),
                "sha256": tape.meta.get("source_sha256"),
                "frames": len(tape),
                "lookahead": tape.meta.get("lookahead", False),
            }
            summary["finish_credits"] = finish_credits
            summary["credits_extension_frames"] = max(0, result.total_frames - len(tape))
        if report_path is None:
            report_path = _default_report(entry_state, until_stage, clean=clean, replay=tape is not None)
        if result.tape is not None and (tape is None or finish_credits):
            result.tape.meta.update(
                {
                    "outcome": result.outcome,
                    "lookahead": lookahead and tape is None,
                    "clean": clean,
                    "until_stage": until_stage,
                    "source_tape": summary.get("source_tape"),
                    "credits_extension_frames": summary.get("credits_extension_frames", 0),
                }
            )
            summary["tape"] = str(result.tape.save(tape_path_for(report_path)))
        if credits and result.success:
            final_state = parse_game_state(env.get_ram(), frame=result.total_frames)
            report = finalize_full_run(
                metrics=result.to_metrics(),
                fps=fps,
                audio_rate=audio_rate,
                width=width,
                height=height,
                frame=result.total_frames,
                final_state=final_state,
                capture_config=capture_config,
                emergency_hp=contract.emergency_hp,
                iframe_hold=contract.iframe_hold,
                require_clean_assists=clean,
                clean_mode=clean,
                dry_run=video is None,
                video_path=video,
                report_path=report_path,
                hard_confirmed=result.hard_confirmed,
                save_state_loads=result.state_loads_after_launch,
                stage_writes=sum(1 for w in result.ram_writes if w["key"] == "stage"),
                lives_writes=sum(1 for w in result.ram_writes if w["key"] in {"lives", "player_lives"}),
                forbidden_a_special_uses=result.a_special_uses,
                post_boot_start_presses=result.post_boot_start_presses,
                extra={"turtle": turtle, "replay": tape is not None, **summary},
            )
            return report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entry": entry_state,
            "until_stage": until_stage,
            "turtle": turtle,
            "contract": contract.name,
            "lookahead": lookahead and tape is None,
            "replay": tape is not None,
            **summary,
        }
        report_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        print(
            f"{result.outcome}: frames={result.total_frames} stage={result.end_stage + 1} "
            f"damage={result.damage_taken} min_hp={result.min_hp} "
            f"lives_lost={result.life_losses} wall={wall:.0f}s -> {report_path}",
            flush=True,
        )
        return payload
    finally:
        if capture is not None:
            capture.abort()
        env.close()


def _default_report(entry: str, until_stage: int | None, *, clean: bool, replay: bool) -> Path:
    if entry == "NONE" and until_stage is None:
        _video, report = default_full_run_paths(clean=clean, dry_run=False)
        return report.with_name(report.stem + "_replay.json") if replay else report
    tag = f"{entry}_to_s{until_stage}" + ("_replay" if replay else "")
    return RECORDINGS_DIR / "dev" / f"{tag}.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="entry", default="NONE", help="entry pin (default NONE = power-on)")
    p.add_argument("--until-stage", type=int, choices=range(2, 12), metavar="N",
                   help="stop when human stage N is reached (10 = Shredder form 2, 11 = ending); "
                   "default = Hard credits")
    p.add_argument("--turtle", choices=sorted(TURTLES), default=DEFAULT_TURTLE)
    p.add_argument("--assisted", action="store_true",
                   help="old contract: emergency HP + form-2 iframe hold (not Clean)")
    p.add_argument("--no-lookahead", action="store_true", help="bare Stage1Policy, no ROM lookahead")
    p.add_argument("--horizon", type=int, default=LookaheadConfig.horizon)
    p.add_argument("--save-points", metavar="PREFIX",
                   help="save natural stage-entry pins as <PREFIX>Stage<N> (dev resume points)")
    p.add_argument("--replay", type=Path, metavar="TAPE", help="replay a tape instead of playing")
    p.add_argument("--finish-credits", action="store_true",
                   help="extend an ending tape with neutral input; fails if gameplay is unfinished")
    p.add_argument("--video", action="store_true", help="encode an mp4 (use with --replay)")
    p.add_argument("--native-video", action="store_true")
    p.add_argument("--hq", action="store_true")
    p.add_argument("--report", type=Path)
    p.add_argument("--max-frames", type=int, default=400_000)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    tape = load_tape(args.replay) if args.replay else None
    entry = tape.entry if tape is not None else args.entry
    turtle = tape.turtle if tape is not None else args.turtle
    until_stage = args.until_stage
    if tape is not None and until_stage is None:
        until_stage = tape.meta.get("until_stage")
    clean = not args.assisted
    video_path = None
    if args.video:
        if args.report is not None:
            video_path = args.report.with_suffix(".mp4")
        elif tape is not None:
            video_path = Path(str(args.replay).replace(".tape.npz", ".mp4"))
        else:
            video_path = default_full_run_paths(clean=clean)[0]
    payload = run(
        entry_state=entry,
        until_stage=until_stage,
        turtle=turtle,
        clean=clean,
        lookahead=not args.no_lookahead,
        config=LookaheadConfig(horizon=args.horizon),
        tape=tape,
        finish_credits=args.finish_credits,
        save_points=args.save_points,
        report_path=args.report,
        video_path=video_path,
        video_config=full_run_video_config(native=args.native_video, hq=args.hq),
        max_frames=args.max_frames,
    )
    ok = payload.get("success", payload.get("status") == "success")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
