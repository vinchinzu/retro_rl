"""Complete Level 1 (The Eagle) and collect Triforce shard 1.

Examples::

    uv run python zelda_i/scripts/run_level1_complete.py --trials 2
    uv run python zelda_i/scripts/run_level1_complete.py \\
      --natural-entry --trials 2 --save-state
    uv run python zelda_i/scripts/run_level1_complete.py \\
      --natural-entry --room-timing --trials 1

    # Clean tip MP4 (power-on → Triforce shard 1)
    uv run python zelda_i/scripts/run_level1_complete.py \\
      --natural-entry --video --trials 1

    # Survival spine (rr-4d53.1): current boot + infinite life; does not
    # overwrite the Clean M5 tape.
    uv run python nes/zelda_i/scripts/run_level1_complete.py \\
      --natural-entry --infinite-life --video --trials 1
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from retro_harness.env import make_env, reset_obs, save_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from retro_harness.video import VideoCaptureConfig, VideoRecorder
from retro_harness.youtube_intro import (
    DEFAULT_INTRO_FRAMES,
    project_intro_lines,
    render_intro_card,
)
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.route.chain import run_controller_stage, run_natural_to_milestone
from zelda_i.dungeon.trace import write_state_provenance
from zelda_i.level1.finish import LEVEL1_TRIFORCE_BIT, level1_triforce_stages
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR, ROOM_TIMINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.room_timer import RoomTimer, bottleneck_visits

def _track_suffix(*, infinite_life: bool) -> str:
    return "_assisted" if infinite_life else ""


def default_video_path(*, natural_entry: bool, infinite_life: bool = False) -> Path:
    """Default showcase path. Assisted runs keep a separate filename."""
    label = "natural" if natural_entry else "isolated"
    return RECORDINGS_DIR / (
        f"level1_complete_{label}{_track_suffix(infinite_life=infinite_life)}.mp4"
    )


def default_report_path(*, natural_entry: bool, infinite_life: bool = False) -> Path:
    """Default JSON report path. Assisted runs do not clobber Clean evidence."""
    label = "natural" if natural_entry else "isolated"
    return RECORDINGS_DIR / (
        f"level1_complete_{label}{_track_suffix(infinite_life=infinite_life)}.json"
    )


def _intro_summary(*, natural_entry: bool, infinite_life: bool) -> str:
    track = "Survival" if infinite_life else "Clean"
    if natural_entry:
        return f"{track} power-on -> Level 1 Triforce shard 1"
    return f"{track} isolated Level 1 finish -> Triforce shard 1"

def _write_intro(
    writer: VideoRecorder,
    *,
    width: int,
    height: int,
    hold_frames: int,
    natural_entry: bool,
    audio_rate: int | None,
    infinite_life: bool = False,
) -> int:
    """Pipe project intro slide frames (silent audio) before gameplay.

    Card is playfield-sized only; ``VideoRecorder`` appends the footer band when
    enabled so geometry matches gameplay frames.
    """
    if hold_frames <= 0:
        return 0
    lines = project_intro_lines(
        game_title="The Legend of Zelda (NES)",
        run_summary=_intro_summary(
            natural_entry=natural_entry,
            infinite_life=infinite_life,
        ),
    )
    card = render_intro_card(
        lines,
        width=width,
        height=height,
        with_footer=False,
    )
    silent = None
    if writer.config.audio and audio_rate is not None and audio_rate > 0:
        n = max(1, int(round(audio_rate / float(writer.config.fps))))
        silent = np.zeros((n, 2), dtype=np.int16)
    for i in range(hold_frames):
        writer.write(card, audio=silent, frame_index=-(hold_frames - i))
    return hold_frames


def run_once(
    *,
    natural_entry: bool = False,
    tag: str = "level1_complete",
    save_checkpoint: bool = False,
    room_timing: bool = False,
    video_path: Path | None = None,
    video_config: VideoCaptureConfig | None = None,
    intro_frames: int = DEFAULT_INTRO_FRAMES,
    infinite_life: bool = False,
) -> dict:
    configure_headless()
    start_state = "NONE" if natural_entry else "Level1Cleared53"
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    prefix = None
    stages = []
    room_timer = RoomTimer() if room_timing else None
    assist = UnlimitedHealthAssist(enabled=True) if infinite_life else None
    frame_base = 0
    writer: VideoRecorder | None = None
    intro_written = 0
    try:
        obs, _ = reset_obs(env)

        on_frame = None
        if video_path is not None:
            config = video_config or VideoCaptureConfig()
            audio_rate: int | None = None
            if config.audio:
                em = getattr(env, "em", None)
                if em is not None and hasattr(em, "get_audio_rate"):
                    audio_rate = int(em.get_audio_rate())
                else:
                    config = replace(config, audio=False)
            writer = VideoRecorder(
                video_path,
                width=int(obs.shape[1]),
                height=int(obs.shape[0]),
                config=config,
                audio_rate=audio_rate,
            )
            intro_written = _write_intro(
                writer,
                width=int(obs.shape[1]),
                height=int(obs.shape[0]),
                hold_frames=intro_frames,
                natural_entry=natural_entry,
                audio_rate=audio_rate,
                infinite_life=infinite_life,
            )

            def on_frame(env_, obs_, action, frame: int) -> None:
                assert writer is not None
                writer.write_from_env(
                    env_,
                    obs_,
                    action=action,
                    frame_index=frame,
                )

        if natural_entry:
            prefix = run_natural_to_milestone(
                env,
                milestone="clear53",
                room_timer=room_timer,
                assist=assist,
                on_frame=on_frame,
                frame_base=frame_base,
            )
            obs = prefix.obs
            prefix_ok = prefix.success
            frame_base = prefix.end_frame
        else:
            idle = nes_idle_action()
            obs, *_ = env.step(idle)
            frame_base = 1
            if assist is not None:
                assist.apply_env(env, frame=frame_base)
            if room_timer is not None:
                room_timer.observe(read_snapshot(env.get_ram()), frame=frame_base)
            if on_frame is not None:
                on_frame(env, obs, idle, frame_base)
            prefix_ok = True

        for name, controller, max_frames in level1_triforce_stages(
            natural_entry=natural_entry,
            survival=infinite_life,
        ):
            if not prefix_ok:
                break
            obs, stage = run_controller_stage(
                env,
                obs,
                name=name,
                controller=controller,
                max_frames=max_frames,
                room_timer=room_timer,
                assist=assist,
                on_frame=on_frame,
                frame_base=frame_base,
            )
            stages.append(stage)
            frame_base = stage.end_frame
            prefix_ok = prefix_ok and stage.success
            if not stage.success:
                break

        snap = read_snapshot(env.get_ram())
        ok = prefix_ok and bool(snap.triforce & LEVEL1_TRIFORCE_BIT)
        checkpoint = None
        provenance = None
        if ok and save_checkpoint:
            checkpoint_path = save_state(env, GAME_DIR, GAME, "Level1Complete")
            checkpoint = str(checkpoint_path)
            provenance = str(
                write_state_provenance(
                    checkpoint_path,
                    source_state_path=(
                        None
                        if natural_entry
                        else GAME_DIR
                        / "custom_integrations"
                        / GAME
                        / "Level1Cleared53.state"
                    ),
                    request={
                        "segment": "level1_complete",
                        "natural_entry": natural_entry,
                        "infinite_life": infinite_life,
                    },
                    selected_trial={
                        "success": ok,
                        "stages": [stage.report() for stage in stages],
                    },
                    natural_entry=natural_entry,
                )
            )

        label = "natural" if natural_entry else "isolated"
        screenshot = RECORDINGS_DIR / f"{tag}_{label}.png"
        save_rgb_png(obs, screenshot)
        video_info = None
        if writer is not None:
            encoded = writer.frames
            closed_path = writer.close()
            writer = None
            video_info = {
                "path": str(closed_path),
                "encoded_frames": encoded,
                "intro_frames": intro_written,
                "gameplay_frames": max(0, encoded - intro_written),
            }
        payload = {
            "ok": ok,
            "natural_entry": natural_entry,
            "prefix_ok": prefix.success if prefix else True,
            "prefix": prefix.report() if prefix else None,
            "stages": [stage.report() for stage in stages],
            "final": {
                "mode": snap.mode,
                "level": snap.level,
                "room": snap.screen,
                "x": snap.link_x,
                "y": snap.link_y,
                "health": snap.health,
                "keys": snap.keys,
                "triforce": snap.triforce,
            },
            "checkpoint": checkpoint,
            "provenance": provenance,
            "screenshot": str(screenshot),
            "end_frame": frame_base,
            "video": video_info,
            "infinite_life": infinite_life,
            "assist": assist.report() if assist is not None else None,
        }
        if room_timer is not None:
            room_timer.finalize(frame=frame_base)
            payload["room_timing"] = room_timer.report(
                source=f"run_level1_complete:{tag}",
                extra={
                    "ok": ok,
                    "natural_entry": natural_entry,
                    "final_room": snap.screen,
                    "final_room_hex": f"0x{snap.screen:02X}",
                    "bottlenecks": bottleneck_visits(room_timer.visits, top_n=8),
                },
            )
        return payload
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        env.close()

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-entry", action="store_true")
    parser.add_argument(
        "--infinite-life",
        action="store_true",
        help="Survival health refill (rr-4d53 spine). Not a Clean STATUS run.",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument(
        "--room-timing",
        action="store_true",
        help=(
            "Opt-in screen/room hop timing via zelda_i.room_timer; "
            f"writes JSON under {ROOM_TIMINGS_DIR}"
        ),
    )
    parser.add_argument(
        "--video",
        nargs="?",
        const="AUTO",
        default=None,
        help=(
            "Record MP4 of the run (ffmpeg). Pass a path or omit the value "
            "for the default under recordings/ "
            "(level1_complete_{natural|isolated}.mp4)."
        ),
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable emulator audio on the MP4 (default: audio on with --video)",
    )
    parser.add_argument(
        "--no-footer",
        action="store_true",
        help="Disable button/frame footer on the MP4",
    )
    parser.add_argument(
        "--no-intro",
        action="store_true",
        help="Skip YouTube intro slide before gameplay",
    )
    parser.add_argument(
        "--intro-frames",
        type=int,
        default=DEFAULT_INTRO_FRAMES,
        help=f"Intro hold frames at 60fps (default {DEFAULT_INTRO_FRAMES})",
    )
    parser.add_argument(
        "--hq",
        action="store_true",
        help="Higher quality encode (scale=3, crf=15, preset=slow)",
    )
    args = parser.parse_args(argv)

    video_path: Path | None = None
    video_config: VideoCaptureConfig | None = None
    if args.video is not None:
        if args.video == "AUTO":
            video_path = default_video_path(
                natural_entry=args.natural_entry,
                infinite_life=args.infinite_life,
            )
        else:
            video_path = Path(args.video)
        if args.hq:
            video_config = VideoCaptureConfig.high_quality(
                audio=not args.no_audio,
                footer=not args.no_footer,
            )
        else:
            video_config = VideoCaptureConfig(
                audio=not args.no_audio,
                footer=not args.no_footer,
            )
        if args.trials > 1:
            print(
                "warning: --video with --trials>1 overwrites the same path "
                f"each trial ({video_path})",
                file=sys.stderr,
            )

    reports = [
        run_once(
            natural_entry=args.natural_entry,
            tag=f"level1_complete_t{trial}",
            save_checkpoint=args.save_state,
            room_timing=args.room_timing,
            video_path=video_path,
            video_config=video_config,
            intro_frames=0 if args.no_intro else max(0, args.intro_frames),
            infinite_life=args.infinite_life,
        )
        for trial in range(args.trials)
    ]
    for trial, report in enumerate(reports):
        final = report["final"]
        failed = next(
            (
                stage["name"]
                for stage in report["stages"]
                if not stage["success"]
            ),
            "-",
        )
        print(
            f"trial={trial} ok={report['ok']} "
            f"prefix_ok={report['prefix_ok']} failed={failed} "
            f"room={final['room']:02X} triforce=0x{final['triforce']:02X} "
            f"end_frame={report.get('end_frame')}"
        )
        if report.get("video"):
            v = report["video"]
            print(
                f"  video={v['path']} frames={v['encoded_frames']} "
                f"(intro={v['intro_frames']} gameplay={v['gameplay_frames']})"
            )
        if args.room_timing and "room_timing" in report:
            rt = report["room_timing"]
            print(
                f"  room_timing visits={rt.get('visit_count')} "
                f"dwell={rt.get('total_dwell_frames')} "
                f"transition={rt.get('total_transition_frames')}"
            )

    label = "natural" if args.natural_entry else "isolated"
    output = default_report_path(
        natural_entry=args.natural_entry,
        infinite_life=args.infinite_life,
    )
    write_json_report(
        output,
        {
            "segment": "level1_complete",
            "bead": "rr-4d53.1" if args.infinite_life else None,
            "natural_entry": args.natural_entry,
            "infinite_life": args.infinite_life,
            "room_timing": args.room_timing,
            "runtime_class": "bronze",
            "intervention_class": "survival" if args.infinite_life else "clean",
            "status_promote": False,
            "trials": args.trials,
            "successes": sum(report["ok"] for report in reports),
            "stop_predicate": "triforce & 0x01",
            "video": str(video_path) if video_path is not None else None,
            "reports": reports,
        },
    )
    print(f"wrote {output}")
    if video_path is not None and video_path.exists():
        size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"video={video_path} ({size_mb:.1f} MB)")
    if args.room_timing:
        ROOM_TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
        timing_out = ROOM_TIMINGS_DIR / f"level1_complete_{label}_timing.json"
        write_json_report(
            timing_out,
            {
                "segment": "level1_complete",
                "natural_entry": args.natural_entry,
                "trials": args.trials,
                "successes": sum(report["ok"] for report in reports),
                "trial_timings": [
                    r.get("room_timing") for r in reports if "room_timing" in r
                ],
            },
        )
        print(f"room_timing={timing_out}")
    return 0 if all(report["ok"] for report in reports) else 1

if __name__ == "__main__":
    raise SystemExit(main())
