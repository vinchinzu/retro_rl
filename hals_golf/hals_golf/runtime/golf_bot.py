"""CLI entrypoint for Hal's Hole in One Golf play / autoplay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from hals_golf.paths import GAME, PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

import stable_retro as retro  # noqa: E402

from hals_golf.runtime.retro_setup import (  # noqa: E402
    backup_mutable_start_state,
    backup_state,
    register_golf_integration,
)
from hals_golf.tasks.menus import Difficulty, PlayMode  # noqa: E402
from hals_golf.tasks.mission import StrokePlayMission  # noqa: E402
from hals_golf.tasks.profile import resolve_club_set  # noqa: E402


def _parse_play_mode(value: str) -> PlayMode:
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"stroke", "stroke-play", "strokeplay"}:
        return PlayMode.STROKE_PLAY
    if normalized in {"vs-hal", "vshal", "vs_hal", "hal"}:
        return PlayMode.VS_HAL
    raise argparse.ArgumentTypeError(
        f"unknown mode {value!r}; expected stroke or vs-hal"
    )


def _parse_difficulty(value: str) -> Difficulty:
    normalized = value.strip().lower()
    if normalized == "amateur":
        return Difficulty.AMATEUR
    if normalized == "pro":
        return Difficulty.PRO
    raise argparse.ArgumentTypeError(
        f"unknown difficulty {value!r}; expected amateur or pro"
    )


def _add_difficulty_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--difficulty",
        type=_parse_difficulty,
        default=Difficulty.AMATEUR,
        help="Title difficulty: amateur (default, verified) or pro",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hal's Hole in One Golf bot")
    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="Play with optional autoplay")
    play.add_argument("--state", default="Title", help="Save state name")
    play.add_argument(
        "--autoplay",
        action="store_true",
        help="Start with stroke-play / VS HAL mission enabled",
    )
    play.add_argument(
        "--mode",
        type=_parse_play_mode,
        default=PlayMode.STROKE_PLAY,
        help="Game mode: stroke (default) or vs-hal",
    )
    play.add_argument(
        "--club-set",
        choices=("auto", "standard", "metal"),
        default="auto",
        help="Club calibration (auto: metal only for a fresh VS HAL boot)",
    )
    _add_difficulty_arg(play)
    play.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip title/menu script (use when already in-round)",
    )
    play.add_argument("--scale", type=int, default=3)
    play.add_argument("--speed", type=float, default=1.0)
    play.add_argument(
        "--power-delay",
        type=int,
        default=42,
        help="Frames before swing power click",
    )
    play.add_argument(
        "--impact-delay",
        type=int,
        default=26,
        help="Frames before swing impact click",
    )
    play.add_argument(
        "--max-holes",
        type=int,
        default=18,
        help="Stop after this many holes",
    )
    play.add_argument(
        "--video",
        nargs="?",
        const="AUTO",
        default=None,
        help="Record full video (default: recordings/play_*.mp4)",
    )
    play.add_argument(
        "--video-scale",
        type=int,
        default=0,
        help="MP4 integer scale (0 = use --scale)",
    )

    clear = sub.add_parser(
        "clear",
        help="Run autoplay headlessly until the course / match is complete",
    )
    clear.add_argument("--state", default="Title", help="Save state name")
    clear.add_argument(
        "--mode",
        type=_parse_play_mode,
        default=PlayMode.STROKE_PLAY,
        help="Game mode: stroke (default) or vs-hal",
    )
    clear.add_argument(
        "--club-set",
        choices=("auto", "standard", "metal"),
        default="auto",
        help="Club calibration (auto: metal only for a fresh VS HAL boot)",
    )
    _add_difficulty_arg(clear)
    clear.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip title/menu script (use when already in-round)",
    )
    clear.add_argument("--power-delay", type=int, default=42)
    clear.add_argument("--impact-delay", type=int, default=26)
    clear.add_argument("--max-holes", type=int, default=18)
    clear.add_argument(
        "--max-frames",
        type=int,
        default=180_000,
        help="Fail if autoplay has not completed within this many frames",
    )
    clear.add_argument(
        "--checkpoint-state",
        default="latest",
        help="State refreshed at each new tee (empty disables checkpoints)",
    )
    clear.add_argument(
        "--tee-state-prefix",
        default="",
        help="Also save each tee as PREFIX<one-based-hole> (diagnostics)",
    )
    clear.add_argument(
        "--video",
        nargs="?",
        const="AUTO",
        default=None,
        help="Record full video of the clear (default: recordings/clear_*.mp4)",
    )
    clear.add_argument(
        "--video-scale",
        type=int,
        default=2,
        help="Video integer scale (native SNES is 256x224)",
    )
    clear.add_argument(
        "--video-fps",
        type=int,
        default=60,
        help="Playback FPS for the recorded video",
    )
    clear.add_argument(
        "--post-complete-frames",
        type=int,
        default=0,
        help="Idle after success so recordings retain result screens",
    )

    sub.add_parser("list", help="List available save states")

    probe = sub.add_parser("probe", help="Cold-boot probe / create states")
    probe.add_argument(
        "--frames",
        type=int,
        default=900,
        help="Frames to run during cold boot",
    )
    probe.add_argument(
        "--save-prefix",
        default="probe",
        help="Prefix for saved debug states/frames",
    )

    search = sub.add_parser(
        "search-hio",
        help="Score hole-in-one tee candidates from a fixed save state",
    )
    search.add_argument(
        "--state",
        default="Hole1_Command",
        help="Tee / command-menu save state to reload per candidate",
    )
    search.add_argument(
        "--mode",
        type=_parse_play_mode,
        default=PlayMode.STROKE_PLAY,
        help="Game mode: stroke (default) or vs-hal",
    )
    search.add_argument(
        "--club-set",
        choices=("auto", "standard", "metal"),
        default="standard",
        help="Club calibration for the planner profile",
    )
    _add_difficulty_arg(search)
    search.add_argument("--impact-delay", type=int, default=26)
    search.add_argument(
        "--max-candidates",
        type=int,
        default=25,
        help="Cap on HIO neighborhood size",
    )
    search.add_argument(
        "--club-deltas",
        default="0",
        help="Comma-separated club DOWN deltas from the base tee club",
    )
    search.add_argument(
        "--power-deltas",
        default="0,-2,2,-4,4",
        help="Comma-separated power deltas from the base tee power",
    )
    search.add_argument(
        "--aim-deltas",
        default="0,-4,4,-8,8",
        help="Comma-separated aim deltas from the base tee aim",
    )
    search.add_argument(
        "--max-frames",
        type=int,
        default=2500,
        help="Per-candidate frame budget after the swing starts",
    )
    return parser


def _list_states() -> int:
    from retro_harness.env import get_available_states

    register_golf_integration(retro, quiet=True)
    states = get_available_states(GAME, PROJECT_DIR)
    if not states:
        print("No save states found.")
        return 0
    for name in states:
        print(name)
    return 0


def _hud(info: dict[str, Any]) -> list[str]:
    lines = ["Hal's Hole in One"]
    for key in (
        "hole_index",
        "stroke_count",
        "rest_distance",
        "lie_type",
        "game_mode",
        "menu_cursor",
        "scene_id",
    ):
        if key in info:
            lines.append(f"{key}={info[key]}")
    return lines


def _open_video(
    video_arg: str | None,
    *,
    prefix: str,
    obs_shape: tuple[int, ...],
    scale: int,
    fps: int = 60,
    audio_rate: int | None = None,
):
    """Create a FrameVideoWriter when ``--video`` is enabled."""
    from hals_golf.runtime.video import FrameVideoWriter, resolve_video_path

    path = resolve_video_path(video_arg, prefix=prefix)
    if path is None:
        return None
    height, width = int(obs_shape[0]), int(obs_shape[1])
    writer = FrameVideoWriter(
        path,
        width=width,
        height=height,
        fps=fps,
        scale=max(1, scale),
        audio_rate=audio_rate,
    )
    audio = f" stereo-s16le@{audio_rate}Hz" if audio_rate is not None else ""
    print(
        f"[VIDEO] recording -> {writer.path} "
        f"({width}x{height} @{fps}fps scale={scale}{audio})"
    )
    return writer


def _close_video(writer) -> None:
    if writer is None:
        return
    path = writer.close()
    seconds = writer.frames_written / max(1, writer.fps)
    size_mb = path.stat().st_size / (1024 * 1024) if path.exists() else 0.0
    print(
        f"[VIDEO] wrote {path} frames={writer.frames_written} "
        f"duration={seconds:.1f}s size={size_mb:.1f}MB"
    )


def _play(args: argparse.Namespace) -> int:
    from retro_harness.env import make_env
    from retro_harness.play_session import PlaySession

    from hals_golf.runtime.bot_runner import GolfBotRunner

    register_golf_integration(retro)
    backup_mutable_start_state(args.state, label="play")

    env = make_env(
        game=GAME,
        state=args.state,
        game_dir=PROJECT_DIR,
        render_mode="rgb_array",
    )

    bot = None
    if args.autoplay:
        mission = StrokePlayMission(
            play_mode=args.mode,
            club_set=resolve_club_set(
                club_set_arg=args.club_set,
                play_mode=args.mode,
                skip_bootstrap=args.skip_bootstrap,
            ),
            difficulty=args.difficulty,
            skip_bootstrap=args.skip_bootstrap,
            power_delay=args.power_delay,
            impact_delay=args.impact_delay,
            max_holes=args.max_holes,
        )
        bot = GolfBotRunner(mission, env=env)

    print(
        "Controls: arrows move, Z=B(confirm), X=A(cancel), Enter=Start | "
        "~ or L+R+SELECT toggles human/autoplay | F5 quicksave | F7/F8 load"
    )
    session = PlaySession(
        env,
        game_dir=str(PROJECT_DIR),
        game=GAME,
        scale=args.scale,
        title=f"Hal's Hole in One: {args.state}",
        bot=bot,
        initial_speed=args.speed,
    )
    session.on_hud = _hud

    video_scale = args.video_scale if args.video_scale > 0 else args.scale
    writer_box: dict[str, Any] = {"writer": None}
    if args.video is not None:

        def _on_step(obs, _reward, _done, _info) -> None:
            writer = writer_box["writer"]
            if writer is None:
                writer = _open_video(
                    args.video,
                    prefix="play",
                    obs_shape=obs.shape,
                    scale=video_scale,
                    fps=60,
                    audio_rate=int(env.em.get_audio_rate()),
                )
                writer_box["writer"] = writer
            if writer is not None:
                writer.write(obs, audio=env.em.get_audio())

        def _on_close() -> None:
            _close_video(writer_box["writer"])
            writer_box["writer"] = None

        session.on_step = _on_step
        session.on_close = _on_close

    session.run()
    return 0


def _clear(args: argparse.Namespace) -> int:
    """Run a fast, noninteractive course-clear verification."""
    import numpy as np

    from retro_harness.env import make_env, save_state
    from retro_harness.protocol import TaskStatus, WorldState

    from hals_golf.core.actions import idle
    from hals_golf.core.ram import (
        WRAM_STROKE_COUNT,
        read_hole_number,
        read_rest_distance,
        read_u8,
    )
    from hals_golf.core.scene import is_command_screen

    register_golf_integration(retro)
    if args.checkpoint_state:
        backup_mutable_start_state(args.checkpoint_state, label="clear")
    hole1_state = (
        "VsHal_Hole1_Command"
        if args.mode is PlayMode.VS_HAL
        else "Hole1_Command"
    )
    complete_state = (
        "VsHalWin" if args.mode is PlayMode.VS_HAL else "CourseComplete"
    )
    refresh_hole_one = not args.skip_bootstrap
    if refresh_hole_one:
        backup_state(hole1_state, label="clear")

    env = make_env(
        game=GAME,
        state=args.state,
        game_dir=PROJECT_DIR,
        render_mode="rgb_array",
    )
    mission = StrokePlayMission(
        play_mode=args.mode,
        club_set=resolve_club_set(
            club_set_arg=args.club_set,
            play_mode=args.mode,
            skip_bootstrap=args.skip_bootstrap,
        ),
        difficulty=args.difficulty,
        skip_bootstrap=args.skip_bootstrap,
        power_delay=args.power_delay,
        impact_delay=args.impact_delay,
        max_holes=args.max_holes,
    )
    writer = None
    exit_code = 1
    try:
        obs, info = env.reset()
        writer = _open_video(
            args.video,
            prefix="clear",
            obs_shape=obs.shape,
            scale=args.video_scale,
            fps=args.video_fps,
            audio_rate=int(env.em.get_audio_rate()),
        )
        if writer is not None:
            writer.write(obs, audio=env.em.get_audio())

        ram = np.asarray(env.get_ram(), dtype=np.uint8)
        world = WorldState(frame=0, ram=ram, info=dict(info), obs=obs)
        mission.reset(world)
        checkpointed_hole = -1

        for frame in range(args.max_frames):
            ram = np.asarray(env.get_ram(), dtype=np.uint8)
            world = WorldState(
                frame=frame,
                ram=ram,
                info=dict(info),
                obs=obs,
            )
            result = mission.step(world)
            if result.status == TaskStatus.SUCCESS:
                card = mission.scorecard()
                total = int(card.get("total", 0))
                save_state(env, PROJECT_DIR, GAME, complete_state)
                if args.mode is PlayMode.VS_HAL:
                    print(
                        f"[CLEAR] complete frames={frame} "
                        f"mode=vs-hal state={args.state} "
                        f"reason={result.reason} "
                        f"match={card.get('holes_won')}-"
                        f"{card.get('holes_lost')}-"
                        f"{card.get('holes_tied')} "
                        f"scorecard={card.get('holes')} "
                        f"opponent={card.get('opponent_holes')}"
                    )
                else:
                    print(
                        f"[CLEAR] complete frames={frame} "
                        f"holes={args.max_holes} state={args.state} "
                        f"total={total} to_par={card.get('to_par')} "
                        f"over_par={card.get('over_par_holes')} "
                        f"scorecard={card.get('holes')}"
                    )
                post_frames = max(0, args.post_complete_frames)
                recorded_post_frames = 0
                for _ in range(post_frames):
                    obs, _reward, terminated, truncated, info = env.step(idle())
                    if writer is not None:
                        writer.write(obs, audio=env.em.get_audio())
                    recorded_post_frames += 1
                    if terminated or truncated:
                        break
                if recorded_post_frames:
                    print(
                        f"[CLEAR] post-complete frames={recorded_post_frames} "
                        f"duration={recorded_post_frames / 60:.1f}s"
                    )
                exit_code = 0
                break
            if result.status in {
                TaskStatus.FAILURE,
                TaskStatus.BLOCKED,
            }:
                card = mission.scorecard()
                print(
                    f"[CLEAR] {result.status.value}: "
                    f"{result.reason or 'no reason'} "
                    f"match={card.get('holes_won')}-"
                    f"{card.get('holes_lost')}-"
                    f"{card.get('holes_tied')} "
                    f"scorecard={card.get('holes')}",
                    file=sys.stderr,
                )
                exit_code = 1
                break

            action = result.action.action if result.action is not None else idle()
            obs, _reward, terminated, truncated, info = env.step(action)
            if writer is not None:
                writer.write(obs, audio=env.em.get_audio())
            if terminated or truncated:
                print(
                    f"[CLEAR] emulator stopped at frame={frame}",
                    file=sys.stderr,
                )
                exit_code = 1
                break

            ram = np.asarray(env.get_ram(), dtype=np.uint8)
            hole = read_hole_number(ram, info)
            strokes = read_u8(ram, WRAM_STROKE_COUNT)
            rest = read_rest_distance(ram, info)
            total = int(mission.scorecard()["total"])
            if (
                args.checkpoint_state
                and hole != checkpointed_hole
                and 1 <= hole <= args.max_holes
                and strokes == 0
                and 0 < rest < 1000
                and is_command_screen(obs)
            ):
                save_state(
                    env,
                    PROJECT_DIR,
                    GAME,
                    args.checkpoint_state,
                )
                if args.tee_state_prefix:
                    save_state(
                        env,
                        PROJECT_DIR,
                        GAME,
                        f"{args.tee_state_prefix}{hole}",
                    )
                if hole == 1 and refresh_hole_one:
                    save_state(
                        env,
                        PROJECT_DIR,
                        GAME,
                        hole1_state,
                    )
                    refresh_hole_one = False
                checkpointed_hole = hole
                match = ""
                if args.mode is PlayMode.VS_HAL:
                    card = mission.scorecard()
                    match = (
                        f" match={card.get('holes_won')}-"
                        f"{card.get('holes_lost')}-"
                        f"{card.get('holes_tied')}"
                    )
                print(
                    f"[CLEAR] hole={hole}/{args.max_holes} "
                    f"rest={rest} total={total}{match} frame={frame} "
                    f"checkpoint={args.checkpoint_state}"
                )
            elif frame > 0 and frame % 10_000 == 0:
                status = mission.mission_status()
                print(
                    f"[CLEAR] progress frame={frame} "
                    f"{status.phase} {status.objective} rest={rest}"
                )
        else:
            status = mission.mission_status()
            card = mission.scorecard()
            print(
                f"[CLEAR] frame limit {args.max_frames} reached: "
                f"{status.phase} {status.objective} scorecard={card}",
                file=sys.stderr,
            )
            exit_code = 1
    finally:
        try:
            _close_video(writer)
        except Exception as exc:  # noqa: BLE001 - surface encode errors
            print(f"[VIDEO] finalize failed: {exc}", file=sys.stderr)
            exit_code = 1
        env.close()
    return exit_code


def _probe(args: argparse.Namespace) -> int:
    from hals_golf.runtime.bootstrap import run_cold_boot_probe

    register_golf_integration(retro)
    run_cold_boot_probe(frames=args.frames, save_prefix=args.save_prefix)
    return 0


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated list of integers for search neighborhoods."""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(int(part) for part in parts)


def _search_hio(args: argparse.Namespace) -> int:
    from hals_golf.runtime.hio_search import search_tee_candidates
    from hals_golf.tasks.shot_policy import SearchSpec

    club_set = resolve_club_set(
        club_set_arg=args.club_set,
        play_mode=args.mode,
        skip_bootstrap=True,
    )
    spec = SearchSpec(
        power_deltas=_parse_int_csv(args.power_deltas),
        aim_deltas=_parse_int_csv(args.aim_deltas),
        club_deltas=_parse_int_csv(args.club_deltas),
        max_candidates=args.max_candidates,
    )
    situation, results = search_tee_candidates(
        state=args.state,
        play_mode=args.mode,
        club_set=club_set,
        difficulty=args.difficulty,
        impact_delay=args.impact_delay,
        max_frames_per_candidate=args.max_frames,
        max_candidates=args.max_candidates,
        spec=spec,
    )
    print(
        f"[HIO] state={args.state} hole={situation.hole} "
        f"strokes={situation.strokes} rest={situation.rest} "
        f"lie={situation.lie} candidates={len(results)}"
    )
    hits = 0
    ranked = sorted(
        results,
        key=lambda row: (
            0 if row.hole_in_one else 1,
            0 if row.end_strokes <= 1 else 1,
            row.end_rest,
            -row.rest_delta,
        ),
    )
    for row in ranked:
        intent = row.intent
        marker = "HIO" if row.hole_in_one else f"dRest={row.rest_delta:+d}"
        if row.hole_in_one:
            hits += 1
        print(
            f"[HIO] #{row.index:02d} power={intent.power} aim={intent.aim:+d} "
            f"club={intent.club_downs} -> rest={row.end_rest} "
            f"strokes={row.end_strokes} frames={row.frames} "
            f"status={row.status} {marker}"
        )
    print(f"[HIO] hole_in_one_hits={hits}/{len(results)}")
    if ranked:
        best = ranked[0].intent
        print(
            f"[HIO] best power={best.power} aim={best.aim:+d} "
            f"club={best.club_downs} rest={ranked[0].end_rest}"
        )
    return 0 if results else 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return _list_states()
    if args.command == "play":
        return _play(args)
    if args.command == "clear":
        return _clear(args)
    if args.command == "probe":
        return _probe(args)
    if args.command == "search-hio":
        return _search_hio(args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
