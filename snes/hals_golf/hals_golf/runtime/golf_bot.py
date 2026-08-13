"""CLI entrypoint for Hal's Hole in One Golf play / autoplay."""

from __future__ import annotations

import sys
from typing import Any

from hals_golf.paths import GAME, PROJECT_DIR, ensure_monorepo_on_path

ensure_monorepo_on_path()

import stable_retro as retro  # noqa: E402

from hals_golf.runtime.cli import build_parser, parse_int_csv  # noqa: E402
from hals_golf.runtime.retro_setup import (  # noqa: E402
    backup_mutable_start_state,
    register_golf_integration,
)
from hals_golf.tasks.mission import StrokePlayMission  # noqa: E402
from hals_golf.tasks.profile import resolve_club_set  # noqa: E402


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


def _play(args) -> int:
    from retro_harness.env import make_env
    from retro_harness.play_session import PlaySession

    from hals_golf.runtime.bot_runner import GolfBotRunner
    from hals_golf.runtime.video import close_cli_video, open_cli_video

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
                writer = open_cli_video(
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
            close_cli_video(writer_box["writer"])
            writer_box["writer"] = None

        session.on_step = _on_step
        session.on_close = _on_close

    session.run()
    return 0


def _probe(args) -> int:
    from hals_golf.runtime.bootstrap import run_cold_boot_probe

    register_golf_integration(retro)
    run_cold_boot_probe(frames=args.frames, save_prefix=args.save_prefix)
    return 0


def _search_hio(args) -> int:
    from hals_golf.runtime.hio_search import search_tee_candidates
    from hals_golf.tasks.shot_policy import SearchSpec

    club_set = resolve_club_set(
        club_set_arg=args.club_set,
        play_mode=args.mode,
        skip_bootstrap=True,
    )
    spec = SearchSpec(
        power_deltas=parse_int_csv(args.power_deltas),
        aim_deltas=parse_int_csv(args.aim_deltas),
        club_deltas=parse_int_csv(args.club_deltas),
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        return _list_states()
    if args.command == "play":
        return _play(args)
    if args.command == "clear":
        from hals_golf.runtime.clear_run import run_clear

        return run_clear(args)
    if args.command == "probe":
        return _probe(args)
    if args.command == "search-hio":
        return _search_hio(args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
