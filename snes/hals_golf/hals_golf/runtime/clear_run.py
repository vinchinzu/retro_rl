"""Headless course / match clear used by ``golf_bot clear``."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import stable_retro as retro
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
from hals_golf.paths import GAME, PROJECT_DIR
from hals_golf.runtime.retro_setup import (
    backup_mutable_start_state,
    backup_state,
    register_golf_integration,
)
from hals_golf.runtime.video import close_cli_video, open_cli_video
from hals_golf.tasks.menus import PlayMode
from hals_golf.tasks.mission import StrokePlayMission
from hals_golf.tasks.profile import resolve_club_set


def run_clear(args: argparse.Namespace) -> int:
    """Run a fast, noninteractive course-clear verification."""
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
        writer = open_cli_video(
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
            if result.status in {TaskStatus.FAILURE, TaskStatus.BLOCKED}:
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
                save_state(env, PROJECT_DIR, GAME, args.checkpoint_state)
                if args.tee_state_prefix:
                    save_state(
                        env,
                        PROJECT_DIR,
                        GAME,
                        f"{args.tee_state_prefix}{hole}",
                    )
                if hole == 1 and refresh_hole_one:
                    save_state(env, PROJECT_DIR, GAME, hole1_state)
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
            close_cli_video(writer)
        except Exception as exc:  # noqa: BLE001 - surface encode errors
            print(f"[VIDEO] finalize failed: {exc}", file=sys.stderr)
            exit_code = 1
        env.close()
    return exit_code
