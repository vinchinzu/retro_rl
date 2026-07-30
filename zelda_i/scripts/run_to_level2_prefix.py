"""Walk from post-Level-1 overworld toward Level 2 (path prefix → 0x4A).

Examples::

    # Isolated from Level1ExitOverworld (post-triforce settle checkpoint)
    uv run python zelda_i/scripts/run_to_level2_prefix.py --trials 2

    # From Level1HeartCollected: collect triforce, settle, walk
    uv run python zelda_i/scripts/run_to_level2_prefix.py --from-heart --trials 2

    # Opt-in per-screen hop timing under recordings/room_timings/
    uv run python zelda_i/scripts/run_to_level2_prefix.py --room-timing --trials 1
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retro_harness.env import make_env, save_state
from retro_harness.nes import nes_idle_action
from snes_oneshot.segment_runner import (
    configure_headless,
    save_rgb_png,
    write_json_report,
)
from zelda_i.chain import run_controller_stage
from zelda_i.level1_finish import (
    TRIFORCE_MAX_FRAMES,
    Level1TriforceController,
)
from zelda_i.level2_overworld import (
    SETTLE_MAX_FRAMES,
    SEGMENT_MAX_FRAMES,
    OverworldToLevel2Controller,
    PostTriforceSettleController,
    level2_path_prefix_success,
    post_triforce_overworld_ready,
)
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR, ROOM_TIMINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.room_timer import RoomTimer, bottleneck_visits


def _collect_and_settle(
    env,
    obs,
    *,
    room_timer: RoomTimer | None = None,
    frame_base: int = 0,
) -> tuple[object, PostTriforceSettleController, dict, int]:
    """From Level1HeartCollected: triforce + settle to overworld 0x37."""
    tf = Level1TriforceController()
    obs, tf_stage = run_controller_stage(
        env,
        obs,
        name="triforce",
        controller=tf,
        max_frames=TRIFORCE_MAX_FRAMES,
        room_timer=room_timer,
        frame_base=frame_base,
    )
    settle = PostTriforceSettleController()
    obs, settle_stage = run_controller_stage(
        env,
        obs,
        name="settle",
        controller=settle,
        max_frames=SETTLE_MAX_FRAMES,
        room_timer=room_timer,
        frame_base=tf_stage.end_frame,
    )
    return (
        obs,
        settle,
        {"triforce": tf_stage.report(), "settle": settle_stage.report()},
        settle_stage.end_frame,
    )


def run_once(
    *,
    from_heart: bool = False,
    max_frames: int = SEGMENT_MAX_FRAMES,
    tag: str = "to_level2_prefix",
    save_checkpoint: bool = False,
    room_timing: bool = False,
) -> dict:
    configure_headless()
    start_state = "Level1HeartCollected" if from_heart else "Level1ExitOverworld"
    env = make_env(GAME, start_state, GAME_DIR, render_mode="rgb_array")
    nav = OverworldToLevel2Controller()
    room_timer = RoomTimer() if room_timing else None
    frame_base = 0
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        obs, *_ = env.step(nes_idle_action())
        frame_base = 1
        if room_timer is not None:
            room_timer.observe(read_snapshot(env.get_ram()), frame=frame_base)
        snap0 = read_snapshot(env.get_ram())
        entry = {
            "from_heart": from_heart,
            "start_state": start_state,
            "mode": snap0.mode,
            "screen": snap0.screen,
            "x": snap0.link_x,
            "y": snap0.link_y,
            "triforce": snap0.triforce,
            "health": snap0.health,
        }
        pre: dict = {}
        if from_heart:
            obs, settle, pre, frame_base = _collect_and_settle(
                env, obs, room_timer=room_timer, frame_base=frame_base
            )
            if not settle.success:
                snap = read_snapshot(env.get_ram())
                png = RECORDINGS_DIR / f"{tag}_settle_fail.png"
                save_rgb_png(obs, png)
                payload = {
                    "ok": False,
                    "stage": "settle",
                    "entry": entry,
                    "pre": pre,
                    "nav": nav.report(),
                    "final": {
                        "mode": snap.mode,
                        "screen": snap.screen,
                        "level": snap.level,
                        "x": snap.link_x,
                        "y": snap.link_y,
                        "health": snap.health,
                    },
                    "screenshot": str(png),
                }
                if room_timer is not None:
                    room_timer.finalize(frame=frame_base)
                    payload["room_timing"] = room_timer.report(
                        source=f"run_to_level2_prefix:{tag}",
                        extra={"ok": False, "stage": "settle", "entry": entry},
                    )
                return payload
        elif not post_triforce_overworld_ready(env.get_ram()):
            # Level1ExitOverworld should already be settled; one idle settle try.
            settle = PostTriforceSettleController()
            obs, settle_stage = run_controller_stage(
                env,
                obs,
                name="settle",
                controller=settle,
                max_frames=SETTLE_MAX_FRAMES,
                room_timer=room_timer,
                frame_base=frame_base,
            )
            pre["settle"] = settle_stage.report()
            frame_base = settle_stage.end_frame

        obs, nav_stage = run_controller_stage(
            env,
            obs,
            name="level2_prefix",
            controller=nav,
            max_frames=max_frames,
            room_timer=room_timer,
            frame_base=frame_base,
        )
        frame_base = nav_stage.end_frame

        snap = read_snapshot(env.get_ram())
        ok = level2_path_prefix_success(env.get_ram()) or nav.success
        png = RECORDINGS_DIR / f"{tag}_{'heart' if from_heart else 'exit'}.png"
        save_rgb_png(obs, png)
        checkpoint = None
        if ok and save_checkpoint:
            checkpoint = str(save_state(env, GAME_DIR, GAME, "Level2Path4A"))
        payload = {
            "ok": ok,
            "stage": "level2_prefix" if ok else "overworld",
            "entry": entry,
            "pre": pre,
            "nav": nav.report(),
            "final": {
                "mode": snap.mode,
                "level": snap.level,
                "screen": snap.screen,
                "x": snap.link_x,
                "y": snap.link_y,
                "sword": snap.sword,
                "triforce": snap.triforce,
                "health": snap.health,
            },
            "screenshot": str(png),
            "checkpoint": checkpoint,
        }
        if room_timer is not None:
            room_timer.finalize(frame=frame_base)
            timing = room_timer.report(
                source=f"run_to_level2_prefix:{tag}",
                extra={
                    "ok": ok,
                    "from_heart": from_heart,
                    "start_state": start_state,
                    "final_screen": snap.screen,
                    "final_screen_hex": f"0x{snap.screen:02X}",
                    "bottlenecks": bottleneck_visits(room_timer.visits, top_n=5),
                },
            )
            payload["room_timing"] = timing
        return payload
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-heart",
        action="store_true",
        help="Start from Level1HeartCollected (collect triforce + settle first)",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=SEGMENT_MAX_FRAMES)
    parser.add_argument(
        "--save-state",
        action="store_true",
        help="Save Level2Path4A.state on success",
    )
    parser.add_argument(
        "--room-timing",
        action="store_true",
        help=(
            "Opt-in screen/room hop timing via zelda_i.room_timer; "
            f"writes JSON under {ROOM_TIMINGS_DIR}"
        ),
    )
    args = parser.parse_args(argv)

    reports = []
    for i in range(args.trials):
        tag = f"to_level2_prefix_t{i}"
        rep = run_once(
            from_heart=args.from_heart,
            max_frames=args.max_frames,
            tag=tag,
            save_checkpoint=args.save_state,
            room_timing=args.room_timing,
        )
        reports.append(rep)
        fin = rep["final"]
        print(
            f"trial={i} ok={rep['ok']} stage={rep.get('stage')} "
            f"sc={fin['screen']:#04x} xy=({fin['x']},{fin['y']}) "
            f"hp={fin['health']:#04x} frames={rep['nav'].get('frames')}"
        )
        if args.room_timing and "room_timing" in rep:
            rt = rep["room_timing"]
            print(
                f"  room_timing visits={rt.get('visit_count')} "
                f"dwell={rt.get('total_dwell_frames')} "
                f"transition={rt.get('total_transition_frames')}"
            )
            for row in (rt.get("extra") or {}).get("bottlenecks") or []:
                print(
                    f"  bottleneck 0x{row['screen']:02X}->0x{row['dest_screen']:02X} "
                    f"loc={row['location_frames']} dwell={row['dwell_frames']} "
                    f"trans={row['transition_frames']}"
                )
    ok_n = sum(1 for r in reports if r["ok"])
    out = RECORDINGS_DIR / (
        "level2_prefix_heart.json"
        if args.from_heart
        else "level2_prefix_isolated.json"
    )
    write_json_report(
        out,
        {
            "segment": "level2_path_prefix",
            "from_heart": args.from_heart,
            "room_timing": args.room_timing,
            "trials": args.trials,
            "passed": ok_n,
            "reports": reports,
        },
    )
    print(f"passed={ok_n}/{args.trials} report={out}")
    if args.room_timing:
        ROOM_TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
        timing_name = (
            "level2_prefix_heart_timing.json"
            if args.from_heart
            else "level2_prefix_isolated_timing.json"
        )
        timing_out = ROOM_TIMINGS_DIR / timing_name
        write_json_report(
            timing_out,
            {
                "segment": "level2_path_prefix",
                "from_heart": args.from_heart,
                "trials": args.trials,
                "passed": ok_n,
                "trial_timings": [
                    r.get("room_timing") for r in reports if "room_timing" in r
                ],
            },
        )
        print(f"room_timing={timing_out}")
    return 0 if ok_n == args.trials else 1


if __name__ == "__main__":
    raise SystemExit(main())
