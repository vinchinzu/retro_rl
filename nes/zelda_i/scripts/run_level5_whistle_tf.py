"""Fixture-live Clean L5: Level5EntranceFromL4 → TF 0x10.

    QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scripts/run_level5_whistle_tf.py \
        --clean --no-video --trials 1

Pin is ``Level5EntranceFromL4`` (Raft/Stepladder). Old ``Level5Entrance``
lacks those items — do not use it. Spine CLI has no ``--from-state``.
``route_eligible=false``. Not a Clean STATUS claim.
"""

from __future__ import annotations

from retro_harness.env import make_env, reset_obs, resync_custom_state
from retro_harness.nes import nes_idle_action
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.anchors import TF_BIT_L5
from zelda_i.level5.spine import run_level5_from_entrance
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import ADDR_TRIFORCE, ADDR_WHISTLE, PLAY_MODE, read_snapshot, read_u8
from zelda_i.runner import (
    add_common_args,
    add_video_args,
    make_assist,
    resolve_video,
    VideoTap,
)

START = "Level5EntranceFromL4"


def _glance(env) -> dict:
    ram = env.get_ram()
    snap = read_snapshot(ram)
    tf = int(read_u8(ram, ADDR_TRIFORCE))
    return {
        "level": int(snap.level),
        "room": int(snap.screen),
        "mode": int(snap.mode),
        "xy": [int(snap.link_x), int(snap.link_y)],
        "whistle": int(read_u8(ram, ADDR_WHISTLE)),
        "triforce": tf,
        "tf_l5": bool(tf & TF_BIT_L5),
        "keys": int(snap.keys),
        "bombs": int(snap.bombs),
        "health": int(snap.health),
        "raft": int(snap.raft),
        "ladder": int(snap.ladder),
        "play": snap.mode == PLAY_MODE and not snap.transitioning,
    }


def _open(from_state: str):
    """Load a named pin. ``runner.open_env`` currently ImportErrors (no load_state)."""
    configure_headless()
    env = make_env(GAME, from_state, GAME_DIR, render_mode="rgb_array")
    reset_obs(env)
    resync_custom_state(env, GAME_DIR, GAME, from_state)
    return env


def run_once(*, from_state: str, tag: str, through: str, clean: bool, video) -> dict:
    env = _open(from_state)
    deaths = 0
    saw_death = False
    orig_step = env.step

    def _step(action, *args, **kwargs):
        nonlocal deaths, saw_death
        result = orig_step(action, *args, **kwargs)
        mode = int(read_snapshot(env.get_ram()).mode)
        if mode == 17 and not saw_death:
            deaths += 1
            saw_death = True
        elif mode != 17:
            saw_death = False
        return result

    env.step = _step
    tap = VideoTap(
        video[0],
        video[1],
        tag=tag,
        intro_summary="Clean L5 EntranceFromL4 -> TF 0x10",
        intro_frames=video[2],
        intervention="Clean no-assist",
    )
    try:
        obs, *_ = env.step(nes_idle_action())
        tap.attach(env, obs)
        start = _glance(env)
        assist = None if clean else make_assist(env)
        run = run_level5_from_entrance(
            env, obs, assist=assist, through=through,
        )
        end = _glance(env)
        png = RECORDINGS_DIR / f"{tag}_final.png"
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        final_obs, *_ = env.step(nes_idle_action())
        save_rgb_png(final_obs, png)
        leftover = {
            "room": end["room"],
            "mode": end["mode"],
            "x": end["xy"][0],
            "y": end["xy"][1],
        }
        ok = bool(run.success) and end["tf_l5"] and deaths == 0
        body = {
            "ok": ok,
            "route_eligible": False,
            "status_claim": None,
            "track": "clean" if clean else "assisted",
            "from_state": from_state,
            "through": through,
            "frames": run.end_frame,
            "deaths": deaths,
            "failed_stage": run.failed_stage,
            "start": start,
            "final": end,
            "leftover": leftover,
            "stages": [
                {"name": s.name, "success": s.success, "frames": s.frames}
                for s in run.stages
            ],
            "screenshot": str(png.resolve()),
        }
        write_json_report(RECORDINGS_DIR / f"{tag}.json", body)
        return body
    finally:
        tap.close()
        env.close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_state=START, default_tag="l5_entrance_tf_clean")
    parser.add_argument(
        "--through",
        default="level5",
        choices=(
            "level5-clear66",
            "level5-east77",
            "level5-whistle",
            "level5-exit04",
            "level5",
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="No infinite-life, no pokes (this runner never assists).",
    )
    add_video_args(parser, default_on=False)
    args = parser.parse_args(argv)
    clean = bool(args.clean or args.no_infinite_life or not args.infinite_life)
    video = resolve_video(args, default_path=RECORDINGS_DIR / f"{args.tag}.mp4")
    results = []
    for trial in range(args.trials):
        tag = args.tag if args.trials == 1 else f"{args.tag}_t{trial}"
        body = run_once(
            from_state=args.from_state,
            tag=tag,
            through=args.through,
            clean=clean,
            video=video,
        )
        results.append(body)
        print(
            "OK", body["ok"],
            "TF", body["final"]["tf_l5"],
            "deaths", body["deaths"],
            "failed", body["failed_stage"],
            "leftover", body["leftover"],
            flush=True,
        )
        if not body["ok"]:
            return 1
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
