"""Continuous Survival spine: power-on, one emulator session, no stitch.

    uv run python nes/zelda_i/scripts/run_survival_spine.py --trials 1
    uv run python nes/zelda_i/scripts/run_survival_spine.py --no-video --trials 1

Power-on first file slot / first quest. Records MP4 + room-transition PNGs
unless ``--no-video``. Does not overwrite Clean M5. No ``--from-state``.
Stop at first failed stage.
"""

from __future__ import annotations

import argparse

from retro_harness.env import make_env, reset_obs
from retro_harness.segment_runner import configure_headless, save_rgb_png, write_json_report
from zelda_i.assist import UnlimitedHealthAssist
from zelda_i.paths import GAME, GAME_DIR, RECORDINGS_DIR
from zelda_i.ram import read_snapshot
from zelda_i.runner import VideoTap, add_video_args, resolve_video
from zelda_i.survival_spine import SPINE_THROUGH, run_survival_spine, spine_final_fields


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", choices=SPINE_THROUGH, default="level1")
    parser.add_argument("--tag", default="survival_spine")
    parser.add_argument("--trials", type=int, default=1)
    add_video_args(parser, default_on=True)
    args = parser.parse_args(argv)

    configure_headless()
    results: list[dict] = []
    for trial in range(args.trials):
        tag = args.tag if args.trials == 1 else f"{args.tag}_t{trial}"
        video_path, video_config, intro = resolve_video(
            args,
            default_path=RECORDINGS_DIR / f"{tag}.mp4",
        )
        env = make_env(GAME, "NONE", GAME_DIR, render_mode="rgb_array")
        tap = VideoTap(
            video_path,
            video_config,
            tag=tag,
            intro_summary="Survival continuous spine, first quest, first file",
            intro_frames=intro,
        )
        assist = UnlimitedHealthAssist(enabled=True)
        try:
            obs, _ = reset_obs(env)
            tap.attach(env, obs)
            # VideoTap wraps env.step; do not also pass on_frame (double encode).
            run = run_survival_spine(
                env,
                obs,
                assist=assist,
                through=args.through,
            )
            snap = read_snapshot(env.get_ram())
            screenshot = RECORDINGS_DIR / f"{tag}_final.png"
            save_rgb_png(run.obs, screenshot)
            payload = {
                **run.report(),
                "trial": trial,
                "final": spine_final_fields(snap),
                "screenshot": str(screenshot),
                "assist": assist.report(),
                "video": tap.close(),
            }
        finally:
            env.close()
        write_json_report(RECORDINGS_DIR / f"{tag}.json", payload)
        results.append(payload)
        video = payload.get("video") or {}
        print(
            f"trial{trial}: ok={payload['ok']} failed={payload.get('failed_stage')} "
            f"tf={payload['final']['triforce']} room=0x{payload['final']['room']:02x} "
            f"keys={payload['final']['keys']} bombs={payload['final']['bombs']} "
            f"boot={payload.get('boot_policy')} video={video.get('path')}"
        )
    n_ok = sum(1 for row in results if row.get("ok"))
    print(f"summary: {n_ok}/{len(results)} continuous Survival {args.through}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
