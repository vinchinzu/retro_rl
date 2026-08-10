"""Record the SM Rando policy from power-on through first-item collection."""

from __future__ import annotations

import argparse
from pathlib import Path

from sm_rando.morph_policy import (
    MORPH_POLICY_REPORT,
    MORPH_POLICY_VIDEO,
    run_morph_policy,
)
from super_metroid.video import continuous_video_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=MORPH_POLICY_VIDEO)
    parser.add_argument("--report", type=Path, default=MORPH_POLICY_REPORT)
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Run the same emulator policy without encoding an MP4.",
    )
    parser.add_argument(
        "--video-start",
        choices=("power_on", "zebes", "after_credits"),
        default="zebes",
        help="Trim only the MP4; policy execution always starts at power-on.",
    )
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args(argv)

    video_path = None if args.no_video else args.video
    video_config = (
        None
        if args.no_video
        else continuous_video_config(
            start=args.video_start,
            fps=args.fps,
            audio=True,
            footer=True,
        )
    )
    report = run_morph_policy(
        video_path=video_path,
        video_config=video_config,
        report_path=args.report,
    )
    final = report.final_state
    room = int(final["room_id"])
    print(
        f"[GREEN] outcome={report.outcome} frames={report.total_frames} "
        f"room=0x{room:04X} morph_ball={final['morph_ball']}"
    )
    print(f"  report: {args.report}")
    print(f"  video:  {video_path if video_path is not None else '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
