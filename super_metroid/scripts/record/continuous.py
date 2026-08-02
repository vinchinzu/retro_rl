#!/usr/bin/env python3
"""Record continuous power-on through a named tip on the KPDR chain.

One CLI for every milestone — do not add ``start_to_*.py`` scripts. Tips are
functions in ``routes/continuous.py`` registered in ``routes/catalog.py``.

Video uses the shared :class:`retro_harness.video.VideoRecorder` (audio, footer,
quality knobs, start gate). Metroid presets live in ``super_metroid.video``.

```bash
# Current tip (Frog Savestation / KPDR K4.0), no video (integrity check)
uv run python super_metroid/scripts/record/continuous.py --no-video
uv run python super_metroid/scripts/record/continuous.py --to frog --no-video

# Showcase: Zebes start, sound, button footer, higher quality
uv run python super_metroid/scripts/record/continuous.py --to frog \\
  --video-start zebes --hq

# List tips
uv run python super_metroid/scripts/record/continuous.py --list

# Clean track (no energy + no ammo); uses *_clean artifact stems by default
uv run python super_metroid/scripts/record/continuous.py --to bombs --clean --no-video
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from super_metroid.routes.catalog import (  # noqa: E402
    CONTINUOUS_TIPS,
    DEFAULT_CONTINUOUS_TIP,
    get_continuous_tip,
)
from super_metroid.routes.continuous import (  # noqa: E402
    default_tip_artifact_paths,
    default_tip_room_timing_path,
    run_to,
)
from super_metroid.routes.runtime import resolve_clean_resources  # noqa: E402
from super_metroid.video import continuous_video_config  # noqa: E402


def main() -> None:
    tip_help = ", ".join(t.tip_id for t in CONTINUOUS_TIPS)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--to",
        dest="tip",
        default=DEFAULT_CONTINUOUS_TIP,
        help=(
            f"Continuous tip to stop at (default: {DEFAULT_CONTINUOUS_TIP}). "
            f"Known: {tip_help}"
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print known continuous tips and exit",
    )
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--state-output",
        type=Path,
        default=None,
        help=(
            "Write a reusable emulator checkpoint only after an integrity-green "
            "tip run (for continuous-like pure probes)."
        ),
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Clean track: disable unlimited energy and unlimited ammo, use "
            "*_clean default artifact stems, and require zero resource writes. "
            "Does not change assisted defaults when omitted."
        ),
    )
    parser.add_argument("--no-unlimited-energy", action="store_true")
    parser.add_argument("--no-unlimited-ammo", action="store_true")
    parser.add_argument(
        "--room-timing",
        action="store_true",
        help=(
            "Opt-in RoomTimer JSON under recordings/room_timings/ "
            "(does not affect continuous integrity or assists; tip must support it)"
        ),
    )
    parser.add_argument(
        "--room-timing-path",
        type=Path,
        default=None,
        help="Explicit room-timing JSON path (implies --room-timing)",
    )

    # Shared VideoRecorder settings (retro_harness.video)
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Video frame rate (default: 60)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=None,
        help="Nearest-neighbor upscale (default: 2, or 3 with --hq)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=None,
        help="x264 CRF quality (lower = better; default: 17, or 15 with --hq)",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="x264 preset (default: medium, or slow with --hq)",
    )
    parser.add_argument(
        "--hq",
        action="store_true",
        help="Higher quality: scale=3, crf=15, preset=slow",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable emulator audio track (default: audio on when recording)",
    )
    parser.add_argument(
        "--no-footer",
        action="store_true",
        help="Disable bottom button/frame footer (default: footer on)",
    )
    parser.add_argument(
        "--video-start",
        choices=("power_on", "zebes", "after_credits", "frame"),
        default="zebes",
        help=(
            "When to begin writing frames (play always power-on). "
            "Default: zebes (Landing Site latch). "
            "after_credits uses --video-start-frame or the default title cutoff."
        ),
    )
    parser.add_argument(
        "--video-start-frame",
        type=int,
        default=None,
        help=(
            "Inclusive frame gate for --video-start frame|after_credits "
            "(after_credits default cutoff if omitted)"
        ),
    )
    args = parser.parse_args()

    if args.list:
        for tip in CONTINUOUS_TIPS:
            marker = "  (default tip)" if tip.tip_id == DEFAULT_CONTINUOUS_TIP else ""
            print(f"{tip.tip_id:12}  {tip.display_name}{marker}")
            if tip.description:
                print(f"{'':12}  {tip.description}")
        return

    tip = get_continuous_tip(args.tip)
    unlimited_energy = not (args.clean or args.no_unlimited_energy)
    unlimited_ammo = not (args.clean or args.no_unlimited_ammo)
    # Full Clean intervention: both resource assists off.
    clean = resolve_clean_resources(
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
    )
    # Any assist-off run uses *_clean stems so assisted baselines stay safe.
    clean_artifacts = not unlimited_energy or not unlimited_ammo

    default_video, default_report = default_tip_artifact_paths(
        tip, clean=clean_artifacts
    )
    video_path = args.video if args.video is not None else default_video
    report_path = args.report if args.report is not None else default_report

    room_timing_path: Path | None = args.room_timing_path
    if args.room_timing and room_timing_path is None:
        if not tip.supports_room_timing:
            parser.error(
                f"tip {tip.tip_id!r} does not support --room-timing "
                f"(use supers or red_tower)"
            )
        room_timing_path = default_tip_room_timing_path(
            tip, clean=clean_artifacts
        )
    elif room_timing_path is not None and not tip.supports_room_timing:
        parser.error(f"tip {tip.tip_id!r} does not support room timing")

    video_config = None
    if not args.no_video:
        overrides: dict = {
            "fps": args.fps,
            "audio": not args.no_audio,
            "footer": not args.no_footer,
        }
        if args.scale is not None:
            overrides["scale"] = args.scale
        if args.crf is not None:
            overrides["crf"] = args.crf
        if args.preset is not None:
            overrides["preset"] = args.preset
        video_config = continuous_video_config(
            start=args.video_start,  # type: ignore[arg-type]
            start_frame=args.video_start_frame,
            hq=args.hq,
            **overrides,
        )

    report = run_to(
        tip.tip_id,
        video_path=None if args.no_video else video_path,
        video_config=video_config,
        report_path=report_path,
        unlimited_energy=unlimited_energy,
        unlimited_ammo=unlimited_ammo,
        room_timing_path=room_timing_path,
        state_output=args.state_output,
        require_clean_resources=clean,
    )
    payload = report.to_dict()
    payload["continuous_tip"] = tip.tip_id
    payload["clean_track"] = clean
    if room_timing_path is not None:
        payload["room_timing_path"] = str(room_timing_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
