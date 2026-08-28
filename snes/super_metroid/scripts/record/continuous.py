#!/usr/bin/env python3
"""Record continuous power-on through a named tip on the KPDR chain.

One CLI for every milestone — do not add ``start_to_*.py`` scripts. Tips are
functions in ``routes/continuous.py`` registered in ``routes/catalog.py``.

Video uses the shared :class:`retro_harness.video.VideoRecorder` (audio,
1080p60 YouTube pad + button sidebars, start gate). Opening credits are
dropped by default (``--video-start after_credits``). Metroid presets live
in ``super_metroid.video``.

```bash
# Current tip (Frog Savestation / KPDR K4.0), no video (integrity check)
uv run python snes/super_metroid/scripts/record/continuous.py --no-video
uv run python snes/super_metroid/scripts/record/continuous.py --to frog --no-video

# Showcase: skip Nintendo/title, 1080p60 sidebars
uv run python snes/super_metroid/scripts/record/continuous.py --to phantoon --hq

# List tips
uv run python snes/super_metroid/scripts/record/continuous.py --list

# Clean track (no energy + no ammo); uses *_clean artifact stems by default
uv run python snes/super_metroid/scripts/record/continuous.py --to bombs --clean --no-video
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
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
        help=(
            "Nearest-neighbor upscale. YouTube auto-fits 1920x1080; "
            "native default is 2"
        ),
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=None,
        help="x264 CRF quality (lower = better; youtube default 17, 15 with --hq)",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="x264 preset (youtube default medium, slow with --hq)",
    )
    parser.add_argument(
        "--hq",
        action="store_true",
        help=(
            "Higher quality encode (CRF 15, preset slow). "
            "YouTube still auto-fits 1920x1080; native uses scale=3"
        ),
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable emulator audio track (default: audio on when recording)",
    )
    parser.add_argument(
        "--no-footer",
        action="store_true",
        help=(
            "Disable bottom button/frame footer (youtube already has none; "
            "native default is footer on)"
        ),
    )
    parser.add_argument(
        "--video-start",
        choices=("power_on", "zebes", "after_credits", "frame"),
        default="after_credits",
        help=(
            "When to begin writing frames (play always power-on). "
            "Default: after_credits (drop Nintendo/title). "
            "power_on keeps opening credits — not for YouTube. "
            "zebes latches Landing Site."
        ),
    )
    parser.add_argument(
        "--native-video",
        action="store_true",
        help="2x gameplay + 16px footer instead of 1080p60 YouTube sidebars",
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
        }
        if args.native_video:
            overrides["layout"] = "native"
        if args.no_footer:
            overrides["footer"] = False
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
    # Full schema stays on disk only (report_path). Stdout is a short human
    # summary — dumping multi-MB JSON here is unreadable in agent/CLI logs.
    _print_run_summary(
        tip_id=tip.tip_id,
        payload=payload,
        report_path=report_path,
        video_path=None if args.no_video else video_path,
        state_output=args.state_output,
    )


def _print_run_summary(
    *,
    tip_id: str,
    payload: dict,
    report_path: Path,
    video_path: Path | None,
    state_output: Path | None,
) -> None:
    """One-screen outcome for agents/humans (not the full report schema)."""
    ok = bool(payload.get("success"))
    outcome = payload.get("outcome", "?")
    frames = payload.get("total_frames", payload.get("frame", "?"))
    final = payload.get("final_state") or {}
    room = final.get("room_id_hex") or (
        f"0x{final['room_id']:04X}" if isinstance(final.get("room_id"), int) else "?"
    )
    beams = final.get("collected_beams")
    beams_s = f"0x{beams:04X}" if isinstance(beams, int) else "?"
    items = final.get("collected_items")
    items_s = f"0x{items:04X}" if isinstance(items, int) else "?"
    integ = payload.get("integrity") or {}
    loads = integ.get("state_loads_zero")
    prog = integ.get("progression_writes_zero")
    deaths = integ.get("deaths_zero")
    integ_bits = []
    if loads is True:
        integ_bits.append("loads=0")
    elif loads is False:
        integ_bits.append("loads≠0")
    if prog is True:
        integ_bits.append("prog=0")
    elif prog is False:
        integ_bits.append("prog≠0")
    if deaths is True:
        integ_bits.append("deaths=0")
    elif deaths is False:
        integ_bits.append("deaths≠0")
    status = "GREEN" if ok else "RED"
    print(f"[{status}] tip={tip_id} outcome={outcome} frames={frames}")
    print(f"  final room={room} beams={beams_s} items={items_s}")
    if integ_bits:
        print(f"  integrity: {', '.join(integ_bits)}")
    if payload.get("error"):
        print(f"  error: {payload['error']}")
    # Tail splits (last 6) — enough to see where the tip landed.
    splits = payload.get("splits") or []
    if splits:
        print("  splits (tail):")
        for s in splits[-6:]:
            sid = s.get("split_id", "?")
            fr = s.get("frame", "?")
            rid = s.get("room_id")
            rh = f"0x{rid:04X}" if isinstance(rid, int) else "?"
            print(f"    {sid} @{fr} {rh}")
    print(f"  report: {report_path}")
    if video_path is not None:
        print(f"  video:  {video_path}")
    else:
        print("  video:  (none — pass without --no-video for .mp4 proof)")
    if state_output is not None:
        print(f"  state:  {state_output}")


if __name__ == "__main__":
    main()
