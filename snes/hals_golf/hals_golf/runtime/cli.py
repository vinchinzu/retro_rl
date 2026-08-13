"""Argparse for ``golf_bot`` play / clear / probe / search-hio."""

from __future__ import annotations

import argparse

from hals_golf.tasks.menus import Difficulty, PlayMode


def parse_play_mode(value: str) -> PlayMode:
    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"stroke", "stroke-play", "strokeplay"}:
        return PlayMode.STROKE_PLAY
    if normalized in {"vs-hal", "vshal", "vs_hal", "hal"}:
        return PlayMode.VS_HAL
    raise argparse.ArgumentTypeError(
        f"unknown mode {value!r}; expected stroke or vs-hal"
    )


def parse_difficulty(value: str) -> Difficulty:
    normalized = value.strip().lower()
    if normalized == "amateur":
        return Difficulty.AMATEUR
    if normalized == "pro":
        return Difficulty.PRO
    raise argparse.ArgumentTypeError(
        f"unknown difficulty {value!r}; expected amateur or pro"
    )


def parse_int_csv(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated list of integers for search neighborhoods."""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return tuple(int(part) for part in parts)


def _add_difficulty_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--difficulty",
        type=parse_difficulty,
        default=Difficulty.AMATEUR,
        help="Title difficulty: amateur (default, verified) or pro",
    )


def _add_mission_args(
    parser: argparse.ArgumentParser,
    *,
    club_default: str = "auto",
    include_skip: bool = True,
    include_power: bool = True,
    include_holes: bool = True,
) -> None:
    parser.add_argument(
        "--mode",
        type=parse_play_mode,
        default=PlayMode.STROKE_PLAY,
        help="Game mode: stroke (default) or vs-hal",
    )
    parser.add_argument(
        "--club-set",
        choices=("auto", "standard", "metal"),
        default=club_default,
        help="Club calibration (auto: metal only for a fresh VS HAL boot)",
    )
    _add_difficulty_arg(parser)
    if include_skip:
        parser.add_argument(
            "--skip-bootstrap",
            action="store_true",
            help="Skip title/menu script (use when already in-round)",
        )
    if include_power:
        parser.add_argument(
            "--power-delay",
            type=int,
            default=42,
            help="Frames before swing power click",
        )
    parser.add_argument(
        "--impact-delay",
        type=int,
        default=26,
        help="Frames before swing impact click",
    )
    if include_holes:
        parser.add_argument(
            "--max-holes",
            type=int,
            default=18,
            help="Stop after this many holes",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hal's Hole in One Golf bot")
    sub = parser.add_subparsers(dest="command", required=True)

    play = sub.add_parser("play", help="Play with optional autoplay")
    play.add_argument("--state", default="Title", help="Save state name")
    play.add_argument(
        "--autoplay",
        action="store_true",
        help="Start with stroke-play / VS HAL mission enabled",
    )
    _add_mission_args(play)
    play.add_argument("--scale", type=int, default=3)
    play.add_argument("--speed", type=float, default=1.0)
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
    _add_mission_args(clear)
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
    _add_mission_args(
        search,
        club_default="standard",
        include_skip=False,
        include_power=False,
        include_holes=False,
    )
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
