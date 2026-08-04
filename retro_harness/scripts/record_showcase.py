"""CLI for generic oneshot segmented showcase recordings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from retro_harness.ladder import LADDER
from retro_harness.showcase import (
    default_output_path,
    load_showcase_game,
    record_showcase,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record a game's segmented completion showcase using generic "
            "footer + button tracking."
        ),
    )
    parser.add_argument(
        "slug",
        nargs="?",
        help="Ladder game slug with a {slug}.showcase module.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output MP4 path (default: game recordings dir convention).",
    )
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--card-frames", type=int, default=60)
    parser.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Record only the first N clips (smoke testing).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List ladder slugs that expose build_showcase().",
    )
    return parser


def _list_showcase_slugs() -> list[str]:
    slugs: list[str] = []
    for entry in LADDER:
        try:
            load_showcase_game(entry.slug)
        except (ImportError, AttributeError, ValueError):
            continue
        slugs.append(entry.slug)
    return slugs


def main(argv: list[str] | None = None) -> int:
    """Record one game's segmented showcase."""
    args = _build_parser().parse_args(argv)
    if args.list:
        for slug in _list_showcase_slugs():
            print(slug)
        return 0
    if args.slug is None:
        parser.error("slug is required unless --list is used")

    game = load_showcase_game(args.slug)
    output = (
        default_output_path(game)
        if args.output is None
        else Path(args.output).resolve()
    )
    manifest = record_showcase(
        game,
        output,
        frame_stride=args.frame_stride,
        scale=args.scale,
        fps=args.fps,
        card_frames=args.card_frames,
        max_clips=args.max_clips,
    )
    print(f"video={output}")
    print(f"manifest={manifest['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
