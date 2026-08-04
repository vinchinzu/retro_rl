"""CLI to extract and link shared ROMs for oneshot ladder games."""

from __future__ import annotations

import argparse
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

from retro_harness.ladder import LADDER, LadderEntry, REPO_ROOT, entry_for
from retro_harness.env import setup_game_rom


@dataclass(frozen=True)
class SetupResult:
    """Outcome of wiring one ladder game's ROM."""

    slug: str
    ok: bool
    rom_path: Path | None = None
    error: str | None = None


def zip_for_entry(entry: LadderEntry, *, repo_root: Path = REPO_ROOT) -> Path:
    """Absolute path to the shared ROM zip for a ladder entry."""
    return repo_root / "roms" / "Super Nintendo" / entry.rom_zip


def setup_entry(entry: LadderEntry, *, repo_root: Path = REPO_ROOT) -> SetupResult:
    """Extract the shared zip and link it into the game integration."""
    from retro_harness.repo import resolve_game_dir

    try:
        game_dir = resolve_game_dir(entry.slug, root=repo_root)
    except FileNotFoundError:
        game_dir = repo_root / entry.slug
    zip_path = zip_for_entry(entry, repo_root=repo_root)
    if not zip_path.is_file():
        return SetupResult(
            slug=entry.slug,
            ok=False,
            error=f"missing zip: {zip_path}",
        )
    try:
        rom_path = setup_game_rom(
            shared_zip=zip_path,
            game_dir=game_dir,
            integration_name=entry.integration,
        )
    except (OSError, zipfile.BadZipFile, FileNotFoundError) as exc:
        return SetupResult(slug=entry.slug, ok=False, error=str(exc))
    return SetupResult(slug=entry.slug, ok=True, rom_path=rom_path)


def setup_slugs(
    slugs: list[str] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
) -> list[SetupResult]:
    """Set up ROMs for selected slugs, or the full ladder when slugs is None."""
    if slugs is None:
        entries = list(LADDER)
    else:
        entries = [entry_for(slug) for slug in slugs]
    return [setup_entry(entry, repo_root=repo_root) for entry in entries]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract and link shared ROMs for oneshot ladder games.",
    )
    parser.add_argument(
        "slugs",
        nargs="*",
        help="Optional game directory slugs (default: all ladder entries).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print ladder slugs and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run ROM setup for ladder games. Returns process exit code."""
    args = _build_parser().parse_args(argv)
    if args.list:
        for entry in LADDER:
            print(f"{entry.rank:2d}  {entry.slug}  {entry.integration}")
        return 0

    results = setup_slugs(args.slugs or None)
    failed = 0
    for result in results:
        if result.ok:
            print(f"ok  {result.slug}: {result.rom_path}")
        else:
            failed += 1
            print(f"fail  {result.slug}: {result.error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
