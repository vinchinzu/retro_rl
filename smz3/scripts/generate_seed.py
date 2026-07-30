"""Generate an SMZ3 seed package (samus.link) and optionally build the combo ROM.

Examples:

  uv run python smz3/scripts/generate_seed.py --test
  uv run python smz3/scripts/generate_seed.py --seed 42 --name my_seed
  uv run python smz3/scripts/generate_seed.py --test --no-rom
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from smz3.seed import generate_seed, generate_test_seed  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate/reload the pinned test seed (1337, uncle sword, original morph)",
    )
    parser.add_argument("--seed", default=None, help="Numeric seed (empty = random)")
    parser.add_argument("--name", default=None, help="Seed package directory name")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default smz3/seeds/<name>)",
    )
    parser.add_argument(
        "--no-rom",
        action="store_true",
        help="Skip combo ROM build (metadata + patch only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --test, re-roll even if package exists",
    )
    parser.add_argument(
        "--smlogic",
        default="normal",
        choices=("normal", "hard"),
        help="Super Metroid logic",
    )
    parser.add_argument(
        "--sword",
        default="uncle",
        choices=("randomized", "uncle", "early"),
        help="Sword placement",
    )
    parser.add_argument(
        "--morph",
        default="original",
        choices=("randomized", "original", "early"),
        help="Morph placement",
    )
    args = parser.parse_args(argv)

    build_rom = not args.no_rom
    if args.test:
        pkg = generate_test_seed(force=args.force, build_rom=build_rom)
    else:
        settings = {
            "smlogic": args.smlogic,
            "swordlocation": args.sword,
            "morphlocation": args.morph,
            "goal": "defeatboth",
            "race": "false",
            "gamemode": "normal",
            "players": "1",
        }
        pkg = generate_seed(
            seed=args.seed,
            settings=settings,
            name=args.name,
            out_dir=args.out,
            build_rom=build_rom,
        )

    print(json.dumps(pkg.to_meta(), indent=2))
    print(f"Package: {pkg.directory}")
    if pkg.rom_path:
        print(f"ROM: {pkg.rom_path} ({pkg.rom_path.stat().st_size} bytes)")
    else:
        print("ROM: not built")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
