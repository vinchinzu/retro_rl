"""Bridge previous-stage clear → fight-ready Stage 2 or 3.

  uv run python -m tmnt_iv.scripts.run_bridge --to 2
  uv run python -m tmnt_iv.scripts.run_bridge --to 3
"""

from __future__ import annotations

from tmnt_iv.run.cli import peek_required_int
from tmnt_iv.run.bridge import BRIDGE_SPECS, bridge_main


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``--to 2|3`` then the shared bridge flags."""
    dest, rest = peek_required_int(
        "--to",
        list(BRIDGE_SPECS),
        argv,
        description="Bridge previous clear → fight-ready Stage 2 or 3.",
        help="Destination human stage (2=Alleycat, 3=Sewer).",
        epilog="Pass --to N --help for that hop's flags.",
    )
    return bridge_main(BRIDGE_SPECS[dest], rest)


if __name__ == "__main__":
    raise SystemExit(main())
