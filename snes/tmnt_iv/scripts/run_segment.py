"""Headless Stage N wave-chain runner.

  uv run python -m tmnt_iv.scripts.run_segment --stage 1
  uv run python -m tmnt_iv.scripts.run_segment --stage 8 --no-heal
"""

from __future__ import annotations

from tmnt_iv.run.cli import peek_required_int
from tmnt_iv.run.segment import STAGE_SPECS, segment_main


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``--stage 1..9`` then the shared segment flags."""
    stage, rest = peek_required_int(
        "--stage",
        list(STAGE_SPECS),
        argv,
        description="Headless TMNT IV stage segment (wave chain).",
        help="Human stage number 1–9 (RAM byte = N-1).",
        epilog="Pass --stage N --help for that stage's flags.",
    )
    return segment_main(STAGE_SPECS[stage], rest)


if __name__ == "__main__":
    raise SystemExit(main())
