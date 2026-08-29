"""Clean (heal=none, pizza-only) proof for stages 1–3.

  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
    uv run python -m tmnt_iv.scripts.probe_clean --stage 1 --suite

  uv run python -m tmnt_iv.scripts.probe_clean --stage 1 --power-on
  uv run python -m tmnt_iv.scripts.probe_clean --stage 2 --from-stage1-clear
  uv run python -m tmnt_iv.scripts.probe_clean --stage 3 --state LiveHardStage3
"""

from __future__ import annotations

from tmnt_iv.run.cli import peek_required_int
from tmnt_iv.run.clean_suite import CLEAN_SPECS, clean_main

_STAGES = sorted(byte + 1 for byte in CLEAN_SPECS)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``--stage 1..3`` then the shared Clean flags."""
    stage, rest = peek_required_int(
        "--stage",
        _STAGES,
        argv,
        description="Clean (pizza-only) probes for TMNT IV stages 1–3.",
        help="Human stage number 1–3 (RAM byte = N-1).",
        epilog="Pass --stage N --help for that stage's flags.",
    )
    return clean_main(CLEAN_SPECS[stage - 1], rest)


if __name__ == "__main__":
    raise SystemExit(main())
