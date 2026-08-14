"""Headless Stage 2 segment: chain Alleycat waves from Stage2."""

from __future__ import annotations

from tmnt_iv.segment import STAGE_SPECS, run_stage2_segment, segment_main

__all__ = ["run_stage2_segment"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the Stage 2 multi-wave segment runner."""
    return segment_main(STAGE_SPECS[2], argv)


if __name__ == "__main__":
    raise SystemExit(main())
