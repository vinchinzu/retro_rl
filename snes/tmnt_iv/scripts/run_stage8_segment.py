"""Headless Stage 8 segment: chain Mode-7 waves from Stage8."""

from __future__ import annotations

from tmnt_iv.segment import STAGE_SPECS, run_stage8_segment, segment_main

__all__ = ["run_stage8_segment"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the Stage 8 multi-wave segment runner."""
    return segment_main(STAGE_SPECS[8], argv)


if __name__ == "__main__":
    raise SystemExit(main())
