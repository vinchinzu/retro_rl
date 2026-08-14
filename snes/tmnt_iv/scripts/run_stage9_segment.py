"""Headless Stage 9 segment: chain Starbase waves from Stage9."""

from __future__ import annotations

from tmnt_iv.segment import STAGE_SPECS, run_stage9_segment, segment_main

__all__ = ["run_stage9_segment"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the Stage 9 multi-wave segment runner."""
    return segment_main(STAGE_SPECS[9], argv)


if __name__ == "__main__":
    raise SystemExit(main())
