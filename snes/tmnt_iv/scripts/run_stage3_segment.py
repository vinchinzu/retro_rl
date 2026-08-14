"""Headless Stage 3 segment: chain Sewer waves from Stage3."""

from __future__ import annotations

from tmnt_iv.segment import STAGE_SPECS, run_stage3_segment, segment_main

__all__ = ["run_stage3_segment"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the Stage 3 multi-wave segment runner."""
    return segment_main(STAGE_SPECS[3], argv)


if __name__ == "__main__":
    raise SystemExit(main())
