"""Headless Stage 6 segment: chain waves from Stage6."""

from __future__ import annotations

from tmnt_iv.segment import STAGE_SPECS, run_stage6_segment, segment_main

__all__ = ["run_stage6_segment"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the Stage 6 multi-wave segment runner."""
    return segment_main(STAGE_SPECS[6], argv)


if __name__ == "__main__":
    raise SystemExit(main())
