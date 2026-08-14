"""Headless Stage 7 segment: chain waves from Stage7."""

from __future__ import annotations

from tmnt_iv.segment import STAGE_SPECS, run_stage7_segment, segment_main

__all__ = ["run_stage7_segment"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the Stage 7 multi-wave segment runner."""
    return segment_main(STAGE_SPECS[7], argv)


if __name__ == "__main__":
    raise SystemExit(main())
