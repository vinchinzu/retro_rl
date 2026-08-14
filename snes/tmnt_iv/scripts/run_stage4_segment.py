"""Headless Stage 4 segment: chain Technodrome waves from Stage4."""

from __future__ import annotations

from tmnt_iv.segment import STAGE_SPECS, run_stage4_segment, segment_main

__all__ = ["run_stage4_segment"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the Stage 4 multi-wave segment runner."""
    return segment_main(STAGE_SPECS[4], argv)


if __name__ == "__main__":
    raise SystemExit(main())
