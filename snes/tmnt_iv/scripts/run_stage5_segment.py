"""Headless Stage 5 segment: chain Prehistoric waves from Stage5."""

from __future__ import annotations

from tmnt_iv.segment import STAGE_SPECS, run_stage5_segment, segment_main

__all__ = ["run_stage5_segment"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the Stage 5 multi-wave segment runner."""
    return segment_main(STAGE_SPECS[5], argv)


if __name__ == "__main__":
    raise SystemExit(main())
