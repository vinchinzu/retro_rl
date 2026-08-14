"""Headless Stage1 segment: chain Foot Clan waves until clear / boss / fail."""

from __future__ import annotations

from tmnt_iv.segment import STAGE_SPECS, run_stage1_segment, segment_main

__all__ = ["run_stage1_segment"]


def main(argv: list[str] | None = None) -> int:
    """CLI entry for the Stage 1 multi-wave segment runner."""
    return segment_main(STAGE_SPECS[1], argv)


if __name__ == "__main__":
    raise SystemExit(main())
