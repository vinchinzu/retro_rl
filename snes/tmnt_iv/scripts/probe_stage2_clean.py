"""Stage 2 Clean proof: heal=none, pizza-only HP recovery, Alleycat clear.

No emergency HP writes. Natural pizza (char 0x30) is allowed. Writes JSON
under ``recordings/stage2_clean_track/``.

Alleycat Blues (stage byte 1) adapter over ``tmnt_iv.clean_suite``. See
``docs/CLEAN_PLAYBOOK.md``.

Status (2026-07-27): early/mid waves still fail heal=none; Metalhead and
pre-boss (w17) already clear pizza-only. Suite tracks both.

  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
    uv run python -m tmnt_iv.scripts.probe_stage2_clean --suite

Single entry:

  uv run python -m tmnt_iv.scripts.probe_stage2_clean --state Stage2
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --state Boss2
  uv run python -m tmnt_iv.scripts.probe_stage2_clean --from-stage1-clear
"""

from __future__ import annotations

from typing import Any

from tmnt_iv.clean_suite import (
    STAGE2_CLEAN,
    clean_main,
    run_clean_probe as _run_clean_probe,
    run_suite as _run_suite,
)

__all__ = ["run_clean_probe", "run_suite", "main"]


def run_clean_probe(
    *,
    state_name: str = STAGE2_CLEAN.default_state,
    max_frames: int = STAGE2_CLEAN.default_max_frames,
    stop_stage_gt: int = STAGE2_CLEAN.stop_stage_gt,
    from_stage1_clear: bool = False,
) -> dict[str, Any]:
    """Fight with zero HP assists until stage advance / death / timeout."""
    return _run_clean_probe(
        STAGE2_CLEAN,
        state_name=state_name,
        max_frames=max_frames,
        stop_stage_gt=stop_stage_gt,
        from_stage1_clear=from_stage1_clear,
    )


def run_suite(*, max_frames: int = STAGE2_CLEAN.cli_max_frames) -> dict[str, Any]:
    """Run Clean probes across checkpoint entries + Stage1_Clear bridge."""
    return _run_suite(STAGE2_CLEAN, max_frames=max_frames)


def main(argv: list[str] | None = None) -> int:
    """CLI entry for Stage 2 Clean (pizza-only) probes."""
    return clean_main(STAGE2_CLEAN, argv, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
