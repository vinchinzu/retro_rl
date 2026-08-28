"""Stage 3 Clean proof: heal=none, pizza-only HP recovery, Sewer clear.

No emergency HP writes. Natural pizza (char 0x30) is allowed. Writes JSON
under ``recordings/stage3_clean_track/``.

Sewer Surfin' (stage byte 2) adapter over ``tmnt_iv.clean_suite``. See
``docs/CLEAN_PLAYBOOK.md``.

**Entry notes (2026-07-27):**

- ``Stage3`` / ``Boss3`` saves are last-life (lives=0) and die on the
  post-kill ``event=0x0B`` fade even after Rat King HP hits 0 — known
  checkpoint artifact (STATUS: dies ~444f into 0x0B). Prefer
  ``LiveHardStage3`` (lives=2) for full stage_advance proof.
- Spike props char ``0x1C``/``0x2C`` (−16) are in ``HAZARD_CHAR_IDS``;
  policy jumps when near (see ``SewerSpikeAvoid``).

  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
    uv run python -m tmnt_iv.scripts.probe_stage3_clean --suite

Single entry:

  uv run python -m tmnt_iv.scripts.probe_stage3_clean --state LiveHardStage3
  uv run python -m tmnt_iv.scripts.probe_stage3_clean --state Boss3
  uv run python -m tmnt_iv.scripts.probe_stage3_clean --from-stage2-clear
"""

from __future__ import annotations

from typing import Any

from tmnt_iv.clean_suite import (
    STAGE3_CLEAN,
    clean_main,
    run_clean_probe as _run_clean_probe,
    run_suite as _run_suite,
)

__all__ = ["run_clean_probe", "run_suite", "main"]


def run_clean_probe(
    *,
    state_name: str = STAGE3_CLEAN.default_state,
    max_frames: int = STAGE3_CLEAN.default_max_frames,
    stop_stage_gt: int = STAGE3_CLEAN.stop_stage_gt,
    from_stage2_clear: bool = False,
) -> dict[str, Any]:
    """Fight with zero HP assists until stage advance / death / timeout."""
    return _run_clean_probe(
        STAGE3_CLEAN,
        state_name=state_name,
        max_frames=max_frames,
        stop_stage_gt=stop_stage_gt,
        from_stage2_clear=from_stage2_clear,
    )


def run_suite(*, max_frames: int = STAGE3_CLEAN.cli_max_frames) -> dict[str, Any]:
    """Run Clean probes across checkpoint entries + Stage2_Clear bridge."""
    return _run_suite(STAGE3_CLEAN, max_frames=max_frames)


def main(argv: list[str] | None = None) -> int:
    """CLI entry for Stage 3 Clean (pizza-only) probes."""
    return clean_main(STAGE3_CLEAN, argv, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
