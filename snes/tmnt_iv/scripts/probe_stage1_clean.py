"""Stage 1 Clean proof: heal=none, pizza-only HP recovery, stage advance.

No emergency HP writes. Natural pizza (char 0x30) is allowed. Writes JSON
under ``recordings/stage1_clean_track/``.

Later-stage Clean suites add a ``CleanProbeSpec`` in ``tmnt_iv.clean_suite``
and a thin CLI adapter here. Do not copy the heal=none loop.
Do not relearn Stage 1 traps; extend stage allowlists carefully.

Path-RNG suite (checkpoint + Baxter + power-on):

  SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
    uv run python -m tmnt_iv.scripts.probe_stage1_clean --suite

Single entry:

  uv run python -m tmnt_iv.scripts.probe_stage1_clean --state Stage1
  uv run python -m tmnt_iv.scripts.probe_stage1_clean --power-on
"""

from __future__ import annotations

from typing import Any

from tmnt_iv.clean_suite import (
    STAGE1_CLEAN,
    clean_main,
    run_clean_probe as _run_clean_probe,
    run_suite as _run_suite,
)

__all__ = ["run_clean_probe", "run_suite", "main"]


def run_clean_probe(
    *,
    state_name: str = STAGE1_CLEAN.default_state,
    max_frames: int = STAGE1_CLEAN.default_max_frames,
    stop_stage_gt: int = STAGE1_CLEAN.stop_stage_gt,
    power_on: bool = False,
) -> dict[str, Any]:
    """Fight with zero HP assists until stage advance / death / timeout."""
    return _run_clean_probe(
        STAGE1_CLEAN,
        state_name=state_name,
        max_frames=max_frames,
        stop_stage_gt=stop_stage_gt,
        power_on=power_on,
    )


def run_suite(*, max_frames: int = STAGE1_CLEAN.cli_max_frames) -> dict[str, Any]:
    """Run Clean probes across checkpoint entries + power-on."""
    return _run_suite(STAGE1_CLEAN, max_frames=max_frames)


def main(argv: list[str] | None = None) -> int:
    """CLI entry for Stage 1 Clean (pizza-only) probes."""
    return clean_main(STAGE1_CLEAN, argv, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
