"""Farm TMNT damage/speed grind trials to a local Ollama model.

Examples::

    # Slash probe, 3 trials on gemma4:12b (default)
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
      uv run python -m tmnt_iv.scripts.run_local_grind --trials 3

    # Technodrome tank focus
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
      uv run python -m tmnt_iv.scripts.run_local_grind \\
        --focus technodrome_tank --trials 5

    # Offline heuristic proposals (no Ollama)
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
      uv run python -m tmnt_iv.scripts.run_local_grind \\
        --skip-model --trials 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tmnt_iv.local_grind.runner import GrindLoopConfig, run_grind_loop  # noqa: E402
from tmnt_iv.local_grind.schema import DEFAULT_TARGETS  # noqa: E402
from tmnt_iv.paths import GAME_DIR  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    labels = [t.label for t in DEFAULT_TARGETS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gemma4:12b",
        help="Ollama model name (default: gemma4:12b)",
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Ollama base URL",
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--focus",
        choices=labels,
        default="slash",
        help="Cheap probe target for this grind session",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=GAME_DIR / "recordings" / "local_grind",
    )
    parser.add_argument(
        "--min-rel-gain",
        type=float,
        default=0.01,
        help="Required relative score improvement to KEEP (default 1%%)",
    )
    parser.add_argument(
        "--no-vision-review",
        action="store_true",
        help="Skip screenshot review calls (propose-only)",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Use deterministic heuristic proposals (no Ollama)",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    records = run_grind_loop(
        GrindLoopConfig(
            model=args.model,
            host=args.host,
            trials=args.trials,
            focus_target=args.focus,
            out_dir=args.out_dir,
            min_rel_gain=args.min_rel_gain,
            use_vision_review=not args.no_vision_review,
            skip_model=args.skip_model,
            timeout=args.timeout,
        )
    )
    keeps = sum(1 for r in records if r.decision.value == "keep")
    errors = sum(1 for r in records if r.decision.value == "error")
    print(f"done: {len(records)} trials, {keeps} keep, {errors} error")
    print(f"artifacts: {args.out_dir}")
    return 1 if errors == len(records) and records else 0


if __name__ == "__main__":
    raise SystemExit(main())
