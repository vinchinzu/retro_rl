"""Run the TMNT local Ollama tool-agent grind harness.

This is the multi-turn tool scaffold (not single-shot propose calls).

Examples::

    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \\
      uv run python -m tmnt_iv.scripts.run_local_grind_agent \\
        --focus slash --max-trials 2

Inspect::

    tmnt_iv/recordings/local_grind_agent/agent_trace.jsonl
    tmnt_iv/recordings/local_grind_agent/summary.json
    tmnt_iv/recordings/local_grind_agent/agent_result.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tmnt_iv.local_grind.agent import AgentConfig, run_grind_agent  # noqa: E402
from tmnt_iv.local_grind.schema import DEFAULT_TARGETS  # noqa: E402
from tmnt_iv.paths import GAME_DIR  # noqa: E402

def main(argv: list[str] | None = None) -> int:
    labels = [t.label for t in DEFAULT_TARGETS]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemma4:12b")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--focus", choices=labels, default="slash")
    parser.add_argument("--max-trials", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=GAME_DIR / "recordings" / "local_grind_agent",
    )
    parser.add_argument("--min-rel-gain", type=float, default=0.01)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)

    result = run_grind_agent(
        AgentConfig(
            model=args.model,
            host=args.host,
            focus=args.focus,
            out_dir=args.out_dir,
            max_trials=args.max_trials,
            max_turns=args.max_turns,
            min_rel_gain=args.min_rel_gain,
            timeout=args.timeout,
        )
    )
    print("---")
    print(f"finished: {result.finished}")
    print(f"turns: {result.turns}  tool_calls: {result.tool_calls}")
    print(f"trials: {result.trials}  keeps: {result.keeps}")
    print(f"summary: {result.summary}")
    print(f"artifacts: {result.out_dir}")
    print(f"  trace:   {result.out_dir / 'agent_trace.jsonl'}")
    print(f"  summary: {result.out_dir / 'summary.json'}")
    print(f"  result:  {result.out_dir / 'agent_result.json'}")
    return 0 if result.finished and result.tool_calls > 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
