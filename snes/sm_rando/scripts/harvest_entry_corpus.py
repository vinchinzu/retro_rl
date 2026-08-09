"""Harvest the real natural Landing Site train/eval state distribution."""

from __future__ import annotations

import argparse
from pathlib import Path

from sm_rando.entry_corpus import (
    LANDING_CORPUS_MANIFEST,
    LANDING_CORPUS_SIZE,
    corpus_summary,
    harvest_landing_entry_corpus,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=LANDING_CORPUS_SIZE)
    parser.add_argument("--output", type=str, default=str(LANDING_CORPUS_MANIFEST))
    args = parser.parse_args(argv)
    corpus = harvest_landing_entry_corpus(
        count=args.count,
        output_path=Path(args.output),
    )
    summary = corpus_summary(corpus)
    print(
        f"states={summary['states']} train={summary['train']} "
        f"eval={summary['eval']} parity={summary['frame_parities']} "
        f"game_states={summary['game_states']} manifest={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
