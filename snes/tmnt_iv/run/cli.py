"""Peek ``--flag`` parsers shared by segment / bridge / Clean CLIs."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence


def peek_required_int(
    flag: str,
    choices: Iterable[int],
    argv: Sequence[str] | None,
    *,
    description: str,
    help: str,
    epilog: str | None = None,
) -> tuple[int, list[str]]:
    """Peek ``--flag N``. If missing, parse with required=True (exits). Return (value, rest)."""
    dest = flag.lstrip("-").replace("-", "_")
    choice_list = list(choices)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(flag, type=int, choices=choice_list)
    peeked, rest = pre.parse_known_args(argv)
    value = getattr(peeked, dest)
    if value is not None:
        return int(value), list(rest)
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument(
        flag,
        type=int,
        required=True,
        choices=choice_list,
        help=help,
    )
    parser.parse_args(argv)
    raise SystemExit(2)
