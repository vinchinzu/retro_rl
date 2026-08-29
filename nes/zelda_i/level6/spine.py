"""Survival-spine L6 catalog + continue. Hop rows live in hops/suffix modules."""

from __future__ import annotations

from zelda_i.level6.hops import l6_prefix
from zelda_i.level6.spine_suffix import l6_suffix_hops
from zelda_i.spine.hops import attach_hops

__all__ = [
    "L6_STOPS",
    "L6_THROUGH",
    "continue_level6_spine",
]


def _l6_rows():
    return l6_prefix(None) + l6_suffix_hops()


L6_THROUGH = tuple(hop.through for hop in _l6_rows())
L6_STOPS = {hop.through: hop.stop for hop in _l6_rows()}


def continue_level6_spine(
    env,
    run,
    *,
    through: str,
    run_stages,
    room_timer=None,
    assist=None,
    on_frame=None,
) -> None:
    """Attach L6 suffix after L5 TF. Mutates ``run``; caller returns it."""
    attach_hops(
        env,
        run,
        l6_prefix(env) + l6_suffix_hops(),
        through=through,
        run_stages=run_stages,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    )
