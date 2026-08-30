"""Public fail-closed Level 9 Survival-spine seam."""

from __future__ import annotations

from zelda_i.level9.hops import l9_hops
from zelda_i.spine.hops import attach_hops

__all__ = [
    "L9_STOPS",
    "L9_THROUGH",
    "continue_level9_spine",
]


def _l9_rows():
    return l9_hops(None)


L9_THROUGH = tuple(hop.through for hop in _l9_rows())
L9_STOPS = {hop.through: hop.stop for hop in _l9_rows()}


def continue_level9_spine(
    env,
    run,
    *,
    through: str,
    run_stages,
    room_timer=None,
    assist=None,
    on_frame=None,
) -> None:
    """Attach L9 after L8; unresolved natural chapters fail on their first frame."""
    attach_hops(
        env,
        run,
        l9_hops(env),
        through=through,
        run_stages=run_stages,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    )
