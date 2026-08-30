"""Public Level 7 Survival-spine seam."""

from __future__ import annotations

from zelda_i.level7.hops import l7_hops
from zelda_i.spine.hops import attach_hops

L7_THROUGH: tuple[str, ...] = (
    "level7-entry",
    "level7-red-candle",
    "level7",
)
L7_STOPS: dict[str, str] = {
    "level7-entry": "level7_entry",
    "level7-red-candle": "level7_red_candle",
    "level7": "level7_complete",
}


def continue_level7_spine(
    env,
    run,
    *,
    through: str,
    run_stages,
    room_timer=None,
    assist=None,
    on_frame=None,
) -> None:
    """Attach L7 after the natural L6 endpoint; current hypotheses fail closed."""
    if through not in L7_THROUGH:
        raise ValueError(f"unknown Level 7 through target: {through!r}")
    attach_hops(
        env,
        run,
        l7_hops(env),
        through=through,
        run_stages=run_stages,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    )


__all__ = ["L7_STOPS", "L7_THROUGH", "continue_level7_spine"]
