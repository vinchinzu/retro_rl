"""Public Level 7 Survival-spine seam.

Wired into ``zelda_i.spine.survival``: for an L7 ``--through`` target the L6
suffix is driven to ``level6-exit`` (the measured post-fanfare OW return, screen
``0x22`` ``(112,125)`` TF ``0x3F``) and L7 continues from there with
``MEASURED_POST_L6_EXIT`` (a shared ``OverworldHandoff``) as the handoff.

``MEASURED_POST_L6_EXIT.verified`` is ``False`` (``selected_item`` not captured
that run + the 42R->60R Bait gap), so the post-L6 controller refuses every
frame and ``--through level7-entry`` runs the whole continuous tape, then stops
fail-closed at ``level7_post_l6_overworld`` (``handoff_unmeasured``).  The
fixture-live ``0x22 -> 0x25`` bait prefix is already wired; it starts walking
the moment the handoff verifies.
"""

from __future__ import annotations

from zelda_i.level7.entry import MEASURED_POST_L6_EXIT
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
    """Attach L7 after the measured L6 fanfare exit; hypotheses fail closed."""
    if through not in L7_THROUGH:
        raise ValueError(f"unknown Level 7 through target: {through!r}")
    attach_hops(
        env,
        run,
        l7_hops(env, handoff=MEASURED_POST_L6_EXIT),
        through=through,
        run_stages=run_stages,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    )


__all__ = ["L7_STOPS", "L7_THROUGH", "continue_level7_spine"]
