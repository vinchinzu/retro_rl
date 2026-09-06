"""Public Level 9 Survival-spine seam.

The L8 leave is measured (rr-6o7.3: ``--through level8`` power-on green), so
``continue_level9_spine`` carries ``MEASURED_POST_L8_HANDOFF`` -- but every
natural L9 chapter is still a fail-closed marker (the OW walk to Spectacle
Rock, the bomb entry, the Old Man TF gate, the interior on natural resources).
Reaching the seam is not greening it.
"""

from __future__ import annotations

from zelda_i.level9.dungeon import MEASURED_POST_L8_HANDOFF, PostLevel8Handoff
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
    handoff: PostLevel8Handoff = MEASURED_POST_L8_HANDOFF,
) -> None:
    """Attach L9 after L8; unresolved natural chapters fail on their first frame."""
    attach_hops(
        env,
        run,
        l9_hops(env, handoff=handoff),
        through=through,
        run_stages=run_stages,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    )
