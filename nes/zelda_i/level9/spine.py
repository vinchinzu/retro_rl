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
    "SPINE_L9_RETOPUP",
    "L9_STOPS",
    "L9_THROUGH",
    "continue_level9_spine",
]


def _l9_rows():
    return l9_hops(None)


L9_THROUGH = tuple(hop.through for hop in _l9_rows())
L9_STOPS = {hop.through: hop.stop for hop in _l9_rows()}


# Bomb-spending L9 gates. The Patra join's contract needs bombs >= 1 (the
# 0x10 bomb hole) and the silver-arrow chapter spends them all; the entry
# contract needs bombs > 0 after the Spectacle Rock blast, and continuous
# power-on run 10 arrived there with one. Same documented Survival count
# top-up as SPINE_L8_RETOPUP (raise-only), at the chapter gates. Not Clean.
SPINE_L9_RETOPUP: frozenset[str] = frozenset(
    {
        "level9_spectacle_rock_bomb",
        "level9_natural_silver_arrows",
        "level9_natural_patra_join",
    }
)


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
        retopup=(
            SPINE_L9_RETOPUP if getattr(run, "allow_pokes", True) else frozenset()
        ),
    )
