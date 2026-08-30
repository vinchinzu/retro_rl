"""Public Level 8 Survival-spine seam.

The three public stops are chapters, not room-by-room CLI targets.  Wave A
defaults fail closed until the L7 handoff, bush burn, and interior topology
have live evidence.
"""

from __future__ import annotations

from zelda_i.level8.dungeon import (
    UNOBSERVED_LEVEL8_CLEAR,
    UNOBSERVED_LEVEL8_TOPOLOGY,
    Level8ClearEndpoint,
    Level8Topology,
)
from zelda_i.level8.entry import (
    UNMEASURED_POST_L7_HANDOFF,
    UNVERIFIED_BUSH_BURN_TARGET,
    BushBurnTarget,
    PostLevel7Handoff,
)
from zelda_i.level8.hops import l8_hops
from zelda_i.overworld.graph import ScreenHop
from zelda_i.spine.hops import attach_hops

__all__ = [
    "L8_STOPS",
    "L8_THROUGH",
    "continue_level8_spine",
]

L8_THROUGH: tuple[str, ...] = (
    "level8-entry",
    "level8-magic-key",
    "level8",
)
L8_STOPS: dict[str, str] = {
    "level8-entry": "level8_entry_live",
    "level8-magic-key": "level8_magic_key_natural",
    "level8": "level8_triforce_0x80",
}


def continue_level8_spine(
    env,
    run,
    *,
    through: str,
    run_stages,
    room_timer=None,
    assist=None,
    on_frame=None,
    handoff: PostLevel7Handoff = UNMEASURED_POST_L7_HANDOFF,
    post_l7_hops: tuple[ScreenHop, ...] = (),
    burn_target: BushBurnTarget = UNVERIFIED_BUSH_BURN_TARGET,
    topology: Level8Topology = UNOBSERVED_LEVEL8_TOPOLOGY,
    clear_endpoint: Level8ClearEndpoint = UNOBSERVED_LEVEL8_CLEAR,
) -> None:
    """Attach the L8 chapter rows after a naturally completed Level 7."""
    if through not in L8_THROUGH:
        raise ValueError(f"unknown Level 8 through target: {through!r}")
    attach_hops(
        env,
        run,
        l8_hops(
            env,
            handoff=handoff,
            post_l7_hops=post_l7_hops,
            burn_target=burn_target,
            topology=topology,
            clear_endpoint=clear_endpoint,
        ),
        through=through,
        run_stages=run_stages,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    )
