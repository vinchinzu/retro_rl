"""Public Level 8 Survival-spine seam.

The three public stops are chapters, not room-by-room CLI targets.  Wave A
defaults fail closed until the L7 handoff, bush burn, and interior topology
have live evidence.

Wired into ``zelda_i.spine.survival``: for an L8 ``--through`` target the L7
suffix is driven to its own last stop (``level7``) and L8 continues from there
with ``UNMEASURED_POST_L7_HANDOFF``, so the entry chapter stops on
``post_l7_handoff_unmeasured`` until rr-8t4.3 measures the real L7 leave.
Reaching the seam is not the same as greening it.
"""

from __future__ import annotations

from typing import Any

from zelda_i.level8.dungeon import (
    LIVE_RECON_LEVEL8_TOPOLOGY,
    UNOBSERVED_LEVEL8_CLEAR,
    UNOBSERVED_LEVEL8_TOPOLOGY,
    Level8ClearEndpoint,
    Level8Topology,
)
from zelda_i.level8.entry import (
    LIVE_RECON_BUSH_BURN_TARGET,
    UNMEASURED_POST_L7_HANDOFF,
    UNVERIFIED_BUSH_BURN_TARGET,
    BushBurnTarget,
    PostLevel7Handoff,
)
from zelda_i.level8.hops import l8_hops
from zelda_i.level8.suffix import (
    FIXTURE_LINEAGE_LEVEL8_SUFFIX,
    Level8SuffixLineage,
)
from zelda_i.overworld.graph import ScreenHop
from zelda_i.spine.hops import attach_hops

__all__ = [
    "L8_STOPS",
    "L8_THROUGH",
    "LIVE_RECON_L8_OVERRIDES",
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

# Disclosed fixture-live recon (rr-6o7.1), for an explicit opt-in caller only:
# ``continue_level8_spine(..., **LIVE_RECON_L8_OVERRIDES)`` or
# ``run_survival_spine(..., level8_overrides=LIVE_RECON_L8_OVERRIDES)``.  The
# default spine passes none of it.  Both constants keep ``route_eligible=False``
# (fixture stand, not a natural post-L7 walk), so ``level8_entry_stop`` still
# refuses and no L8 stop can green off this bundle; it only lets a recon caller
# replay the observed burn aim and entry room instead of the empty defaults.
LIVE_RECON_L8_OVERRIDES: dict[str, Any] = {
    "burn_target": LIVE_RECON_BUSH_BURN_TARGET,
    "topology": LIVE_RECON_LEVEL8_TOPOLOGY,
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
    suffix: Level8SuffixLineage = FIXTURE_LINEAGE_LEVEL8_SUFFIX,
) -> None:
    """Attach the L8 chapter rows after a naturally completed Level 7.

    ``suffix`` is the rr-6o7.3 composition gate (``level8.suffix``).  The
    default lineage is fixture-only, so the clear chapter keeps its
    fail-closed ``level8_return_passage`` row.
    """
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
            suffix=suffix,
        ),
        through=through,
        run_stages=run_stages,
        room_timer=room_timer,
        assist=assist,
        on_frame=on_frame,
    )
