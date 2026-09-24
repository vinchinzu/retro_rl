"""Public Level 8 Survival-spine seam.

The three public stops are chapters, not room-by-room CLI targets.  Wave A
defaults fail closed until the L7 handoff, bush burn, and interior topology
have live evidence.

Wired into ``zelda_i.spine.survival``: for an L8 ``--through`` target the L7
suffix is driven to its own last stop (``level7``) and L8 continues from
``MEASURED_POST_L7_HANDOFF``. Default hops are the reverse pond table plus the 0x42 west ring
(``PostLevel7ToBushController``); the south-shore fixture controller
stays fail-closed on the north strip.
"""

from __future__ import annotations

from typing import Any

from zelda_i.level8.dungeon import (
    LIVE_RECON_LEVEL8_TOPOLOGY,
    MEASURED_LEVEL8_CLEAR,
    MEASURED_LEVEL8_ENTRY_TOPOLOGY,
    Level8ClearEndpoint,
    Level8Topology,
)
from zelda_i.level8.entry import (
    LIVE_RECON_BUSH_BURN_TARGET,
    MEASURED_POST_L7_HANDOFF,
    BushBurnTarget,
    PostLevel7Handoff,
)
from zelda_i.level8.hops import l8_hops
from zelda_i.level8.suffix import (
    NATURAL_LINEAGE_LEVEL8_SUFFIX,
    Level8SuffixLineage,
)
from zelda_i.level8.overworld import L7_POND_VIA_SHOP_E5_HOPS
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

# No inventory top-up (rr-doua). Power-on, L7 leaves bombs=4 / keys=1. The
# magic-key ascent bombs 0x6E and 0x3E north (throwing more at 0x3E's
# Darknuts) and opens two key doors (0x4E->0x3E, 0x2E->0x1E) with the
# carried key plus 0x5E's. The return passage bombs 0x4C north, so the walk
# in buys a 20R pack at 0x44 (``L7_POND_VIA_SHOP_E5_HOPS``).

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
    handoff: PostLevel7Handoff = MEASURED_POST_L7_HANDOFF,
    post_l7_hops: tuple[ScreenHop, ...] = L7_POND_VIA_SHOP_E5_HOPS,
    burn_target: BushBurnTarget = LIVE_RECON_BUSH_BURN_TARGET,
    topology: Level8Topology = MEASURED_LEVEL8_ENTRY_TOPOLOGY,
    clear_endpoint: Level8ClearEndpoint = MEASURED_LEVEL8_CLEAR,
    suffix: Level8SuffixLineage = NATURAL_LINEAGE_LEVEL8_SUFFIX,
) -> None:
    """Attach the L8 chapter rows after a naturally completed Level 7.

    ``suffix`` is the rr-6o7.3 composition gate (``level8.suffix``).  As of
    rr-6o7.3 the default lineage is composable (the Gleeok suffix is
    spine-green from the power-on Magical-Key frontier) and ``clear_endpoint``
    is the measured OW ``0x6D`` leave, so ``--through level8`` runs the full
    ordered suffix and ``level8_clear_stop`` greens on the settled leave.
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
