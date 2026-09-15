"""Overworld HC pickup recon: raft_heart 0x2F via dock 0x3F (rr-ps7.4.1).

**Partial live recon only. Not spliced into the spine. Not a route
promotion.** ``overworld/locations.py``'s ``raft_heart`` row stays
``EVIDENCE_SOURCE`` until the dock/launch/pickup leg is live-verified.

Grid: screen id = ``(row << 4) | col`` (16 cols x 8 rows,
``overworld/graph.py``). Dock ``0x3F`` = col 15, row 3. Island
``0x2F`` = col 15, row 2 (directly north of the dock — same shape as
the live L4 dock ``0x55`` -> island ``0x45``, ``level4/overworld.py``).
Doc hypothesis (``OVERWORLD_DOORS.md``): from start ``0x77`` (col 7,
row 7), 8 screens east then 4 north lands on ``0x3F``.

Live (2026-09-14, ``Level3ExitOverworld`` fixture: post-L3 OW ``0x74``,
raft=1, tf=0x04, heart_containers=7, health full, via
``retro_harness.env.resync_custom_state`` -- not power-on. Driven with
the shared ``OverworldPathController`` (``overworld/path.py``), no RAM
writes)::

    0x74 (128,125) --LEFT align_y=141--> 0x73         [hop0, 217f]
    0x73            --UP   align_x=128--> 0x63         [hop1, 617f total]
    0x63            --RIGHT y_band 145-155--> 0x64      [hop2, 833f total]
    0x64            --RIGHT align_y=141--> 0x65         [hop3, 1205f total]
    0x65            --RIGHT y_band 140-162--> 0x66      [hop4, needs
                                                          SWING_PERIOD/
                                                          SWING_HOLD below]

hop0/hop1 replay the live-verified prefix of ``LEVEL4_HOPS_FROM_POST_L3``
(``level4/overworld.py``, rr-0fx) -- 0x74 is the L3 door mouth itself
(post-dungeon-leave-on-mouth-tile); any direction but LEFT re-entered the
dungeon in this same probe.

0x65 (col 5, row 6) is a river-and-bridge screen with 3 land enemies
camped at the east-bank bridge exit; the default hop tuning
(``swing_period=10, swing_hold=3, farm_below_hearts=3``) died there
3/3 (frames 2845, 2867, 3062, always at the bridge/east-bank). Raising
swing aggression and disabling the low-heart farm diversion (it does not
defend well mid-crossing) survived it 1/1: ``SWING_PERIOD=6``,
``SWING_HOLD=4``, ``FARM_BELOW_HEARTS=0``.

0x66 (col 6, row 6) is a dense ~9-enemy Armos-shaped formation with
coastal (beach-corner) borders. Every exit tried from it died or stalled
under those same tuned params:

- RIGHT y_band (140,162): alive, stalls forever at exactly
  ``(232, 165)`` (screen-edge threshold, ``overworld/common.py``
  ``EDGE_EAST_X``) -- reads as a real wall, not a slow crossing.
- RIGHT y_band (163,183): alive, same stall at ``(232, 165)``.
- RIGHT y_band (196,210): dies at ``(16, 189)`` (west side, never
  reached the east band).
- UP align_x=32 (toward 0x56): dies at ``(32, 85)``.
- DOWN align_x=16 (toward 0x76, back onto start's row 7): alive to
  ``(16, 189)`` under a 3000f cap, dies under 3535f (chip damage, not a
  hard block).

None of these confirm a screen-5 (index 5) hop out of 0x66. The
row-7-from-start hypothesis (``0x77`` -> E x8 -> ``0x3F``) was not
attempted independently; ``0x74``'s door-mouth block (LEFT-only) means
reaching row 7 east of the mouth would need its own bypass, not a
straight replay of hop0.

**Frontier: live through 0x66 (hop index 4 of an unknown-length chain).
Dock 0x3F, launch tile/direction into 0x2F, and the HC touch are all
still hypothesis.**
"""

from __future__ import annotations

from zelda_i.anchors import SCREEN_RAFT_HEART_DOCK, SCREEN_RAFT_HEART_ISLAND
from zelda_i.overworld.graph import ScreenHop, path_screens_from_hops

SOURCE_HYPOTHESIS = True  # only hop0-hop4 below are live; the rest is planning.

SCREEN_POST_L3_RETURN = 0x74  # == level4.overworld.SCREEN_POST_L3_RETURN
RAFT_HEART_DOCK = SCREEN_RAFT_HEART_DOCK  # 0x3F
RAFT_HEART_ISLAND = SCREEN_RAFT_HEART_ISLAND  # 0x2F

# Live-confirmed prefix only (2026-09-14, rr-ps7.4.1). Do not extend this
# tuple without a fresh live probe -- see module docstring for what was
# tried and failed past 0x66.
RAFT_HEART_HOPS_LIVE: tuple[ScreenHop, ...] = (
    ScreenHop(0x73, "LEFT", align_y=141),
    ScreenHop(0x63, "UP", align_x=128),
    ScreenHop(0x64, "RIGHT", y_band_lo=145, y_band_hi=155),
    ScreenHop(0x65, "RIGHT", align_y=141),
    ScreenHop(0x66, "RIGHT", y_band_lo=140, y_band_hi=162),
)
RAFT_HEART_SCREENS_LIVE: tuple[int, ...] = path_screens_from_hops(
    SCREEN_POST_L3_RETURN, RAFT_HEART_HOPS_LIVE
)
assert RAFT_HEART_SCREENS_LIVE[0] == SCREEN_POST_L3_RETURN
assert RAFT_HEART_SCREENS_LIVE[-1] == 0x66

# Tuning required to survive hop3->hop4 (0x65 bridge + 0x66 Armos field)
# live, 2026-09-14. The shared-default tuning (period=10, hold=3,
# farm_below_hearts=3) died 3/3 at the 0x65 bridge/east-bank chokepoint.
RAFT_HEART_SWING_PERIOD = 6
RAFT_HEART_SWING_HOLD = 4
RAFT_HEART_FARM_BELOW_HEARTS = 0


def raft_heart_recon_report() -> dict[str, object]:
    """Machine-readable partial-recon summary. Not a route/planning claim."""
    return {
        "bead": "rr-ps7.4.1",
        "status": "partial_live_recon",
        "source_hypothesis": SOURCE_HYPOTHESIS,
        "route_eligible": False,
        "natural_entry": False,
        "fixture_source": "Level3ExitOverworld",
        "screens": {
            "post_l3_return": hex(SCREEN_POST_L3_RETURN),
            "dock_hyp": hex(RAFT_HEART_DOCK),
            "island_hyp": hex(RAFT_HEART_ISLAND),
            "live_frontier": hex(RAFT_HEART_SCREENS_LIVE[-1]),
        },
        "live_hops": [
            {"target": hex(h.target), "dir": h.direction} for h in RAFT_HEART_HOPS_LIVE
        ],
        "tuning": {
            "swing_period": RAFT_HEART_SWING_PERIOD,
            "swing_hold": RAFT_HEART_SWING_HOLD,
            "farm_below_hearts": RAFT_HEART_FARM_BELOW_HEARTS,
        },
        "not_verified": [
            "hop_from_0x66_onward",
            "dock_0x3F_arrival",
            "raft_launch_tile_and_direction",
            "island_0x2F_arrival",
            "heart_container_touch",
        ],
        "docs": "nes/zelda_i/docs/OVERWORLD_DOORS.md",
    }


__all__ = [
    "RAFT_HEART_DOCK",
    "RAFT_HEART_FARM_BELOW_HEARTS",
    "RAFT_HEART_HOPS_LIVE",
    "RAFT_HEART_ISLAND",
    "RAFT_HEART_SCREENS_LIVE",
    "RAFT_HEART_SWING_HOLD",
    "RAFT_HEART_SWING_PERIOD",
    "SCREEN_POST_L3_RETURN",
    "SOURCE_HYPOTHESIS",
    "raft_heart_recon_report",
]
