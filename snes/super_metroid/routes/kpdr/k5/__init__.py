"""K5 Alpha PB pure stack — post-Ice pre-Moat (``rr-dbu.8``).

Tape recon only (``docs/tasks/SM-SPEED-ICE-MOAT-HUMAN.md`` Phase B return +
Phase C). **Do not invent hops** outside this table.

Ice return (package ``routes/kpdr/ice/`` — shared predecessor)::

    Ice 0xA890
      → Snake 0xA8B9              ✅ pure dual 538f ×2  (ice_to_snake)
        → Tutorial 0xA865         ✅ pure dual 2386f ×2 (ice_snake_to_tutorial)
          → Ice Gate 0xA815       ✅ pure dual 969f ×2  (ice_tutorial_to_gate)
            → Business 0xA7DE     ✅ pure dual 879f ×2  (ice_gate_to_business)

K5 reverse tunnels + Red climb (this package — implement when predecessor pure)::

    Business 0xA7DE
      → Warehouse 0xA6A1          ✅ pure dual 10255f ×2 (business_to_warehouse; Super fall+ladder)
        → East Tunnel 0xCF80      ✅ pure dual (warehouse_to_east; reverse east_to_warehouse)
          → Glass 0xCEFB          ✅ pure dual (east_to_glass; reverse glass_to_east)
            → West Tunnel 0xCF54  ✅ pure dual (glass_to_west; reverse west_to_glass)
              → Below Spazer 0xA408 ✅ pure dual (west_to_below; reverse below floor→west)
                → Bat 0xA3DD      ✅ pure dual (below_to_bat; reverse bat_to_below_spazer)
                  → Red Tower 0xA253 ✅ pure dual (bat_to_red; reverse red_tower_to_bat)
                    → Hellway 0xA2F7 ✅ dual 6199f ×2 (warehouse_to_red hop 6 body)
                      → Caterpillar 0xA322 ✅ pure dual 2218f ×2
                        → Alpha PB 0xA3AE PLM ⬜  (scaffold ``alpha_pb.py``)

Public controllers land here only after pure-green dual from a real
tape-backed handoff. Continuous / STATUS promote is planner-only after the
full pure stack greens.
"""

from __future__ import annotations

from super_metroid.routes.kpdr.k5.bat_to_red import play_bat_to_red
from super_metroid.routes.kpdr.k5.below_to_bat import play_below_to_bat
from super_metroid.routes.kpdr.k5.east_to_glass import play_east_to_glass
from super_metroid.routes.kpdr.k5.glass_to_west import play_glass_to_west
from super_metroid.routes.kpdr.k5.hellway_to_caterpillar import (
    play_hellway_to_caterpillar,
)
from super_metroid.routes.kpdr.k5.red_to_hellway import play_red_to_hellway
from super_metroid.routes.kpdr.k5.warehouse_to_east import play_warehouse_to_east
from super_metroid.routes.kpdr.k5.west_to_below import play_west_to_below

__all__ = [
    "play_bat_to_red",
    "play_below_to_bat",
    "play_east_to_glass",
    "play_glass_to_west",
    "play_hellway_to_caterpillar",
    "play_red_to_hellway",
    "play_warehouse_to_east",
    "play_west_to_below",
]
