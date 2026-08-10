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
      → Warehouse 0xA6A1          (reuse ``play_business_to_warehouse``)
        → East Tunnel 0xCF80      ⬜ reverse of east_to_warehouse
          → Glass 0xCEFB          ⬜ reverse of glass_to_east
            → West Tunnel 0xCF54  ⬜ reverse of west_to_glass
              → Below Spazer 0xA408 ⬜ reverse of below_spazer_to_west
                → Bat 0xA3DD      ⬜ reverse of bat_to_below_spazer
                  → Red Tower 0xA253 ⬜ climb reverse of red_tower_to_bat
                    → Hellway 0xA2F7 ⬜
                      → Caterpillar 0xA322 ⬜
                        → Alpha PB 0xA3AE PLM ⬜  (scaffold ``alpha_pb.py``)

Public controllers land here only after pure-green dual from a real
tape-backed handoff. Continuous / STATUS promote is planner-only after the
full pure stack greens.
"""

from __future__ import annotations

__all__: list[str] = []
