## Residual — rr-0hjh Moonfall core skill (Climb first descent)

**Continue:** Keep the Climb moonfall controller. Next trajectory: jump at the
right lip (~x=372) and **do not** LEFT-steer during the first 200px of fall
(LEFT cancelled moonfall back onto the start platform). Alternative: buffer
from a Parlor leave pin through the vertical door.

**Status:** Skill API + `$09E4` poke + Climb policy + probe landed.
Moonfall **initiates** on the warp pin (movement 16 → spin pose 25, vd=0).
It does **not** yet clip the first floater or reach Pit. Assisted Climb seed
is still product. `CLIMB_MOONFALL_ON_CLEAN = False`.

**Pin in:** `scratch/climb_descent_enter.state` — Climb `0x96BA` gs=8
~(357,49) p42 facing left. **Warp** via parlor door `0x898E` from
`full_start_v1_morph.state` (items `0x0004`, no Hi-Jump). Not power-on
evidence (Ceres elev checkpoint dies at f23123 on this capture path).
Moonwalk-on twin: `scratch/climb_descent_enter_moonwalk.state` (`$09E4=1`).

**Goal:** Pit `0x975C` gs=8, faster than the seed. Restore moonwalk **off**
after Pit so later seeds stay valid.

### Bench (same warp pin, 2026-08-28)

| Policy | Result | frames | seconds | clock |
|--------|--------|-------:|--------:|-------|
| seed (before) | Pit gs=8 **GREEN** | 895 | 14.892 | 00:14.92 |
| moonfall (after) | timeout y=67 jump **RED** | 1200 | 19.967 | 00:20.00 |
| Δ | seed faster | +305 | +5.075 | |

Trace: moonwalk mt=16 at y=91, moonfall mf=1 at ~f42. First floater ~x=395
y=107 eats the fall before vy uncapped. Three setup variants (air-buffer
RIGHT, lip jump, LEFT-steer) same miss class: **cancelled at the top
floater**. Stop repeating; new trajectory or Parlor-door buffer.

### Probe

```bash
uv run python snes/super_metroid/scripts/probe/climb_descent.py capture-warp
uv run python snes/super_metroid/scripts/probe/climb_descent.py bench
# power-on (Ceres elev currently RED on this path):
uv run python snes/super_metroid/scripts/probe/climb_descent.py capture
```

Overwrite `scratch/climb_descent_bench.json` only.

### Already green (do not re-prove)

| Layer | Notes |
|-------|-------|
| ROM-free skill / flag / Climb action | `tests/test_moonfall.py`, `tests/test_ram.py` |
| `$09E4` poke | moonwalk pin reads 1; gameplay moonwalk mt=16 |
| Map Rando labels | `canMoonwalk` Hard, `canMoonfall` Very Hard; project-core override |
| Seed from warp pin | **895f** Pit gs=8 |

Did not STATUS-promote. Did not change `DEFAULT_CONTINUOUS_TIP`. Did not
overwrite `recordings/morph_clean.json`. Did not flip
`CLIMB_MOONFALL_ON_CLEAN`.
