## Residual — rr-0hjh Moonfall core skill (Climb + Parlor first descent)

**Continue:** Both practice duals are **GREEN** from their pins. Do **not**
flip `CLIMB_MOONFALL_ON_CLEAN` or `PARLOR_MOONFALL_ON_CLEAN` until a
power-on / product chain dual-greens (not door-warp) and the next hop still
clears with moonwalk restored off. Then splice into clean Morph.

**Status:** Left-lip Climb spinning moonfall reaches Pit from the warp pin.
Parlor moonfall (top dash + shaft lip + LEFT+X+L floor-door clip) reaches
Climb from the product landing-seed handoff. Assisted seeds are still
product. Both clean flags stay False.

### Climb

**Pin in:** `scratch/climb_descent_enter.state` — Climb `0x96BA` gs=8
~(357,49) p42 facing left. **Warp** via parlor door `0x898E` from
`full_start_v1_morph.state` (items `0x0004`, no Hi-Jump). Not power-on
evidence. Moonwalk-on twin: `scratch/climb_descent_enter_moonwalk.state`.

**Trajectory (do not revert to right-lip):** land start ledge y=91, face
right, moonwalk **left** to x≤349, jump+spin, **hold LEFT** down the shaft
(skips pirate floater at ~395,107), aim-down from y≈1600 to clip the
bottom platform, run **RIGHT** into Pit door ~(493, 2187). Poke `$09E4`
on at entry, **off** after Pit.

**Goal:** natural-enter dual green + faster than 895f; then flip the clean
flag and splice. Restore moonwalk off so pit/elev/morph seeds stay valid.

#### Bench (same warp pin, 2026-08-28)

| Policy | Result | frames | seconds | clock |
|--------|--------|-------:|--------:|-------|
| seed (before) | Pit gs=8 **GREEN** | 895 | 14.892 | 00:14.92 |
| moonfall (after) | Pit gs=8 **GREEN** | 503 | 8.370 | 00:08.38 |
| Δ | moonfall faster | −392 | −6.523 | |

Right-lip / RIGHT-steer setups land on the pirate floater (pose 137 at
395,107) — same miss class as the first probe. Left-lip + LEFT fall is
the replacement trajectory.

Moonfall Pit leave is pose 11 vs seed pose 9 at the same xy ~(39,139).
Climb seed from that seat still needs a natural-enter confirm.

### Parlor

Wiki: moonfall down, then double downbacks to the Climb floor door
(https://wiki.supermetroid.run/Parlor_and_Alcatraz). Listed save 0.20s
(8.10s regular fall → 7.50s moonfall). Dense grass platforms; uncapped
vy never built (max_vy=5). Room cut vs our 1095f seed is 0.47s.

**Pin in:** `scratch/parlor_descent_enter.state` — Parlor `0x92FD` gs=11
~(19,1163) p12 facing left (landing→parlor door still settling). Product
handoff: `ceres_elev_leave_v1_end.state` ship settle → landing seed.
Items `0x0000`. **Not** the morph-state door-warp (awake Geemers; seed
desyncs). Load with `settle=0`. Moonwalk twin:
`scratch/parlor_descent_enter_moonwalk.state`. Door-ledge dump:
`scratch/parlor_descent_door_ledge.state` ~(435,1183) p42.

**Trajectory:** dash LEFT+B through the landing door and top corridor;
jump at x≈1127 to clear the first Geemer ledge; moonwalk at the shaft
lip x≤360 y≈235; LEFT off grass platforms; **LEFT+X+L** from y≥1170
clips the floor Climb door. `$09E4` on at Parlor, off after Climb.

Morph-state warp `0x8916` dumps awake parlor (Geemer at the y=171 ledge)
— seed stays put. Do not use that pin for A/B.

#### Bench (same product handoff, 2026-08-28)

| Policy | Result | frames | seconds | clock |
|--------|--------|-------:|--------:|-------|
| seed (before) | Climb gs=8 **GREEN** | 1095 | 18.220 | 00:18.25 |
| moonfall (after) | Climb gs=8 **GREEN** | 1067 | 17.754 | 00:17.78 |
| Δ | moonfall faster | −28 | −0.466 | |

Leave seats differ: seed Climb ~(393,84) p42; moonfall ~(357,84) p112
(moonwalk restored off). Climb **seed** from the moonfall leave is still
Pit gs=8 **GREEN** 895f pose 9 ~(39,139).

### Probe

```bash
uv run python snes/super_metroid/scripts/probe/kpdr.py pure climb-to-pit-moonfall \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/climb_descent_enter.state
uv run python snes/super_metroid/scripts/probe/kpdr.py pure parlor-to-climb-moonfall \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/parlor_descent_enter.state
```

Overwrite `scratch/climb_descent_bench.json` and
`scratch/parlor_descent_bench.json` only.

### Already green (do not re-prove)

| Layer | Notes |
|-------|-------|
| ROM-free skill / flags / Climb+Parlor action | `tests/test_moonfall.py`, `tests/test_ram.py` |
| `$09E4` poke | on at room, off after dest (bench after-row moonwalk=0) |
| Map Rando labels | `canMoonwalk` Hard, `canMoonfall` Very Hard; project-core override |
| Climb seed from warp pin | **895f** Pit gs=8 |
| Climb moonfall from warp pin | **503f** Pit gs=8, −6.52s vs seed |
| Parlor seed from product handoff | **1095f** Climb gs=8 |
| Parlor moonfall from product handoff | **1067f** Climb gs=8, −0.47s vs seed |
| Climb seed from parlor-moonfall leave | **895f** Pit gs=8 |

Did not STATUS-promote. Did not change `DEFAULT_CONTINUOUS_TIP`. Did not
overwrite `recordings/morph_clean.json`. Did not flip
`CLIMB_MOONFALL_ON_CLEAN` or `PARLOR_MOONFALL_ON_CLEAN`.
