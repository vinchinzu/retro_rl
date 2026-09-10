## Residual — rr-20w.3 D3 spring second plot + daily grape

**Status:** two mountain grapes + potato shop ROM GREEN from
`Y1_D3_Morning` / `Y1_D3_PostTwoGrape`. **Second potato ring ROM GREEN**
(2026-09-09): established + watered beside the D2 rows from
`Y1_D3_PostShop`. `CROP_ESTABLISH` / `CROP_WATER` now walk the two
pocket rings in order. "Water all 16" (both rings same evening) is left
to rr-3ae8 (refill). Do not STATUS.

### Second plot placement (SOLVED)

`SECOND_POCKET_PLANT_CENTER = (19, 28)` (`harvest.maps.farm_pond`).
`crop_planner.plan_crop_field` on live `Y1_D3_PostShop` RAM (map warmed
~60f, D2 ring `(13,28)`+8 protected) with fence-lip (y30) rings excluded
picks **(19, 28)** uniquely:

| center | score | route_cost | note |
|--------|-------|-----------|------|
| **(19,28)** | **170** | **45** | chosen — crop tiles y27-29, off well + fence lip |
| (18,29) | 218 | 37 | higher score but bottom row on the y30 fence lip (hoe stand = y31 wall) — excluded |
| (19,29) | 188 | 42 | also fence-lip bottom row |
| (17,29)/(16,29)/(18,28) | — | — | REJECTED: top-row crop tiles have no watering stand (well body 0xA1 at x15-17 y26-27) |
| (13,25) north of D2 | — | — | REJECTED: top row can't get a stand (y23 is the 0xA8 bank) |

crop_tiles `(18,27)(19,27)(20,27)(18,28)(20,28)(18,29)(19,29)(20,29)`.
Beside D2 (same y band, east of the well), not an east island. Potato
day 3 → harvest Spring 9, profit 8×80−200 = 440.

The space east of D2 is boxed: **well** (0xA1, x15-17 y26-27) west/north,
**0xA8 bank** at **x21** east, **fence lip** y30 south. The east crop
column (x20) is hoed from the west/south, not the (walled) east.

### ROM

| Hop | Pin | Frames | Clock | Result |
|-----|-----|--------|-------|--------|
| 2 grapes | `Y1_D3_Morning` | 6194 | 06:00→13:12 | shipping 0→300 |
| potato shop | `Y1_D3_PostTwoGrape` | 2628 | 13:12→16:08 | 0x1C, potato 0→1, $250→$50, farm 0x00, +240f settle |
| second ring establish | `Y1_D3_PostShop` | ~3190 | 16:08→18:12 | (19,28) 8 tilled + 8 planted (0x54), bag spent, on farm |
| second ring + water | `Y1_D3_PostShop` | — | 16:08→18:08 | +8 wet (0x55), can 12→4, both rings 16 planted, on farm |

Reports: `recordings/d3_buy_seeds_after_two_grape.json`,
`recordings/d3_second_plot.json` (last = the watered run).

Pins (all local): `Y1_D3_PostShop` (16:08, farm, potato 1, $50, D2 ring
planted), `Y1_D3_PostSecondPlot` (establish only), `Y1_D3_PostSecondPlotWatered`
(18:08, new ring wet, can=4).

### Wiring

- `farm_pond.SECOND_POCKET_PLANT_CENTER` / `POCKET_PLANT_CENTERS` /
  `next_unplanted_pocket_center` / `pocket_water_center` — one source of truth.
- `day_phase_registry._build_crop`: CROP_ESTABLISH → `farm_pocket_plant_skill(
  center=next_unplanted_pocket_center(ram), ram=ram)`; pocket CROP_WATER →
  `farm_pocket_water_skill(center=pocket_water_center(ram))`.
- `skills.farm_pocket_plant_skill` / `farm_pocket_water_skill` /
  `farm_nav_pocket_*` take `center=` (default west pocket unchanged).
- `crop_skills`: `pocket_hoe_ring_skills(center, ram=)` +
  `remap_pocket_hoe_stand(..., ram=)` — with `ram`, hoe-stand remap skips
  non-soil / live-wall stands (`_HOE_STAND_TILE_IDS`) and prefers the north
  (face-down) alt so nav does not settle a row south of a face-up stand.
- `crop_establish._west_pocket_plant_center` iterates `POCKET_PLANT_CENTERS`,
  returns the first unplanted ring — no second hardcoded `(13,28)`.
- `buy_seeds_probe` / `d2_plant_probe`: `--save-end-state`; probe
  `--second-plot` (implies `--skip-clear`); probe warms 90f on load and
  dismisses the 5pm ShippingScene mid-sequence.

### Tests (all green)

`test_crop_planner.SecondPlotPlacementTests`,
`test_crop_skills.SecondPocketPlotTests`,
`test_day_plan_crop_phases.Day3SecondPlotTests.test_crop_establish_targets_second_ring_once_west_pocket_is_planted`.
Full crop/day-plan suite: **549 tests OK**.

### Still open

- "Water all 16" — re-water the D2 ring the same evening / D4 (needs a
  refill: can 12, 16 tiles). rr-3ae8.
- Second-ring tiles are ROM-tuned once from `Y1_D3_PostShop`; re-validate
  if the D3 lineage is re-minted.

### Non-claims

- No STATUS. Did not start from `Y1_D2_Morning_After_D1`.
- Did not record a BFS-closable walk. Did not treat CrossMap origin-return
  as shop success. Did not re-prove the two grapes / shop hop / a 3rd grape.
- Did not claim rr-3ae8 (refill) or a second bead.
- 5pm shipping credit for the last grape not verified to next morning
  (farmer was on farm at 17:00).
