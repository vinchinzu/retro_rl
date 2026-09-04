# Residual — rr-n91a L7-C 0x7B B→A dest 0x29 (2026-09-04)

Did not STATUS. Did not poke doors/TF/position. Did not touch `level8/**`,
`AGENTS.md`, harvest, or parent `rr-8t4.3` (still in_progress). Spine
`make_tip_stairs_controller` / `make_aquamentus_heart_controller` /
`make_level7_shard_leave_controller` stay fail-closed.

## Glance (pin start)

`Level7Interior0DNoseCellarReconFixture`: L7 mode **9** cellar **`$EB=0x7B`**
`(192,93)` tile 111, keys 2, bombs 6, candle 2, TF 0. 4× keese 0x1B hp=0.
`route_eligible=false`. Existing disclosed poke is in fixture provenance;
no new position pokes.

## RAM claim (written before first live trial)

From this pin, floor-cross left, first settled play `$EB` is **0x29**
(ROM AttrA). Miss if dest is 0x0D (CheckSubroom AttrB / UP on the source
ladder). Never UP at `x>=$80`.

## Dest 0x29 — live 2/2

Policy: DOWN the east/source column, floor LEFT to `x=48`, UP left ladder.
Never UP on the right/source ladder. OccupancyWalker banned. Inverse of
`cellar_cross_dir` (L6 A→B east then UP).

| tag | dest | pose | controller frames | total frames |
|-----|------|------|-------------------|--------------|
| `20260904_C1` | play **0x29** | `(96,157)` | 396 | 856 |
| `20260904_C2` | play **0x29** | `(96,157)` | 396 | 856 |

Reports: `recordings/l7_7b_cellar_cross_20260904_C1.json`,
`recordings/l7_7b_cellar_cross_20260904_C2.json`.
Pin: `Level7Interior29PreBossReconFixture`. Goriya 0x05/0x06 in 0x29.
`progression_writes=0` `capacity_writes=0` `deaths=0` `position_poke=0`.

C1 first miss: `pit_tile_250` at west floor `(48,189)` — false L8 y=141
trap. Policy now only fails 250 at the y=141 ledge, not the west ladder.

## Suffix (C1, after dest 0x29)

- **0x29 bomb-E → 0x2A** 1/1. Survival bomb-count top-up 6→8 at the
  verified gate (`apply_owned_inventory`, never `max_bombs`). Used 8→7.
  Dest play **0x2A** `(32,141)` west mouth. Pin
  `Level7Interior2AAquamentusReconFixture`.
- **0x2A Aquamentus** 1/1. Reused `Level1AquamentusController`
  ALIGN/FACE/ATTACK/DODGE/COLLECT_HEART, `tank_hits=True`, room aliased
  from live `$EB=0x2A` (not L1 0x35). Type **0x3D** at entry. Fireballs
  0x55 not on the west-mouth census (spawn hp=0); one damage event in
  0x2A while tanking. Sword-only. HC **3→4**. Leftover `(120,141)`.
- **0x2A E-shutter → 0x2B** 1/1. Play **0x2B** `(16,141)` west mouth.
  Pin `Level7Interior2BTriforceReconFixture`. Diamond floor; DOWN at
  x=16 does not move. South-around `(32,141)→(32,189)→(120,189)→(128,141)`
  collects the shard. Fanfare mode 18, then OW.
- **OW leftover (fixture-lineage, 1/1, not Survival):** play **0x42**
  `(96,93)` mode 5, TF **0x40**, candle 2, whistle 1, keys 2, bombs 7,
  HC 4. Pin started at TF 0 so this is **not** the Survival `0x7F`
  packet. `MEASURED_POST_L7_EXIT.verified` stays False.
  `PostLevel7Handoff.verified` untouched.

## Rooms live / selected min

L7 remaining selected suffix: cellar `0x7B` → play `0x29` → boss `0x2A`
→ tf `0x2B` = **4/4 fixture-live** (dest 0x29 is 2/2; 0x2A/0x2B/OW are
1/1 from C1). Prefix is live through TIP_OF_NOSE `0x0D`. `NOSE_CELLAR.ram_id`
stays None (no 0x0D walk-on). Graph ram_ids: PRE_BOSS `0x29`, AQUAMENTUS
`0x2A`, TRIFORCE `0x2B`, evidence `fixture-live`, `route_eligible=false`.

## Not done / leftover

- **0x0D south-face squeeze / walk-on of cellar 0x7B is still OPEN.**
  0x0D is not dead. Parent `rr-8t4.3` keeps that TAS.
- Spine factories stay fail-closed. Do not wire the poke-spawned cellar
  cross onto `level7-red-candle` / `level7_complete`.
- Survival OW leave packet (TF `0x7F`, 8+1 HC) is unmeasured. Do not
  fill `MEASURED_POST_L7_EXIT` from the TF-0 recon pin.

Next leftover: 0x0D walk-on (parent), or Survival-lineage shard leave
once the poke-fixture TF=0 is no longer the start.
