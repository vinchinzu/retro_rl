# Residual — rr-n91a / rr-8t4.3 L7-C 0x0D walk-on (2026-09-04)

Did not STATUS. Did not poke doors/TF/position. Did not touch `level8/**`,
`AGENTS.md`, harvest, or parent `rr-8t4.3` (still in_progress). Spine
`make_tip_stairs_controller` / `make_aquamentus_heart_controller` /
`make_level7_shard_leave_controller` stay fail-closed. Cellar-cross dest
0x29 (below) left untouched.

## 2026-09-04 sitting — north-corridor UP from y=141 MISS

### Glance (pin start)

`Level7Interior0DClearedReconFixture`: L7 play **`$EB=0x0D` mode 5**
`(63,149)` tile 118, doors=2, keys 2, bombs 6, candle 2, TF 0.
`0x68` `(192,144)` state 0.

### RAM claim (before run)

From pin to `(144,141)` (known live), UP toward y≤99 between upper
plus-corners `(128,125)`/`(160,125)`. Miss if UP pins at y≥117.

### Trials

| tag | dest $EB/mode/xy | frames | PNG | vs claim |
|-----|------------------|--------|-----|----------|
| `20260904_NU` col | play 0x0D `(142,141)` tile 119 | 63 | `20260904_NU_nu_144_141.png` | **HIT** `(144,141)` |
| `20260904_NU` UP x=144 | play **0x0D** m5 `(144,117)` tile **179** | 122 stuck | `20260904_NU_nu_after_up_144.png` | **MISS** y≥117 |
| `20260904_NU176` col | play 0x0D `(174,141)` tile 119 | 87 | `20260904_NU176_nu_176_141.png` | **HIT** `(176,141)` |
| `20260904_NU176` UP x=176 | play **0x0D** m5 `(176,117)` tile **179** | 146 stuck | `20260904_NU176_nu_after_up_176.png` | **MISS** y≥117 |

`progression_writes=0` `capacity_writes=0` `deaths=0` `position_poke=0`.
Reports `recordings/20260904_NU.json`, `20260904_NU176.json`. Probe
`--north-up`. y≤99 not entered. No push-then-walk-on. Did not retry
DOWN or plug clips. `NOSE_CELLAR.ram_id` stays None.

### New dead beliefs

- Dead: UP from `(144,141)` between upper plus-corners reaches y≤99.
  Pins **`(144,117)` tile 179** (same class as west `(48,117)`).
- Dead: UP from `(176,141)` (west of 0x68) reaches y≤99. Pins
  **`(176,117)` tile 179**.

### Leftover glance

L7 play **`0x0D` mode 5** `(63,149)` tile 118, doors=2, Candle **2**, TF **0**,
keys 2, bombs 6, whistle 1, ladder 1, `0x68` `(192,144)` state 0.
`route_eligible=false`.

Next leftover: 0x0D walk-on still OPEN. Do not poke onto the stairs.

## 2026-09-04 sitting — plus-corner x=144 DOWN MISS

### Glance (pin start)

`Level7Interior0DClearedReconFixture`: L7 play **`$EB=0x0D` mode 5**
`(63,149)` tile 118, doors=2, keys 2, bombs 6, candle 2, TF 0.
`0x68` `(192,144)` state 0. PNG `recordings/l7_0d_walkon_20260904_start.png`.

### RAM claim (before run)

From pin `(63,149)` to `(96,141)` (known live), RIGHT to `(144,141)`,
DOWN between plus-corners (statues x=128 and x=160 at y=157) toward
y=165. Miss if DOWN pins at y≤157 or x slides off 144.

### Trial `20260904_PC`

| step | dest $EB/mode/xy | frames | PNG | vs claim |
|------|------------------|--------|-----|----------|
| y=141 gap | play **0x0D** m5 `(94,141)` tile 119 | 27 | `20260904_PC_pc_96_141.png` | **HIT** |
| `(144,141)` | play 0x0D `(142,141)` tile 119 | 63 | `20260904_PC_pc_144_141.png` | **HIT** |
| DOWN x=144 | play **0x0D** m5 `(144,157)` tile **178** | 117 stuck | `20260904_PC_pc_after_down.png` | **MISS** y≤157; x held 144 |

`progression_writes=0` `capacity_writes=0` `deaths=0` `position_poke=0`.
Report `recordings/20260904_PC.json`. Probe `--plus-corner`. Did not
retry Trial B. No dest fixture. `NOSE_CELLAR.ram_id` stays None.

### New dead belief

- Dead: x=144 between plus-corner statues `(128,157)` / `(160,157)` is a
  walkable DOWN corridor to y≥160. Moving Link pins **`(144,157)` tile
  178**. x does not slide off 144.

### Leftover glance

L7 play **`0x0D` mode 5** `(63,149)` tile 118, doors=2, Candle **2**, TF **0**,
keys 2, bombs 6, whistle 1, ladder 1, `0x68` `(192,144)` state 0.
`route_eligible=false`.

Next leftover: 0x0D walk-on still OPEN. Do not poke onto the stairs.

## 2026-09-04 sitting — Trial A y=141 lure + Trial B plug clip MISS

### Glance (pin start)

`Level7Interior0DClearedReconFixture`: L7 play **`$EB=0x0D` mode 5**
`(63,149)` tile 118, doors=2, `room_all_dead=4`, keys 2, bombs 6, candle 2,
whistle 1, ladder 1, TF 0. `0x68` `(192,144)` state 0.
PNG `recordings/l7_0d_walkon_20260904_start.png`.

### RAM claim (before Trial A)

From pin `(63,149)`, walk to `(96,141)` via y=141 gap (UP/RIGHT, never
DOWN at x=63), then DOWN to `(96,165)` (known lure), then RIGHT along
y=162-165 toward x=192, then UP to the 0x68 south face. Miss if
`(96,165)` is not reached, or RIGHT pins at x≤176, or south face
unreachable.

### Trials

| tag | dest $EB/mode/xy | frames | PNG | vs claim |
|-----|------------------|--------|-----|----------|
| `20260904_A` gap | play **0x0D** m5 `(94,141)` tile 119 | 27 | `20260904_A_a_96_141.png` | **HIT** y=141 gap |
| `20260904_A` lure | play **0x0D** m5 `(96,157)` tile **178** | 81 stuck DOWN | `20260904_A_a_96_165.png` | **MISS** never `(96,165)` |
| `20260904_B` push | play 0x0D `(177,141)` block **(208,96)** s2 | — | `20260904_B_b_pushed.png` | RIGHT push as known |
| `20260904_B` LEFT+UP | play 0x0D `(191,133)` tile 118 | 1 | `20260904_B_b_clip_LEFT.png` | still y=133 |
| `20260904_B` RIGHT+UP | play 0x0D `(192,133)` tile 179 | 1 | `20260904_B_b_clip_RIGHT.png` | no move |
| `20260904_B` grade | play **0x0D** m5 `(192,133)` | — | `20260904_B_b_plug.png` | **MISS** boxed y≥133 |

`progression_writes=0` `capacity_writes=0` `deaths=0` `position_poke=0`.
Reports: `recordings/20260904_A.json`, `recordings/20260904_B.json`.
Probe `--trial-a` / `--trial-b`. No dest fixture. No Trial C (south face
never stood). `NOSE_CELLAR.ram_id` stays None.

### New dead beliefs

- Dead: DOWN from y=141 gap `(96,141)` reaches lure `(96,165)`. Pins
  **`(96,157)` tile 178** (plus SW diamond). The y=141 gap itself is live.
- Dead: one-frame LEFT+UP / RIGHT+UP at the `(192,133)` plug after the
  verified RIGHT push crosses north onto the x=192-204 column. LEFT+UP
  slides 1px west, y stays 133; RIGHT+UP is tile 179 no-move.

### Leftover glance

L7 play **`0x0D` mode 5** `(63,149)` tile 118, doors=2, Candle **2**, TF **0**,
keys 2, bombs 6, whistle 1, ladder 1, `0x68` `(192,144)` state 0.
`route_eligible=false`. Do not leave the post-push pin as leftover.

Next leftover: 0x0D walk-on still OPEN. Do not poke onto the stairs.

## 2026-09-04 sitting — 0x0D walk-on south-of-gap / south-strip MISS

### Glance (pin start)

`Level7Interior0DClearedReconFixture`: L7 play **`$EB=0x0D` mode 5**
`(63,149)` tile 118, doors=2 (LEFT), `room_all_dead=4`, keys 2, bombs 6,
candle 2, whistle 1, ladder 1, TF 0, food 0, 3 HC. `0x68` `(192,144)`
state 0. 3× bubble `0x2b`. PNG:
`recordings/l7_0d_walkon_20260904_start.png`. `route_eligible=false`.

### RAM claim (written before first live trial)

From this pin, DOWN to y=162 (south-of-gap band between unwalkable
y136-158 and south wall y~166), then RIGHT along y=161-164 to x=192.
Claim: that band is walkable floor to the 0x68 south-face column. Miss if
y never enters 161-164, or RIGHT pins at x<=176 (old gap pin).

### Trials

| tag | dest $EB/mode/xy | frames | PNG | vs claim |
|-----|------------------|--------|-----|----------|
| `20260904_sog` | play **0x0D** m5 `(64,157)` tile **178** | 48 stuck DOWN | `recordings/20260904_sog_sog_after_down.png` | **MISS** y never entered 161-164 |
| `20260904_ss` | play **0x0D** m5 `(64,157)` tile **178** | 48 stuck DOWN | `recordings/20260904_ss_ss_after_down.png` | **MISS** y never reached 180 (L9 y=189 one-shot) |

`progression_writes=0` `capacity_writes=0` `deaths=0` `position_poke=0`.
Reports: `recordings/l7_0d_walkon_20260904_sog.json`,
`recordings/l7_0d_walkon_20260904_ss.json`. Probe:
`scratch/probe_l7_room0d_squeeze.py --south-of-gap` / `--south-strip`.

Did not UP-push (south face unreachable). Did not retry west-above-band
y<=99. Did not save a dest fixture. Did not flip spine factories. Did
not fill `MEASURED_POST_L7_EXIT`. `NOSE_CELLAR.ram_id` stays None.

### New dead beliefs

- Dead: y=161-164 (between gap y136-158 and south wall y~166) is reachable
  by DOWN from pin `(63,149)`. Moving Link pins **`(64,157)` tile 178**
  (SW diamond). PNG `20260904_sog_sog_after_down.png`.
- Dead: L9 `room30_stairs_step` y=189 south-strip is reachable from
  `(63,149)` in 0x0D. Same pin; ROM south door is WALL (code 1). One shot,
  not looped.

### Leftover glance

L7 play **`0x0D` mode 5** `(63,149)`, tile 118, doors=2, `room_all_dead=4`,
Candle **2**, TF **0**, keys 2, bombs 6, whistle 1, food 0, ladder 1,
3 hearts. Block `0x68` `(192,144)` state 0. `route_eligible=false`.
(Trials left Link at `(64,157)`; leftover is the uncleared-push pin.)

Next leftover: 0x0D walk-on still OPEN. Do not poke onto the stairs.

---

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
