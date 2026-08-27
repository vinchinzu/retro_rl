# D2 farm-clear issues

Facts for Spring D2 section work (`rr-20w.2`). Product path is
**grape → shop → shed hoe+seeds → pocket `CLEAR_PLOT` → hoe/plant/water → evening leftover**,
not a morning whole-farm wipe. Do not STATUS-promote Gate B from this list.

Walk invariant (must stay true through nav refactors): **BFS never routes
onto stumps, large rocks, damage tiles, or the small boulder.** Stand on a
neighbor and swing/lift. Tool-swing frames must not hold the d-pad.
WEED `0x03` is ROM-walkable but pins travel — travel BFS must not route onto
it either (`rr-20w.2.2`).

**2026-08-21 landed (unit, no live pin):** travel denylist
(`TRAVEL_SOLID_TILES` + `Pathfinder.is_walkable`); honest unbounded
`FarmClearTask` SUCCESS; lift-only drops only unarmed types; 2×2 stamina
is an 8-swing budget (16); shipping dismiss no longer A-pulses `lock==0`
after 17:00. Live pocket
probe is still `rr-20w.2.3`. Do not restore a morning whole-farm wipe.

**2026-08-21 landed (`rr-jq9q`, live `Y1_After_Buy_Potato`):** hoe+seed is
one shed visit (`exit_when_done=False` on hoe); splice is
`ENSURE_CROP_SEEDS` then `CLEAR_PLOT` then plant (clear-then-shed sealed
the return). NavTask leaves shed door `(26,30)` west, never UP/A into
the shed. Hoe face is a 1f tap (holding UP walked onto the target).
Probe `d2_plant_probe`: collect 1140f, notch clear, `CROP_ESTABLISH`
sequence complete, `(13,28)=0x54`. One-cell does not spend the bag.
**Next (`rr-m7mk` / `rr-bvam` / `rr-w14t` / `rr-5aaw`):** D2 work catalog
is `harvest.planner.d2_work`. Shop splice concatenates plant+water+leftover
so a 06:08 plan still gets evening smash (not `hour>=17` at plan time).
Quotas: **8** plant/water, **10** bushes (pick+toss), **all** fence posts
to ponds, **10** stones to ponds, **4** large 2×2 (hammer), **2** stumps
(axe). Hammer then axe (2-slot carry). Spa
inserts when stamina cannot finish an 8-swing 2×2. Hoe ring must not stand
on the well `(15,27)`. Tune from `Y1_After_Buy_Potato`. **2026-08-23 live pin GREEN**
(`rr-m7mk` / `rr-bvam`): `d2_plant_probe --water` from
`Y1_After_Buy_Potato` tills 8, spends the bag, fills the empty shelf
can (y=31 corridor + F0 stand `(32,34)`), wets 8 (`0x55`), can 20→12,
6292f (`recordings/d2_plant_water.json`). Redo power-on after leftover
smash. Do not STATUS-promote Gate B.

**2026-08-23 (`rr-m7mk` lift residual):** pocket stone at **(11,29)** (not
the stand to its north) was a false-success. Lift emptied the cell, then
`toss_pulse` held UP for 24f and walked the rock onto **(11,28)**; CLEAR
counted it cleared and `failed_tiles` skipped the re-drop. Farmer then
stood at (11,29) boxed by pond `0xA6` / stone / tilled ring / fence.
Fix: claim a lift only after hands are empty and the origin stays
non-debris; toss A is d-pad-free; hoe stands + origin are not landings;
boxed west-lip carry runs east instead of throwing north. Live
`Y1_After_Buy_Potato`: CLEAR `plot_ring_clear` with `(11,28)=0x01`,
`(11,29)=0x02`, first hoe tills `(13,29)=0x07`. Hoe well stand is remapped
in `remap_pocket_hoe_stand`: `(14,27)` from `(14,28)` face-up, not
`nav_hoe_ring_1_left` onto `(15,27)`. East-bottom `(14,29)` is not
`(15,29)` face-left — leftover stone `(16,29)` seals the rightward
nudge (`nav_hoe_ring_3_left` timeout); stand `(13,29)` face-right
instead. Left-middle `(12,28)` is not `(12,29)` face-up — leftover
stone `(12,30)` seals the southward nudge (`nav_hoe_ring_6_up`);
stand on the untilled notch `(13,28)` face-left. Young potato `0x54`/`0x55`
is farm-walkable so the can-fetch can leave the boxed notch.
`pocket_to_shed` skips `farm_to_shed`'s west hop `(137,375)` (shipping
ditch). Live 8-plant/`--water` from `Y1_After_Buy_Potato` is still the
pin. Shots: `recordings/d2_stone_probe/`.

**2026-08-24 leftover pin (`rr-w14t`, not green):** `d2_leftover_probe`
from `Y1_After_Buy_Potato`. Dump: 506 weeds, **80 fence posts** (house
paddock + y=31 wall x=11–29 + east column x=32), **0× `0x06`**, 185
stones (`0x04`), 51 large 2×2, 38 stumps.
CLEAR_BUSHES 10 in the west pocket is GREEN (3007f). Leftover order is
now **bushes pick+toss → dump all fence posts in ponds → toss 10 stones
in ponds → walk off shed door onto loaded a8/a1 → hammer 4 large → axe
2 stumps**. ENSURE_HAMMER RAM shelf is GREEN but used to land on shed
door `(26,30)` `0xFF` (farm map unloads; counts look like a wipe with
`cleared_count=0`). Hammer fetch now NavTasks to `(25,28)` a1 before
SUCCESS; FarmClearer keeps holding west/NW until `farm_map_loaded`.
`0x06` is absent — leftover "10 small" is 10 pond-tossed `0x04` stones,
not hammered.

Fence dump (`--section fences`, not green): y=31 wall first, F0 south-lip
toss, skip a stuck post instead of burning the day on one target. Live
`Y1_After_Buy_Potato`: **80→19** (61 pond-dumped) in 90 017f / 18:00;
remaining 19 are house north y=13 + west x=2, boxed in house `0xA6`.
Do not toss into `0xA6` (regression 80→79).

Stones (`--section stones`): first 10 pond-tossed `0x04` were GREEN from
`Y1_After_Buy_Potato` (185→175, 7797f). CLEAR_STONES is exhaustive (all
remaining `0x04` to ponds). Live from `Y1_D2_After_Stumps`: axe+hoe still
lifts (no stow); 175→48 in 200001f, then 48→45 and `too many fence
failures` at the barn walls. Pin `Y1_D2_Stones_Frontier` (45 left).
Horse-barn sprite walls are `FARM_NO_GO_TILES` (push-into cells, not
stand-on stasis). Takeoff from `(17,20)` leaves south onto `(17,21)`.
North-of-barn leftover dumps at `(46,16)` face-up into `0xFA` (trimmed
`horse_barn_edges` slice), not the F0 south lip. Do not STATUS-promote
Gate B.

Rocks (`--section rocks`, pin GREEN from `Y1_D2_After_Stones`): 4/4
large 2×2 in 3747f, `51 → 47`, stam 65→17. Hammer stays planted after the
first face — a d-pad re-center STZs `$096D`. Spa-return from
`Y1_D2_After_Rocks` is pin-green (`Y1_D2_After_Spa`, 17→100). Stumps
quota 2/2 is pin-green from that spa pin (`Y1_D2_After_Stumps`, 38→36).
Do not STATUS-promote Gate B.

## Invariants

| Tile | IDs | `FARM_WALKABLE` | Travel BFS | Clear |
|------|-----|-----------------|------------|-------|
| Soil / path | `0x00`, `0x01`, `0xA0`–`0xA3`, … | yes | yes | — |
| WEED | `0x03` | yes (ROM) | **no** | adjacent lift |
| STONE / FENCE | `0x04` / `0x05` | no | no | adjacent |
| Small ROCK | `0x06` | **no** | **no** | adjacent hammer |
| Stump 2×2 | `0x09`–`0x0C` | **no** | **no** | adjacent axe |
| Large rock 2×2 | `0x0D`–`0x10` | **no** | **no** | adjacent hammer, ROM 6 hits / 8-swing stam budget |
| Rock damage | `0x11`–`0x14` | **no** | **no** | same 2×2, keep swinging |

Push-facing (`player_action==0` + no pixel motion) must not seal the
**approach cell** next to a rock/stump being cleared.

## Issues

### P0 — travel walks onto weeds (`rr-20w.2.2`)

`FARM_WALKABLE` includes `WEED`. `Pathfinder.is_walkable` / `NavTask` BFS
onto bushes; MultiNav already no-gos them. FarmClearer rock-first then paths
**through weeds** to boulders. Live pin on `(13,27)`.

**Fix:** travel BFS denylist (weed + every stump/rock/damage ID) even when
the ROM walkable set includes the tile. Clear still stands on a walkable
neighbor. Do not remove `0x03` from scanner dumps.

### P0 — `CLEAR_PLOT` must not stand on bushes (`rr-20w.2.3`)

Pocket clear is weeds/stones, `fetch_tools=False`. Approach is `NavTask`,
which still walks onto `0x03`. After `rr-20w.2.2`, pocket approach uses the
same travel denylist.

### P0 — stumps/rocks must stay non-walkable (`rr-20w.2.10`)

Already absent from `FARM_WALKABLE`, but no unit test. Uncommitted
`Navigator.follow_path` push-facing can `temp_block` the **approach** tile
after 20f of zero motion (same byte as idle). Hammer swing must stay
direction-free (`use_tool` is Y-only; do not add d-pad to hit frames).

### P1 — shop success deletes leftover `CLEAR_FIELD`

`_splice_plant_after_shop` drops every remaining `CLEAR_FIELD` and inserts
pocket plant (`day_plan_orchestrator.py`). **Morning wipe is not the D2
goal** — keep the splice. Evening leftover is `rr-20w.2.8` and must not
depend on a 06:08 `late_day` expansion (currently `_evening_field_clear_phases`
only runs if `hour >= 17` **at plan time**, so a 6am plan never attaches it).

### P1 — unbounded `CLEAR_FIELD` SUCCESS-lies (`rr-20w.2.11`)

`FarmClearTask` returns SUCCESS on `stamina_low`, `partial_clear`,
`clear_budget`, lift-only leftovers, and **off-farm empty scans**
(`can_start` is True off-farm for shed recordings that are disabled).
Orchestrator `_advance`s as real work. Pocket `CLEAR_PLOT` plant-notch
SUCCESS is still correct.

**Fix:** whole-farm SUCCESS only when remaining clearable debris is empty.
Incomplete → not `_advance` success. Off-farm unbounded clear must not
SUCCESS in one tick.

### P1 — hammer never fetched; missing axe drops ROCK (`rr-20w.2.12`)

`FETCH_CLEAR_TOOL_RECORDINGS` stays off. Leftover uses RAM shelf
`ENSURE_HAMMER` / `ENSURE_AXE` (`ensure_tool` → `EnsureCarryToolTask`),
not recorded `GET_HAMMER` / `GET_AXE`. `SHED_TOOL_SPECS` bottom-shelf
stands from DATA16_81BE0F (stand y = sprite_y+24, same as can): hammer
`(176,168)`, axe `(192,168)`, sickle `(144,168)`. Two carry slots:
ENSURE_HAMMER → CLEAR_ROCKS → ENSURE_AXE → CLEAR_STUMPS (never both).

**Fix:** drop only the debris types whose tool is actually missing. If
hammer is in the pair, still smash rocks.

### P1 — stamina gate cannot finish a large rock (`rr-20w.2.14`; spa is `rr-pzw`)

Hammer/axe is −2/swing; ROM breaks a 2×2 at **6** registered hits (`$096D`).
Y-holds miss, so clear will not *start* a rock/stump below an **8-swing**
budget (16 stamina). `MIN_CLEAR_STAMINA=4` is only the lift floor. Damage
tiles collapse to the 2×2 TL. Lifts continue at stamina 1–3. Do not spa
on D2 morning.

**Fix:** `Stamina.from_ram` + `can_finish_multi_hit()`. Evening leftover
clear (`rr-20w.2.8`) inserts `HOT_SPRING_STAMINA` (fill to max,
`return_to_farm`) when stamina cannot finish the 8-swing budget (`rr-pzw`
wiring). Splice-time spa uses **live stamina after water**; it does not
spa on D2 morning. Mid-phase spa+retry if `CLEAR_ROCKS`/`CLEAR_STUMPS`
fail `stamina_low` is a follow-up (orchestrator-owned, not a
`farm_clearer` thrash `if`).

### P2 — 5pm A-pulse steals tool frames (`rr-20w.2.13`)

`shipping_scene_needs_dismiss` treats farm + `hour>=17` + `lock!=1` as the
shipper box. Orchestrator intercepts every phase and pulses A. A hammer
swing after 17:00 looks like a lock. Keep dismiss on text `0x031A`/`0x031B`
and pending flag `0x0400`, not “any locked farm evening.”

### P2 — file size / spaghetti

`day_plan_phases.py` 1002, `farm_clearer.py` 999, `multi_nav.py` 1062.
Do not add spa / hammer / evening branches as thrash `if`s. Extract a
helper first.

## Not D2 morning

| Item | Bead | Note |
|------|------|------|
| Whole-farm 800-target wipe | — | Starves hoe; catalog 3500f is keep-alive only |
| Hot spring | `rr-pzw` | Evening leftover insert is wired. D2 night pin `Y1_D2_Night_Farm`: grape-corridor farm→spa→farm GREEN (outbound 1956f, return SUCCESS 3813f). Do not route the east fish pond. |
| Farm-bush `SHIP_BERRY` | `rr-r3he` | Not D2 grape |
| Gate B soak | `rr-5in` | Do not close on one D2 pin |

## Tests that currently hide this

- `_make_farm_ram(..., stamina=100, tool=HAMMER)`
- `test_stamina_stop_completes_clearer` (low stam → complete SUCCESS)
- `test_missing_startup_tools_do_not_fail_farm_clear_task` (lift-only SUCCESS)
- `test_input_lock_zero_is_locked_not_missing` (farm 17:00 lock=0 → shipping)
- No test: stump/rock/damage IDs not walkable
- No test: hammer-in-backpack still targets ROCK
- No test: NavTask BFS must not enter `WEED`
