# Pre-L1 loadout

Primary walkthrough: [Zelda Dungeon — The Gathering](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/the-gathering/)
(2015-01-23). Secondary: [IGN Preparation](https://www.ign.com/wikis/the-legend-of-zelda/Preparation)
(2025-07-16). First quest only. Grid: `screen = (row << 4) | col`, start
`0x77` = H8.

M5 Clean is still power-on → L1 Triforce on 3 containers and the wooden sword
(18909f, TF `0x01`). This prefix is the combat-budget answer: gather **before**
the 0x37 mouth, then re-enter L1 with 6 containers and the White Sword. Do
not overwrite the 18909f claim.

No pokes. Do not STATUS from a pin.

## This sitting (2026-09-16) — the frames, not the tuning

`OverworldPathController.step` now censuses the *stem* of every
`FrameAction.reason` per screen (`report()["reason_by_screen"]`). That census
is what this sitting is: four stalls were found by reading it, none of them
by tuning a number.

| Screen | Before | What owned the frames | After |
|--------|--------|-----------------------|-------|
| `0x7B` | 1863f, 6 hits | `hop_ay` 394 — aligning to a row nothing needed | 480f, 1 hit |
| `0x7C` | 24877f timeout | `hunt_slash_recover` 19965 — one body holding the contact rung | crossed |
| `0x7C` | 24914f timeout | `hop` 12459 / `hop_lane` 12455 alternating one frame each | 4197f |
| `0x6F` | died f593 | `shop_p_hunt_settle` 593 — an idle *stand* in guard | finishes |

**The align was the big one.** `probe_coast_lane` already said 0x7B, 0x7C and
0x7D scroll east from **every** row; the hop table still carried
`align_y=131/130/130` across them. On a leever screen that is not a lane, it
is a vertical shuffle in a swarm. Those three hops now carry
`SCREEN_ANY_ROW_BAND = (77, 205)` — the sweep's own extent, so the push never
corrects — and the natural drift lands inside `SCREEN_7E_EAST_BAND` anyway.
`pre_l1_anyrow1` walked all nine hops and **arrived 0x6F for the first time**:
28 kills, 14R.

Four structural caps went in with it, each priced by the census above:

- `ScreenHunter._strike_budget` — the contact strike was the top of the ladder
  with **no budget at all** (not `TargetBook`, three rungs down; not the
  screen budget, also below it). One body that will not die owned 24877
  frames. Identity is `(slot, type)`; the frames also spend the screen budget.
- `OverworldPathController._grinding` — per `(hop_index, screen)` frame
  budget (`DEFAULT_HOP_SCREEN_MAX_FRAMES = 4000`). Past it the optional rungs
  (scoop, hunt, occupied-lane) are switched off and the push is all that is
  left. The local caps do not compose: a lane steer that picks the travel
  direction resets its own counter, and `track_stuck` sees a Link who is
  moving.
- `destination_hunted` returns True in guard — `_after_hops` answers a
  declining hunt with an **idle**, and the hunt declines for the whole guard
  branch, so on the destination screen the two met as a 2400-frame stand.
- `HUNT_PICKUP_RADIUS = 72` / `HUNT_HEAL_MAX_FRAMES = 120` — 0x7E is four
  `octorok_fast` plus a Zora and one pass spent 240 frames of `hunt_heal`
  and 179 of `hunt_scoop` crossing it for one heart and 2R, taking five of
  twelve hits doing it.

### Picking the rupees up

- `ScreenHunter.cleared` is now separate from `done`. `done` means "stop
  chasing here"; a budget **retire** lands in `done` with the wave still
  walking. Only `cleared` scoops money — that is the 5R-on-the-floor gap
  (`pre_l1_beam4`: 24 dropped, 19 banked): the kill that empties a screen
  drops on the frame the screen goes `done`, and `done` used to collect
  `heal_only`. `screen_table` stopped calling a retire "cleared" too.
- `combat.heal_wanted` replaces `filled_hearts < heart_containers` in
  `path._rupee_scoop`. That nibble is whole hearts *minus one*, so the old
  test was true at full health on every container count and the walk detoured
  for hearts it could not bank.
- A heal rung (`hunt_heal`) sits **above the beam**: one `$0670` chip is
  exactly what takes the beam away, so on every frame the heal can claim, the
  shot below it is already dead. `scoop_heal_radius` is 96 where rupees keep
  48 — a heart is worth crossing a screen for and a rupee is not.

### Arrive short → fight next door (`overworld/topup.py`)

`bomb_topup` is a new stage between the walk and the buy. It is a no-op on any
pass that is not short: `_at_stop` is `rupees >= price` **on the shop screen**,
so it finishes on frame 1 when the walk already banked the pack.

Not the corridor behind it. `overworld.respawn` is the ROM's rule and it is
decisive here: `RoomHistory` is six slots, so on arrival at 0x6F the five
screens behind it are all still in the ring and none of their waves come back.
Walking back gives nothing until the **sixth** screen (0x7A) — twelve screens
of leevers and Zoras for one respawn.

The fresh fights are the neighbours the walk never entered, and both are now
measured out **and back** (`scratch/probe_6f_neighbours.py`, tag `n4`: one
boot, the real hop table to 0x6F, state saved on arrival, every candidate
row/column restored, pushed, censused, pushed back):

| Exit | Lands | Lanes | Out | Home | Wave (peak 6) |
|------|-------|-------|-----|------|----------------|
| `0x6F` UP | `0x5F` | every column 72–232 (align clamps at x=128) | 193–240f | DOWN 85f | 3 octorok_fast, 2 octorok_blue, 1 zora |
| `0x6F` LEFT | `0x6E` | y 93–197; **77 / 85 / 205 dead** | 155–377f | RIGHT 106f | 3 moblin, 2 octorok_blue_fast, 1 moblin_blue |

0x5F is first in the table: octoroks are two wooden hits and the ROM's drop
table pays them, where 0x6E is four moblins. Both are six-body waves — the
same size as the corridor screens that cost the walk its hearts, so this is a
real fight, not a lap of an empty screen.

**The stage has not run live yet.** The walk has to survive to 0x6F for it to
get a frame, and only `pre_l1_anyrow1` has.

## Current walk (2026-09-15)

`--through pre-l1` is sword → `overworld/shop_p7.py` hunting walk → 0x6F
buy. Screens: `0x77 → 0x78 → 0x79 → 0x7A → 0x7B → 0x7C → 0x7D → 0x7E →
0x7F → 0x6F`. Overlay Map-1.png gives the screen sequence; it does **not**
give a row that walks. `0x4A` is the later arrows cave
(`overworld/bomb_shop.py`). Do not join via `0x68` / `0x5C` maze / `0x5E`
candle.

### The east lanes are measured, not painted

`scratch/probe_coast_lane.py` (tag `l1`, 2026-09-15): one boot, the emulator
state saved on arrival, then 17 candidate rows a screen — walk to the row,
hold EAST, did it scroll?

| Screen | Rows that scroll east |
|--------|-----------------------|
| `0x79` | **165 only** (the south beach) |
| `0x7A` | **133 and 141 only** — 125 and 149 are dead, and ≤117 / ≥157 are not even reachable from the west mouth |
| `0x7B` | every row (Link converges on 133/141 anyway) |
| `0x7C` | every row |
| `0x7D` | every row |
| `0x7E` | 117, 125, **141 and below**; **133 is dead** |
| `0x7F` | none — the hop out of `0x7F` is UP into `0x6F` |

Two of those rows are the walk timeout. `align_y` carries `y_tol=5`
(`overworld.common.align_and_push`), so the painted 131 on `0x7A` **accepts
y=126**, which does not scroll: the 2026-09-15 retest sat 27501f at y≈126
pressing RIGHT with 53 occupancy misses. `0x7E`'s painted 130 accepts the
dead 133 the same way. Both are now `y_band`s of the measured corridor
(`SCREEN_7A_EAST_BAND`, `SCREEN_7E_EAST_BAND`) — a band is the honest shape,
because a point plus a tolerance reaches outside what was measured.

Do not BFS the `OccupancyWalker` across these screens to find a lane. `0x78`
is a tree maze and the 1px learned grid walled in its own start cell after
1173 frames (probe `c1`): the sweep has to be a row sweep from a restored
state.

Unassisted leftover (`pre_l1_beam3` / `pre_l1_beam4`, reproduced 2/2): died
**0x7C `(192,85)`** at 7500f on hop 5, **19R**, 24 kills, streak best 17,
2.996 hearts over 6 hits. One rupee short of the 20R pack, with 5R of the
24R dropped still on the floor — the scoop is the gap, not the kill rate.
0x6F has not been arrived live. Stop is `ADDR_BOMBS >= 1`, not arrival on
the shop screen.

| | retest (before) | measured lanes | + the sword shot |
|---|---|---|---|
| leftover | timeout 0x7A | died 0x7D | died 0x7C |
| rupees | 2 | 5 | **19** |
| kills | 8 | 18 | **24** |
| streak best / resets | — | 6 / 6 | **17 / 3** |
| hearts spent | — | 3.996 (8 hits) | **2.996 (6 hits)** |
| beam fired / aimed / ready | — | 0 / 0 / 539 | **47 / 567 / 3016** |

Remaining damage is **4 of 6 hits from `0x55`** (the Zora spit), plus two
leevers. The bodies are no longer the bill.

### Do not (this sitting)

- ButtonsPressed is an edge. Hunt `_a_edge`: press A, then idle.
  Held A does not re-swing.
- Travelling frames offer `ScreenHunter.take_beam` **above** stall-escape
  (`path._do_hop`; 600f commits used to zero the weapon).
- Zora facing `0x03` is missing from `threat._FACING_AXIS`. The `0x55` spit
  is that hole, not a shop_p7 special case.
- Scoop vs restock: `scoop_rupees` vs `need_rupees`. Coast scoops; `need_rupees=0`
  so no 0x78 farm loop. `laps=0`. Do not poke Food/bombs/keys.

`SHOP_P7_HOPS` is measured (`SCREEN_79_BEACH_Y`, `SCREEN_7A_EAST_BAND`,
`SCREEN_7E_EAST_BAND`), not the overlay. `take_beam` then scoop sit above
`_stall_escape`. 5R on the floor is still the gap.

## Do not walk (measured traps)

ZD counts screens on the 16×8 grid. Two of those counts are not corridors.

| ZD text | Grid decode | Why it fails |
|---------|-------------|--------------|
| 0x79 overlay centre / inland joins | `0x79` y≈125; `0x68`/`0x5C`/`0x5E` | Overlay centre lane is a rocky bowl (no east cell `x>=232`, no north scroll). `0x6A`/`0x6B` south are tree walls; `0x7B` north is a bomb wall. `0x68` east from x=48 is a bush wall. `0x6C` east is dead. `0x5E` east is a tree wall. The live walk does not take these; it skirts 0x79 on the beach |
| After candle, down then left two, climb to White Sword | `0x0C` → `0x1C` → **`0x1B`** → `0x1A` → `0x0A` | `0x1B` is Lost Hills (wraps all four ways; 4th UP is L5 `0x0B`). Bypass: `0x0C` → `0x1C` → `0x2C` → west → `0x1A` → N `0x0A` |
| Heart-1 sidequest "up 2, right 2" from `0x7B` | `0x7B` → `0x5B` → **`0x5C`** maze | `0x5C` needs `LEVEL2_5C_MAZE_WAYPOINTS`. Do not treat as a free RIGHT |

`0x67` is a dead-end **from start** (`0x77` north). Coming onto `0x67` from the
east (`0x68` west) is the ZD 30R bomb wall, and is fine.

## Validated destinations

Catalog names are `overworld/locations.py`. "Walk" is what we would actually
drive. "Grid" is ZD's screen-count, even when the corridor is blocked.

| ZD § | Dest | Catalog | Open | Walk | Notes |
|------|------|---------|------|------|-------|
| 1.1 | `0x6F` `shop_p7` | `CAVE_SHOP_ARROWS` | open | **Map-1 south coast.** `overworld/shop_p7.py`: 0x79 beach east into 0x7A, then `0x7B`…`0x7F` UP `0x6F`. `0x4A` is the later arrow cave, same shop family, not this errand | Bombs 20R. Cave mouth `(48, 77)` |
| 1.2 | `0x7B` `heart_l8` | take-any | bomb N wall | from `0x6F` D1 L4 (`0x7F`→`0x7B`) | 4 HC. Approach from the east so we never enter 0x79 |
| 1.2 | `0x2C` `heart_m3` | take-any | bomb lower-right of center rock | after NE rupee stops | 5 HC. Same stand as IGN |
| 1.3 | `0x0F` `rupees_100_p1` | 100R | secret N wall of `0x1F` | `0x2C` → `0x2D` → `0x1D` → `0x1E` → `0x1F`, hug N wall | IGN's "NE corner past the gambling den" |
| 1.3 | `0x0E` `letter` | letter | open | `0x1F` → `0x1E` → stairs N | Needed for the potion shop. No pickup controller yet |
| 1.3 | `0x0C` `shop_m1` | `CAVE_SHOP_CANDLE` | **open** | `0x0E` → `0x1E` → `0x1D` → `0x0D` → `0x0C` | ZD candle shop. Better than IGN's bomb-open `0x66` and better than the long `0x5E` L8 corridor. 60R |
| 1.3 | `0x0A` `white_sword` | white sword | open, **5 HC** | **not through Lost Hills** | Blue Lynel on the 0x1A→0x0A climb |
| 1.4 | `0x48` `rupees_i5` | rupees | secret / burn | we already walk 0x48 on L1 | 30R burn, top-right bush |
| 1.4 | `0x47` `heart_h5` | take-any | burn 5th bush from the right | `0x48` LEFT y=141 | 6 HC. Pocket measured (`HEART_H5_*`) |
| 1.4 | `0x46` `shop_g5` | `CAVE_SHOP_ALT` | burn corner bush | `0x47` LEFT | Magical Shield **90R**. ZD: bait is also on this counter (buy later, not now) |
| 1.5 | `0x4A` `arrow_shop` | `CAVE_SHOP_ARROWS` | open | existing L2 prefix | Arrows 80R. Buy controller exists. Farm is still the gap |
| 1.5 | `0x6B` `rupees_100_l7` | 100R | secret / burn | `0x4A` east then south | Third-column lower bush |
| 1.5 | `0x67` `rupees_h7` | rupees | bomb N wall | `0x6B` L4 onto 0x67 from the east | 30R. Do not reach this by walking N from start |
| 1.5 | `0x64` `potion_e7` | potion | secret | `0x67` L3 | Show the Letter. 2nd Potion |
| 1.6 | `0x62` `rupees_100_c7` | 100R | burn, 3rd bush from top in the center | from the south-west 30R | IGN's brown-shrub 100R, ZD has the stand |
| 1.6 | `0x34` `special_shop_e4` | bait_or_blue_ring | Armos, **top-middle** | `0x51` R3 U2 | Blue Ring 250R. Pin slot order before a buy. Do not poke `ADDR_FOOD` |

Open-method mismatches (ZD vs catalog), live-pin before a hop:

- `0x3D` `rupees_n4`: ZD "right Armos 30R", catalog `OPEN_BURN`
- `0x56` `gamble_g6`: ZD burn 10R, catalog `OPEN_BOMB`
- `0x51` `gamble_b6`: ZD burn 10R, catalog gamble

## Order (ZD, not IGN)

Sword → farm while walking to `0x6F` bombs → `0x7B` heart → `0x2C` heart (5 HC)
→ `0x0F` 100R → Letter → candle `0x0C` → White Sword `0x0A` → `0x47` heart
(6 HC) + 90R shield `0x46` → arrows `0x4A` if 80R → potion `0x64` → Blue Ring
`0x34` → L1 `0x37`.

IGN bought the candle at `0x66` (bomb-open, next to start) and the White Sword
before the burn heart. ZD's NE-coast cluster (100R, Letter, candle `0x0C`,
White Sword) is one trip and skips `0x66`.

## Wiring

Dedicated `--through pre-l1`. Not spliced onto `level1_survival_tf_stages`.
`gathering.py` is the Composer row (sword + walk + buy). M5 18909f stays
the wooden 3HC oracle. Re-measure L1 after this prefix greens.
Hop dest and screen sequence come from Map-1.png (`overworld.zd_map`).
Live 0x79 east is the beach (`shop_p7.SCREEN_79_BEACH_Y`), not the overlay
centre lane and not the L8/candle corridor. `SHOP_P7_HOPS` is that
measured table (beach + the two east `y_band`s), not painted centre
rows. `take_beam` then scoop sit above `_stall_escape` on the hop
ladder. Coast `scoop_rupees=True` with `need_rupees=0`.

The 3HC L2 door suffix still dies on 0x4C even with evade-on (`rr-8t4.4-residual`
2026-09-14 census): last playable `(121,133)` hp `0x30` 0/4. Two whole hearts
were already gone. 6 HC would have had budget left; occupied-lane on hop 5
is still required so Link does not walk onto the body.

## ROM facts (any hunting walk)

These hold on any hunting walk. Corridor-specific numbers from the old
inland 0x4A prefix are labeled as such; they are not this dest.

**ButtonsPressed is an edge.** `Link_HandleInput` (`Z_05.asm`) wields on
`ButtonsPressed AND #$80`. `ButtonsPressed` is the edge (`Z_07.asm`:
`new EOR ButtonsDown AND new`), so a **held** A swings once and never
again. `ScreenHunter._strike` presses A one frame, then idles. The release
frame is an idle, never the direction. `_approach`'s blocked-align fallback
goes through the same edge.

**`$066F` lo nibble is whole hearts minus one.** `0x22` is 3/3.
`ram.whole_hearts` is the honest read. `filled_hearts` keeps the raw nibble
because the L1 chain is frame-perfect against it. `hits_taken` watches
`$066F` only; a wooden chip lands in `$0670` (`$80`) and never moves the
whole-heart byte. Read `hunt.report()["damage_taken"]`.

**`Link_BeHarmed` zeros `$50` / `$627`**, then subtracts damage and grants
`$04F0=24`. Survival assist writes `$0670` back to `$FF` the same frame, so
`hits_taken`, `damage_taken`, and `assist.damage_events` all read 0.
`--through pre-l1` forces assist off even if the caller passed one.

**20R pack is 36 unbroken kills.** `$0627` counts every kill; only
`Link_BeHarmed` zeroes it. `$0050` caps at 10 and any forced drop zeroes it.
`$0627 == 16` is tested **before** `$0050 >= 10`, so the 16-kill fairy
spends six kills of 5-rupee progress. A clean streak pays at 10, 26, 36, 46
— not 10, 20, 30. 20R is 36 unbroken kills in expectation, 46 if every
random roll cancels.

**Overworld waves are one-shot.** `ModifyObjCountByHistoryOW` clears a
screen's kill flags only when it is **absent** from the six-entry
`RoomHistory` (`$621`) and those flags read 7.
`RunCrossRoomTasksAndBeginUpdateMode` appends a room **only if it is not
already in the history**. An out-and-back evicts nothing at any depth. A
lap needs 7 distinct screens. (Old inland prefix, not this walk: the
`0x4A<->0x49` restock never was a farm for this reason.) `laps` stays 0
until a coast lap is measured better than one clean pass.

**Zora spit.** 195-frame surfacing clock on `$00AC`. The `0x55` shot sits
motionless on the muzzle for 17 frames, so `ObjectTracker` reads zero
velocity and `threat.assess` calls it safe for the whole dodge window.
Type `0x55` is not small-shield blockable. A zora's facing byte reads
`0x03`, which is missing from `_FACING_AXIS`, so `in_firing_line` has
never returned True for one.

**Scoop is the money gap.** Drops hit the floor. A kill census is not a
banked-rupee census. `CombatLedger.report` has `rupees_dropped` /
`rupees_left`.

**The flying sword is a real weapon on this corridor** (`zelda_i/beam.py`).
`Z_07.asm MakeSwordShot` puts the shot in object slot `$0E` when the blade
(slot `$0D`) reaches state 3, the low nibble of `$066F` equals the high
nibble, and `$0670 >= $80`. `Z_01.asm
CheckMonsterSwordShotOrMagicShotCollision` hands it the **blade's own** damage
points (`$10` wooden) and damage type (1), and its kill runs
`HandleMonsterDied` — so a beam kill ticks `$0627` / `$0050` and rolls the
same drop table as a melee kill. It is free reach, not a second economy.
Wooden `$10` one-shots a blue tektite, a red tektite and a red octorok (ROM
`ObjectTypeToHpPairs`: all `$10`); a blue octorok and a zora are `$20`, a
blue leever `$40`.

Measured live (`scratch/probe_beam.py`, 0x77, hp `0x22`/`0xFF`, all four
directions): **3.0 px/frame**, flying state `$10`, spreading state `$11` for
~22f, and the shot flies to the room bound — there is no distance decay, so
"range" is the screen. Muzzle offset from Link is ±19 px horizontally,
−16 UP and +30 DOWN. A live shot is its own cooldown: `MakeSwordShot`
returns early while slot `$0E` is non-zero.

**One chip takes the weapon away.** `Link_BeHarmed` subtracts damage from
`HeartPartial` *before* borrowing a whole heart, so a wooden octorok's `$80`
takes a full `$FF` to `$7F` — one below the gate. The beam is an *at full
health* weapon, which is why the hunt fires it early and why a heart on the
floor is now worth more than its buffer: it hands the weapon back.

**Reach is worth nothing to a layer that never spends the edge.**
`scratch/probe_beam.py --phase lane` (tag `b3`, the live coast walk): the
shot was up for **807** of 6218 frames and a body stood in a 9 px lane on
**361** of them — and `ScreenHunter.step` reached its own beam branch on
**262** and aimed on **0**. Those are disjoint sets. While Link is
*travelling*, the hop table owns the frame and `common.walk_or_swing` only
presses A at contact range; a body straight ahead in a lane is exactly the
geometry a hop produces. `ScreenHunter.take_beam` is the shot offered
to that layer from `_do_hop` before recovery.

**Where it sits in the hop ladder is the whole weapon.** Offering it *below*
the movement-recovery layers changed nothing at all (`pre_l1_beam2`: 539
ready frames, 0 aimed): probe `b4` caught a red octorok sitting 3 px off
Link's row 132 px ahead while one `hop1_escape` commit — `_stall_escape` is
600 frames — owned every frame of it. `take_beam` then scoop now run
above `_stall_escape` / occupancy / `unstick_wiggle` /
`recover_off_edge`, and below the reactive evader (`path._threat_action`,
which still runs first). It is self-limiting: `MakeSwordShot` refuses while
slot `$0E` is live, so a held lane costs one press per shot, not one per
frame.

**Stop is `ADDR_BOMBS >= 1`**, not arrival on a shop screen. Arrival cannot
tell a walk that banked 20R from one that banked one rupee.
