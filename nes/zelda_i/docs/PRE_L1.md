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

## Do not walk (measured traps)

ZD counts screens on the 16×8 grid. Two of those counts are not corridors.

| ZD text | Grid decode | Why it fails |
|---------|-------------|--------------|
| From start, right 8 then up 1 (bomb shop) | `0x77` → … → `0x7F` → `0x6F` | `0x79` is a rocky pocket (`graph.SCREEN_LABELS`, L8 bush trap). Enter from `0x78` east @ y≈180, **no east exit**. Live east-of-start is `0x77` → E `0x78` → N `0x68`. 0x68 east from the west sand column is a bush wall (live 2026-09-14). Next bypass: `0x58`/`0x59` SOUTH onto row 6 |
| After candle, down then left two, climb to White Sword | `0x0C` → `0x1C` → **`0x1B`** → `0x1A` → `0x0A` | `0x1B` is Lost Hills (wraps all four ways; 4th UP is L5 `0x0B`). Bypass: `0x0C` → `0x1C` → `0x2C` → west → `0x1A` → N `0x0A` |
| Heart-1 sidequest "up 2, right 2" from `0x7B` | `0x7B` → `0x5B` → **`0x5C`** maze | `0x5C` needs `LEVEL2_5C_MAZE_WAYPOINTS`. Do not treat as a free RIGHT |

`0x67` is a dead-end **from start** (`0x77` north). Coming onto `0x67` from the
east (`0x68` west) is the ZD 30R bomb wall, and is fine.

## Validated destinations

Catalog names are `overworld/locations.py`. "Walk" is what we would actually
drive. "Grid" is ZD's screen-count, even when the corridor is blocked.

| ZD § | Dest | Catalog | Open | Walk | Notes |
|------|------|---------|------|------|-------|
| 1.1 | `0x6F` `shop_p7` | `CAVE_SHOP_ARROWS` | open | **bypass 0x79** via `0x78`→`0x68`; **0x68 east from x=48 is dead** (bush / occupancy). Next: `0x58`/`0x59` SOUTH onto `0x69` then east | Bombs 20R. Same shop family as `0x4A`. Farm drops on the way **and** the `0x4A` tektite wave — six row-1 bodies at 0.891 R/kill are 5.3R of the corridor's 8.4R (2026-09-15 budget) |
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
M5 18909f stays the wooden 3HC oracle. Re-measure L1 after this prefix greens.

The 3HC L2 door suffix still dies on 0x4C even with evade-on (`rr-8t4.4-residual`
2026-09-14 census): last playable `(121,133)` hp `0x30` 0/4. Two whole hearts
were already gone. 6 HC would have had budget left; occupied-lane on hop 5
is still required so Link does not walk onto the body.

## This sitting (2026-09-14) — 4.5.1 Bomb Shop Route Landed (0x4A)

Claimed `rr-ps7.4` (ladder row `pre_l1`). Dedicated `--through pre-l1`
boots power-on, clears sword cave on `0x77`, then walks to bomb shop on `0x4A`.

### Corridor Geometry & Row 6 Blockade
- Attempted row 6 corridor (`0x69` -> `0x6A` -> `0x6B` -> `0x6C`): cleanly
  walked to `0x6C`, but `0x6C` east is **dead** (walled west pocket at `x <= 64`;
  solid vertical bush column at `x≈80` blocks through-access to `0x6D`).
  `0x6D` also has no west entrance, and `0x5E` east is a tree wall.
- `0x4A` and `0x6F` belong to the identical `CAVE_SHOP_ARROWS` shop family in ROM
  (`AttrsB >> 2 == 0x1D`), selling Bombs 4-pack for 20R on the mid pedestal.
- `0x4A` is reached via the proven live corridor:
  `0x77` --RIGHT align_y=140--> `0x78` --UP align_x=48--> `0x68` --UP align_x=48--> `0x58` --RIGHT y148-162--> `0x59` --UP align_x=112--> `0x49` --RIGHT align_y=141--> `0x4A`.

### Continuous Spine Measured
- Continuous spine `--through pre-l1 --no-video --trials 1` is **GREEN** (1/1):
  `trial0: ok=True failed=None tf=0 room=0x4a keys=0 bombs=0 rupees=1 set_state=0`.
  Walk time after sword: 1924 frames (~32s), 0 hits taken, full 3/3 hearts (`hp=0x22`).

### Rupee Farm Mechanics Analysis
- When short of 20R, `RupeeFarmController` previously stalled due to three root causes:
  1. `DEFAULT_FARM_Y_LO = 120` in `rupee_farm.py` filtered out all overworld
     octoroks on both `0x78` (spawn y=90..109) and `0x4A` (spawn y=93..110).
  2. Slot 11 (`type_id=0x64`, `hp=240`, cave/NPC trigger) was parsed as prey
     by `obj.slot >= 1 and obj.type_id not in (0, 0xFF, 0x60)`.
  3. `farm_leave` pulsed for a single frame before zeroing `empty_frames`,
     causing `farm_wait` oscillation without screen scroll.
- Cave entry on `0x4A` (`(176,77)`) is verified and reached in 3031 frames total from power-on.
  Next action: bank 20R on the walk / via unsticking rupee farm, then buy bombs in cave.

## This sitting (2026-09-14) — the walk is the farm (kills are a metric)

`--through pre-l1` still ends on `0x4A`, but it now clears every screen on the
way instead of crossing it. New module `overworld/hunt.py` (`ScreenHunter`),
wired by `OverworldPathController.hunt` (off by default, on for
`ShopP7WalkController`). The rupee scoop is no longer gated at `BOMB_SHOP_PRICE`
— bombs are purchase 1 of 14.

### Measured, 1/1 green (`prel1_hunt7`)

| | |
|---|---|
| result | `ok=True room=0x4a`, walk stage 4010f (was **1924f** unhunted), end 4958f |
| kills | **14** slot census / **13** ROM counters |
| per screen | `0x68` 3, `0x58` 4, `0x59` 1, `0x49` 6 |
| peak live per screen | `0x78` 3, `0x68` 4, `0x58` 4, `0x59` 5, `0x49` 6 |
| damage | `hits_taken` 0 **and** `damage_taken` 0 (the partial-heart census) |
| rupees | **0** |
| screens | 4 cleared, `0x59` retired on its budget (2 slots skipped) |

M5 Clean re-measured after the change: `--natural-entry --trials 2` 2/2,
TF `0x01`, **18909f** both trials. Unchanged.

### The corridor's random table is thin; the forced 5-rupee died to contact

Fourteen kills produced **two** floor drops — a fairy on `0x58`, a heart on
`0x49` — and **no rupee**. Not a pickup bug: every `0x60` slot that appeared
was logged. Red octoroks (`0x07`/`0x08`) are drop-table row 0 (Baxter A,
31%): hearts and 1-rupees. That is not CLEANUP_PLAN 4.5.1's group B (blue
octorok `0x09` / blue moblin `0x03`, 41%, bombs). Random drops will not
fund 20R here.

The forced 5-rupee at 10 consecutive kills is the reliable money, and it
never fired because **Link walked onto the bodies**. `Link_BeHarmed`
(aldonunez `Z_01.asm`) zeros `$50`/`$51`/`$627` on collision, *then*
subtracts damage. A wooden octorok chip is `$0670` `$80` — `$066F` does not
move. Survival assist writes `$0670` back to `$FF` the same frame, so
`hits_taken`, `damage_taken`, and `assist.damage_events` all read 0.

Re-measured 2026-09-15 (`scratch/probe_kill_streak.py`, assist on, 1/1
green, 14/13 kills, 0 rupees): **6 streak resets, each with `$04F0=24` and
knockback 32**, health still `0x22`/`$FF`. Peak 4 is the 0x68 three-kill
plus the first 0x58 kill, then contact. Two more iframe arms on 0x59 peahats
after the streak was already 0. `hunt.report()["hurt_events"]` watches
`$04F0`; that is the census that agrees with the resets.

2026-09-15 sword-reach + assist off (`prel1_reach`, 1/1 green, walk 3978f):
**3 rupees**, 11/9 kills, peak 7, 2 resets, `hurt_events` 4, `damage_taken`
5, leftover `0x4A (0,141)` hp `0x20` 0/3. The refill is gone (`assist=None`)
and the hunt no longer occupancy-walks onto the sprite (peel / in-place A /
`sword_stand`). Two contacts remain, so the forced 5-rupee still does not
fire. Next: kill those two resets (0x49 `hunt_hurt` is one). Do not STATUS.

### Two traps this sitting paid for

- **A hop cannot leave a pocket.** `align_and_push` holds one direction and
  `unstick_wiggle` waits forever once its wiggle is spent. Hunting walks Link
  off the lane, and two runs ended wedged at `(56,125)` — one on `0x49`, one on
  `0x58` — burning 27,809 and ~28,000 frames on `unstick_wait` / `band_down`.
  Fixed with `_stall_escape`: an occupancy walk toward the hop's exit on the
  grid the chase just learned. It is **started by the stall** (`stuck >
  stuck_threshold`) and only where `hunt` is on.
- **A stall escape must be committed.** Consulting it per frame is useless:
  `stuck` resets the instant it moves Link a pixel, which hands the frame back
  to the rule that wedged him. Measured 27,809 `band_down` against 531
  `hop3_escape`, Link never leaving the pocket. It now owns the screen until
  the hop scrolls or 600 frames pass.

- `hits_taken` is blind to half-heart damage: it watches `$066F` (whole
  hearts) and a chip hit lands in `$0670`. `hunt.report()["damage_taken"]`
  watches both. Do not read `hits_taken=0` as "took no damage".

## This sitting (2026-09-15b) — the errand is wired end to end, and it is short

`--through pre-l1` is now **sword → hunting walk → buy**, and the stop is the
4-pack (`ADDR_BOMBS >= 1`), not arrival on `0x4A`. Arrival cannot tell a walk
that banked 20R from one that banked one rupee. The run is therefore **red**,
and the last stage says exactly why: `shop_need_20_have_2`.

Live `prel1_land`, `--no-video --trials 1`, `assist=None`, `set_state=0`:

| | 8c7162cc | now |
|---|---|---|
| kills (census / ROM counters) | 10 / 9 | **16 / 15** |
| `streak_best` | 3 | **7** |
| `streak_resets` | 2 | 3 |
| rupees | 1 | **2** |
| screens fought | `0x68` `0x58` `0x49` | + `0x59`, + **`0x4A` (4 of 6 tektites)** |
| hearts at the shop | `0x20` (1 of 3) | `0x21` (2 of 3) |
| `damage_taken` / `hurt_events` | 4 / 4 | 4 / 4 |
| stop | arrival on `0x4A` (green) | bombs (red, `shop_need_20_have_2`) |

### Five root causes, all measured off `scratch/probe_contact.py`

- **The heart gate was reading one heart low.** `$066F`'s low nibble is whole
  hearts *minus one* — `0x22` is 3/3 and the walk ended alive on `0x20`, which
  is one heart, not zero — so `snap.filled_hearts` is `hearts - 1`.
  `HUNT_MIN_HEARTS = 1` against it retired `0x58`, `0x59` **and** `0x49` after
  a single chip hit, while Link still held two of three hearts, and then the
  hop — which has no combat at all — walked him into two octoroks at 9px. New
  `ram.whole_hearts` is the honest read; `filled_hearts` keeps the raw nibble
  because the L1 chain is frame-perfect against it.
- **Giving up is not a policy.** Below `min_hearts` the hunt now *guards*: it
  still answers a body in the pad with the blade, still banks a heart off the
  floor, and still spends the screen budget — it only stops chasing.
- **The swing was on a blind cadence.** `frames % 8 < 3` idled up to five
  frames per swing while a body closed ~1px a frame (`0x49` f=3653: ten frames
  standing while slot 4 walked 16 → 9). Link's own object state (`$00AC`
  slot 0) is non-zero for the whole animation, so the hunt now presses A the
  frame the blade box fills and idles only while the swing runs — which is
  also the release edge the ROM needs before the next swing.
- **Two reactive layers, and the wrong one won.** `OverworldPathController`
  runs its `ReactiveEvader` ahead of every hop rule, the hunt included. A red
  octorok is one wooden hit, so stepping away from a body already inside the
  blade box trades a kill for a frame of separation it gives straight back.
  The evader now yields on `evade_yield_to_sword` (306 frames live) when
  `ScreenHunter.striking` says the blade reaches.
- **The hop table ends on the frame Link scrolls onto the destination**, so
  `0x4A`'s six blue tektites — drop-table row 1, the richest bodies on the
  corridor — were the one wave a hunting walk never saw.
  `hunt_destination` + `ScreenHunter.take_destination` fight it on a budget of
  its own (2400f / 420f per target, because tektites hop). 1 kill → 4.

The shield is on and instrumented (`shield_frames` 46 = 41 holds + 5 turns).
It is a *modifier* now, never a mode: it can hold a swing Link was going to
make while already facing a shot, or spend a frame the hunt was going to idle,
and it is silenced by a body that is **closing**, not by a flat pad. The flat
36px pad is why three gatings were byte-identical last sitting — on this
corridor something is always inside 36px, so the shield never got a frame.

### `0x48` is not worth taking on three containers

Four leevers at drop-table row 1 are 3.6R, more than the whole five-screen
octorok corridor, and a there-and-back detour off `0x58` uses only hops
`LEVEL2_PATH_HOPS` already proves. It **killed the run twice** (`fixG`,
`fixH`, both mode 17): Link scrolls onto `0x48` at y≈205, *below* `HUNT_BOX`,
so the hunt cannot fight there at all, and he arrives with one heart because
`0x58` comes first. Reverted. Revisit after the `0x7B` / `0x2C` containers.

Also reverted: skipping the `0x59` peahat wave. The arithmetic is sound
(row 3, 0.081 R/kill, invulnerable in flight, and two of one run's four
contacts) but the one live measurement came back worse on every axis
(13 kills / best 5 / 5 resets / 0R). One deterministic trajectory cannot
settle it, and a change that cannot be measured does not land.

### The drop table is the real blocker, not the fighting

`drops_by_state` (new, on `combat.CombatLedger`) separates "row 0 rolled no
rupee" from "the rupee was there and the hunt walked past it". Live: **16
kills produced three floor drops** — two 1-rupees and one heart, no 5-rupee.

Across this sitting's runs that is **53 kills for 7 drops, ~13%**, against the
31% / 59% `DropItemRates` rows `scratch/bomb_budget.py` bills the corridor at.
At the measured rate one pass pays ~2R, and a *perfect* 26-kill streak on 28
bodies would add 10R forced against ~3.5R random — **13.5R, not the 22R the
budget table predicts.** One pass cannot buy the 4-pack. Chase the 13% before
chasing more screens: either the table is being read wrong (class per enemy,
`$052A` cycle) or something is eating drops before the census sees them.

`0x78` is the one screen with a wave and no fight, and that is **correct, not
a gap**. New `peak_prey_by_screen` (prey *inside* `HUNT_BOX`) against the
ledger's `peak_live_by_screen` (live bodies anywhere) reads `live 0x78: 3`,
**no `0x78` prey entry at all**: its three octoroks never enter the box. They
sit on the west scroll column Link arrives through, and a chase there scrolls
him back onto `0x77` under the hop table. The 0.62R stays on the floor.

### Next

1. The 13% drop rate against the ROM's 31%/59% tables. This is the errand.
2. The ≥6-screen respawn loop is still untested, and is still the only way
   past one pass — but on three containers a second lap is a death, not a
   farm (see `0x48`). Hearts first.
3. `shield_swings_held` is 0 live: the swing-hold half of the shield has
   never fired. Either the corridor has no frame for it or the gate is wrong.

## This sitting (2026-09-15) — what 20R actually costs, and the rocks

`scratch/bomb_budget.py` is the arithmetic (ROM rows + the live spawn table);
`scratch/probe_contact.py` is the measurement (a 48-frame ring buffer dumped
on every `$04F0` arm, so the collider is named rather than guessed).

### The bomb pack costs 36 unbroken kills, or 128 kills without a streak

Per-kill expectation, from `DropItemRates` / `Types0..3` (see
`scratch/drop_mechanics_rom.md`):

| ROM row | letters | P(drop) | R/kill | kills per 1R | bomb packs/kill |
|---|---|---|---|---|---|
| 0 | A / A | 0.312 | 0.156 | 6.4 | — |
| 1 | **B / C** | 0.594 | **0.891** | 1.1 | — |
| 2 | **C / B** | 0.406 | 0.122 | 8.2 | **0.122** |
| 3 | D / D | 0.406 | 0.081 | 12.3 | — |

**The letters collide.** `scratch/drop_mechanics_rom.md` follows Baxter
(row 1 = B, the two-5-rupee table); `overworld/locations.py` calls that same
table `DROP_C` and the bomb table `DROP_B`. Contents and rates agree — only
the letter differs. Key on the ROM row. CLEANUP_PLAN 4.5.1's "group B" is
`locations.py`'s B: blue octorok, the *bomb* table, not the 5-rupee one.

The forced drops, on one unbroken streak (`$0627` counts every kill and only
`Link_BeHarmed` zeroes it; `$0050` caps at 10 and any forced drop zeroes it):

| kill | drop | E[rupees] |
|---|---|---|
| 10 | forced 5R | 6.4 |
| 16 | **forced fairy** | 7.2 |
| 26 | forced 5R | 13.6 |
| 36 | forced 5R | **20.0** |
| 46 | forced 5R | 26.4 |

`$0627 == 16` is tested **before** `$0050 >= 10`, so the 16-kill fairy
spends six kills of 5-rupee progress: a clean streak pays at 10, 26, 36, 46
— not 10, 20, 30. **20R is 36 unbroken kills in expectation, 46 if every
random roll cancels, and 128 kills on row-0 octoroks with no streak at all.**

### The corridor's supply, and where the money actually is

One pass, from the ROM spawn table (`locations.q1_spawns`) with the two
grouped screens taken from the live census instead — a grouped spawn byte is
a group index, not an ObjType, and `0x49`'s reads `group_28` where `0x28` is
also Rope (row 1), which would credit it 5.3R it does not have:

| screen | prey | n | row | random |
|---|---|---|---|---|
| `0x78` | octorok | 4 | 0 | 0.62R |
| `0x68` | octorok | 4 | 0 | 0.62R |
| `0x58` | octorok ×2 / octorok_fast ×2 (live) | 4 | 0 | 0.62R |
| `0x59` | peahat | 4 | 3 | 0.33R |
| `0x49` | octorok_fast ×5 + octorok_blue ×1 (live) | 6 | 0, 2 | 0.90R + 0.12 packs |
| **`0x4A`** | **tektite_blue ×6** | 6 | **1** | **5.34R** |
| `0x48` (one hop off) | leever ×4 | 4 | 1 | 3.56R |

**`0x4A`'s six tektites are worth more than the whole five-screen octorok
corridor that leads to them** — 5.3R of the walk's 8.4R, from 6 of 28 bodies.
The 1.1 row in the table above says "Farm drops on the way, not the 0x4A
tektite wave"; on the economics that is backwards, and `0x48` (four leevers,
same row) is one hop off `0x58`.

What a single pass can pay, 32 bodies:

| best unbroken streak | forced | random | total |
|---|---|---|---|
| 0 | 0R | 12.0R | 12.0R |
| 10 | 5R | 12.0R | 17.0R |
| **26** | **10R** | 12.0R | **22.0R** |

So one pass **can** buy the pack, but only by killing `0x4A` and `0x48` and
holding a 26-kill streak. Today's walk holds 3. Overworld waves are one-shot
at depth 1–2 (AGENTS.md), which matches the ROM: OW respawn wants the screen
out of the **6-room history** with its kill bits at max, and a depth-1 or
depth-2 round trip never leaves the history. A ≥6-screen loop is untested.

### Two of the four contacts are octorok rocks, not bodies

`probe_kill_streak.py` filtered slots to `0 < hp < 200` and so could not see
the thing that was hitting Link. With hp-0 slots kept, the baseline walk
(`scratch/contact1.json`, the committed b0328ea0 code) reads:

| f | screen | collider | reset |
|---|---|---|---|
| 1763 | `0x68` | octorok #3 body, d=17 | no (streak was 0) |
| 4500 | `0x49` | octorok_fast #4 body, d=9 | **yes, lost 7** |
| 4567 | `0x49` | octorok_fast #6 body, d=15 | no |
| 4732 | `0x49` | **`rock_projectile` #11** | **yes, lost 1** |

and the current walk (`contact7`) reads two rocks (`f=2056` mid-swing at
reach, `f=2275` walking to a drop with no body inside 70px) and two bodies
taken **after** `hunt_hurt_49` retired the screen and the hop drove.

`Link_BeHarmed` does not care which. **"The hunt walks onto the bodies" was
half the story** — half the streak resets on this corridor are shots.

### The engage ladder: react to the nearest body, no peel inside the pad

Two named defects, both from the `contact1` windows:

- **A 17–18px dead band.** `in_sword_hitbox` is true out to 20, but the
  turn-to-face branch was guarded by `pad > MIN_DODGE_BODY + 2`. At `0x68`
  `f=1763` Link stood ten frames facing WEST, pulsing A at the wall, with the
  octorok 17px EAST. The swing now carries its direction (`nes_action(face,
  "A")` — the facing byte is read before the sword and the attack state pins
  Link), so no frame turns without swinging and none swings the wrong way.
- **The pad was target-only, and the peel was wall-blind.** At `0x49`
  `f=4500` the 7-kill streak died to slot 4 closing 9px on Link's lane while
  the walk was aimed at a stand 30px north of slot 1; at `f=4567` slot 6 sat
  at 15px, also not the target. `_engage` now reacts to `_closest_body` and
  only *walks* toward the held target. Inside `MIN_DODGE_BODY` it swings
  instead of peeling: a sidestep has to walk the whole pad before it clears
  the hitbox and the body closes ~1px/frame, so the measured peel pressed
  LEFT into a bush for eight frames while slot 4 closed 16 → 8.

Live `contact7` (`assist=None`, `set_state=0`): `ok=True` room `0x4a`,
**1 rupee**, 10/9 kills, `streak_best` **3**, 2 resets, 4 contacts. Both
named defects are gone from the contact list. **The headline moved down**
(`contact1`: 3R, best 7): on a deterministic single-trajectory spine any
timing change reshuffles the whole downstream RNG stream, so 1-vs-1 rupee
counts are not a controlled comparison. Do not quote this as an improvement.

### The shield is wired, tested, and off

Two contacts are blockable rocks and the small shield eats them for free
while Link faces them and is not attacking (`behaviors.shield_blocks`), so
`ScreenHunter.shield` exists: `ObjectTracker` velocity, `threat.assess` to
pick the soonest hazard, face the `approach_side`, never press A. **Every
live walk with it on ran Link out of hearts on `0x49`** (`contact4`,
`contact5`, `contact6` — mode 17, and byte-identical trajectories under
three different gatings, so the gating was never what bound). It defaults to
`shield=False` until a sitting can measure which frames it is stealing.

### Next

1. Instrument the shield (`shield_frames`, per-reason counts) and find the
   frames it steals before turning it on.
2. `hunt_hurt` at `filled_hearts <= 1` retires the screen and hands a live
   wave to a hop with no combat at all — both `contact7` body contacts are
   after a retire. The hunt gives up exactly when the hearts matter most.
3. Hunt `0x4A` and `0x48` (row 1, 8.9R between them) instead of skipping them.
4. Measure a ≥6-screen respawn loop against the 6-room history.

Do not STATUS. Do not overwrite M5 18909f.

## This sitting (2026-09-15c) — the bill, screen by screen

The run totals were hiding the answer. "16 kills, 4 hits, 2 rupees" reads the
same whether the damage was one screen or five, and whether the money came
from the row-0 octoroks or the one row-1 wave that is worth more than all of
them. Everything is now billed per screen, and damage is in **1/256 of a
heart** — a wooden chip is `$0670 -= 0x80` and never moves `$066F`, so a
whole-heart census reads the corridor's damage as almost zero.

New: `combat.ScreenTally` / `CombatLedger.screens` (frames, kills by ObjType,
distinct slot-lifetimes by ObjType, damage and heal units, rupees, hearts
in/out, streak in/out, drops), and `ScreenHunter.damage` — the dungeon's
`postmortem.DamageLog`, wired to the overworld tracker, so each hit is
attributed to the body or shot that was there a frame before the knockback.
`ScreenHunter.screen_table()` merges the two. Renderer and run:
`scratch/probe_screen_tables.py` (`tables1.json` / `tables1.md`).

Live `tables1`, `--through pre-l1`, `assist=None`, `set_state=0`. Same
trajectory as `prel1_land` — 16/15 kills, best 7, 3 resets, 2R, 4 hits — so
this is that run, instrumented, not a new one.

| screen | frames | hunt f | hearts in -> out | damage | hits | cause | R | kills | spawned | streak | cleared |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `0x77` | 183 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 0 | 0->0 | yes |
| `0x78` | 114 | 0 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 0 | 3 | 0->0 | no |
| `0x68` | 566 | 322 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 2 | 4 | 0->2 | yes |
| `0x58` | 646 | 397 | 3.00/3 -> 3.00/3 | 0.00 | 0 | - | 0 | 3 | 4 | 2->5 | no |
| `0x59` | 739 | 533 | 3.00/3 -> 1.99/3 | **1.00** | 2 | `fireball_or_statue_projectile_E` `peahat_S` | 0 | 1 | 5 | **5->0 (reset)** | no |
| `0x49` | 684 | 434 | 1.99/3 -> 1.49/3 | 0.50 | 1 | `octorok_blue_W` | 1 | 6 | 6 | 0->4 (reset) | yes |
| `0x4a` | 2618 | 2401 | 1.49/3 -> 1.99/3 | 0.50 | 1 | `tektite_blue_E` | 1 | 4 | 6 | 4->1 (reset) | yes |

| screen | spawned red | spawned blue | other | killed red | killed blue | killed other | ROM row | drops | R |
|---|---|---|---|---|---|---|---|---|---|
| `0x78` | 3 | 0 | - | 0 | 0 | 0 | 0 | - | 0 |
| `0x68` | 4 | 0 | - | 2 | 0 | 0 | 0 | - | 0 |
| `0x58` | 4 | 0 | - | 3 | 0 | 0 | 0 | - | 0 |
| `0x59` | 0 | 0 | **peahat x4 zora x1** | 0 | 0 | 1 | 3 | - | 0 |
| `0x49` | 5 | 1 | - | 5 | 1 | 0 | 0,2 | 1R x1 | 1 |
| `0x4a` | 0 | 0 | tektite_blue x6 | 0 | 0 | 4 | 1 | 1R x1 heart x1 | 1 |

Red is `octorok`/`octorok_fast` (`Types0`, 31%); blue is `octorok_blue`/
`octorok_blue_fast` (`Types2`, 41%, the bomb table). Drop rate this run:
**3 floor drops on 16 kills, 19%** against 39% billed across the rows
actually killed (10 row-0, 4 row-1, 1 row-2, 1 row-3). E[R] 5.33, banked 2.

### Not one red octorok landed a hit

Nine red-octorok kills across `0x68`, `0x58` and `0x49` for **0.00 hearts**.
All four hits — and all three streak resets — came from the four things that
are not a red octorok: a zora's spit, a peahat, a blue octorok, a blue
tektite. The chase, the sword cadence and the evade are solved *for the enemy
the corridor is mostly made of*. Tuning them further buys nothing.

### `0x59` is the whole loss, and its wave is not what the spawn table says

The ROM spawn table bills `0x59` as four peahats. Live it is **peahat x4 plus
a zora**, and the zora is what costs the heart: `fireball_or_statue_projectile`
from the east is its spit. The screen takes **739 frames and a full heart**,
which is the only whole heart the walk loses, and pays **0 rupees for 1 kill**
— row 3, 0.081 R/kill, the cheapest bodies on the corridor. Both hits come
from the two things a wooden sword cannot answer: a peahat is invulnerable in
flight and a zora submerges.

**And it kills the money.** Link enters `0x59` on a 5-kill streak and leaves
on 0. `0x49`'s six kills then run 0->4 instead of 5->11, so the forced 5-rupee
at ten kills never fires. The 2026-09-15 skip-`0x59` experiment was reverted
because one trajectory came back worse on every axis; that measurement had no
way to see that the screen it skipped was the one spending the streak. This
is the row to re-open, and now there is a number to judge it by.

### `0x4a` is 43% of the walk for 1 rupee

2618 of the walk's 6117 frames, the full `HUNT_DESTINATION_FRAMES` 2400
budget (`hunt_budget_4a`), for 4 of 6 tektites. 808 of those are `guard`
frames — Link arrives at 1.49 hearts, under `min_hearts`, so he stops chasing
and only answers what closes. The tektite wave is the richest on the corridor
and the walk reaches it with the least health to spend on it.

`0x78` is still three red octoroks that never enter `HUNT_BOX` (peak live 3,
peak prey 0, 0 hunt frames). Unchanged, and still correct.

### Next

1. `0x59`: skip it, or answer the zora and the flying peahats. It is one
   heart, 739 frames, a 5-kill streak and zero rupees.
2. Hearts before `0x4a`. The richest wave is fought on the least health.
3. The drop rate is still short (19% here, ~13% cumulative) of the ROM rows.

One caveat: `hunt.py` is 1064 LOC, over the ~1000 soft max, from the damage
log wiring and `screen_table`. `ShieldPolicy`/`TargetBook` are the seam if it
grows again.

## This sitting (2026-09-15d) — the zora has a clock, and the corridor has a price list

Three of the four open rows above are one sitting's work, because all three
are the same mistake: the hunt knew *where* things were and nothing about
what they were **worth** or **when** they fire.

### The zora fires on a 195-frame clock, and the shot waits 17 frames to move

`scratch/probe_zora.py` logs every frame a zora (`0x11`) or a fireball
(`0x55`) holds a slot on the live walk. Four surfacings on `0x59`
(`scratch/zora1.json`), **identical to the frame every time**. `ObjState`
(`$00AC`) is the cycle; the zora does not move inside one and picks a new tile
for the next.

| ObjState | frames | what it is |
|---|---|---|
| `0x00` | 2 | surfacing begins |
| `0x01` | 32 | rising |
| `0x02` | 15 | surfaced, mouth open — **the tell** |
| `0x03` | 34 | firing; the `0x55` slot appears 2f in |
| `0x04` | 16 | submerging |
| `0x05` | 96 | submerged |

Two numbers matter, and the second is the bug:

1. The shot is born **2 frames into `0x03`**, so the `0x01 -> 0x02` edge is
   **~50 frames** of warning.
2. **The shot then sits on the muzzle for 17 frames before it moves at all.**
   Measured 17/17/17/17.

That dwell is why nothing dodged it. `ObjectTracker` measures a motionless
slot at zero velocity, so `threat.assess` scores it *safe* for the entire
window in which a 1 px/frame Link could still walk out of its way — and by
the time it has a velocity (1.65-1.76 px/frame, closing on Link's 1.0) the
dodge window is gone. `behaviors.shield_blocks` correctly says `0x55` needs
the Magical Shield, so every shield rule passed on it, and nothing replaced
them. `threat.off_line_step` could not help either: **a zora's facing byte
reads `0x03`**, which is in no `_FACING_AXIS` entry, so `in_firing_line` has
never once returned True for one.

`0x59` f2888 is the whole heart: the ball spawned at (196,157) and held there
while Link walked **east from x=31 to x=100 along y=157** — straight down its
line, into it.

The aim is quantized at launch, not a clean bearing: the four shots left at
180.0, 180.0, -171.1 and -124.2 degrees against bearings to Link of 180.0,
-172.7, -162.9 and -119.3. So there is no line to solve and **no pre-emptive
rule worth having** — walking before the launch only moves the target. The
dwell is the one window where the line is fixed and Link is not yet on it.
`ShotPolicy.duck` steps perpendicular to the muzzle bearing for exactly that
window (`hunt_duck`), and never along it.

### The price list: a kill is a drop row plus a streak tick

`overworld/prey.py` is `scratch/bomb_budget.py`'s arithmetic promoted to
where policy can read it (the scratch CLI now imports the tables instead of
keeping a second copy).

| ROM row | prey on this corridor | random R/kill | + streak | total |
|---|---|---|---|---|
| 0 | red octorok, red tektite, blue leever | 0.156 | 0.385 | **0.541** |
| 1 | **blue tektite**, red leever, ghini | 0.891 | 0.385 | **1.275** |
| 2 | blue octorok, blue moblin | 0.122 | 0.385 | **0.507** |
| 3 | peahat, armos, zora | 0.081 | 0.385 | **0.466** |

The streak column is the one the old policy could not see. Ten unbroken kills
force a 5-rupee and sixteen force a fairy, so **every kill is worth ~0.385R
on top of its own table** — more than row 0's entire drop. It also prices the
damage: a contact at streak 7 throws away 2.7R of forced-drop progress, which
is more than all five octorok screens' random drops put together.

### A red octorok is worth chasing. The arithmetic is not close.

This is where the sitting's instinct and the numbers disagreed, so the numbers
won. One chase frame costs 0.00137R of walk time (8.4R billed over 6117 live
frames) plus 0.00101R of streak risk (4 hits over those frames at the mean
streak) = **0.0024R/frame**. Against a red octorok's 0.541R that pays back a
**227 px** walk; `HUNT_BOX` is 182 px wide. There is no distance on the screen
at which walking away from a red is the better trade, and `tables1` measured
nine of them landing 0.00 hearts.

What *is* true is that the same chase stops paying at **113 px on short
health**, because a contact then costs the streak *and* a heart — and the
measured price of that heart is `0x4a`: Link arrived on 1.49, spent 808 of
2401 frames guarding, and left two of six row-1 tektites alive. 808 frames
plus two row-1 bodies is 3.66R, which triples the cost of a chase frame.

So the gate is **health, not distance** (`PreyPolicy.thrifty_below_hearts`),
and the drop row's real job is the **order**: at equal range the blue tektite
is held over the red octorok, every time. The bodies that are simply never
targets are the ones that are not kills — a zora submerges via
`DestroyMonster` with no `HandleMonsterDied`, so its slot vanishing is not
even a streak tick (`prey.SKIP_TYPES`, with armos and boulder).

### `0x59` is crossed, not cleared

`ShopP7WalkController.hunt_transit_screens = {0x59}`. On a transit screen the
hunt still strikes a body in the blade box, still blocks a rock, still ducks a
fireball and still scoops a drop it walks past — it never *chases* and never
spends the screen budget. Declining the wave is not the same as standing in
it, which is what the reverted 2026-09-15 skip experiment could not express.

### Measured: `tables1` -> `tables3`

| screen | `tables1` (baseline) | `tables3` (this sitting) |
|---|---|---|
| `0x68` | 566f / 322 hunt / 2 kills | **566f / 322 hunt / 2 kills** — byte-identical |
| `0x58` | 646f / 397 hunt / 3 kills | **646f / 397 hunt / 3 kills** — byte-identical |
| `0x59` | 739f, **1.00 heart**, 2 hits, streak 5->0 | **203f, 0.00 hearts, 0 hits, streak 5->5** |
| `0x49` | in on 1.99 hearts / streak 0; 6 kills, 1 hit | in on **3.00 hearts / streak 5**; 4 kills, 2 hits |
| `0x4a` | 2618f, 4 of 6 tektites, 1R | 2554f, 2 of 6 tektites, 0R |
| run | 16 kills, 2R, 2.01 hearts, streak best 7 | 11 kills, 1R, 2.01 hearts, streak best 5 |
| stop | `bomb_buy` `shop_need_20_have_2` | `bomb_buy` `shop_need_20_have_1` |

Read that table by screen, not by total.

**The two screens the policy touched are unambiguous.** `0x59` went from the
walk's single largest loss to 203 frames and nothing else; the fireball landed
no hit in either run with `duck` wired (22 duck frames). And `0x68`/`0x58` came
back **byte-identical to the baseline**, which is the check that matters for
the value ordering: `PreyPolicy` changed no decision on a wave of four
identical red octoroks, exactly as the arithmetic above says it should not.

**The two screens after it are a reshuffle, not a regression.** Link now
enters `0x49` 536 frames earlier, on a full 3 hearts and a live 5-kill streak
instead of 1.99 and 0 — a completely different wave state — and `prey_passed`
is empty for the whole run, so no rule this sitting declined a single body
there. `0x4a` spent the same full 2400-frame budget for 2 kills instead of 4.
The corridor is chaotic with respect to entry timing (this is the same
sensitivity `L1` has), so **a single deterministic trajectory cannot grade a
policy change end to end** — which is exactly how the first skip-`0x59`
experiment got reverted on a bad total. Grade per screen; the totals move for
reasons no rule chose.

### Two defects the first live run found

`tables2` **died on `0x49`** (5 contacts, `link_death`). Both causes were
real, both are fixed and pinned:

1. **`duck` stepped into a body.** `ShotPolicy.face` has refused to stand
   still while a body closes since `0x49` killed three shield walks; the new
   dodge shipped without the same rule, so it bought distance from a shot
   that had not fired by spending it on an octorok that was already touching
   Link. `perpendicular` now takes the live bodies and skips a step whose
   landing cell is inside `MIN_DODGE_BODY` of one.
2. **The target order used chebyshev.** `combat.nearest_to` — what the value
   order replaced — measures *manhattan*, because the order is a walk cost
   and contact is the square pad. Ranking on chebyshev silently re-picked a
   different octorok on every off-axis wave, which is what reshuffled
   `0x58` from 646 frames to 1407. The gate still uses chebyshev; only the
   order changed back.

### Next

1. **`0x49` is the new whole loss.** Link now arrives there on a 5-kill
   streak with full health — the exact setup for the forced 5-rupee at ten —
   and two `octorok_fast` contacts (E and S) take it back to 0. That is the
   only screen between this walk and the first forced drop it has ever
   earned.
2. Hearts before `0x4a` is still open, and still the richest wave fought on
   the least health.
3. The drop rate is still 18% against 36% billed. One pass cannot fund 20R.
4. `PreyPolicy.thrifty_below_hearts` has **not fired live yet** (whole hearts
   never reached 2 on a chase frame), so the short-health chase cap is
   unit-tested arithmetic, not a measured result.

Standing caveat, now larger: `hunt.py` is **1335 LOC** against the ~1000 soft
max (was 1064). `overworld/prey.py` took the value model out; `ShotPolicy` is
the next seam if it grows again, and `dungeon/threat.py` already owns the
shot-timing vocabulary it would land in.


## This sitting (2026-09-15e) — the sword was never swinging, and the farm is a lap

Two ROM facts, one per open row. The first is why the corridor cost two hearts
and three streak resets; the second is why one pass was never going to fund the
pack whatever the fighting looked like.

### `ButtonsPressed` is an edge, so a held A swings once and never again

`Link_HandleInput` (`Z_05.asm`) wields the sword on

```
LDA ObjState / BNE @CheckMovement      ; only while Link is idle
LDA ButtonsPressed / AND #$80 / JSR WieldSword
```

and `ButtonsPressed` is **not** "A is down". `Z_07.asm` builds it every poll as
`new EOR ButtonsDown AND new` — "down now instead of before". `ScreenHunter.
_strike` returned `nes_action(face, "A")` on every frame it owned, so after the
first press A stayed down, no further swing started, and `$00AC` never went
non-zero — which is the byte `link_busy` watches to decide when to release. The
loop is closed: **A held starts no swing, a swing that never starts never sets
the state, and the state is the only thing that released A.**

What that looks like live is not a missed swing, it is a walk. The four
contacts of `base_head` are one window each, all the same
(`scratch/probe_contact.py`, now logging the driving rule and the buttons):

| f | screen | window |
|---|---|---|
| 3197 | `0x49` | 12 frames of `hunt_49_slash`, state 0, Link walking y157 -> y148 into `octorok_fast` #2 |
| 3532 | `0x49` | **24 consecutive `hunt_49_slash` frames of UP+A, state 0 throughout**, Link walking 1.4 px/f from y=138 to y=103 while slot 4 holds 8 px off his shoulder |
| 4946 | `0x4a` | `hunt_4a_recover` x4 while `tektite_blue` #3 closes 15 -> 7 |
| 5167 | `0x4a` | 20 frames of `hunt_4a_slash` oscillating y117<->119, tektite #2 closing |

The fix is one frame: after a press that did not take, idle. The release frame
is an **idle**, not the direction — a held direction is what closes the last
8 px. (Two traps paid for on the way: the probe's first button decoder used
`enumerate(NES_BUTTON_NAMES)`, which is off by one past index 1 and drops A at
index 8 entirely, so the trace read `DOWN` for UP and never showed the A that
was the whole story. And `_approach` is the *second* producer of a `_slash`
frame; it held A the same way and now goes through `_strike`.)

### Measured: `base_head` -> `rel1`, and the first forced drop this walk has earned

Same spine, same trajectory, `assist=None`, `set_state=0`.

| screen | `base_head` (head) | `rel1` (release edge) |
|---|---|---|
| `0x77` `0x78` `0x68` `0x58` `0x59` | 183 / 114 / 566 / 646 / 203 f | **byte-identical**, same kills, same streak |
| `0x49` | 843f, 4 of 6, **2 hits / 1.00 heart**, 1R, streak 5->2 (2 resets) | 907f, **6 of 6**, 0 hits, **2R**, streak **5->11** |
| `0x4a` | 2554f, 2 of 6, **2 hits / 1.00 heart**, 0R, streak 2->0 | **2190f**, **6 of 6**, 0 hits, **6R**, streak **11->17** |
| run | 11 kills, 2 drops, **1R**, 4 hits, 2.01 hearts, best 5, 3 resets | **17 kills, 8 drops, 8R, 0 hits, 0.00 hearts, best 17, 0 resets** |
| hearts at the shop | `0x20` (1 of 3) | **`0x22` (3 of 3)** |
| stop | `shop_need_20_have_1` | `shop_need_20_have_8` |

The five screens the fix could not touch came back byte-identical, which is the
control: nothing on them was ever a contact. The two that were are unambiguous
— both cleared, both hitless, and `0x4a` **faster** (2190f against 2554f) while
killing three times as many bodies, because a swing that lands ends the chase.

**The streak crossed 10 and 16 for the first time.** `0x49` banked the forced
5-rupee at kill ten and `0x4a` the forced fairy at sixteen; the run ends on a
live 17-streak with every heart. Six of the eight drops are on the two fixed
screens.

**And the drop-rate blocker was the same bug.** 8 floor drops on 17 kills is
**47% against 42% billed** — the 13-19% of the last three sittings was a census
of a walk whose kills were all on row 0 and whose row-1 wave it never fought.
There is nothing wrong with `DropItemRates`.

### One pass still cannot pay, and the ROM says exactly what can

8R against 20R. The supply has to be a second wave, and
[`overworld/respawn.py`](../overworld/respawn.py) is the rule, from the
disassembly rather than from folklore:

* `ModifyObjCountByHistoryOW` (`Z_05.asm`) runs inside `CreateRoomObjects` on
  every room load. It **clears** a screen's kill-count flags — the full
  respawn — only when the screen is absent from the six-entry `RoomHistory`
  (`$621`) **and** those flags already read the max, 7. A screen that *is* in
  the history has its kill count subtracted from the spawn count instead.
* `SaveKillCountOW` writes 7 exactly when `RoomKillCount >= RoomObjCount` (the
  screen was cleared of whatever it spawned) and otherwise adds the partial
  count in, capped at 7 — so partial clears converge on 7 over visits.
* `RunCrossRoomTasksAndBeginUpdateMode` (`Z_07.asm`) appends the room to the
  history **only if it is not already in it**, and leaves
  `CurRoomHistoryIndex` alone when it is.

That last rule is the one that decides route shape, and it is why
`rupee_farm`'s `0x4A <-> 0x49` restock was never a rupee supply: **every screen
on the way back is already in the history, so an out-and-back evicts nothing,
at any depth.** Eviction needs *new* rooms. This corridor has **seven** distinct
screens against six slots, so the smallest thing that works is a full lap back
to `0x77`: walking onto `0x77` evicts `0x78`, and from there each screen Link
enters evicts the next one in front of him, so the whole eastbound leg comes
back whole — and every lap after it does the same.
`gathering.PRE_L1_LAP_HOPS` is that table (`laps=N` on
`make_shop_p7_walk_controller`); `respawn.respawn_visits` is the arithmetic and
`tests/test_respawn.py` asserts it.

Measured live (`scratch/probe_respawn_lap.py`, `lap1`): the history filled
`0x77 0x78 0x68 0x58 0x59 0x49` in order and `0x4A` overwrote `0x77` at the
wrap, exactly as `RoomHistory` models it; re-entering `0x49` four kills later
read flags **4** and spawned **6 - 4 = 2** bodies. That run also proved the
westbound hops as far as `0x59` — and then **died in mode 17 on it**, because
it was lapping on the one heart the old sword left. The lap was never blocked
on the hops. It was blocked on the sword.

### The lap runs, the waves come back, and it still banks 8R

`scratch/probe_respawn_lap.py --laps 1` (`lap2`), same release-edge tree,
`assist=None`. The westbound hops all held this time and the respawn happened
exactly where `respawn.respawn_visits` says, screen by screen:

| visit | f | screen | wave | flags in | what the ROM did |
|---|---|---|---|---|---|
| 7 | 4240 | `0x49` | empty | **7** | in history -> suppressed |
| 9 | 5103 | `0x58` | `octorok_fast` x1 | 3 | in history -> 4 - 3 = 1 |
| 10 | 5531 | `0x68` | octorok x2 | 2 | in history -> 4 - 2 = 2 |
| 12 | 6339 | `0x77` | - | 0 | **not in history** -> appended at idx1, evicting `0x78` |
| 13 | 6445 | `0x78` | octorok x2 | 0 | fresh; evicts `0x68` |
| 14 | 6644 | `0x68` | **octorok x4** | **0** | flags hit 7 on visit 10, so this entry **cleared them**: full wave |
| 15 | 8125 | `0x58` | **octorok x2 + `octorok_fast` x2** | 0 | full wave |
| 17 | 9035 | `0x49` | **6 bodies** | 0 | full wave, **+6 rupees** |
| 18 | 9670 | `0x4a` | **`tektite_blue` x6** | 0 | full wave |

That is the ROM's two halves in one run: a screen *in* the history has its kill
count subtracted (visits 7, 9, 10), and a screen out of it with flags at 7 has
them cleared (visits 14-18). `0x68` shows the whole cycle — 2 of 4 on the first
pass, the other 2 on visit 10 (which is what writes 7), and all 4 back on
visit 14.

**And the money did not move.** 33 kills against `rel1`'s 17, for the same
8 rupees. Three named reasons, in size order:

1. **7R was left on the floor.** The drops happened: `0x0f` x2 and `0x18` x5 is
   **15R of rupees on the ground** against 8 banked. (`rel1` leaks the same
   way: 14R down, 8 banked.) The scoop, not the drop table, is now the biggest
   single line on this errand.
2. **Five streak resets, best 11.** The costly one is the westbound `0x59`
   (visit 8, streak **11 -> 0**): eastbound the hop crosses it along y≈155,
   but coming back from `0x49` Link enters at the top on x=112 and has to walk
   the length of the peahat-and-Zora screen to reach the west exit. Damage
   2.508 hearts over 5 hits — `fireball_or_statue_projectile_S`,
   `rock_projectile_S`, `octorok_fast_E`, `tektite_blue_N` x2.
3. **The lap skips the first `0x4a` wave.** With `laps=1` the first arrival is
   a mid-table hop, not the destination, so Link turns round after 106 frames
   and the six row-1 tektites — the richest wave on the corridor — go
   unfought until the very last visit.

So `laps` stays **0** by default. The lap is wired, ROM-derived, unit-tested
and now measured working as a *respawn mechanism*; it is not yet a better
rupee-per-frame deal than one clean pass, and a route change that cannot be
measured better does not land.

### Next

1. **The scoop.** 15R fell and 8R was banked. Everything else on this list is
   smaller than that.
2. Westbound `0x59`. Eastbound it is 203 frames and free; westbound it cost an
   11-streak and a heart. Either enter it lower or route the lap around it.
3. `laps=1` should hunt `0x4a` on the first arrival too (or put the lap
   *before* the destination hunt rather than after it).
4. `hunt.py` is 1377 LOC against the ~1000 soft max. `ShotPolicy` is still the
   seam.
