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
