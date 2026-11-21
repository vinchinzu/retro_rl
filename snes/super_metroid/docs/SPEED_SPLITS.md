# Grapple Skip speed splits

The sub-55 clock runs from first Ceres control to ship launch. This
snapshot uses the vanilla ROM scan of `v3_no_crocomire` at 60 frames per
second. At 60.0988 fps, the same 359,943 frames take about 9.9 seconds less.
The reference image gives cumulative splits in the BEST and 56:28.88
columns. Each reference segment below is the difference between two
consecutive splits in its own column. Positive deltas mean our tape is
slower. † marks a different preceding split due to early Hi-Jump or Ice.

## v3 baseline against the reference

| # | Split | Ours | Ref BEST | Δ vs BEST | Ref 56:28.88 | Δ vs run |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Ceres Station | 01:58.25 | 01:31.01 | +00:27.24 | 01:31.31 | +00:26.94 |
| 2 | Morph Ball | 01:52.73 | 01:38.69 | +00:14.04 | 01:42.82 | +00:09.91 |
| 3 | First Missiles | 00:31.93 | 00:22.35 | +00:09.58 | 00:23.30 | +00:08.63 |
| 4 | Bomb | 02:38.80 | 01:43.50 | +00:55.30 | 01:42.73 | +00:56.07 |
| 5 | First Super | 08:32.03 | 02:10.98 | +06:21.05 | 02:22.54 | +06:09.49 |
| 6 | Charge Beam | 01:43.33 | 00:57.11 | +00:46.22 | 00:58.02 | +00:45.31 |
| 7 | Kraid † | 02:28.68 | 02:38.31 | -00:09.63 | 03:09.02 | -00:40.34 |
| 8 | Varia Suit | 00:07.07 | 00:08.28 | -00:01.21 | 00:05.79 | +00:01.28 |
| 9 | Hi-Jump Boots † | 03:48.90 | 02:24.99 | +01:23.91 | 02:01.82 | +01:47.08 |
| 10 | Speed Booster † | 05:14.85 | 02:02.61 | +03:12.24 | 01:56.61 | +03:18.24 |
| 11 | Wave Beam | 01:24.87 | 01:12.82 | +00:12.05 | 01:04.61 | +00:20.26 |
| 12 | First Power Bomb † | 04:37.87 | 02:45.88 | +01:51.99 | 02:57.27 | +01:40.60 |
| 13 | Phantoon | 06:07.77 | 03:13.96 | +02:53.81 | 03:02.16 | +03:05.61 |
| 14 | Gravity Suit | 02:57.42 | 02:34.17 | +00:23.25 | 02:39.25 | +00:18.17 |
| 15 | Draygon | 10:41.82 | 07:07.41 | +03:34.41 | 06:29.71 | +04:12.11 |
| 16 | Space Jump | 00:05.88 | 00:07.43 | -00:01.55 | 00:09.94 | -00:04.06 |
| 17 | Plasma Beam | 04:19.63 | 01:51.85 | +02:27.78 | 02:00.18 | +02:19.45 |
| 18 | Ice Beam † | 02:12.27 | 02:20.84 | -00:08.57 | 02:14.20 | -00:01.93 |
| 19 | Ridley † | 13:40.95 | 05:37.91 | +08:03.04 | 05:38.88 | +08:02.07 |
| 20 | G4 | 14:39.57 | 05:32.83 | +09:06.74 | 05:30.62 | +09:08.95 |
| 21 | Mother Brain 1 | 04:23.18 | 04:24.43 | -00:01.25 | 04:18.64 | +00:04.54 |
| 22 | Mother Brain 2 | 00:57.35 | 01:49.43 | -00:52.08 | 01:35.23 | -00:37.88 |
| 23 | Ship | 04:53.90 | 02:54.96 | +01:58.94 | 02:54.23 | +01:59.67 |
| | **Total** | **1:39:59.05** | **57:11.75** | **+42:47.30** | **56:28.88** | **+43:30.17** |

## Combined times

Charge → Varia on Grapple Skip includes Hi-Jump. The reference collects
Hi-Jump after Varia, so the comparable block ends at the final pickup in
each route. Early Ice and First Power Bomb are grouped here for budgeting.
They are two nonconsecutive segments in the reference. Mother Brain 1 and
2 share row 331, so their combined time provides the fight comparison.

| Block | Ours | Ref BEST | Δ vs BEST |
|---|---:|---:|---:|
| Kraid + Varia + Hi-Jump | 06:24.65 | 05:11.58 | +01:13.07 |
| First Power Bomb + Ice Beam | 06:50.13 | 05:06.72 | +01:43.41 |
| Mother Brain 1 + 2 | 05:20.53 | 06:13.86 | -00:53.33 |

[Speed plan and room priorities](plan.md#sub-5500-benchmark--route-policy-roadmap)

## v4 direct Red Tower return

The v4 tape is `v4_grapple_skip_red` at **1:39:04.47**, a gain of
**3,275 frames (54.58s)** over v3. Both tapes use the same 60 fps clock.
Three segment durations changed.

| Segment | v3 | v4 | Frames saved |
|---|---:|---:|---:|
| G4 | 14:39.57 | 13:48.08 | 3,089 |
| Mother Brain 2 | 00:57.35 | 00:57.33 | 1 |
| Ship | 04:53.90 | 04:50.82 | 185 |
| **Ceres → Ship** | **1:39:59.05** | **1:39:04.47** | **3,275** |

The v4 run is 41:52.72 above reference BEST, 42:35.59 above the 56:28.88
run, and 44:04.47 above 55:00. The largest remaining G4 cost on the tape is
the 10,465-frame Brinstar Reserve and ammo detour in row 310.
[Power-on and audit evidence](GRAPPLE_SKIP_SCOPE.md#direct-red-tower-return)

## Space Jump to Plasma Beam

The `grapple_skip_space_jump` profile makes each jump decision from Samus's
live pose, velocity, and position. The Space Jump Room exit uses one takeoff
from the lower floor. Halfie Climb and West Cactus Alley use controlled
drops through the shafts.

| Row | Room | v4 frames | Skill frames | Jump presses before → after |
|---:|---|---:|---:|---:|
| 212 | Space Jump Room | 795 | 258 | 6 → 1 |
| 213 | Draygon's Room | 577 | 447 | 3 → 2 |
| 214 | The Precious Room | 558 | 413 | 5 → 4 |
| 216 | Halfie Climb Room | 554 | 323 | 6 → 0 |
| 218 | West Cactus Alley | 558 | 301 | 1 → 0 |

The route saves **1,291 frames (21.52s)** from the settled Space Jump pickup
through the settled Plasma Beam pickup, including the retimed Colosseum and
East Cactus Alley joins. The Colosseum refill starts at 27 energy and leaves
Halfie Climb entry at 333/399. Plasma Beam is collected with 259/399 energy
and the return through Plasma Tutorial Room settles at 289/399.

## v5 through credits

The default Clean replay uses `master_tape_grapple_skip_v5_clean`. The full
tape saves **1,846 frames (30.77s)** over v4 and finishes at **1:38:33.70**
from Ceres control to ship. The guarded replay and independent vanilla ROM
audit pass through row 339 and the final credits screen at frame 379,723.

Ridley leaves four Power Bombs and ten Supers for the Lower Norfair
return. At Mother Brain, v5 enters with 498/499 energy, 50 Missiles, and
12 Supers. Row 331 clears in 6,523 frames, 194 fewer than v4. Samus spends
the available ammunition on the barriers, glass, and MB1, then uses Charge
against MB2. She enters the Rainbow kneel with 67 energy and boards the
ship with 376. Rows 332–339 use the live closeout and escape controllers.

The seven v5 rules rebuild the same button digest from the v4 stem.
Run from the repository root with headless drivers configured.

```bash
speed=snes/super_metroid/recordings/speed_benchmarks
uv run python -m super_metroid.splice chain --replay --require-credits
uv run python -m super_metroid.splice chain --replay --require-credits \
  --stem "$speed/master_tape_grapple_skip_red_clean"
uv run python snes/super_metroid/scripts/tools/audit_master_tape.py \
  --stem speed_benchmarks/master_tape_grapple_skip_v5_clean

uv run python -m super_metroid.splice policy build grapple_skip_space_jump \
  --source-stem "$speed/master_tape_grapple_skip_red_clean" \
  --only space_jump_room_exit_r212 --only draygon_room_exit_r213 \
  --only precious_space_jump_r214 --only halfie_return_drop_r216 \
  --only west_cactus_drop_r218 --only ridley_no_crocomire_r271 \
  --only mb_tourian_v2_r331 --stem "$speed/v5_rebuild_clean"
```

[Guarded replay](../recordings/speed_benchmarks/v5_replay.log) ·
[Independent audit](../recordings/speed_benchmarks/v5_audit.log) ·
[ROM split scan](../recordings/speed_benchmarks/master_tape_grapple_skip_v5_clean_splits.json)

## Reactive opening on Grapple Skip

The `start_to_bombs` profile reuses the TAS-derived Ceres arm pumps,
Ridley tail tank, and elevator wall jump. Climb uses platform landings
to advance to the next jump. At Bomb Torizo the skills open the orb with
the beam, steer near the statue side of the fight, and spend naturally
collected missiles. Each skill runs from the live predecessor.

| Cumulative from Ceres control | v4 | Reactive opening | Gain | 56:28.88 reference |
|---|---:|---:|---:|---:|
| Ceres Station | 01:58.25 | 01:32.70 | 25.55s | 01:31.31 |
| Morph Ball | 03:50.98 | 03:25.45 | 25.53s | 03:14.13 |
| First Missiles | 04:22.92 | 03:57.38 | 25.53s | 03:37.43 |
| Bombs pickup | 07:01.72 | 06:30.17 | 31.55s | 05:20.16 |
| Post-Torizo Parlor (row 33) | 08:56.92 | 07:24.55 | 92.37s | |

Climb saved 5.18s through the next two rooms on both tested entries. The
fight follows the Bombs split and clears into Flyway in 2,828 frames
with 99 energy, up from 37 on v4.

The guarded replay and independent vanilla ROM audit passed through row
38 in 42,392 frames from power-on with zero loads, RAM writes, or deaths.
Building all five rules on v4 and splicing only Climb into the earlier
candidate produced identical button streams. Evidence and the RAM split
scan are in `recordings/speed_benchmarks/start_to_bombs/opening_rebuilt_clean.*`.
The opening comparison used the v4 full-game tape.

## V6 Moonfall opening

V6 enables Moon Walk with the controller in Special Setting Mode before the
intro. The Ceres Elevator descent takes **412 frames**, the same as the
vendored Sniq 100% reference from first control to the settled Falling Tile
entry. Vanilla ROM traces show uncapped Moonfall in Ceres and Climb, peaking
at 12 and 22 pixels per frame.

Continue V6 from `master_tape_grapple_skip_v6_fast_opening_clean`. It
steers around the grass platforms in Parlor, turns toward the Pit hatch
during Climb's Moonfall, and carries the exit speed into running jumps
across Pit. Samus brakes on the Blue Brinstar elevator with angle-up and
boards as soon as the pad is ready. The four rooms save **363 frames
(6.05s)** through Morph Ball and take no damage.

| Room | Prior V6 frames | Latest V6 frames | Saved |
|---|---:|---:|---:|
| Parlor | 932 | 870 | 62 |
| Climb | 480 | 449 | 31 |
| Pit Room | 691 | 474 | 217 |
| Blue Brinstar Elevator | 340 | 287 | 53 |

| Cumulative from Ceres control | V5 | Prior V6 | Latest V6 | KPDR 56:28.88 | Latest delta |
|---|---:|---:|---:|---:|---:|
| Ceres Station | 1:58.25 | 1:28.65 | **1:28.65** | 1:31.31 | **−2.66s** |
| Morph Ball | 3:50.98 | 3:16.67 | **3:10.62** | 3:14.13 | **−3.51s** |
| First Missiles | 4:22.92 | 3:48.60 | 3:42.55 | 3:37.43 | +5.12s |
| Bombs pickup | 7:01.72 | 6:00.18 | 5:54.13 | 5:20.16 | +33.97s |
| Post-Torizo Parlor (row 33) | 8:56.92 | 6:51.00 | **6:47.90** | | |

Samus keeps her first five Missiles by turning back in Construction Zone.
Skipping the second pack saves 1,384 frames (23.07s). The Flyway controller
opens the hatch on those five shots. Torizo starts at 79 energy with empty
ammo. The statue-side firing position saved 196 frames (3.27s) on the
prior V6 entry. The same controller takes 2,791 frames on the latest entry,
177 more than on the prior entry, and exits to Flyway with 99 energy and
five naturally dropped Missiles. The return to Parlor is **2:09.02 ahead
of V5**.

The latest opening reaches Big Pink at row 40 in **40,228 frames** from
power-on, 243 frames (4.05s) earlier than the prior V6. It reaches the
same leave with 109/199 energy and 0/5 Missiles. The guarded replay and
independent vanilla ROM audit match the endpoint with zero loads, RAM
writes, or deaths. The V7 Early Supers branch starts on the Green Brinstar
elevator leave after row 38.
[V6 route plan](plan.md#v6-speed-candidates)

```bash
speed=snes/super_metroid/recordings/speed_benchmarks
uv run python -m super_metroid.splice chain --replay --through-row 40 \
  --stem "$speed/master_tape_grapple_skip_v6_fast_opening_clean"
uv run python snes/super_metroid/scripts/tools/audit_master_tape.py \
  --stem speed_benchmarks/master_tape_grapple_skip_v6_fast_opening_clean

uv run python -m super_metroid.splice policy build v6_opening \
  --source-stem "$speed/master_tape_grapple_skip_v6_opening_clean" \
  --only parlor_moonfall_r14 --only climb_moonfall_r15 \
  --only pit_crossing_r16 --only blue_brinstar_elevator_r17 \
  --through-row 40 --max-delay 4 --stem "$speed/v6_fast_opening_rebuild_clean"
```

[Sniq opening trace](../tas/bodies/sniq_100_ceres_lsnes_hops.json) ·
[Route policy](../policies/route_profiles/v6_opening.json) ·
[Guarded replay](../recordings/speed_benchmarks/v6_fast_opening_replay.log) ·
[Independent audit](../recordings/speed_benchmarks/v6_fast_opening_audit.log) ·
[ROM split scan](../recordings/speed_benchmarks/master_tape_grapple_skip_v6_fast_opening_clean_splits.json) ·
[Prior V6 split scan](../recordings/speed_benchmarks/master_tape_grapple_skip_v6_opening_clean_splits.json)

## V7 Early Supers

The replay and times below predate the decision to put the no-Hi-Jump Warehouse route directly into V7. They do not verify the revised route. See [the V7 plan](plan.md#v7-early-supers).

On the V6 opening, the Main Shaft skill farms three Zeelas for the five
Missiles needed at the Early Supers hatch. Samus runs up, jumps, and taps
Down twice just above the first crumble blocks. She lands during the morph
animation and carries running momentum under all three shutters. The right
chute wall jump reaches the Super Missile tank. The way back picks up Bridge
Missile, which pays for the Dachora hatch. [Mockball input timing](https://tasvideos.org/GameResources/SNES/SuperMetroid#Mockball)

The branch enters Dachora with five Supers, then collects Charge Beam in
Big Pink and enters Green Hill Zone on four Supers. Skipping Spore Spawn
and its farming loop saves **17,418 frames (4:50.30)** over rows 39–47.
The recorded row-204 tape keeps the Hi-Jump-first Warehouse order and uses the Sloaters
refill, a timed Bat Cave drop, and a Beetom farm to restore resources for
Spazer, Wave, and the Ice return.

| Cumulative from Ceres control | V5 | V7 |
|---|---:|---:|
| First Super | 15:33.75 | 8:31.98 |
| Charge Beam | 17:17.08 | 10:12.20 |
| Ice Beam | 32:33.72 | 26:21.92 |

The guarded replay and independent vanilla ROM audit matched through row
126 in **110,852 frames** from power-on, **25,262 frames (7:01.03)** ahead
of V5. Samus reaches Red Tower with 35/399 energy, three Missiles, two
Supers, and Charge, Spazer, Wave, and Ice. The replay observed 118 room
transitions with zero loads, RAM writes, or deaths.

The row-141 prefix also passes both gates. The shifted Phantoon fight uses
charged shots and takes 9,798 frames with the loot exit, saving 1,346 frames.
Samus reaches the powered Wrecked Ship Main Shaft after **141,967 frames**,
**7:34.20** ahead of V5. [Row 141 replay](../recordings/speed_benchmarks/master_tape_grapple_skip_v7_row141_clean.json) ·
[Row 141 audit](../recordings/speed_benchmarks/v7_row141_audit.log)

The latest power-on replay and independent audit reach Botwoon at row 204
in **177,723 frames**, **26,607 frames (7:23.45)** ahead of V5. The replay
matched 167 room transitions with zero loads, RAM writes, or deaths.
Samus enters with Gravity, 287/399 energy, 20 Missiles, three Supers, and
three Power Bombs. [Row 204 replay](../recordings/speed_benchmarks/master_tape_grapple_skip_v7_row204_clean.json) ·
[Row 204 audit](../recordings/speed_benchmarks/v7_row204_audit.log)

```bash
speed=snes/super_metroid/recordings/speed_benchmarks
uv run python -m super_metroid.splice policy build v7_early_supers \
  --prefix-stem "$speed/master_tape_grapple_skip_v6_fast_opening_clean" \
  --source-stem "$speed/master_tape_grapple_skip_v5_clean" \
  --through-row 126 --max-delay 12 --stem "$speed/v7_rebuild_row126_clean"
uv run python -m super_metroid.splice chain --replay --through-row 204 \
  --stem "$speed/master_tape_grapple_skip_v7_row204_clean"
uv run python snes/super_metroid/scripts/tools/audit_master_tape.py \
  --stem speed_benchmarks/master_tape_grapple_skip_v7_row204_clean
```

[Route policy](../policies/route_profiles/v7_early_supers.json) ·
[Replay report](../recordings/speed_benchmarks/master_tape_grapple_skip_v7_row126_clean.json) ·
[ROM split scan](../recordings/speed_benchmarks/master_tape_grapple_skip_v7_row126_clean_splits.json) ·
[Independent audit](../recordings/speed_benchmarks/v7_row126_audit.log)

## First Red Tower ascent on Grapple Skip

The `red_tower_first_ascent` profile replaces row 127, the first ascent
after Ice Beam. Samus enters with 5 energy, uses the Sloaters energy
station, freezes the bottom Ripper, then spin-jumps into the right wall and
kicks left onto the frozen ledge at y=2351. The kick and landing succeeded
on the first attempt at all five tested timings from the natural
predecessor. The rest of the ascent reuses the Ice checkpoint climb,
including its existing recorded inputs and timed wall jumps in the
mid-shaft section.

The skill finishes in 6,418 frames versus 7,070 on v4, saving 10.87s
through Alpha Power Bombs. It hands off in Morph Ball at the Hellway door,
and rows 128–130 replay their original inputs. Samus collects five Power
Bombs and leaves the pickup with 312 energy, up from 304 on v4. The entry
trials and A/B report are in `recordings/speed_benchmarks/red_first_ascent/`.
The power-on replay and independent vanilla audit passed through row 130
in 145,208 frames with zero loads, RAM writes, or deaths. The first splice
stopped at the Wrecked Ship Main Shaft join on row 138. V5 repairs that join.

## v5 through Gravity Suit

The `grapple_skip_v5` profile builds from v4 and carries the first Red Tower
ascent through Gravity Suit. The RAM split scan uses the 60 fps Ceres-control
clock.

| Cumulative split | v4 | v5 | Gain |
|---|---:|---:|---:|
| Ice Beam | 32:33.72 | 32:33.72 | 0.00s |
| Alpha Power Bombs | 37:11.58 | 37:00.72 | 10.87s |
| Phantoon | 43:19.35 | 42:12.80 | 66.55s |
| Gravity Suit | 46:16.77 | 45:16.02 | **60.75s** |

A 31-frame wait in Crateria Kihunter gets the Waver drop. Samus charges the
Moat spark from Kihunter and skips the return through the Moat door. Rows
134–136 save 1,645 frames and leave 83 energy and the Super needed for the
Main Shaft hatch. The charge fight and loot exit save another 1,734 frames
on row 140. Attic drops take Samus from 71 to 106 energy. Item Cancel on
the next frame restores beam fire for West Ocean. Bowling Alley costs 90
energy, and Samus collects Gravity Suit with 13 energy and three Power Bombs.

The guarded vanilla ROM replay passed through row 148 in 174,633 frames
from power-on with zero state loads, RAM writes, or deaths. It recorded
153 room transitions and matched every planned crossing. The independent
ROM audit matched the final emulator snapshot `6a8cc014aa49…`.
[Route policy](../policies/route_profiles/grapple_skip_v5.json) ·
[Replay report](../recordings/speed_benchmarks/master_tape_grapple_skip_v5_gravity_clean.json) ·
[RAM split scan](../recordings/speed_benchmarks/master_tape_grapple_skip_v5_gravity_clean_splits.json) ·
[Independent ROM audit](../recordings/speed_benchmarks/v5_gravity_audit.log)

```bash
uv run python -m super_metroid.splice chain --replay --through-row 148 \
  --stem snes/super_metroid/recordings/speed_benchmarks/master_tape_grapple_skip_v5_gravity_clean

uv run python -m super_metroid.splice policy build grapple_skip_v5 \
  --source-stem snes/super_metroid/recordings/speed_benchmarks/master_tape_grapple_skip_red_clean \
  --through-row 148 \
  --stem snes/super_metroid/recordings/speed_benchmarks/v5_gravity_rebuild_clean
```
