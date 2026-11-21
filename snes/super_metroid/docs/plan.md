# Super Metroid Clean finish plan

Finish vanilla any% KPDR on stable-retro snes9x from power-on, through the
final “SEE YOU NEXT MISSION” screen. Use Grapple Skip for the main Clean speed
route. The deliverables are the controller tape, an independent replay report
with zero loads, RAM writes and deaths, the final screen, and elapsed real
time from the input frames. Speed work starts with the post-Rainbow
splice. The 100% route follows the first finish.
[Clean contract](CLEAN_TRACK.md) · [Verified leaves](STATUS.md)

## Start here next session

The original Clean full-game finish is preserved on
`recordings/master_tape_clean.{json,npz}` (450,538 frames from power-on,
`2:05:08.97` @ 60 fps; 437,121f to ending entry + 13,417f credits tail;
439,054f / `2:01:57.57` from Ceres control; 363 room transitions; final screen
`recordings/master_tape_finish_screen_clean.png`, "SEE YOU NEXT MISSION" 34%).
Both `splice chain --replay --require-credits --stem snes/super_metroid/recordings/master_tape_clean`
(proof saved in the report) and the zero-harness audit (`audit_master_tape.py`,
snapshot `0f98bfacc32c…`, covering rows 1–339 + credits tail) passed on the
unmodified vanilla ROM with zero loads, RAM writes, or deaths.
[Clean evidence](CLEAN_TRACK.md)

The next priorities are:
1. **Sub-55:00 speed campaign & segment benchmarking** (`198,000` frames Ceres→Ship @ 60 fps, benchmarked against the standard 23-split `0:56:28.88` KPDR Any% Late Ice reference). Use `python -m super_metroid.splice benchmark` and `python -m super_metroid.splice policy` to A/B test, swap, and roll back individual segment Skills without touching the verified `master_tape_clean` baseline (see [Sub-55:00 benchmark & route-policy roadmap](#sub-5500-benchmark--route-policy-roadmap)).
   Carry the V7 Early Supers candidate forward from its actual leave. The current proof and full-game replay are listed in [STATUS.md](STATUS.md#clean-track).
2. **Tape-to-policy conversion**: replace detour loops, stored-button rooms, and the four rollout-lookahead controllers (rows 41, 43, 73, 210) with restore-free RAM-driven Skills registered in `splice/route_policy.py` (see [From tape to policy](#from-tape-to-policy)).
3. **100% route** after speed milestones land.

### V7 Early Supers

**Route decision:** V7 takes the direct Warehouse route without the initial Hi-Jump detour. This is part of V7 itself, not an optional branch. Rebuild from the V7 predecessor, repair Kraid and downstream joins without assuming Hi-Jump, and audit from power-on before treating the old Botwoon replay as current. The existing row-204 tape still includes Hi-Jump.

To watch a tape, run `uv run python snes/super_metroid/scripts/tools/audit_master_tape.py --stem speed_benchmarks/<tape_stem> --headed` on the desktop. The shared pygame viewer uses `rgb_array`; the emulator’s `human` rendering produced a white window here. The viewer starts at 1x; `[`, `]` change speed and Escape closes it. Do not set the dummy video driver for watching. Confirm the window displays the game before telling the user playback is working.

The [V7 profile](../policies/route_profiles/v7_early_supers.json) uses the
fast V6 opening through the Green Brinstar elevator. The Main Shaft skill
farms the first red hatch on the upper Zeelas, crosses the three shutters
with a mockball and collects Early Supers and the bridge Missile pack. It
spends five Missiles on the Dachora hatch and reaches Big Pink with all
five Supers. The Charge route descends from Dachora and enters Green Hill
Zone on four Supers. Together rows 39–47 save 17,418 frames against V5.

Resume at row 206, Full Halfie, from the natural V7 Botwoon exit in
`custom_integrations/SuperMetroid-Snes/scratch/pilot17_clean.state`. The
row-205 profile rule beats Botwoon from the audited row-204 leave and
settles in the Energy Tank room with 203/399 energy, no Missiles, three
Supers, and three Power Bombs. The existing Full Halfie inputs fail to
rejoin from this state. Budget one Super for the Colosseum hatch and one
for Draygon's eye door. Then revisit the late Brinstar Reserve detour and
Tourian ammo budget. V7 already has Early Supers and Bridge Missile on
the Brinstar return. Promote after a full power-on replay and independent
ROM audit through credits.

### V6 speed candidates

- **Moonfall and one-pack opening.** Use the verified V6 opening as the
  starting point for early speed work. Ceres is 2.66s and Morph is 3.51s
  ahead of the KPDR reference. First Missiles trails by 5.12s and the Bombs
  pickup by 33.97s. Shorten the remaining Morph-to-Missiles and Parlor ascent movement.
  The V7 profile uses this opening for Early Supers.
  [Opening splits](SPEED_SPLITS.md#v6-moonfall-opening)
- **Kraid before Hi-Jump.** The `kraid_first` profile replaces rows 61–68
  with the Warehouse wall-jump Skill. The direct crossing takes 764 frames,
  spends one Super, and reaches Zeela with the starting 60 energy. The
  crossing saves 6,876 frames (114.60s) through Zeela. A retimed Zeela hatch
  then rejoins the original Kihunter and Mini-Kraid inputs and reaches the
  eye door at row 71 with 53/199 energy, four Supers, and zero Missiles.
  The guarded replay and independent ROM audit match through row 71 in
  87,685 frames from power-on. The net gain through this leave is 6,815
  frames (113.58s). The approach probe crosses the pillars without damage
  and leaves Samus at `(423, 411)` below the hatch. Clear the eye with one
  Super and find two Missiles for the existing quick kill or adapt its
  opener. Time the route through Varia and the return to Business Center.
  Compare a delayed Hi-Jump pickup there with going
  straight to Cathedral. The skipped detour included its Energy Tank
  and Missile pack, so compare resources through Speed Booster as well.
  [Candidate report](../recordings/search_development_clean/v6_kraid_first/kraid_first_kraid_attempt_clean.json) ·
  [Eye-door approach inputs](../recordings/search_development_clean/v6_kraid_first/eye_door_approach_clean.npz) ·
  [Independent audit](../recordings/search_development_clean/v6_kraid_first/kraid_first_through_71_audit.log) ·
  [Wall-jump reference](https://maprando.com/logic/room/78/4/3/38)
- **Red Tower, Phantoon, and Gravity.** The Red Tower and Phantoon candidate
  saves 60.75s through Gravity on its own run. Repair the row 152 Kihunter
  join and carry the route into the v5 Space Jump and Ridley finish.
  [Gravity splits](SPEED_SPLITS.md#v5-through-gravity-suit)

### Watch review: movement, bosses, and detours

Observations from watching the v4 full-run replay. These are targets for
investigation, not verified time savings; compare from natural predecessor
states through the next dependent gate, with resources intact.

- [ ] **High — Gravity Suit return:** start from the candidate's row 148 leave
  with 13 energy and three Power Bombs. Local development clears rows
  149–151 from that leave. Find a Kihunter crossing on row 152 that survives
  at 13 energy, or recover more energy in Bowling Alley. Carry the join into
  Red Brinstar with enough Power Bombs for the next two hatches.
- [ ] **High — later Red Tower ascents:** the Gravity candidate carries the first-visit wall
  jump and Sloaters refill through Gravity Suit. Convert the climb above
  the temporary floor to live ledge checks and review the later ascents.
- [ ] **High — post-Draygon Space Jump:** v5 carries the five live rooms
  through the rebuilt Ridley and Mother Brain fights to credits.
  The next movement targets are East Cactus Alley (217), Plasma Spark and
  Kassiuz (220–221), and the exit ledge in The Precious Room on shifted entries.
- [ ] **High — early Climb:** review the watched full-run ascent and missed
  jumps against the newer reactive opening. The new opening is not yet joined
  to a complete run; retain its gains when repairing the Big Pink join.
- [ ] **High — early farming:** investigate the unnecessary-looking room exit
  and return, including rows 41–47 and their embedded farming transitions.
  Compare shorter routes through the Spazer gate with adequate energy and ammo.
- [ ] **High — count missed jumps by room:** define an observable failed-jump
  or failed-landing event from input and live pose/position, distinguish
  intentional jumps and retries, then scan the full replay and review ambiguous
  cases. Report counts per room and visit, timestamps, and examples; compare
  after each Skill change. Button presses alone cannot establish missed jumps.
- [ ] **Medium — bosses:** review the first two watched boss encounters and
  measure faster live fight Skills through their exits. Audit every remaining
  boss for phase waits, movement, shots, and resource cost; prioritize by total
  time saved while preserving downstream survival and ammo.
- [ ] **Medium — room exits and reentries:** scan actual room transitions,
  flag short A → B → A loops and longer return paths, then classify each by
  pickup, boss, refill, route requirement, or avoidable detour. Rank candidates
  by full downstream cost rather than assuming every return is wasteful.

Quick structural scan of the v4 report found immediate return candidates at
rows **90–92** (Bubble Mountain/save room), **156–159** (Red Tower/Noob Bridge
and Fireflea), **263–264** (Red Kihunter/Fireflea), and **296–298** (Skree
Boost/Red Tower). Other returns include boss, item, and refill rooms and may be
intentional. This scan only compares adjacent report rows: it misses embedded
transitions such as early farming, and does not establish necessity or count
missed jumps. A full live-state scan remains to do.

### Full v4 watch scan

Replayed all **381,569 frames** of `master_tape_grapple_skip_red_clean` from
power-on; the final emulator snapshot matched the recorded endpoint
`99d07adf0df695d7…`. The actual room-ID trace contains **318 visits** and
**56 overlapping A → B → A patterns**. These are return patterns, not 56
confirmed unnecessary detours. Frame numbers below are zero-based absolute
power-on tape frames; 60 frames is one second.

- **Early farming:** Big Pink/Dachora alternates at frames 50,822–53,476,
  with three Dachora visits of 293–294 frames. Spore Super/Farming alternates
  at 68,079–71,768: five 220-frame Super Room returns between farming visits.
  Compare stock and downstream gates before removing them.
- **Red Tower (`0xA253`):** the visits starting at **136,858** (6,124f) and
  **186,813** (4,792f) are long; the latter is surrounded by Noob Bridge and
  refill-related returns. Prioritize visual review of both ascents and the
  resource consequences of their exits.
- **Post-Draygon:** Space Jump Room starts at **216,755**; `0xD81A → 0xD9D4
  → 0xD81A` occurs at **220,240–221,344**. This visit restores 27 energy
  to 399 in the Energy Recharge Room. Keep the refill while shortening
  the climb and westward crossing in Colosseum.
- **Early Climb (`0x96BA`):** the visit at **31,964** lasts 2,315 frames;
  assess it against the newer reactive opening, which is not part of v4.

For a first-pass jump inventory, count rising edges of **A** (the route's jump
button) during ordinary game state with at least 20 frames remaining in the
room. Flag an edge when Samus rises less than 8 pixels in the next 20 frames.
This yields **3,642 eligible presses and 1,123 no-rise candidates**, not a
count of missed jumps: door inputs, combat, motion already in progress, and
intentional low jumps can be flagged, while missed landings and other failures
can be missed. For example, the early Climb visit has **33 presses and zero
no-rise flags** despite the visibly poor climb. Do not use these numbers as a
performance score. Red Tower's long visits have 15/64 and 13/48 flagged/presses;
Space Jump Room has 4/12. Review with pose, trajectory, and video before
classifying failures and develop a detector for missed landings and retries.

[Per-visit room and candidate inventory](../recordings/speed_benchmarks/v4_watch_scan.json)
contains every visit, frame bounds, counts, sample candidate frames, and the
return list. [Frame trace](../recordings/speed_benchmarks/v4_watch_scan_trace.npz)
contains room, x, y, game state, pose, item flags, movement type, and actions
in that order. Room names are report-derived labels and can describe a route
segment rather than the room itself; use room IDs when reviewing.

### Tourian & finish summary (rows 307–339 + credits)

- Row 307 recharges at the Landing Site Gunship (499E / 40M / 10S / 5PB).
- Row 310 detours through Brinstar (`0x99BD→…→0x9C07→0x9BC8→…→0xA5ED`) to pick
  up the Brinstar Reserve Tank (+100R), Early Supers (+5S), and two +5 Missile
  packs, raising max ammo to 50M / 15S (95 Missile-eq vs. Mother Brain's 82-eq
  budget).
- Rows 316–319 clear Metroid Rooms 1–4 with Ice+Wave+Plasma; rooms 3–4 spend
  Missiles so natural drops top Supers to 15/15 and Reserve to 100/100.
- Rows 320–321 keep `E=33/499` + `100R` without triggering Auto-Reserve, letting
  Big Boy Room (322–323) replay verbatim. Row 325 recharges to 499E / 50M / 15S
  / 100R, and 1-Super door opens on rows 327 and 330 enter `0xDD58` with
  488/499E, 100/100R, 45/50M, and 13/15S (84 Missile-eq).
- Row 331 re-equips Screw Attack via the pause menu (`0x7327→0x732F`), breaks
  all four Zebetites (`12S+9M`), shatters the glass and clears MB1 (`1S+35M`),
  clears MB2 with Charge+Wave+Ice+Plasma (`pre-Rainbow E=388/499`), and reaches
  the Rainbow kneel at `(210,165)` with `E=88/499`.
- Rows 332–339 join directly into the live `finish_controller_for` controllers
  (`play_mother_brain_closeout` + `combat/escape.py`, all eight rows
  `room-controller`), reaching Landing Site ship launch with `E=453/499` and
  `1:20` on the escape timer, followed by the 13,417-frame neutral credits tail.
- All Tourian development tapes, solvers, and the audit log live under
  `recordings/search_development_clean/tourian/`. Prior row-283 and row-315
  masters are kept as `master_tape_row283_clean.*` and `master_tape_row315_clean.*`.

Verify or audit the promoted master at any time:

```bash
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python -m super_metroid.splice chain --replay --require-credits
# Independent audit of v5 through the final credits screen
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run python snes/super_metroid/scripts/tools/audit_master_tape.py \
  --stem speed_benchmarks/master_tape_grapple_skip_v5_clean
```

## Backward finish work

Rows 332–339 now use live state to finish Mother Brain after the Rainbow
kneel, cross the four Tourian escape rooms, climb through Crateria and
board the ship. The credits controller idles through the final message.
The Composer selects these controllers on their escape crossings and
records the resulting buttons for the power-on replay.

The ROM splice gate starts once at the Rainbow kneel or one of the seven
escape room entries, then follows the controller sequence to the final
screen. Delayed starts vary the entry motion and enemy timing. Loads and
RAM writes are blocked after startup.

The original Rainbow-kneel tape replayed to the same final state over
31,454 frames with zero loads or RAM writes after startup. Ship launch
left 1:05 on the escape timer.

```bash
QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
  uv run pytest snes/super_metroid/tests/test_credits_finish.py -m rom -q
```

Work backward next from the row-331 kneel. The archived barrier and MB1
strategy spends about 81–82 Missile-equivalents. The current Clean
capacities of 40 Missiles and 10 Supers give 70 equivalents at full stock.
Solve the budget before adopting that strategy at the natural entry.
Use the Tourian recharge in row 325, and measure energy before the
Rainbow Beam. The power-on replay and independent audit still need to
join the natural Tourian route to the finish.

## Speed work from the Rainbow kneel

The speed budget is 198,000 frames at the repo's 60 fps convention, with
the clock starting at first Ceres control. Keep splits for both ending
entry and the fully drawn final screen. The verified row-315 prefix
already costs 1:43:17 from Ceres, 48:17 over budget before the remaining
Tourian fights, escape and credits.

The first pass delays the jump in Escape Room 4 until Samus is under
the west shaft and walks into the first Climb jump to control momentum.
From the Rainbow kneel, it saves 416 frames in Escape 4, 158 in Climb,
and 27 in Parlor. Ending entry is at 17,437 frames and the final screen is
at 30,854 frames, a 600-frame improvement. The escape clock shows
1:16.07 at ending entry, and all eight entry cases pass the ROM gate.
The alternative kneel produced by the archived row-331 fight also replays
to the final screen. Climb takes 1,627 frames on that branch, so start
the next tuning pass with both Climb entries. The reports and buttons are
under `recordings/search_development_clean/finish_speed/`. Both
predecessor leaves are in the integration's `scratch/` directory as
`finish_speed_rainbow_climb.state` and `finish_speed_row331_climb.state`.

Run the existing finish controllers through the Composer from the same
pin on both sides of a change. Save the first run's JSON as the baseline.

```bash
export QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
pin=snes/super_metroid/tasks/g4_tourian_human_mb_anchors/f000000_boot_0xDD58.state
out=snes/super_metroid/recordings/search_development_clean/finish_speed
mkdir -p "$out"
uv run python -m super_metroid.splice credits --splice-state "$pin" \
  --tape-out "$out/before.npz" > "$out/before.json"
# Change the room's controller in combat/escape.py and test on the same pin.
uv run python -m super_metroid.splice credits --splice-state "$pin" \
  --compare "$out/before.json" --tape-out "$out/after.npz" > "$out/after.json"
```

The benchmark reports input frames per room, leave coordinates, energy,
and escape timer. It plays the recorded buttons in a fresh emulator
from the same pin and compares the splits and final RAM. Both runs
enforce zero loads, RAM writes, and deaths after startup.
`--compare` matches the ROM, pin, and `--entry-delay` before reporting
frames saved, with positive deltas indicating gains.

For room tuning, start with the real predecessor's settled leave. Use an
archived room entry and varying `--entry-delay` values to check motion
and enemy timing. Keep tactics tied to position, pose, and velocity.
Rerun the whole suffix from the Rainbow kneel after a room beats its
baseline. The extra traversal exposes changes in later rooms caused by
the faster leave. Run the eight-start ROM gate above before keeping the
change. Once the power-on join reaches the Rainbow kneel, run both
power-on replay gates with that row-331 leave.

## Sub-55:00 benchmark & route-policy roadmap

**End goal:** **Sub-55:00 RTA** (`198,000` frames @ 60 fps, first Ceres control → Ship boarding) on vanilla Super Metroid, Clean (0 state loads, 0 RAM writes, 0 deaths, credits complete), gradually replacing stored tape buttons with reusable, restore-free RAM-driven Skills.

### Foundation & CLI (voice / agent friendly)

- **Benchmarking (`splice/benchmark.py`)**: compares any `master_tape_*_clean.{json,npz}` against the standard 23-split KPDR Any% reference (`0:56:28.88` Late Ice / `57:11.75` PB) and the `55:00.00` target, tracks Skill % per segment, flags route-order shifts (e.g., Early Hi-Jump, Early Ice), ranks single-row hotspots, and logs every verified snapshot to `recordings/speed_benchmarks/history.json`.
- **Swappable Route Policies (`splice/route_policy.py` + `policies/route_profiles/*.json`)**: use `--only RULE_ID` to apply an individual rule to the current verified tape. `--use-tape RULE_ID` keeps the selected source's inputs for that span. A full profile build starts from the original chain.
- **Natural-entry checkpoints** are cached in `scratch/clean_checkpoints/<tape_sha256>/`. An A/B cache miss reconstructs the predecessor from the selected source tape. Generate every checkpoint in one pass with `benchmark --scan --save-checkpoints`.

```bash
export QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy

# Compare any Clean tape against the standard 0:56:28.88 splits + <55:00 goal
uv run python -m super_metroid.splice benchmark
uv run python -m super_metroid.splice benchmark --scan --save-checkpoints --record-history --label <tag>

# Inspect route policies & restore-free Skills
uv run python -m super_metroid.splice policy list
uv run python -m super_metroid.splice policy show speed_detours_v1

# Local checkpoint A/B (tests Skill vs old tape + downstream Sync, zero risk to master)
uv run python -m super_metroid.splice policy ab speed_detours_v1 business_to_frog_r237

# Build and power-on prove the Red Tower route from the v3 source
uv run python -m super_metroid.splice policy build grapple_skip_fast \
  --only red_tower_space_jump_r301 --only tourian_direct_r313 \
  --only metroid_rooms_r316 --only big_boy_room_r322 \
  --only rinka_shaft_direct_r328 --only mb_tourian_v2_r331 \
  --source-stem snes/super_metroid/recordings/speed_benchmarks/master_tape_no_crocomire_clean \
  --stem snes/super_metroid/recordings/speed_benchmarks/red_tower_rebuild_clean
```

### Baseline macro-block gap (`v0_baseline` `1:58:13.95` vs `0:56:28.88` reference)

| Macro Block | Ref `56:28` | `v0_baseline` | Gap | Primary Recovery Levers |
|---|---:|---:|---:|---|
| 1. Ceres Station | `01:31.31` | `01:58.25` | `+00:26.94` | `routes/kpdr/ceres/` arm-pump + fast Ceres Ridley Skill |
| 2. Crateria + Blue Brinstar (→ Bomb) | `03:48.85` | `05:03.47` | `+01:14.62` | Movement polish + menu-toggled Clean Moonfall |
| 3. Early Brinstar (Bomb → Charge) | `03:20.56` | `10:15.37` | `+06:54.81` | Fast Bomb Torizo; **Early Supers Mockball** policy (skips Spore Spawn + farms) |
| 4. Kraid + Varia (incl. Hi-Jump diff) | `03:14.81` | `06:24.65` | `+03:09.84` | Fast Baby Kraid return Skill; standard post-Varia Hi-Jump order |
| 5. Upper Norfair (Varia → First PB, Early Ice) | `08:00.31` | `13:29.85` | `+05:29.54` | Bubble Mountain direct Skill; skip Red Tower refill (row 127) |
| 6. Wrecked Ship (First PB → Gravity) | `05:41.41` | `09:05.18` | `+03:23.77` | Short-charge Moat/WO (arrive w/ ammo+E) → `combat/phantoon_doppler.py` |
| 7. Maridia (Gravity → Plasma, w/ Croc+Grapple) | `08:39.83` | `27:14.30` | `+18:34.47` | **#1 gap**: Business Center direct (**+1:21 done**), cut Red Tower PB farm, **skip Crocomire + Grapple** (~10–12 min!) |
| 8. Lower Norfair (Plasma → Ridley) | `07:53.08` | `17:03.90` | `+09:10.82` | **#3 gap**: Map Room direct (**+1:31 done**), cut Spring Ball (**+2:35**), fast Screw/Ridley |
| 9. Return to G4 (Ridley → Tourian) | `05:30.62` | `15:08.28` | `+09:37.66` | **#2 gap**: cut late Brinstar Early Supers/Reserve detour (**+2:57**), fast Red Tower/LN return |
| 10. Tourian + Mother Brain (G4 → MB2) | `05:53.87` | `07:37.82` | `+01:43.95` | Upper Tourian Save bypass (**+0:18 done**), faster Metroid Rooms 1–4 Skills |
| 11. MB3 + Escape → Ship | `02:54.23` | `04:52.88` | `+01:58.65` | Tune live `combat/escape.py` Skills (Escape 1–4, Climb, Parlor) |
| **TOTAL (Ceres → Ship)** | **`56:28.88`** | **`1:58:13.95`** | **`+1:01:45.07`** | **`+1:03:13.95` to Sub-55:00 (`55:00.00` / `198,000f`)** |

### Five-wave strategy to Sub-55:00

1. **Wave 1 — Foundation + Zero-Route-Risk Detour, Tourian & Escape Skills (`speed_v1_verified` / `speed_v2_tourian` / `speed_detours_v1`):**
   - `speed_v1_verified` (`v1_escape_climb`): **POWER-ON VERIFIED** on `recordings/speed_benchmarks/master_tape_speed_v1_clean.{json,npz}` (`450,390f` total, `1:58:11.48` RTA Ceres→Ship, **-148f / -2.47s** net across Climb + Parlor, 9.6% Skill).
   - `speed_v2_tourian` (`v2_tourian_skills`): **POWER-ON VERIFIED** on `recordings/speed_benchmarks/master_tape_speed_v2_clean.{json,npz}` (`443,259f` total, `1:56:12.63` RTA Ceres→Ship, **-7,279f / -2:01.32** vs `v0_baseline`, **12.2%** Skill coverage, 0 loads, 0 RAM writes, 0 deaths, `rom_vanilla: true`, `credits_complete: true`, `all_planned_crossings_match: true`). Combines `climb_fast_r337` with 4 new restore-free Tourian RAM Skills (`routes/kpdr/tourian/{entry,metroid_rooms}.py`):
     - `tourian.first_room_direct` (`r313–315`, `+1,056f / +17.60s`): skips Upper Tourian Save Room (`0xDF1B`).
     - `tourian.metroid_rooms_all` (`r316–319`, `+4,111f / +68.52s`): freeze-and-blast crowd control + 46f drop sweep + PB latch escape, exits with `S=15/15, M=40/50`.
     - `tourian.big_boy_room` (`r322–323`, `+1,545f / +25.75s`): energy-independent Baby cutscene wait (`enemy0_x > 60000`) + shot-block staircase clear.
     - `tourian.rinka_shaft_direct` (`r328–330`, `+580f / +9.67s`): skips Lower Tourian Save Room (`0xDE23`), enters `0xDD58` with `E=498/499, M=45/50, S=13/15`, paired with Clean MB solve `row331_tourian_v2_clean.npz` (`7,143f`, hitless MB2).
   - `norfair.business_to_ice_gate` (rows 169–177): local A/B **PASS**, **+80.70s** saved, `+5` energy, rows 178–180 `saved+idle:0`; full-chain blocker: row 181 (Crocomire open-loop RNG tape — eliminated completely by `no_crocomire`!).
   - `norfair.business_to_frog` (rows 237–239): local A/B **PASS**, **+91.28s** saved, all 11 rooms 240–250 `tape+idle:0` to `(39,139)`; full-chain blocker: row 251 (Golden Torizo open-loop RNG tape — enters at `E=31–52/399`, charged Plasma+Wave+Ice deals 900 dmg/shot at `$0CD0 >= 62`, needs a low-HP dodge/drop-harvest Skill).
   - `norfair.bubble_to_bat_cave` (rows 88–93, 1-Super Clean variant): local climb saves `+14.55s`; needs Bat Cave entry alignment / Super stock check before enabling.
2. **Wave 2 — Same-Route Boss & Movement Skills (~12–15 min saved, target ~1:40):**
   - Clean Moat + West Ocean short-charge Skills (preserve ammo + E) → `combat/phantoon_doppler.py` (~2:20 saved).
   - Fast Ceres Skills (`routes/kpdr/ceres/`, ~0:25 saved), fast Bomb Torizo (~1:00 saved), fast Escape (`combat/escape.py`, ~1:00 saved), fast Ridley & Metroid Rooms (~4:00 saved).
   - Convert the 4 rollout controllers (rows 41, 43, 73, 210) to restore-free RAM Skills.
3. **Wave 3 — Minor Route Cuts on Swappable Boards (~15–18 min saved, target ~1:22):**
   - Grapple Skip already bypasses Spring Ball at row 229. Work next on the late Brinstar Reserve and ammo detour (310), the Red Tower refill and Power Bomb farm (127 and 158), the Colosseum refill (215), and the Golden Torizo and Screw Attack rooms.
4. **Wave 4 — Major Any% KPDR Route Policy Shifts (~20–24 min saved, target ~0:58):**
   - Grapple Skip is the main line. The audited v3 starting point was `1:39:59.05`. Biggest named gaps versus BEST were G4 `+9:07`, Ridley `+8:03`, and First Super `+6:21`. Ridley's preceding split differs because of Early Ice. [Grapple Skip proof](GRAPPLE_SKIP_SCOPE.md) · [Split data](SPEED_SPLITS.md)
   - `early_supers` policy: Mockball Early Supers, skipping Spore Spawn + Big Pink/Spore farms (~6.5–7.5 min saved).
   - Standard KPDR order (`late_ice` or fast `early_ice` — both selectable via route policy!).
5. **Wave 5 — Sub-55:00 Execution Polish (~4–6 min saved, target `< 55:00` / `< 198,000f`):**
   - Menu-toggled Clean Moonfall (`Climb`, `Parlor`), short-charge shinesparks, continuous room-by-room Skill hill-climbing via `splice policy ab` + `splice benchmark`.

### Opening skill chain

The `start_to_bombs` profile has a power-on proof through row 38. Its
remaining early gap is 70.01s at Bombs pickup against the 56:28.88
reference, with 50.05s in the First Missiles-to-Bombs split. Build a
reactive Parlor return (row 29, 1,667f) from the faster Climb exit (row 28,
2,000f). Carry the changed entry through Flyway, the Bombs pickup, and
Torizo's drop and exit.

The pickup and fight skills also passed a separate power-on run through
row 46 and First Supers in 67,222 frames, a gain of 4,690 frames through
that join. In a fight-only full-chain build, the row 47 Big Pink inputs
turned back into the Farming Room from the shifted entry. Adapt the
archived Big Pink controller for the Farming Room leave at `(1240,1419)`
and validate Charge Beam, the green hatch and the remaining Super stock.
[Opening splits and proof](SPEED_SPLITS.md#reactive-opening-on-grapple-skip)

```bash
uv run python -m super_metroid.splice policy build start_to_bombs \
  --source-stem snes/super_metroid/recordings/speed_benchmarks/master_tape_grapple_skip_red_clean \
  --through-row 38 \
  --stem snes/super_metroid/recordings/speed_benchmarks/start_to_bombs/rebuild_clean
```

### Highest-leverage next tests

| Rows | Current cost | Test and downstream acceptance |
|---|---:|---|
| 310 | 10,465f | Price Early Supers and earlier ammo pickups against the late Reserve and missile detour. Keep the Tourian barrier and Mother Brain ammo budget valid through row 331. |
| 41–44 | 10,989f for Spore Spawn alone | Prototype Early Supers with mockball. Verify Charge pickup and current and maximum stock through the Spazer gate. |
| 158 | 9,184f | Shorten the Red Tower refill and Power Bomb farm. Entry has 11 energy. Measure through the tube break at 164 and the grappleless Botwoon/Full Halfie join. |
| 127 | 7,070f | Reuse the Ice climb and improve the refill earlier in the return. Natural entry has 5 energy. Time through Alpha Power Bombs and Phantoon entry. |
| 135–140 | 11,144f for Phantoon alone | Time Moat and West Ocean with the doppler fight and drops. Verify the Attic and Gravity join with actual remaining ammo. |
| 88–95 | 873f local opportunity | The direct Bubble Mountain climb reaches Bat Cave but spends a Super and arrives spinning. Row 95 fails. Match the pose on the ledge and confirm usable stock through Speed Booster before enabling. |
| 297–301 | 1,464f net saving in A/B | The Skree Boost Room bypass reaches Hellway with 195 energy and 10 Supers and joins rows 302–310. Rebuild through row 331 to check the changed fight timing. Candidate and A/B are in `recordings/speed_benchmarks/grapple_skip_work/`. |

## Major-section speed work

Use boss and major pickup milestones or settled hub entrances as section
boundaries. Keep each farm or refill with the traversal or fight it funds.
The splits below are elapsed input time from the verified row-315 master at
60 fps. Tourian rows 316–331 remain the finish priority.

| Rows | Section and finish checkpoint | Baseline | What to try |
|---|---|---:|---|
| 2–44 | Ceres control through Brinstar Supers | 15:35.17 | Shorten the Big Pink farm and Spore Spawn loop (41–44). Check the stock needed for the return farm and Spazer gate. |
| 45–83 | Supers through Kraid and Varia to Business Center | 11:01.93 | Time the post-Supers farm (46) with the Spazer approach. Check ammo and energy at Kraid and at the Business Center exit. |
| 84–134 | Upper Norfair, Speed/Wave/Ice, first Power Bombs to the Moat | 12:03.67 | Test a direct Bubble Mountain crossing around the save-room loops (88–92, 1,927 frames). Keep the row-127 refill unless a faster Ice return leaves enough energy to reach the Moat. |
| 135–148 | Moat through Phantoon to Gravity | 7:39.13 | Time Moat and West Ocean together. Shorten Phantoon (140, 11,144 frames) while keeping the stock for the Attic and Gravity exit. |
| 149–185 | Gravity through the tube break and Crocomire to Grapple | 11:43.85 | Shorten the Red Tower refill and Power Bomb farm (158, 9,184 frames) against the later yellow hatches and tube break. Include Crocomire's exit in fight timing. |
| 186–223 | Grapple through Maridia, Botwoon and Draygon to Plasma | 15:33.40 | Time the Colosseum approach, Draygon and the exit refill (208–215) together. Carry enough energy into the Plasma route and Lower Norfair. |
| 224–264 | Plasma through the Norfair descent and Screw Attack to Red Kihunter Shaft | 13:46.03 | Test a direct Business Center exit around the Map Room loop (237–238, 6,279 frames). Keep the Spring Ball route for the first pass and verify stock at row 264. |
| 265–283 | Lower Norfair through Ridley and his tank to Bubble Mountain | 7:42.93 | Shorten the tunnels and pirate fights with the Ridley drop and return budget in view. Keep enough stock for the row-284 floor opening. |
| 284–315 | Bubble Mountain through the statues to Metroid Room 1 | 8:11.42 | After the Tourian join works, try bypassing the Upper Tourian Save Room (313–314, 1,635 frames). Recheck ammo and the row-316 fight from the new entry. |

First try the Business Center and Bubble Mountain detours as separate
candidates. Their frame counts include travel back to the hub, so measure the
actual gain through the next common settled exit (239 and 93 respectively).
The Map Room loop gains 15 energy on the current tape. Check what the direct
route carries into Lower Norfair.
Then take on the Brinstar and Wrecked Ship blocks, where the longest early
farm and fight can be timed from established entries.

For local A/B, load the preceding row's `scratch/clean_checkpoints/rowNNN_leave.state`
once with zero added settle frames. For rows 265 and 284 use
`scratch/master_tape_row264_clean.state` and `scratch/master_tape_row283_clean.state`.
The current master buttons reproduced the opening rows of all nine sections
from those pins, including 88–93 and 237–239 from the detour entries.
Compare through the common exit and the next resource-dependent rooms. Check
settled room, pose and momentum, energy, ammo and selection, item/boss/event
flags, and RNG-dependent drops or enemies that the next section relies on.

Carry the candidate's actual leave forward and try the existing master
buttons first. If the successor diverges, retime or re-solve from that leave
and count the net gain through the rejoin. Build accepted candidates on
separate stems with `chain --from-row` and retain the committed buttons for
unchanged rows. Use a separate candidate board if a route removes crossings.
Before promotion, replay from power-on through row 315 or the latest verified
frontier. Once the natural Tourian join lands, require the final screen and
the independent audit for the combined run.

## Speedrun techniques and build order

Use the major sections for timing and integration. Within each section,
search a single trick from the last settled entry shared with the baseline.
Keep setup, recovery, and the successor's cost in that comparison. The
[tech tree](TECH_TREE.md) tracks builder availability and speedrun sources.
Its Map Rando difficulty labels describe execution difficulty. The risk
below describes how far a candidate can change the Clean run.

| Risk | What can change | Evidence to collect |
|---|---|---|
| Low | A local input window with the same goal and resource needs. Entry pose or momentum may shift. | Same-pin A/B, settled exit, and the next crossing from the actual new leave. |
| Medium | A tight movement window, carry through a door, or new enemy timing. Recovery or a bad entry can erase the gain. | Local A/B and the rest of the section. For a reusable Skill, vary the approach or input timing and show how it recovers when the window is missed. |
| High | Different route, drops, resource spend, boss phase, or game options. Failure may surface sections later. | Compare through the next dependent gate, fight, or refill. Check boss/item/event flags and current and maximum stock. Rebuild and prove the candidate prefix on its own stem before adopting it. |

| Technique | Risk and failure mode | Build or benchmark next |
|---|---|---|
| Jump timing, braking and exit input | Low for one stable crossing. An early jump hits the lip, excess momentum misses the landing, or a held button changes the successor's first input. | Reuse `combat/escape.py`. Escape 4 → Climb is the model, with a 600-frame net gain from the Rainbow kneel through the final screen. Tune the first Climb jump on both saved entries next. |
| Arm pumping | Low on a clear runway. Higher speed changes takeoff position and door timing. | Benchmark the existing [Ceres arm-pump spans](../routes/kpdr/ceres/spine.py) from row 1 through the next settled Ceres exits. Try later runways after measuring the Ceres gain. |
| Early door shots and elevator exits | Low with the same weapon and route. Medium when carried speed or an enemy changes the next doorway. | Try firing on approach and leaving the elevator from the exit side. Check input acceptance, one successful shot, and the next settled leave around Ice Beam Gate (118–119) or the Norfair elevator return (236–239). |
| Wall jumps, consecutive wall jumps, and ledge grabs | Medium. Contact side, spin, aim, and entry velocity decide whether the climb lands. | Use [walljump](../routes/skills/walljump.py) and [ledge grab](../routes/skills/basic_moves.py) with a plain jump fallback. Benchmark Bubble Mountain to Bat Cave (row 93) from both the save-room return and the direct approach. |
| Mockball | Medium on the same route. Morph timing can lose speed or leave Samus in the wrong pose for the exit. | The [TAS extractor](../tas/skills_extract.py) finds input patterns. Build a ROM-checked movement candidate in a same-route Lower Norfair morph tunnel (row 265), with the regular bomb path as the comparison. Early Supers is a separate route change. |
| Short charge, stutter step, and shinespark | Medium for equivalent landings. High if the faster path changes energy, the charge runway, or the next door shot. | Compare full charge with the existing [short-charge and stutter helpers](../routes/skills/shinespark.py) across rows 134–138. Time the runway detour, spark crash and recovery, and Wrecked Ship hatch together. |
| Pause and idle trims | Medium. A pause may change input edges or RNG phase, and a long idle can be part of a required transition. | Use the [pause-span finder](../scripts/tools/cut_pause_tape.py) on a human source, then confirm removable frames on the master entry. Time resumed control and the next enemy or door that relies on it. |
| Damage boost | High. The contact can spend needed energy or send Samus the wrong way. | Benchmark a controlled boost only where the baseline already takes damage. Bowling Alley into Gravity (147–148) is a first test. Compare total energy and time through the pickup and first exit. |
| Kraid quick kill, Phantoon doppler and Double Super | High. The firing window, ammo selection, enemy pattern, and later drops can change together. | Kraid's rollout already clears row 73 in 1,595 frames. Reuse [Phantoon doppler](../combat/phantoon_doppler.py) against the Clean charge-beam baseline (row 140, 11,144 frames). Test Double Super if the natural entry can spare two Supers. Include the Basement entrance, loot sweep, and Attic lock. |
| Grappleless Draygon | High. The approach leaves 59 energy and the fight must preserve enough for the exit. | Row 210 takes 6,876f. Time shot placement and Dodge windows using Charge, Ice, Wave, and Spazer. Compare through Space Jump and the Colosseum refill. |
| Farm and refill routing | High. A shorter farm can leave too little stock several hatches later. | Shorten Big Pink and Spore Spawn (41–47) through the Spazer gate, then test smaller Red Tower farms (158) through the tube break and Ice Gate floor (178). Track the cost of recovering missed drops. |
| Moonfall | High. It changes vertical speed and leaves Moonwalk active in the file. | Climb development probes have a 392-frame local gain against their seed. Build a controller-only Moonwalk setup from the menu and verify Parlor → Climb → Pit plus the next Morph approach. Carry the option state into the candidate's full downstream replay. [Moonfall probe evidence](tasks/rr-0hjh-residual.md) |
| Side-room and item detours | High. The next entrance and natural resource path can both change. | Start with the Map Room and Bubble Mountain loops above. Put a Spring Ball bypass or a route that changes major pickups on a separate candidate board and carry it through Lower Norfair. |
| Tourian ammo routing and barrier/fight shortcuts | High. Current stock is tight and Mother Brain needs a natural phase transition. | Close the row-316 to Rainbow join first. Then time Metroid ammo use, recharge, barriers, and the actual row-331 kneel as one section. A different finale approach needs its own replay from that kneel through credits. |

### Candidate workflow

1. **Choose the comparison.** For a movement or fight technique, name the
   start row and common exit, the next dependency, and the expected gain or
   measured baseline cost. Start with the current controller or committed
   buttons. Use the human trace or an authored TAS demonstration to seed the
   search window. Run the census when checking a human slice for sync.
2. **Search on the ROM.** Use the existing room or combat owner and the
   Composer. Reuse a Skill when its entry checks apply. Search timing and
   actions from the same pin and measure which live pose, position, momentum,
   or enemy phase predicts success. Save the winning buttons and the leave
   for replay. For the finish use the [credits A/B command](#speed-work-from-the-rainbow-kneel).
3. **Record and replay.** Keep the example/source, risk, ROM and pin identity,
   baseline and candidate frame counts, resource changes, first downstream
   divergence, and replay result with the candidate under
   `recordings/search_development_clean/`. A ROM replay of the recorded
   buttons provides the local result. Use actual successor state to measure
   the net section gain and replay through the risk's downstream dependency.
4. **Integrate one candidate at a time.** Re-run the same downstream section
   from the current accepted predecessor before combining gains. Promote
   after `splice chain --replay` reaches the verified frontier, using
   `--require-credits` once the run finishes. Use the independent audit for
   milestones and published claims. Track any remaining sync work in this plan.

Report the local and integrated deltas separately. A faster input window can
cost additional time at the next door, refill, or enemy. Record the furthest
verified row or final-screen result alongside each delta.

The first build pass is two separate detour A/Bs, then arm pumping in Ceres,
the Bubble Mountain wall-jump line, and short charge across Moat–West Ocean.
Use the resulting section timings to decide whether to invest in new
mockball, damage-boost, or boss-variant controllers. Keep the natural Tourian
join on the finish path while these candidates are being measured.

## From tape to policy

`chain` prints `policy_coverage` beside its proof. It now diffs each row's
committed buttons against its human source slice. The ledger label cannot
decide this: an override replaces the slice and the row still reads
`tape+idle:0`. At the full-game finish (rows 1–339 + credits tail, 450,538 frames),
44.3% of non-tail frames are verbatim human slices (193,645f), 1.0% are delayed
copies (4,340f), 45.4% are other stored buttons (overrides, searches, detours,
198,276f), and 9.3% are controllers (40,860f, including all eight live
post-Rainbow rows 332–339). CONTEXT.md calls a RAM-driven controller a Skill.

- Restore-free controllers today: rows 75 (Varia return), 135 (Moat), 136
  (West Ocean), 3,464 frames. Rollout users: rows 41 Big Pink, 43 Spore
  Spawn, 73 Kraid, 210 Draygon, 20,980 frames. Make those four read RAM and
  drop the restore. Draygon's one lookahead only finds the latch frame.
- About 70 rows commit buttons that are not their source slice. Rows built from now
  on are labelled `override:` with the npz hash. Rows 32-205 (and 215-262) were
  built before that and carry no pointer to their buttons; recover each from
  `recordings/search_development_clean/` before any of them is modified.

## End-to-end coverage

The Clean candidate board has 340 rows, including one boot metadata row,
the power-on sequence, 316 planned room crossings and 22 same-room rows
through Landing Site ship launch. Its 27 source tapes and two existing
crossing controllers connect every planned visit. The physical room graph
contains all 316 crossings. The route visits 168 of the 262 rooms in that
graph. The other 94 rooms each have a development entry in
`maps/room_problems.json`. An all-room run still needs an ordered route
through those rooms and recordings that connect them.

| Rows | Route section | Crossings | Open source-entry work |
|---|---|---:|---|
| 1–83 | Power-on to Business Center after Kraid | 76 | Verified from power-on. The actual replay includes 16 additional farming transitions in rows 41 and 46. |
| 84–134 | Upper Norfair to the Moat | 46 | Verified from power-on, including the Sloaters Energy Refill detour in row 127. |
| 135–148 | Moat, Wrecked Ship and Gravity | 13 | Verified from power-on, including Phantoon, the Attic enemy lock, and Gravity Suit. |
| 149–185 | Return to Norfair, Crocomire and Grapple | 36 | Verified from power-on, including the Maridia Tube break, Crocomire defeat and Grapple pickup. |
| 186–223 | Maridia, Botwoon, Draygon and Plasma | 35 | Verified from power-on, including the Colosseum refill detour (215) and Plasma (223). |
| 224–272 | Lower Norfair, Screw Attack and Ridley | 46 | Verified through Ridley and the Energy Tank. Includes the Spring Ball detour in 229, Golden Torizo Supers, and Screw Attack. |
| 273–315 | Lower Norfair return, statues and Tourian entry | 43 | Verified from power-on through row 315, including the Landing Site recharge (307), Brinstar Early Supers + Reserve Tank detour (310), Statues Room, and Tourian entry. |
| 316–339 | Tourian, Mother Brain, escape and ship launch | 21 | Verified from power-on through row 339 + 13,417-frame credits tail (`credits_complete=true`, 450,538f total). Rows 316–321, 324–325, 327, 330–331 use Tourian development overrides; rows 322–323, 326, 328–329 replay human slices verbatim; rows 332–339 run live `finish_controller_for` controllers. |

The Survival report independently covers 120 crossings through Phantoon.
Every one appears in the Clean board's physical links. The runs take
different sequences through those links, so resume Clean from its own master
state. The source-entry census remains a record of which archived candidates
sync from their source states. Cathedral, Rising Tide, Bubble Mountain, and
the Ice return needed different inputs on the Clean chain. Each remaining
candidate still needs to clear from its actual Clean predecessor.

## Route milestones

Rows below are inclusive IDs on the rebuilt Clean board. A row can be a
same-room pickup or a controller crossing. Each milestone uses the same
master tape. Promote it when an independent replay reaches its row **and**
shows the listed result. Check boss flags in bank 7E, collected item/beam
bits and actual ammo capacities at the appropriate split.

| Through row | Milestone | What the replay must show |
|---:|---|---|
| 44 | Brinstar Supers | Morph, Bombs, Bomb Torizo and Spore Spawn cleared, first five-Super capacity |
| 55 | Charge and Spazer | Big Pink Charge pickup, Super gate open, Spazer collected in A447 |
| 74 | Kraid and Varia | Hi-Jump and its tank, Kraid defeat flag, Varia collected in A6E2 |
| 130 | Upper Norfair and Power Bombs | Speed, Wave and Ice collected, first Power Bomb capacity in A3AE |
| 148 | Phantoon and Gravity | Cross the Moat and West Ocean, Phantoon flag set, Gravity collected in CE40 |
| 185 | Crocomire and Grapple | Crocomire defeated, Grapple collected in AC2B |
| 223 | Draygon and Plasma | Botwoon and Draygon defeated, Space Jump and Plasma collected |
| 272 | Ridley | Lower Norfair traversed with Screw Attack, Ridley defeated, Ridley Tank Room reached |
| 315 | Tourian access | All four major boss flags, natural statue opening, Tourian entrance and save-room return |
| 338 | Mother Brain and escape | Mother Brain defeated, escape rooms traversed before timeout, Landing Site reached |
| 339 + credits tail | Finish | Samus boards the ship. Replay all neutral tail inputs, verify the fully drawn final message, capture final screen and report elapsed time |

## Parallel work and integration (first dispatch, now closed)

| Owner during the first dispatch | Input | Work returned to the integrator |
|---|---|---|
| Frontier Sol | The last accepted master row | Solve consecutive leaves until the next failure. Save the best viable prefix and its replay, with entry/exit energy and ammo. This worker alone changes the canonical tape. |
| Kraid Sol | Repaired s10 doorway, then row 72's actual leave when available | Record buttons for the quick kill, Varia pickup and return. Report required ammo, first damage and the successor's settled position. |
| Credits Sol | Final Landing Site source, then the natural early-game farming state | Validate the credits detector and tail against the cartridge. On completion, search for a natural resource farm upstream of the Spazer gate. |
| Integrator | Clean board, source pins, independent replay | Grade room inputs on the same row IDs used by the solver. Test the two Moat/West Ocean crossings together. Update this handoff and STATUS only with the appropriate evidence. |

Independent workers may use an exact room-entry state, read RAM and restore
to search. Deliver SNES-12 buttons or a reusable room controller, source
identity, frame range, energy and ammo costs, and the next room. When the
master reaches that entry, try the candidate and run the next row. A different
loadout, input phase, or enemy RNG can change the result.

After Kraid, move the fight worker to Phantoon, then Botwoon/Draygon, then
Ridley. Move the terminal worker to the first resource shortage or uncovered
connection ahead of the master. A failed downstream room that needs ammo sends
work back to the nearest affordable pickup, enemy drop or refill station.
Develop alternative prefixes under separate artifact stems so the integrator
can compare the downstream result before replacing the accepted tape.

The 340-row source census graded 320 as recorded, one after a three-frame
delay (row 319), and 16 red. Two crossings require controllers and one row is
boot metadata. The saved results are in `recordings/tape_census_clean.json`.
Rows 331–332 are consecutive Mother Brain visits from different takes.

## Known resource and route gaps

- **Early Brinstar.** Super selection, the remaining pickup animation and
  enemy drops differ at the s6→s8 handoff. Spend Supers deliberately on gates.
  Keep enough energy and ammo to survive Green Hill Zone, get Spazer and kill
  Kraid. Measure this on the actual prefix.
- **Maridia, row 212.** The source ammo jumps from 30 to 60 Missiles,
  4 to 15 Supers and 5 to 10 Power Bombs. These are current counts.
  Measure actual capacities and stock on the Clean entry and budget
  natural pickups or drops for later fights.
- **Tourian, row 307 onward.** At the Landing Site seam, source energy
  jumps from 344 to 499, Missiles from 65 to 70 and Power Bombs from 2
  to 15. Budget Metroid rooms, barriers, Mother Brain's drain and escape
  from the actual Clean loadout. The end-of-credits detector has been
  verified from the ending pin. The master must continue past state 39
  to the fully drawn final message.

## Commands

Run from the repository root with the local ROM and human tapes installed.

```bash
export QT_QPA_PLATFORM=offscreen SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy

# Validate joins, physical room links, input slices and the ship-launch row.
uv run python snes/super_metroid/scripts/tools/tape_census.py --audit

# Source-entry candidates on the current board, including the late game.
uv run python snes/super_metroid/scripts/tools/tape_census.py
uv run python snes/super_metroid/scripts/tools/tape_census.py --rows 70-76 \
  --out snes/super_metroid/recordings/kraid_census_clean.json

# Advance to a milestone, then independently verify that milestone again.
uv run python -m super_metroid.splice chain --through-row 55
uv run python -m super_metroid.splice chain --replay --through-row 55

# Include the actual room sequence and its owning rows in the replay output.
uv run python -m super_metroid.splice chain --replay --proof-json

# Final acceptance returns success only for a complete loadless replay.
uv run python -m super_metroid.splice chain --replay --require-credits
```

Use `--override ROW=path.npz` while extending an unaccepted room. An override
for a row already on the master requires a new prefix ending before that row.
The board can be regenerated with `chain --rebuild-board` during a solver run.
Keep all development output under Super Metroid. The older assisted Phantoon
run, reference TAS movies and parked lsnes work are described in the
[game instructions](../AGENTS.md).
