# Status — Zelda I (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M5 |
| Best verified result | Power-on → defeat Aquamentus → collect Triforce shard 1 |
| Last verification | 2026-07-28 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **Level 1 complete** (`triforce & 0x01`, 2/2 natural + 2/2 isolated) |
| Integration | `LegendOfZelda-Nes` |
| ROM zip | `roms/Nintendo/NES/Legend of Zelda, The.zip` |
| Ready frame (probe) | ~567 (BOOT_PERIOD=50; was ~1749 @ period 180) |
| Checkpoints | `Level1.state`, `Level1Entrance.state`, `Level1FirstKey.state`, `Level1North.state`, `Level1Cleared63.state`, `Level1Cleared53.state`, `Level1Complete.state` |
| Evidence | [level1_complete_natural.json](../recordings/level1_complete_natural.json), [level1_complete_isolated.json](../recordings/level1_complete_isolated.json), [Level 1 route notes](LEVEL1_ROUTE.md) |

## Assisted full-game tip (does not change the M5 Clean gate)

| Segment | Result | Evidence |
|---------|--------|----------|
| L4 complete → L5 entry | 1/1 to room `0x76` in 5,031 path frames; bombs=7, Raft=1, Stepladder=1, Triforce=`0x0c` preserved | `l4_to_l5_assisted_v4.json`; `Level5EntranceFromL4` |
| L5 entry → room `0x66` key | 1/1 assisted clear in 1,254 frames; three Gibdos dead, keys 0→1 | `l5_clear66_entrance_assisted.json`; `Level5Cleared66` |

Both runs used the Survival health refill and reported zero progression writes
and zero capacity writes. They are development checkpoints, not Clean or
power-on STATUS promotions. The paused forward boundary is return
`0x66→0x76`, then open the east key door to `0x77`; the active backward pass
is documented below and in `docs/plan.md`.

## Backward endgame recon (fixture-only; does not change either gate)

| Segment | Result | Evidence |
|---------|--------|----------|
| Fully loaded final-Patra room `0x52` → Ganon `0x42` → Zelda `0x32` → credits/final page | **1/1**; Ganon type `0x3E`, brown ObjState nonzero, Silver Arrow kill sets `$0672=1`; rolling credits at frame 3,395, final page at 4,595 | `l9_ganon_credits_recon.json`; `Level9BeforeGanonReconFixture`; `Level9CreditsReconFixture`; `Level9FinalScreenReconFixture` |
| Live final Patra `0x52` → naturally earned north door → same Ganon/Zelda/ending suffix | **2/2 exact**; body `0x47` HP `0xB0`, 8 eyes `0x25` HP `0x60`; Patra clear 1,883f, credits 5,342f, final page 6,542f; inventory preserved through Patra, runtime controller writes 0 | `l9_patra_credits_recon.json`; `Level9FinalPatraReconFixture`; `Level9FinalPatraClearedReconFixture`; `Level9PatraFinalScreenReconFixture` |

Both backwards-development proofs compose the full inventory and room-loader
setup before their start state, so they remain `route_eligible=false`. The new
Patra run does **not** inherit the earlier object-removal or north-door writes:
after reset it uses controller input only, naturally raises door bit `0x08`,
and continues without another state load. Survival restores four heart units
per trial (two in `0x52`, two in `0x42`) with zero deaths and zero progression/
capacity writes. This proves the final-Patra/Ganon/Zelda/ending suffix, **not**
a Survival-assisted or Clean Level 9 route. Next backward boundary: inspect
candidate predecessor room `0x62` under `rr-sz8.3`; see
[LEVEL9_ROUTE.md](LEVEL9_ROUTE.md).

## Verified segments

| Segment | Entry | Result | Frames (typ.) | Evidence |
|---------|-------|--------|---------------|----------|
| Wooden sword cave | `Level1.state` (isolated) | sword=1 on screen 0x77 | ~796 | `sword_cave_isolated.json` (2/2) |
| Wooden sword cave | power-on boot (natural) | sword=1 on screen 0x77 | ~758 | `sword_cave_natural.json` (2/2) |
| Sword → Level 1 interior | isolated (post-sword from state) | `level==1` | ~2193 nav | `to_level1_isolated_dungeon.json` (2/2) |
| Boot → Level 1 interior | power-on (natural chain) | `level==1` | ~758+2328 | `to_level1_natural_dungeon.json` (2/2) |
| Entrance 0x73 → first key | `Level1Entrance.state` | `level==1 && keys>=1` in 0x74 | 4272 | `level1_first_key_isolated.json` (2/2) |
| Power-on → first key | power-on natural chain | `level==1 && keys>=1` in 0x74 | 758+2328+1091 | `level1_first_key_natural.json` (2/2) |
| First key → north room 0x63 | `Level1FirstKey.state` | 0x63, mode 5, 3 Stalfos spawned | 1002 | `level1_north_isolated.json` (2/2) |
| Power-on → north room 0x63 | power-on natural chain | same room-ready predicate | 758+2328+1091+1004 | `level1_north_natural.json` (2/2) |
| Room 0x63 clear | `Level1North.state` | 0 live Stalfos, RoomAllDead≥20 | 2706 | `level1_clear63_isolated.json` (2/2) |
| Power-on → room 0x63 clear | power-on natural chain | same clear predicate | 758+2328+1091+1004+2922 | `level1_clear63_natural.json` (2/2) |
| Room 0x63 clear → room 0x53 key | `Level1Cleared63.state` | 0 live Stalfos, RoomAllDead≥20, keys=1 | 1506 | `level1_clear53_isolated.json` (2/2) |
| Power-on → room 0x53 key | power-on natural chain | same clear + collected-key predicate | 758+2328+1091+1004+2922+1508 | `level1_clear53_natural.json` (2/2) |
| Room 0x53 key → room 0x54 clear | `Level1Cleared53.state` | 0 live Keese, RoomAllDead≥20 | 1223 | `level1_clear54_isolated.json` (2/2) |
| Power-on → room 0x54 clear | reusable natural milestone chain | same room-clear predicate | prefix + 1665 | `level1_clear54_natural.json` (2/2) |
| Room 0x53 key → Triforce shard 1 | `Level1Cleared53.state` | room 0x36 and `triforce & 0x01` | 14,391 suffix | `level1_complete_isolated.json` (2/2) |
| Power-on → Triforce shard 1 | reset / no state load | room 0x36 and `triforce & 0x01` | 29,039 total | `level1_complete_natural.json` (2/2) |
| Post-L1 OW → Level 2 path 0x4A | `Level1ExitOverworld.state` | screen 0x4A, triforce & 0x01 | ~2,886 | `level2_prefix_isolated.json` (3/3); hop timings `room_timings/level2_prefix_isolated_timing.json` (1/1, 2026-07-29) |

Natural-entry Level 1 chain uses `SwordCaveController`,
`OverworldToLevel1Controller`, `Level1FirstKeyController`,
`Level1UnlockNorthController`, `Level1Clear63Controller`, and
`Level1Clear53Controller`, followed by the generic `DungeonRoomSpec`
controller for room 0x54 (no RAM writes or state loads).

The complete Level 1 runner extends that same natural prefix through rooms
`0x52→0x42→0x41→0x43→0x33→0x23→0x44→0x45→0x35→0x36`.
It defeats Aquamentus with a projectile-aware controller, collects the Heart
Container, and accepts only the persistent first-shard bit. It remains
**Bronze / Clean**: read-only RAM plus controller input, with no state load or
RAM write during the natural attempt.

## Overworld path (probe-stable)

```
0x77 ─E@y140─► 0x78 ─N@x48─► 0x68 ─N@x48─► 0x58
  ─N@x112─► 0x48 ─N─► 0x38 ─W─► 0x37 ─UP@x112─► Level 1
```

**Traps:** 0x67 (north of start) is a tree-locked dead end; 0x47 is a lake (raft). Do not route col-7 straight north.

## Level 1 route (probe-stable)

```
entry 0x73 ─E─► first-key room 0x74 ─key─► W to 0x73
  ─spend key at north door─► room 0x63 (3 Stalfos)
  ─clear─► no drop; N→0x53 open ─clear 5 Stalfos─► fixed key@(128,109)
  ─W─► 0x52→0x42→0x41→0x43→0x33→0x23
  ─backtrack─► 0x43 ─E─► 0x44→0x45 ─N─► 0x35 Aquamentus
  ─E─► 0x36 Triforce shard 1
```

The walkthrough-informed correlation and required/optional branches are
documented in [LEVEL1_ROUTE.md](LEVEL1_ROUTE.md). Room `0x54` is the optional
Compass branch; the accepted speed route also skips the Map, Bow, and
Boomerang pickups.

Room 0x74 has five Stalfos and two block clusters. The natural policy acquires
the carried key without requiring a full room clear, returns via the lower lane
(y≈181), and spends it at the locked north door.

Room 0x63 clear uses a hybrid chase/patrol sword policy (2706 frames isolated /
2922 natural from room-ready). RoomItemId stays `0x03`; keys/rupees/bombs do not
change. North of 0x63 is room **0x53** (five Stalfos, RoomItemId=`0x19` key).

Room 0x53 reuses the chase/patrol combat, then collects the fixed room-clear
key at `(128,109)`. It succeeds in 1506 frames isolated / 1508 natural from the
0x63-clear endpoint with health unchanged at `0x20`; keys go 0→1 while
rupees/bombs remain unchanged. `RoomAllDead>=20` is the clear signal. The
transient type `0x60` object seen at some enemy death positions is a green
rupee, not the room key.

Door probes from the saved endpoint confirm south→`0x63`, west→`0x52`, and
east→`0x54` are open; north is closed. Room `0x52` has six Keese (type `0x1B`,
RoomItemId=`0x03`). Room `0x54` has eight Keese (type `0x1B`,
RoomItemId=`0x16`).

Room 0x54 is the first data-driven `DungeonRoomSpec` segment. Keese liveness
must use object type because their HP bytes remain zero. A 16-trial,
four-process lab sweep went 16/16; attack phase 0 + engage distance 48 ranked
first at 1223 isolated frames. The promoted policy then passed 2/2 isolated
and 2/2 full power-on natural-entry trials (1665 natural suffix frames).
Clearing causes no known inventory change because the policy does not collect
the item. The walkthrough correlates `0x16` with the optional Compass.
West returns to 0x53 and a physical east-door probe is blocked.

The Zelda-local dungeon lab now provides parallel policy sweeps, full traces
and first-divergence reports, 120-frame failure tails, phase RAM deltas with
known/unknown symbols, physical exit probes, generated reports/spec
suggestions, reusable milestone chaining, and SHA-256 checkpoint provenance.
See `docs/DUNGEON_LAB.md`.

## Done

- Directory layout and NES integration stubs
- `scripts/setup_rom.py` / `scripts/boot_probe.py`
- **M2 instrumentation** — mode, level, screen, Link x/y, facing, sword, bombs, rupees, health, cave vs overworld (`ram.py`, `data.json`)
- **Shared graph core** — `retro_harness.adventure` (`RouteGraph`, capability BFS, leg planning)
- **Overworld + early route graph** — verified path screens + Level 1 portal
- **M3–M5 sword segment** — enter NW cave on 0x77, wooden sword, return to start
- **M3–M5 Level 1 overworld** — sword → tree door → dungeon interior
- **M3–M5 Level 1 first rooms** — entrance 0x73 → first key in 0x74 → locked
  north door → clear 0x63 → clear/key 0x53 → east → clear eight Keese in 0x54
- **M3–M5 Level 1 completion** — required west route, switch/hint, Map room,
  two more keys, Goriya/Wallmaster rooms, Aquamentus, Heart Container, and
  Triforce shard 1; 2/2 isolated and 2/2 Clean natural-entry
- **Level 2 approach scaffolding** — post-triforce settle to 0x37, walk prefix
  to 0x4A (controllers, route graph, runner); suffix to 0x3C open
- **Screen/room timer** — `room_timer.py` + opt-in `--room-timing` on
  Level 1 complete / Level 2 prefix via `chain.run_controller_stage`
- **Dungeon instrumentation** — room item/count, live object types/positions/HP,
  key inventory, opened-door bits, and room-ready/clear stop predicates
- **Dungeon laboratory** — room specs, parallel sweeps, trace diff/failure
  tails, RAM deltas, exit probing, provenance, and generated handoffs

## Level 2 overworld + entry (in progress)

After Triforce fanfare the engine returns Link to **overworld 0x37** (~704
idle frames). From there the agent **walks** (no save-state warp).

Verified walk prefix (controller target 0x4A, 3/3 isolated from
`Level1ExitOverworld`):

```
0x37 E@y140 → 0x38 S → 0x48 S → 0x58 E → 0x59 N → 0x49 E → 0x4A
```

Stop: `level2_path_prefix_success` on screen 0x4A (~2886 frames). See
[LEVEL2_ROUTE.md](LEVEL2_ROUTE.md). Evidence:
`recordings/level2_prefix_isolated.json`. Checkpoint fixture:
`Level1ExitOverworld.state`.

**Door-path geometry** (probe-verified; Clean health not yet):

```
0x37→38→48→58→59→5A→5B→5C(maze)→5D@x52↑→4D→4C→3C door
```

Avoid `0x4B→0x5B` (north entry seals east). Enter 0x5B from **0x5A**. 0x5C
needs BFS maze waypoints (`LEVEL2_5C_MAZE_WAYPOINTS`). Dev fixtures:
`Level2DoorOW`, `Level2Entrance`, `Level2EntryFresh`.

**Interior first facts:** entry **0x7d** empty combat; north **0x6d** = **5×
Rope `0x28`**, spawn ~100f, clear sets LEFT door bit `0x02` (LEFT @ y≈141 →
**0x6c** 6 Ropes + fixed key `0x19` @ ~**(136,141)**); east **0x7e** = **5×
Rope + key `0x19`** via diamond-nav (y≈157 wall-first, not naive y≈141 RIGHT).
Recon: `recordings/l2_recon_probe.json`. Specs: `ROOM_7D_SPEC` / `ROOM_6D_SPEC` /
`ROOM_6C_SPEC` / `ROOM_7E_SPEC` / `ROOM_6E_SPEC` / `ROOM_6F_SPEC`. Walkthrough:
[LEVEL2_ROUTE.md](LEVEL2_ROUTE.md).

**Isolated pure (Clean, checkpoint — not natural-entry STATUS):** 0x6d clear
2/2 from `Level2Entrance` (`level2_clear6d_isolated.json`, 674f); 0x6c west
key 2/2 from `Level2RopesCleared` + 2/2 chain from entrance
(`level2_clear6c_isolated.json`, `level2_clear6c_from_entrance_isolated.json`);
0x7e east key 2/2 from `Level2Entrance` (`level2_clear7e_isolated.json`,
1110f, checkpoint `Level2EastKey.state`); 0x6f gels+compass 2/2 from
`Level2EastKey` (`level2_clear6f_isolated.json`, 1443f, checkpoint
`Level2Compass.state`). Stops: `level2_room_6d_cleared`,
`level2_room_6c_key_success`, `level2_room_7e_key_success`,
`level2_room_6f_compass_success`.

**Assisted Moon entry (not Clean STATUS):** 2/2
`level==2` mode 5 room **0x7d** ~(120,205) via
`probe_level2_suffix.py --infinite-life --enter-dungeon`
(`recordings/l2_entry_assisted_t{0,1}_probe.json`). Stop:
`level2_entrance_success`. Checkpoint `Level2Entrance.state` = room-ready.

**Assisted L2 complete (not Clean STATUS, 2026-08-07):** Boom → Dodongo →
south-band TF `0x0d` sets `triforce & 0x02` (**2/2**). Evidence:
`l2_complete_assisted.json`; checkpoints `Level2Complete`,
`Level2ExitOverworld`. Magical Boomerang / Dodongo geometry documented in
LEVEL2_ROUTE.

**Assisted post-L2 → L3 enter (not Clean STATUS):** OW `0x3C` → reverse door
corridor → Manji door `0x74` → room `0x7c` (**2/2**). Evidence:
`l2_to_l3_assisted.json`. Runner: `run_l2_to_l3.py --infinite-life`.

**Not yet Clean STATUS:** natural-entry L2 full clear; continuous power-on →
L3; Clean door-path health. (Clean **At4A→0x3C** isolated exists via `rr-hxs`
but is not power-on natural STATUS.)

### Measured door-path breakpoint (2026-07-29)

Opt-in `room_timer` on `LEVEL2_DOOR_HOPS` from `Level1ExitOverworld` (2/2
identical fail, Clean input only):

| Hop | location_frames | hearts on arrival |
|-----|-----------------|-------------------|
| 0x37→38 | 423 | 3/4 |
| 0x38→48 | 448 | 3/4 |
| 0x48→58 | 477 | 2/4 (−1 on 0x48/58) |
| 0x58→59 | 498 | 2/4 |
| 0x59→5A | **659** | 1/4 (−1 on 0x59/5A) |
| 0x5A→5B | 598 | 1/4 |
| 0x5B→5C | **718** | **0/4** (−1 on 0x5B/5C) |

Stop: **death on 0x5C** at ~(16,93), mode 17, hop 7/11 (next would be maze
0x5C→0x5D). Artifact:
`recordings/room_timings/level2_door_path_probe_timing.json`.

Verified prefix 0x37→0x4A remains 1/1 with hop timing
(`room_timings/level2_prefix_isolated_timing.json`, 6 hops, slowest
0x49→0x4A ~539f). **No controller change** after this measurement: death is
heart-starvation before the maze, not a single misaligned hop to tweak.

## Not done

- L5 remainder (Whistle → Digdogger → TF `0x10`) then L6–L9 under assist
- Clean residual after full-game assist pass
- Natural-entry continuous power-on chain (deferred under assist-first)
- Full eight-dungeon/Ganon route graph
- Broader overworld bomb / white-sword chain
- Continuous multi-dungeon dry run (M6–M8)

## Dual track (2026-08-06)

| Track | Tooling | STATUS role |
|-------|---------|-------------|
| Clean | default runners | only path that promotes verified Clean gates |
| Survival-assisted | `--infinite-life` / `assist.UnlimitedHealthAssist` | first-pass geometry; contract `ASSIST_CONTRACT.md` |

Infinite life is implemented and CLI-wired; **no assisted end-to-end segment
is STATUS-promoted yet**. Process: `docs/tasks/PROCESS.md`. Work: `bd ready -l zelda_i`.

## Next

1. **Active backward tip:** candidate Level 9 room `0x62` → live final Patra
   `0x52` → proven credits suffix — `rr-sz8.3`; exact command and evidence
   boundary in `docs/plan.md`.
2. **Paused forward tip:** L5 east key (`0x66→0x76→0x77`) under Survival
   assist — `rr-28p`, then Whistle → Digdogger → TF `0x10`.
3. Clean residual only after full-game assist pass (`rr-4oz`).
