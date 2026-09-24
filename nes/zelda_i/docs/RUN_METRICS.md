# Run metrics — milestones toward a clean credits video

One row per measured continuous run, newest last. The row comes from
`scripts/run_metrics.py <report.json>`; the ledger behind it is
`spine/ledger.py` (wraps `env.step`, observation only).

- **frames**: power-on to the end of the last stage.
- **flutter**: Link's position reversing on one axis within 8 frames outside
  knockback, meaning a visible left-right or up-down twitch. Lower is better to watch.
- **drops picked**: enemy floor drops picked up / seen. The rest expired or
  were left behind on a room change.
- **hearts missed**: heart/fairy drops not picked up. Under the refill each one
  is healing the assist paid for instead.
- **hits / refills**: assist damage events / refill writes. Under strict
  `--engage-hearts 1`, each refill is a death the assist prevented. With
  `--observed-damage-guard`, the report also separates target and safety
  refills; safety writes protect against a larger hit observed earlier.
- **poked b/k/R**: bombs, keys and rupees granted by Survival top-ups.
- **slowest**: the longest single room visit, which is the first stall to look at.
- **damage**: hearts lost over the whole run (the ledger's per-room book; the
  worst rooms print with every run).
- **room items missed**: dungeon rooms whose item (`$00AB`) was never taken:
  its world-flag item bit (`$06FF`/`$077F` + room) was still clear when Link
  left. A missed key is one a Survival top-up pays for.

A deterministic emulator gives one outcome per config, so one run is the
measurement. Compare a row against the row above it only when the code changed.

| run | result (assist) | through | frames | flutter | drops picked | hearts missed | hits / refills | poked b/k/R | slowest | damage | room items missed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full_poweron12 (92e7a315, = poweron11 code) | ok (unlimited_health) | level9-credits | 292742 | 17942 | 67/176 | 32 | 441 / 667 | 82 / 6 / 9 | 9:03 12842f |
| **full_poweron27 (b0a328e9)** | **ok (unlimited_health)** | level9-credits | **277687** | **8178** | 66/163 | 42 | 371 / 596 | 67 / 6 / 0 | 7:0d 14024f |
| full_poweron26 (3ce0a056), stopped in L9 | level9_natural_silver_arrows (unlimited_health) | level9-credits | 270230 | 6529 | 68/177 | 47 | 324 / 533 | 55 / 5 / 0 | 9:4f 13993f |
| lastheart_poweron28 (01e2011f + raft fix) | level5_whistle_fight_64 (last_heart) | level9-credits | 125417 | 3873 | 57/101 | 8 | 100 / 12 | 50 / 3 / 0 | 4:20 4346f |
| blue_ring_full_poweron2 (438f56a9 + Dodongo turn) | level7_red_candle_pickup (unlimited_health) | level9-credits | 222646 | 5804 | 57/133 | 28 | 74 / 342 | 29 / 3 / 177 | 7:4a 35911f |
| blue_ring_full_poweron3 (+ L4 HC, 0x4A floor) | level6_clear_0x19 (unlimited_health) | level9-credits | 168184 | 5274 | 50/115 | 26 | 62 / 222 | 29 / 3 / 177 | 6:19 15310f |
| blue_ring_full_poweron4 (+ 0x19 ladder escape, reachable_only, BFS rewrite; damage includes a 16h boot artifact, since fixed) | enter_level6 (unlimited_health) | level9-credits | 142193 | 3311 | 44/105 | 23 | 42 / 168 | 29 / 3 / 177 | 5:65 4042f | 138.52 | 1:43=map 2:1e=bombs 2:3e=key 2:3f=bombs 3:5d=rupee5 3:6b=key 5:26=key 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 |
| blue_ring_full_poweron5 (+ OW 0x15 lattice start, 0x29 fixes) | level7_room59_up (unlimited_health) | level9-credits | 179712 | 4024 | 73/141 | 29 | 81 / 282 | 38 / 3 / 177 | 7:59 5150f | 199.45 | 1:43=map 2:0e=heart_container 2:1e=bombs 2:3e=key 3:5d=rupee5 5:26=key 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:19=map 6:28=rupee5 6:29=key 6:2d=key 7:58=rupee5 7:68=bombs 7:69=bombs |
| blue_ring_full_poweron7 (+ L7 0x59 door, arrow-budget rupee floor, L8 Gleeok HC walk, L2 Dodongo HC) | level8_return_passage_east_3e (unlimited_health) | level9-credits | 218026 | 6074 | 68/152 | 36 | 100 / 372 | 60 / 4 / 177 | 7:0d 8448f | 242.55 | 1:43=map 2:1e=bombs 2:3e=key 3:5d=rupee5 5:26=key 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:19=map 6:28=rupee5 6:2d=key 6:58=key 7:0c=bombs 7:1b=bombs 7:38=rupee5 7:58=rupee5 7:68=bombs 7:69=bombs 8:2e=map 8:4e=rupee5 8:6e=rupee5 |
| blue_ring_full_poweron8 (+ L8 0x3E door, L9 0x10/0x05 engine clears, 0x10 re-entry) | level9_natural_patra_join (unlimited_health) | level9-credits | 265684 | 6859 | 72/170 | 41 | 213 / 566 | 72 / 5 / 177 | 9:04 15296f | 430.53 | 1:43=map 2:1e=bombs 2:3e=key 3:5d=rupee5 5:26=key 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:19=map 6:28=rupee5 6:2d=key 6:58=key 7:0c=bombs 7:1b=bombs 7:38=rupee5 7:58=rupee5 7:68=bombs 7:69=bombs 8:2e=map 8:3f=bombs 8:4c=key 8:4e=rupee5 8:6e=rupee5 9:15=rupee5 9:16=bombs 9:61=key 9:62=rupee5 |
| **blue_ring_full_poweron9 (+ L9 0x04 north-aisle lattice leg)** | **ok (unlimited_health)** | level9-credits | **260248** | 7354 | 73/172 | 42 | 199 / 570 | 72 / 5 / 177 | 7:0d 8448f | 387.56 | 1:43=map 2:1e=bombs 2:3e=key 3:5d=rupee5 5:26=key 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:19=map 6:28=rupee5 6:2d=key 6:58=key 7:0c=bombs 7:1b=bombs 7:38=rupee5 7:58=rupee5 7:68=bombs 7:69=bombs 8:2e=map 8:3f=bombs 8:4c=key 8:4e=rupee5 8:6e=rupee5 9:15=rupee5 9:16=bombs 9:61=key 9:62=rupee5 |
| **blue_ring_full_poweron10 (+ Patra outside the orbit)** | **ok (unlimited_health)** | level9-credits | **257943** | 7094 | 76/171 | 39 | 166 / 549 | 72 / 5 / 177 | 7:0d 8448f | 356.33 | 1:43=map 2:1e=bombs 2:3e=key 3:5d=rupee5 5:26=key 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:19=map 6:28=rupee5 6:2d=key 6:58=key 7:0c=bombs 7:1b=bombs 7:38=rupee5 7:58=rupee5 7:68=bombs 7:69=bombs 8:2e=map 8:3f=bombs 8:4c=key 8:4e=rupee5 8:6e=rupee5 9:15=rupee5 9:16=bombs 9:61=key 9:62=rupee5 |
| **blue_ring_full_poweron14 (+ room-item sweep, lattice bomb approach, leave_wall latch)** | **ok (unlimited_health)** | level9-credits | 262777 | 6461 | 71/163 | 30 | 200 / 593 | 72 / 4 / 177 | 7:0d 12100f | 402.83 | 1:43=map 3:5d=rupee5 5:26=key 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:19=map 6:28=rupee5 6:29=key 6:2d=key 7:0c=bombs 7:1b=bombs 7:38=rupee5 7:58=rupee5 7:68=bombs 7:69=bombs 8:2e=map 8:3f=bombs 8:4c=key 8:4e=rupee5 8:6e=rupee5 9:15=rupee5 9:16=bombs 9:61=key 9:62=rupee5 |
| **blue_ring_full_poweron16 (d8b8235c: L9 join on the engine, Ganon windows, Gleeok turn-node stand, L6 0x09 south row, L7 0x0D ring lure)** | **ok (unlimited_health)** | level9-credits | **251516** | 5983 | 76/169 | 37 | 221 / 515 | 69 / 3 / 177 | 7:0d 6875f | 380.15 | 1:43=map 3:5d=rupee5 5:26=key 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:19=map 6:28=rupee5 6:2d=key 7:0c=bombs 7:1b=bombs 7:38=rupee5 7:58=rupee5 7:68=bombs 7:69=bombs 8:2e=map 8:3f=bombs 8:4c=key 8:4e=rupee5 8:6e=rupee5 9:15=rupee5 9:16=bombs 9:61=key 9:62=rupee5 |
| **blue_ring_full_poweron17 (f549259a: + Patra lane stand)** | **ok (unlimited_health)** | level9-credits | **249503** | 6167 | 74/168 | 37 | 173 / 473 | 74 / 3 / 177 | 7:0d 6875f | 337.66 | 1:43=map 3:5d=rupee5 5:26=key 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:19=map 6:28=rupee5 6:2d=key 7:0c=bombs 7:1b=bombs 7:38=rupee5 7:58=rupee5 7:68=bombs 7:69=bombs 8:2e=map 8:3f=bombs 8:4c=key 8:4e=rupee5 8:6e=rupee5 9:15=rupee5 9:16=bombs 9:62=rupee5 |

Blue Ring power-on 4-9 (2026-09-23, commit 82fd55ac and after): each run
stopped one stage later; each stall was fixed from its save point with
`scripts/stage_replay.py` and the next run started from power-on. Run 6 died
on an `ImportError` from an edit made while it was in flight (no row).
Run 9 is the first Blue Ring run to the credits: one session, zero state
loads, 14 containers (L2 and L4 hearts now taken), and 199 assist hits
against 371 in full_poweron27. Per-room damage and missed room items come
from the ledger books added the same day. Runs 11-13 stopped on stalls the
room-item sweep exposed (L2 0x1E bomb approach, L4 0x21 water stand, L6
0x39 leave_wall tug); run 14 reached the credits with 50 of 76 room items
taken (L2-L4 complete). Hearts per tape vary widely (356-403h across runs
10-14 on similar code): score combat on offsets, not on these rows.
Run 16 (2026-09-23): 11,261 fewer frames and 23 fewer hearts than run 14.
Run 15 had stopped in L6 0x09 (the stairs' south halt idled); the fix was
replayed from its save point, resumed to the credits, then run 16 started
from power-on. Run 16 predates the Patra lane stand (f549259a): 0x61 cost
it 39 hearts, which that change scores at 1.6h over 36 offsets.
Run 17 (the Patra lane stand) reached the credits at 249,503 frames and
337.7 hearts: 0x52 0h, 0x61 1h, Ganon 5h, L8 Gleeok 5.1h (run 14: 12, 22,
18, 26). The worst room is now L8 0x3E's six Blue Darknuts (23.5h).

## Stabilization loop after the lattice walkers (2026-09-23)

Each continuous run failed one stage later than the last; each stall was
fixed from its `R<n>_<stage>` save point and the next run started from
power-on. Frames and flutter count only up to the failure.

| run | failed stage | TF | frames | flutter | slowest visit | fix |
|---|---|---|---|---|---|---|
| full_poweron13 | north | 0x00 | 39136 | 716 | 1:74 3552f | L1 return-west on the lattice; dungeon door lanes |
| full_poweron14 | enter_6f_key | 0x01 | 62619 | 3033 | 2:6e 5806f | 0x6E key door lattice-only |
| full_poweron16 | level5_whistle_stairs_64 | 0x0F | 129348 | 5823 | 5:64 7131f | stairs_step first; wait for the wave; door retry |
| full_poweron17 | level7_room19_east_bomb | 0x3F | 195890 | 11521 | 0:52 7194f | bomb re-place (3x) |
| full_poweron18 | level5 west_did_not_enter_24 | 0x0F | 140771 | 5293 | 2:6e 5118f | L5 west hops = exit_door |
| full_poweron19 | level7_pond_approach | 0x3F | 209046 | 37106 | 0:55 29890f | raft 0x61 not a combatant |
| full_poweron20 | enter_level4 | 0x07 | 132867 | 1889 | 0:45 37835f | L4 mouth approach row south of the mouth |
| full_poweron21 | level5 west_did_not_enter_25 | 0x0F | - | - | 5:26 | moat: ladder_release; west hops retry then clear |
| full_poweron22 | level4_west_0x31 | 0x07 | - | - | 4:32 | L4 west door lattice; maze ladder release |
| full_poweron23 | level6_clear_0x29 | 0x1F | - | - | 6:29 | latched ladder crossing heading |
| full_poweron25 | level6_clear_0x29 | 0x1F | - | - | 6:29 | engine backs off a ladder that goes nowhere |
| full_poweron26 | level9_natural_silver_arrows | 0xFF | 270230 | 6529 | 9:4f 13993f | passes on later code from the same pose |

## Last-heart refill (`--engage-hearts 1`)

The refill writes only at the last heart, so `refills` counts deaths the
assist prevented. This is the number potions and better combat must drive to zero.

| run | through | result | frames | refills (deaths prevented) | hits | note |
|---|---|---|---|---|---|---|
| lasth_l1 (fed68421) | level1 | ok | 48874 | 2 | 11 | gather chain keeps its own last-heart refill |
| lastheart_poweron28 (01e2011f + raft fix) | level9-credits | L5 0x64 death | 125417 | 12 | 100 | Power-on, no state loads; L3 Raft and L4 TF clear. Blue Darknuts hit for two hearts, skipping the one-heart refill window. |

The L3 0x0F raft pickup initially timed out at `(176,149)`: the controller
treated a row eight pixels below the Raft as the pickup lane and pressed LEFT
into the corridor wall. Narrowing that lane to three pixels cleared the L3
Triforce from the saved predecessor state (`l3h_fix1`, 17,004 frames, one
disclosed state load), then cleared L3 on the continuous run above. The L5
death is the next boundary. At 0x64, the room starts with five blue Darknuts;
the existing fight policy took 28 hearts of damage there in full-refill run
27. The last-heart run entered with four hearts and died after two two-heart
contacts before killing one. The threshold stayed at one heart; no new
inventory or progression writes were added.

## Pre-l1 prefix (assist off)

| run | result | frames | flutter | note |
|---|---|---|---|---|
| baseline (92e7a315) | ok | 13133 | 3100 | 0x79/0x7A 2px bounce, ~2600 of the flutters |
| lattice walker (257dfeef) | ok | 6985 | 493 | pixel BFS obeys the ROM turn lattice |
