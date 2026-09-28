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
| **blue_ring_full_poweron24 (92f1031d: hidden rupee caves, potions; no rupee writes)** | **ok (unlimited_health)** | level9-credits | 283010 | 10127 | 82/184 | 36 | 156 / 484 | 75 / 4 / **0** | 9:52 11407f | 336.4 | 1:44=boomerang 3:5d=rupee5 3:6b=key 5:26=key 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:19=map 6:28=rupee5 6:2d=key 7:0c=bombs 7:1b=bombs 7:38=rupee5 7:58=rupee5 7:68=bombs 7:69=bombs 8:2e=map 8:3f=bombs 8:4c=key 8:4e=rupee5 8:6e=rupee5 9:15=rupee5 9:16=bombs 9:62=rupee5 |
| **blue_ring_full_poweron31 (b8cc4ab6: Patra eye aim)** | **ok (unlimited_health)** | level9-credits | **275135** | 8219 | 82/183 | 35 | 149 / 479 | 75 / 4 / **0** | 7:49 4302f | 329.41 | same 25 as run 24 |
| lastheart_poweron34 (Patra melee; guarded last heart) | level8_magic_key_stairs | level9-credits | 263596 | 8859 | 79/146 | 26 | 207 / 15 (7 target, 8 safety) | 66 / 3 / **0** | 8:1f 16150f | 272.03 | 19 rooms; see report |
| **lastheart_poweron37 (0x1F inner bombs, 0x75 ladder)** | **ok (guarded_last_heart)** | level9-credits | **293488** | 9472 | 92/170 | 30 | 315 / **23** (7 target, 16 safety) | 91 / 3 / **0** | 7:0d 5039f | 388.33 | 24 rooms; see report |
| **natl8_3 (rr-doua: no bomb/key writes)** | **ok (unlimited_health)** | level8 | **238187** | 6175 | 76/179 | 53 | 115 / 389 | 0 / 0 / 0 | 0:5f 3506f | 260.25 | 1:43=map 1:44=boomerang 3:5d=rupee5 5:26=key 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:19=map 6:28=rupee5 6:2d=key 7:0c=bombs 7:1b=bombs 7:38=rupee5 7:58=rupee5 7:68=bombs 7:69=bombs 8:2e=map 8:3f=bombs 8:4c=key 8:4e=rupee5 8:6e=rupee5 |
| **natural_credits_poweron39 (rr-ps7: zero pokes, natural buys)** | **ok (unlimited_health)** | level9-credits | **289154** | **7478** | 84/187 | 41 | 151 / 493 | **0 / 0 / 0** | 7:1a 5712f | 341.25 | 1:43=map 1:44=boomerang 3:5d=rupee5 5:26=key 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:19=map 6:28=rupee5 6:2d=key 7:09=rupee5 7:0c=bombs 7:1b=bombs 7:38=rupee5 7:58=rupee5 7:68=bombs 7:69=bombs 8:2e=map 8:3f=bombs 8:4c=key 8:4e=rupee5 8:6e=rupee5 9:15=rupee5 9:16=bombs 9:62=rupee5 |
| **natural_credits_poweron53 (L3/L8 rupees, 0x67 L9 shop detour)** | **ok (unlimited_health)** | level9-credits | **310175** | 11147 | 95/230 | 57 | 167 / 525 | **0 / 0 / 0** | 7:0d 5376f | 337.42 | 1:43=map 1:44=boomerang 5:26=key 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:19=map 6:28=rupee5 6:2d=key 7:09=rupee5 7:0c=bombs 7:1b=bombs 7:38=rupee5 7:58=rupee5 7:68=bombs 7:69=bombs 8:2e=map 8:3f=bombs 8:4c=key 8:4e=rupee5 9:15=rupee5 9:16=bombs 9:62=rupee5 |
| **clean_poweron64 (gathering G1 + melee, 2026-09-25)** | **raft_0x0f (off: Clean)** | level9-credits | 110963 | 4994 | 58/121 | 12 | 0 / 0 | **0 / 0 / 0** | 1:63 2885f | 30.33 | 1:43=map 1:44=boomerang 3:69=bombs |
| **clean_poweron69 (L3: pond, potion, flank, traps, Manhandla dodge)** | **level4_triforce_0x08 (off: Clean)** | level9-credits | 150867 | 8229 | 65/139 | 15 | 0 / 0 | **0 / 0 / 0** | 4:20 7149f | 40.87 | 1:43=map 1:44=boomerang 4:13=heart_container |
| **clean_poweron73 (L3 five-rupee scoop + L4 raft dismount)** | **level4_triforce_0x08 (off: Clean)** | level9-credits | **150867** | **8229** | 65/139 | 15 | 0 / 0 | **0 / 0 / 0** | 4:20 7149f | 40.87 | 1:43=map 1:44=boomerang 4:13=heart_container |
| **clean_poweron74 (L4 potion before arrows, Gleeok drinks)** | **arrow_restock_l4 (off: Clean), L4 TF 0x0F** | level9-credits | **155195** | 10027 | 60/130 | 15 | 0 / 0 | **0 / 0 / 0** | 4:20 4794f | 40.91 | 1:43=map 1:44=boomerang |
| **clean_poweron76 (L5 0x26 ladder step-off latch, Digdogger)** | **ok (off: Clean), L5 TF 0x1F** | level5 | **190444** | 11115 | 72/148 | 16 | 0 / 0 | **0 / 0 / 0** | 4:20 6938f | 58.5 | 1:43=map 1:44=boomerang 5:26=key 5:27=key 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 |
| **clean_poweron82 (coast hearts, Magical Sword, L6 door-table reroute)** | **level6_south_0x29 (off: Clean), Rod taken** | level6 | 235873 | 15052 | 103/196 | 24 | 0 / 0 | **0 / 0 / 0** | 4:20 7410f | 101.29 | 1:43=map 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:28=rupee5 |
| **clean_poweron83 (c903cb4b: 0x13 wallet gate, 0x29 straight walk, no 0x7a)** | **ok (off: Clean), L6 TF 0x3F** | level6 | **235596** | 14696 | 106/200 | 24 | 0 / 0 | **0 / 0 / 0** | 4:20 7410f | 99.06 | 1:43=map 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:28=rupee5 6:2d=key |
| **clean_poweron84 (0x28 bomb, door hop unstick, Goriya retreat)** | **ok (off: Clean), L7 TF 0x7F** | level7 | **264227** | 15462 | 122/225 | 30 | 0 / 0 | **0 / 0 / 0** | 4:20 7410f | 117.88 | 1:43=map 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:28=rupee5 6:2d=key 7:0c=bombs 7:1b=bombs 7:38=rupee5 7:58=rupee5 7:68=bombs 7:69=bombs |
| **clean_poweron98 (L7/L8 rupee detour, blue potion, return passage)** | **ok (off: Clean), L8 TF 0xFF** | level8 | **296423** | 16570 | 115/230 | 36 | 0 / 0 | **0 / 0 / 0** | 4:20 7410f | 136.04 | 1:43=map 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:28=rupee5 6:2d=key 7:0c=bombs 7:1b=bombs 7:58=rupee5 7:69=bombs 8:3f=bombs 8:4c=key 8:4e=rupee5 |
| clean_poweron_h1 (HEAD edf97dd6 alone) | level4_triforce_0x08 (off: Clean) | level9-credits | 157393 | – | – | – | 0 / 0 | 0 / 0 / 0 | – | – | L3 TF at 126,643f |
| natural_credits_poweron67 (edf97dd6: Dodongo stand, restock letter) | level8_return_passage_bomb_north_4c (unlimited_health) | level9-credits | 251358 | 9105 | 75/189 | 39 | 106 / 371 | 0 / 0 / 0 | 8:3e 8667f | 211.89 | see report |
| **survival_credits_reroute (L6 0x28 bomb, L8 0x4C bomb, natural credits)** | **ok (unlimited_health)** | level9-credits | **312446** | 14785 | 98/235 | 56 | 87 / 409 | **0 / 0 / 0** | 8:3e 9282f | 212.47 | 1:43=map 1:44=boomerang 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:28=rupee5 6:2d=key 7:0c=bombs 7:1b=bombs 7:38=rupee5 7:58=rupee5 7:68=bombs 7:69=bombs 8:2e=map 8:3f=bombs 8:4c=key 8:4e=rupee5 9:15=rupee5 9:16=bombs 9:62=rupee5 |
| **clean_poweron_c12 (L9 lanes A-F, PolicyGuard; 21,164 rollout restores, 0 other loads; MP4 = tape replay)** | **ok (off: Clean), credits TF 0xFF** | level9-credits | **334763** | 17477 | 127/251 | 37 | 0 / 0 | **0 / 0 / 0** | 4:20 7410f | 150.81 | 1:43=map 5:37=compass 5:47=key 5:56=bombs 5:57=rupee5 6:28=rupee5 6:2d=key 7:0c=bombs 7:1b=bombs 7:58=rupee5 7:69=bombs 8:3f=bombs 8:4c=key 8:4e=rupee5 9:15=rupee5 9:62=rupee5 |

`natural_credits_poweron53` is one continuous power-on session: no state
loads, deaths, or inventory/progression/capacity writes. The L3 0x5D and
L8 0x6E five-rupee room items were collected in play. At L9's post-L8
shop, Link bought one 20R pack, opened 0x67 for 30R with one bomb, returned
to 0x4A for the second pack, and reached credits with 5 bombs. The health
refill remained active (Survival); the real Clean frontier is still the
gathering White Sword walk.

natural_credits_poweron39 (2026-09-24, rr-ps7): first continuous power-on run
to credits with zero inventory pokes or writes of any kind: `ok=True`,
`set_state_count=0`, 289,154 frames, all 8 Triforce pieces naturally collected,
Ganon defeated and Zelda rescued. Wooden arrows bought naturally for 80R at 0x4A,
Bait bought naturally for 60R at 0x34, bombs bought naturally at 0x44 and 0x4A,
and Blue Gohma 0x1E arrow fire gated on vulnerability and alignment saving 19
rupees (4 connecting shots vs 23 blind shots) to fully fund both post-L8 bomb
packs. `inventory_assist=None`.


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
Run 24 (2026-09-24, 92f1031d) is the first run to the credits with no
rupee write: one session, zero state loads, the Blue Ring and candle paid
from hidden caves. It is 33,507 frames slower than run 17, and a third of
that is one room: final Patra 0x52 took 11,407 frames because its last eye
orbited below the room (rr-e59v). Bomb/key/arrow/Food writes remain
(ledger: keys@1:23, L2-L4 bombs, arrows@6:1c, food@0:42; rr-doua).
Run 31 (b8cc4ab6) is run 24 with the Patra eye aim (rr-e59v): every room
before 0x52 replays frame for frame, and 0x52 fell 11,407 -> 3,362 frames
(flutter there 2,077 -> 0), so the run is 275,135 frames.

natl8_3 (2026-09-24, rr-doua) is power-on to the L8 leave with no bomb or
key write: one session, zero state loads, 238,187 frames. Bombs come from
20R restocks bought only when short (0x4A 5->8 before L2, 0x44 1->5 on the
L4 walk and 0->4 on the L8 walk) and room drops; L1 takes 0x72's key for
0x43 E. The arrows@6:1c and food@0:42 writes remain. Getting there took
five resumed fixes the 16 poked bombs had hidden (L4 0x60 knockback over the
ladder, 0x12 lattice stand, L7 cellar re-climb, an off-stand wall placement,
0x4C failing while its last bomb burned).

Last-heart power-on 34 had no state loads and reached L8 0x1F, where the
Darknut clear timed out without a full-heart beam (`rr-secm`). Its last-heart
guard made 7 target and 8 safety refills through that failure. The new Patra
melee arm was not reached on this continuous run. From the saved L9
predecessor, the same no-beam 0x61 fight and final Patra cleared through
credits in 35,288 frames with one disclosed state load: 0x61 cost 3 hearts,
final Patra 2 hearts / 1,087 frames, and the L9 suffix used 1 target plus 5
safety refills. That suffix is development evidence, not a power-on result.

Last-heart power-on 37 is the first continuous run of this arm to credits:
zero state loads or deaths, TF `0xFF`, 14 containers, and no rupee write.
It remains Survival with 23 health refills (7 target, 16 safety), 91 bombs
and 3 keys granted in total, plus the L6 wooden arrows and L7 Food writes.
L8 0x1F now spends 0–5 held bombs on the two inner Darknuts after the
outer wave; run 37 used 3 and cleared the whole Magical Key stage in 3,805f
(run 34 timed out after 16,000f of the clear). A cellar 0x75 knockback
recovery at x>192 let the L9 join finish. Final Patra entered with 6.49
hearts, took 7, and needed a refill despite its no-beam melee clear.

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

## Clean (`--clean`: no refill, no pokes)

Before 2026-09-24 `--clean` was stripped from `sys.argv` on import, so every
earlier "Clean" spine tape here was Survival. These are the first real ones.

| run | result | frames | note |
|---|---|---|---|
| clean_poweron42 | death, `walk_pond` 0x4A | 24635 | Waypoint walk into octoroks/moblins; 7 half-heart hits from 3.5/4. |
| clean_poweron46 (defend layer, 0x34 Armos door) | death, `white` 0x17 | 40801 | Past the pond, 0x2C, the NE cluster, letter, candle and 0x28; reached `white` on 1/5 hearts. |

## Last-heart refill (`--engage-hearts 1`)

The refill writes only at the last heart, so `refills` counts deaths the
assist prevented. This is the number potions and better combat must drive to zero.

| run | through | result | frames | refills (deaths prevented) | hits | note |
|---|---|---|---|---|---|---|
| lasth_l1 (fed68421) | level1 | ok | 48874 | 2 | 11 | gather chain keeps its own last-heart refill |
| lastheart_poweron28 (01e2011f + raft fix) | level9-credits | L5 0x64 death | 125417 | 12 | 100 | Power-on, no state loads; L3 Raft and L4 TF clear. Blue Darknuts hit for two hearts, skipping the one-heart refill window. |
| lastheart_poweron29 (67d2973c, `--observed-damage-guard`) | power-on → L2 entry | 0x5C maze stall | 90778 | 0 | 7 | Hidden-rupee chain, red potion bought; L1 Triforce with no refill. |
| 29r (7602ea58) | L2 entry → L4 0x40 key | align stall | +48999 | 0 | 24 | 2 potion drinks (L2 0x3E, L3 raft 0x0F); the refill held 315 frames for them. |
| 29r2 (7820e227) | L4 0x40 → stepladder | pickup-pose stall | +7984 | 0 | 2 | |
| 29r3 (e7d25c8d) | L4 stepladder → L6 heart | full-hearts stop | +73389 | 5 + 6 safety | 110 | No potion left (restock at 0x64 landed after this pin); worst L5 0x05 17h, 0x64 12h. |
| 29r4 (bb546dd4) | L6 heart → L8 0x1F | Darknut clear timeout | +60630 | 4 | 65 | 0x1F's clear needs the full-heart beam. |
| lastheart_poweron34 | power-on → L8 0x1F | Darknut clear timeout | 263596 | 7 + 8 safety | 207 | No state loads; 0x1F spent 16,150f with 8.5h damage. Fresh predecessor pin: `LastHeart34_level8_magic_key_stairs`. |
| lastheart_poweron37 | power-on → credits | ok, Survival | 293488 | 7 + 16 safety | 315 | Zero state loads or deaths. L8 0x1F clears by bombs; both L9 Patras clear by melee. Refill count and remaining inventory writes are the Clean gap. |

Run 29 is power-on in five pieces: each stop was a stall, fixed from its
save point and resumed (a resume proves the pose only; the next continuous
run is the check). Every stall except L8 0x1F was a Survival assumption or
a hand walk: a pose-only stop, a full-hearts stop predicate, a greedy
align, or an unstick that idled forever. Totals to L8 0x1F: 15 refills
(9 at the last heart, 6 safety), 2 drinks.

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
