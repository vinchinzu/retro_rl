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
| Power-on → L1 TF → L2 entry (`rr-4d53.1` closed) | **1/1 Survival**; first-quest slot 1; `aquamentus_heart` 877f; TF `0x01`; enter L2 `0x7d` at (120, 205); 31828f; deaths 0; progression/capacity writes 0 | `survival_spine.json` / `.mp4` (not Clean M5) |
| Power-on → L2 Magical Boomerang (`rr-4d53.2.1` closed) | **1/1 Survival**; boom in `0x4f`; 44551f; L2 entry bombs=4 keys=0; final bombs=2 keys=1; deaths 0; poke_bombs=false; progression/capacity writes 0 | `survival_spine.json` / `survival_spine_l2_boom_v14.json` |
| L2 Boom → Dodongo → TF `0x02` | **1/1 Survival** on the continuous spine; 50529f; room `0x0d` mode 18; `tf=0x03`; deaths 0; progression/capacity writes 0; **documented bombs 2→16 + keys 1→2 + B-slot bombs** (owned counts only) | `survival_spine.json` / `.mp4` / `survival_spine_l2_tf_v10.json` |
| Power-on → L3 entrance `0x7c` (`rr-4d53.3.0` closed) | **1/1 Survival**; 53918f; `tf=0x03`; L2 entry bombs=0; L3 `0x7c` (120,205) bombs=8 keys=4; deaths 0; progression/capacity writes 0; **documented bombs/keys count top-up** (`poke_bombs=16`); dest `0x5b` not on this stop | `l3_entrance_bombtopup.json` / `_final.png` |
| Power-on → L3 west key `0x7b` (`rr-4d53.3.1.1` closed) | **1/1 Survival**; 54589f; room `0x7b` keys=5 (entry 4); bombs=8; `tf=0x03`; west_key 671f; deaths 0; progression/capacity writes 0; **documented bombs/keys count top-up**; dest `0x5b` not on this stop | `l3_west_key_spine.json` / `_final.png` |
| Power-on → L3 dest `0x5b` (`rr-4d53.3.1.2` closed) | **1/1 Survival**; 57256f; room `0x5b` mode 5; keys=5; bombs=8; `tf=0x03`; deaths 0; progression/capacity writes 0; **documented bombs/keys count top-up** | `l3_dest_0x5b_v12.json` / `_final.png` |
| Power-on → L3 Compass room `0x5a` (`rr-4d53.3.3.1` closed) | **1/1 Survival**; 57648f; room `0x5a` mode 5; keys=5; bombs=8; `tf=0x03`; deaths 0; progression/capacity writes 0; exact 392f west-door chunk | `l3_compass_0x5a_v1.json` / `_final.png` |
| Power-on → L3 west Darknuts `0x59` (`rr-4d53.3.3.2` closed) | **1/1 Survival**; room `0x59` mode 5; keys naturally 5→4; bombs=8; `tf=0x03`; deaths 0; progression/capacity writes 0 | `l3_west_darknuts_0x59_v1.json` / `_final.png` |
| Power-on → L3 south Darknuts `0x69` (`rr-4d53.3.3.3` closed) | **1/1 Survival**; room `0x69` mode 5; 5 Darknuts cleared in `0x59`; keys=4; bombs=8; deaths/progression/capacity writes 0 | `l3_south_darknuts_0x69_v2.json` / `_final.png` |
| Power-on → natural L3 Raft (`rr-4d53.3.3.4` closed) | **1/1 Survival**; room `0x0f` mode 9; Raft bit set; keys=4; bombs=8; `tf=0x03`; 8 Darknuts cleared in `0x69`; deaths/progression/capacity writes 0 | `l3_raft_spine_v2.json` / `_final.png` |
| Power-on → L3 Manhandla → TF `0x04` | **1/1 continuous Survival**; post-fanfare OW `0x74`, TF=`0x07`, Raft=1, keys=4; 92,948f; deaths 0; progression/capacity writes 0; `state_restores=0`; documented bomb-count top-up 8→16 at natural Raft (farm deferred) | `l3_tf_continuous_video_v1.json` / `.mp4` / `_final.png` |
| Power-on → L4 entry `0x71` | **1/1 continuous Survival**; live `0x74→0x73→0x63→0x64→0x65→0x55→0x45→0x71`; 95,281f; TF=`0x07`, Raft=1, keys=4, bombs=0; deaths 0; progression/capacity writes 0; no state load | `l4_entry_continuous_v1.json` / `_final.png` |
| Power-on → L4 natural key `0x51` → clear `0x50` | **1/1 continuous Survival**; `0x71→0x61`, Vires clear, documented bomb-count top-up 0→16, bomb north consumes one, natural key raises keys 4→5, then `0x50` Vires clear; TF=`0x07`, bombs=15; deaths 0; progression/capacity writes 0; no state load | `l4_clear50_continuous_v1.json` / `_final.png` |
| Power-on → L4 natural key `0x40` | **1/1 continuous Survival**; coordinate-gated `0x50→0x40`, Zols clear, natural key raises keys 5→6; 103,630f; TF=`0x07`, bombs=15; deaths 0; progression/capacity writes 0; no state load | `l4_room40_key_continuous_v7.json` / `_final.png` |
| Power-on → L4 enter `0x30` | **1/1 continuous Survival**; existing north controller from `(136,125)` free-UP into `0x30` in 227f; 103,857f; TF=`0x07`, keys=6, bombs=15; leftover `(120,205)`; deaths 0; progression/capacity writes 0; no state load | `l4_room30_continuous_v1.json` / `_final.png` |
| Power-on → L4 enter `0x31` | **1/1 continuous Survival**; `0x30` Vire clear from `(120,205)` (ignore `0x2b`) then KEY-RIGHT @y141; 104,524f; hop 667f; TF=`0x07`, keys 6→5, bombs=15; leftover `(16,141)`; deaths 0; progression/capacity writes 0; no state load | `l4_room31_continuous_v1.json` / `_final.png` |
| Power-on → L4 clear `0x31` | **1/1 continuous Survival**; west-alcove RIGHT+UP clip then coordinate waypoints to mid-maze, Vire clear 4,818f; 109,514f; TF=`0x07`, keys=5, bombs=15; leftover `(112,141)`; deaths 0; progression/capacity writes 0; no state load | `l4_clear31_continuous_v7.json` / `_final.png` |
| Power-on → L4 enter `0x32` | **1/1 continuous Survival**; UP to y=113, RIGHT+DOWN clip into the east column `(160,125)`, south-U waypoints into `0x32`; 109,890f; hop 376f; TF=`0x07`, keys=5, bombs=15; leftover `(16,141)`; deaths 0; progression/capacity writes 0; no state load | `l4_room32_continuous_v11.json` / `_final.png` |
| Power-on → L4 clear `0x32` | **1/1 continuous Survival**; Zol+LikeLike clear from `(16,141)` (ignore `0x2b`/`0x68`); 113,702f; hop 3,812f; leftover `(80,109)`; TF=`0x07`, keys=5, bombs=15; deaths 0; progression/capacity writes 0; no state load | `l4_clear32_continuous_v1.json` / `_final.png` |
| Power-on → L4 stepladder `0x60` | **1/1 continuous Survival**; east-dock waypoints, `ADDR_LADDER` at `(136,141)`; 118,292f; TF=`0x07`, keys=5, bombs=15; leftover `(136,141)` mode-9 `0x60`; deaths 0; progression/capacity writes 0; no state load | `l4_stepladder_continuous_v34.json` / `_final.png` |
| Power-on → L4 exit `0x60→0x32` | **1/1 continuous Survival**; reverse-dock waypoints after 150f item freeze, DOWN x=175/176 (v1 LEFT mid-dock solid); 118,806f hop 514f; TF=`0x07`, keys=5, bombs=15, ladder set; leftover `(192,189)` play `0x32`; deaths 0; progression/capacity writes 0; no state load | `l4_exit60_continuous_v2.json` / `_final.png` |
| Power-on → L4 west `0x32→0x31` | **1/1 continuous Survival**; south-U around pushed 0x68; 119,211f hop 405f; leftover `(208,141)` play `0x31`; TF=`0x07`, keys=5, bombs=15, ladder set; deaths 0; progression/capacity writes 0; no state load | `l4_west31_continuous_v1.json` / `_final.png` |
| Power-on → L4 KEY-UP `0x20` | **1/1 continuous Survival**; reverse 0x31 maze east-U + clip + inland west, KEY-UP @x120; 120,079f hop 868f; leftover `(120,205)` play `0x20`; keys 5→4; TF=`0x07`, bombs=15, ladder set; deaths 0; progression/capacity writes 0; no state load | `l4_keyup20_continuous_v1.json` / `_final.png` |
| Power-on → L4 enter `0x21` | **1/1 continuous Survival**; 0x20 Vire clear then north-around + RIGHT+DOWN clip + y=141 RIGHT; 121,775f hop 447f; leftover `(16,141)` play `0x21`; keys=4 bombs=15 TF=`0x07` ladder set; deaths 0; progression/capacity writes 0; no state load | `l4_room21_continuous_v22.json` / `_final.png` |
| Power-on → L4 map `0x21` | **2/2 continuous Survival**; spawn RIGHT+UP to `(48,93)` then RIGHT+DOWN clip; `ADDR_MAP|0x08` at `(208,181)` in 297f; 122,072f; map=`0x0A`; keys=4 bombs=15 TF=`0x07` ladder set; deaths 0; progression/capacity writes 0; no state load | `l4_map_continuous_v15.json` / `_final.png` |
| Power-on → L4 bomb-UP `0x11` | **2/2 continuous Survival**; east-column UP to y=93 then LEFT (v1 y=109 LEFT is a 16px pillar); bomb-UP stand `(120,105)`; 122,507f hop 435f; leftover `(120,189)` play `0x11`; map=`0x0A`; keys=4 bombs 16→15 TF=`0x07` ladder set; deaths 0; progression/capacity writes 0; no state load | `l4_bomb11_continuous_v2.json` / `_final.png` |
| Power-on → L4 0x01 natural key | **2/2 continuous Survival**; v1 hold-UP leftover `(120,93)` is north wall; bomb-UP `(120,105)` 377f then pickup `(120,141)` 819f; 123,703f hop 1196f; leftover `(120,133)` play `0x01`; keys 4→5 bombs 15→14 map=`0x0A` TF=`0x07` ladder set; deaths 0; progression/capacity writes 0; no state load | `l4_key01_continuous_v3.json` / `_final.png` |
| Power-on → L4 clear `0x12` | **2/2 continuous Survival**; DOWN 0x01→0x11 then bomb-RIGHT `(192,141)`; Vire clear ignore `0x68`; 124,993f hop 1290f; leftover `(128,117)` play `0x12`; keys=5 bombs 14→13 TF=`0x07`; deaths 0; progression/capacity writes 0; no state load | `l4_clear12_continuous_v1.json` / `_final.png` |
| Power-on → L4 enter Gleeok `0x13` | **2/2 continuous Survival**; x-first to push stand `(112,144)` (v1 y-first leftover `(128,141)` DOWN solid); hold4 `PATH_12_TO_GLEEOK`; 125,407f hop 414f; leftover `(32,141)` play `0x13`; keys=5 bombs=13 TF=`0x07`; deaths 0; progression/capacity writes 0; no state load | `l4_gleeok13_continuous_v2.json` / `_final.png` |
| Power-on → L4 TF `0x08` | **2/2 continuous Survival**; south-stand Gleeok 3564f; TF `0x07→0x0F`; mode 18 room `0x03` `(120,149)`; 128,971f; keys=5 bombs=13; HC not mid-room; deaths 0; progression/capacity writes 0; no state load | `l4_tf_continuous_v1.json` / `_final.png` |
| Power-on → L5 entry `0x76` | **1/1 continuous Survival**; L4 fanfare settle 284f onto island `0x45`, `POST_L4_TO_LEVEL5_HOPS` (not old At4A); 134,393f hop 5,138f; leftover `(120,205)`; TF=`0x0F` keys=5 bombs=13; deaths 0; progression/capacity writes 0; no state load | `l5_entry_continuous_v1.json` / `_final.png` |
| Power-on → L5 clear `0x66` | **1/1 continuous Survival**; occupancy miss-block (v1 cardinal timeout `(119,173)` 2/3); 138,634f hop 4,241f; leftover `(32,101)` keys 5→6; deaths 0; progression/capacity writes 0; no state load | `l5_clear66_continuous_v2.json` / `_final.png` |
| Power-on → L5 east key `0x77` | **1/1 continuous Survival**; north-bank to ladder x=56 then DOWN; Pols Voice clear leftover `(136,165)` keys 7; 142,958f; deaths 0; progression/capacity writes 0; no state load | `l5_east77_continuous_v1.json` / `_final.png` |
| Power-on → L5 Recorder `0x04` | **1/1 continuous Survival**; East Key → 0x66 bomb-west → 0x04; 160,648f hop 17,690f; mode 9 `(135,141)`; keys 7→6 bombs 13→8; whistle earned; deaths 0; progression/capacity writes 0; no state load | `l5_whistle_continuous_v1.json` |
| Power-on → L5 TF `0x10` | **1/1 continuous Survival**; cellar `0x04` → Digdogger `0x24` → room `0x14` mode 18 `(120,149)`; 173,961f hop 13,311f; TF=`0x0F→0x1F`; keys 6→5 bombs=8; deaths 0; progression/capacity writes 0; no state load | `l5_tf_continuous_v1.json` / `_final.png` |
| Power-on → L6 entry `0x79` (`rr-g3c1`) | **1/1 continuous Survival**; L5 fanfare settle 510f onto `0x0B`, 0x1B y=141 LEFT after south-around x≈72, west chain y=141, 0x14/0x23 SE blue paths; 179,355f hop 4,884f; leftover `(120,205)`; TF=`0x1F` keys=5 bombs=8; whistle earned; deaths 0; progression/capacity writes 0; no state load | `l6_entry_continuous_v2.json` / `_final.png` |
| Power-on → L6 east key `0x7a` | **1/1 continuous Survival**; wall-first RIGHT then 0x7a clear; 181,199f hop 1,844f; leftover `(120,141)`; keys 5→6; deaths 0; progression/capacity writes 0; no state load | `l6_east_key_continuous_v1.json` / `_final.png` |
| Power-on → L6 west `0x78` | **1/1 continuous Survival**; free LEFT 0x79, key-LEFT 0x78, wizzrobe clear; 182,415f hop 1,216f; leftover `(144,141)`; keys 6→5; deaths 0; progression/capacity writes 0; no state load | `l6_west_continuous_v1.json` / `_final.png` |
| Power-on → L6 compass room `0x68` | **1/1 continuous Survival**; occupancy UP from `(144,141)` (8 miss-blocks at x=144) into play `0x68`; 182,636f hop 221f; leftover `(120,205)`; keys=5 bombs=8 TF=`0x1F`; 5× Zol live; deaths 0; progression/capacity writes 0; no state load | `l6_compass_continuous_v1.json` / `_final.png` |
| Power-on → L6 0x68 compass | **1/1 continuous Survival**; occupancy-patrol Zol/gel clear then `ADDR_COMPASS|0x20`; 187,575f hop 4,939f; leftover `(120,149)`; keys=5 bombs=8 TF=`0x1F`; deaths 0; progression/capacity writes 0; no state load | `l6_clear68_continuous_v1.json` / `_final.png` |
| Power-on → L6 Keese `0x58` | **1/1 continuous Survival**; occupancy UP from `(120,149)` into play `0x58`; 187,784f hop 209f; leftover `(120,205)`; 8× Keese live; north door sealed; deaths 0; progression/capacity writes 0; no state load | `l6_keese_continuous_v1.json` / `_final.png` |
| Power-on → L6 clear `0x58` | **1/1 continuous Survival**; occupancy-patrol 8× Keese TYPE-only; 188,666f hop 882f; leftover `(112,167)`; keys=5 (no key pickup); north still sealed; deaths 0; progression/capacity writes 0; no state load | `l6_clear58_continuous_v1.json` / `_final.png` |
| Power-on → L6 enter `0x48` | **1/1 continuous Survival**; occupancy long-UP from `(112,167)` is **free** (keys=5); 189,007f hop 341f; leftover `(120,205)` play `0x48`; 4 blade traps live; deaths 0; progression/capacity writes 0; no state load | `l6_room48_continuous_v1.json` / `_final.png` |
| Power-on → L6 enter `0x38` | **1/1 continuous Survival**; occupancy run-UP through 0x48 traps (no clear); 189,268f hop 261f; leftover `(120,189)` play `0x38`; Like-Like + wizzrobe live; deaths 0; progression/capacity writes 0; no state load | `l6_room38_continuous_v1.json` / `_final.png` |
| Power-on → L6 clear `0x38` | **1/1 continuous Survival**; occupancy-patrol 7× wizzrobe/Like-Like; 194,755f hop 5,487f; leftover `(32,149)` play `0x38`; Bubble residual; blocks unpushed; keys=5 bombs=8 TF=`0x1F`; deaths 0; progression/capacity writes 0; no state load | `l6_clear38_continuous_v1.json` / `_final.png` |
| Power-on → L6 enter `0x28` | **1/1 continuous Survival**; live left 0x68 UP (slot11 `96,144→136`) then west-aisle north; 197,962f hop 3,207f; leftover `(120,189)` play `0x28`; keys=5 bombs=8 TF=`0x1F`; deaths 0; progression/capacity writes 0; no state load | `l6_room28_continuous_v6.json` / `_final.png` |
| Power-on → L6 clear `0x28` | **1/1 continuous Survival**; occupancy-patrol 2× orange wizzrobe `0x24`; 200,549f hop 2,587f; leftover `(120,181)` play `0x28`; keys=5 bombs=8 TF=`0x1F`; deaths 0; progression/capacity writes 0; no state load | `l6_clear28_continuous_v1.json` / `_final.png` |
| Power-on → L6 enter `0x18` | **1/1 continuous Survival**; LEFT+UP at y=181, hold UP, RIGHT+UP at y=109; 200,829f hop 280f; leftover `(120,189)` play `0x18`; diamond floor, north stairs, east shutter; Gleeok not on leftover PNG (spawn/RAM residual); keys=5 bombs=8 TF=`0x1F`; deaths 0; progression/capacity writes 0; no state load | `l6_room18_continuous_v7.json` / `_final.png` |
| Power-on → L6 settle `0x18` | **1/1 continuous Survival**; IDLE 512f; leftover `(120,189)` play `0x18`; spawn type **`0x44`** (never `0x43`/`0x46`) + fireball `0x56`; room_item_id `0x03`; doors 0/0; TF=`0x1F` keys=5 bombs=8; deaths 0; progression/capacity writes 0; no state load | `l6_settle18_continuous_v1.json` / `_final.png` |
| Power-on → L6 Gleeok `0x18` | **1/1 continuous Survival**; LEFT+UP y=189 then L4 south-stand on `0x44`; 204,189f hop 2,848f; leftover `(121,133)` body-gone; `0x46` mid-fight; east shutter still closed; north stairs live; map=`0x0A` (no L6 map); keys=5 bombs=8 TF=`0x1F`; deaths 0; progression/capacity writes 0; no state load | `l6_gleeok18_continuous_v1.json` / `_final.png` |
| Power-on → L6 post-Gleeok census `0x18` | **1/1 continuous Survival**; IDLE 192f; leftover `(156,133)`; no `0x44`/`0x46`; `0x56` then gone; `cur_opened_doors` 0→5 `mask` 0; 204,381f hop 192f; keys=5 bombs=8 TF=`0x1F` map=`0x0A`; deaths 0; progression/capacity writes 0; no state load | `l6_postgleeok18_continuous_v2.json` / `_final.png` |
| Power-on → L6 enter `0x19` | **1/1 continuous Survival**; occupancy y=141 RIGHT from 0x18 (PNG-black shutter + mask 0 walkable); 204,632f hop 251f; leftover `(16,141)` play `0x19`; keys=5 bombs=8 TF=`0x1F` map=`0x0A`; deaths 0; progression/capacity writes 0; no state load | `l6_room19_continuous_v1.json` / `_final.png` |
| Power-on → L6 clear `0x19` | **1/1 continuous Survival**; idle census 160f then occupancy-patrol 2× Zol `0x13` + 2× Like-Like `0x17`; 208,845f hop 4,213f; leftover `(176,158)`; RoomItemId `0x17` Map on floor; map still `0x0A`; keys=5 bombs=8 TF=`0x1F`; deaths 0; progression/capacity writes 0; no state load | `l6_clear19_continuous_v1.json` / `_final.png` |
| Power-on → L6 enter `0x09` | **1/1 continuous Survival**; skip-Map KEY-UP axis LEFT x=136 then occupancy north; 209,120f hop 275f; leftover `(120,205)` play `0x09`; keys 5→4 bombs=8 TF=`0x1F` map=`0x0A`; deaths 0; progression/capacity writes 0; no state load | `l6_room09_continuous_v2.json` / `_final.png` |
| Power-on → L6 clear `0x09` | **1/1 continuous Survival**; idle census 160f then occupancy-patrol 3× blue `0x23` + 2× orange `0x24`; 210,699f hop 1,419f; leftover `(112,173)`; left 0x68 `(96,144)` unpushed; keys=4 bombs=8 TF=`0x1F` map=`0x0A`; deaths 0; progression/capacity writes 0; no state load | `l6_clear09_continuous_v1.json` / `_final.png` |
| L4 complete → L5 entry | 1/1 to room `0x76` in 5,031 path frames; bombs=7, Raft=1, Stepladder=1, Triforce=`0x0c` preserved | `l4_to_l5_assisted_v4.json`; `Level5EntranceFromL4` |
| L5 entry → room `0x66` key | 1/1 assisted clear in 1,254 frames; three Gibdos dead, keys 0→1 | `l5_clear66_entrance_assisted.json`; `Level5Cleared66` |
| East Key Pols Voice `0x77` → natural Recorder → Whistle basement `0x04` | **1/1 Survival** from `Level5EastKey`; whistle `$065C` 0→1; room `0x04` mode 9; deaths 0; progression/capacity writes 0; `route_eligible=false` | `l5_e2w_t2.json`; `Level5WhistleFrom77` |
| Whistle basement `0x04` → Digdogger `0x24` → L5 Triforce room `0x14` | **1/1 continuous Survival session** (not a seamed tape); 10,776 route frames; Triforce `0x0c→0x1c`; zero deaths and zero resource/progression/capacity pokes; 43 damage units logged for later hardening | `Level5Complete` (development-only; attach only via the continuous spine) |

These runs used the Survival health refill and reported zero progression writes
and zero capacity writes. They are development checkpoints, not Clean or
power-on STATUS promotions. Power-on → L3 west key `0x7b` is on the
continuous tape (`rr-4d53.3.1.1` closed) with documented bomb/key count pokes —
not Clean. The continuous spine now holds the natural `0x40` key and has
cleared `0x32`, collected `ADDR_LADDER` on `0x60`, exited to play `0x32`,
and walked west into `0x31` then KEY-UP `0x20` `(120,205)` keys 5→4. 0x20
Vire clear and `0x20→0x21` are on the continuous tape (`l4_room21_continuous_v22`,
leftover `(16,141)` play `0x21`). Map pickup is on the tape
(`l4_map_continuous_v15` 2/2, `(208,181)`, map=`0x0A`). Bomb-UP `0x11`
is on the tape (`l4_bomb11_continuous_v2` 2/2, leftover `(120,189)`).
0x01 natural key is on the tape (`l4_key01_continuous_v3` 2/2, leftover
`(120,133)`, keys 4→5). 0x12 Vire clear is on the tape
(`l4_clear12_continuous_v1` 2/2, leftover `(128,117)`). Gleeok enter and
L4 TF `0x08` are on the tape (`l4_tf_continuous_v1` 2/2, TF=`0x0F`).
`.6` closed. HC was not mid-room. Power-on spine now holds L6 west 0x78
(`l6_west_continuous_v1` 1/1, room `0x78` `(144,141)`). `.7` / `.4` / `rr-g3c1`
closed. 0x38 clear is on the tape (`l6_clear38_continuous_v1` 1/1,
leftover `(32,149)`). 0x28 enter is on the tape
(`l6_room28_continuous_v6` 1/1, leftover `(120,189)`). 0x28 wizzrobe
clear is on the tape (`l6_clear28_continuous_v1` 1/1, leftover
`(120,181)`, max_live=2). Gleeok enter is on the tape
(`l6_room18_continuous_v7` 1/1, leftover `(120,189)` play `0x18`).
Gleeok fight / Rod / Gohma / TF `0x20` residual.
Isolated poke-16 tapes remain recon only.
Isolated `Level3*` pins cannot close spine beads. The East Key
→ Recorder seam (`rr-4d53.5`) is attached on the continuous tape through
L5 TF `0x10` (`rr-4d53.7` / `rr-4d53.4` closed). The active backward pass
is documented below and in `docs/plan.md`.

## Backward endgame recon (fixture-only; does not change either gate)

| Segment | Result | Evidence |
|---------|--------|----------|
| Fully loaded final-Patra room `0x52` → Ganon `0x42` → Zelda `0x32` → credits/final page | **1/1**; Ganon type `0x3E`, brown ObjState nonzero, Silver Arrow kill sets `$0672=1`; rolling credits at frame 3,395, final page at 4,595 | `l9_ganon_credits_recon.json`; `Level9BeforeGanonReconFixture`; `Level9CreditsReconFixture`; `Level9FinalScreenReconFixture` |
| Live final Patra `0x52` → naturally earned north door → same Ganon/Zelda/ending suffix | **2/2 exact**; body `0x47` HP `0xB0`, 8 eyes `0x25` HP `0x60`; Patra clear 1,883f, credits 5,342f, final page 6,542f; inventory preserved through Patra, runtime controller writes 0 | `l9_patra_credits_recon.json`; `Level9FinalPatraReconFixture`; `Level9FinalPatraClearedReconFixture`; `Level9PatraFinalScreenReconFixture` |
| Candidate `0x62` as cardinal predecessor of `0x52` | **RETARGET**; 8 Keese `0x1B`; doors 0; ROM north wall; no live `0x62`→`0x52` | `l9_room62_patra_credits_recon_probe.json`; `Level9Room62ReconFixture` |
| Play `0x40` key-north → `0x30` → cellar `0x67` right → `0x04` → `0x03` → Patra → credits | **1/1** recon; credits 16015 / final 17215 / total 17305; zero forbidden writes; `route_eligible=false` | `l9_room40_dump.json`; `l9_play40_keynorth_patra_credits_recon.json`; `Level9Room40KeyNorthReconFixture` |
| Play `0x21` south → `0x31` | **DEST NO**; south shutter sealed at (120,189); still 0x21. 0x11 south → 0x21 is live. `route_eligible=false` | `l9_room21_dump.json`; `l9_probe11_south_21.json` |
| Blade-trap/Like-Like room `0x41` clear + north → east-bomb `0x31` → block-stairs `0x30` → Patra stairs `0x03` → credits | **1/1** recon; north is blocked by live Like-Likes and opens after controller clear; credits 25858 / final 27058 / total 27148; zero runtime controller writes; `route_eligible=false` | `l9_room41_dump.json`; `l9_play41_north_patra_credits_recon.json`; `Level9Room41NorthReconFixture` |

These backwards-development proofs compose the full inventory and room-loader
setup before their start state, so they remain `route_eligible=false`. The new
Patra run does **not** inherit the earlier object-removal or north-door writes:
after reset it uses controller input only, naturally raises door bit `0x08`,
and continues without another state load. Survival restores four heart units
per trial (two in `0x52`, two in `0x42`) with zero deaths and zero progression/
capacity writes. This proves the final-Patra/Ganon/Zelda/ending suffix, **not**
a Survival-assisted or Clean Level 9 route.

`rr-sz8.3` materialized uncleared `0x62` (8 Keese `0x1B`, doors 0) via loader
`0x72`+UP. Live north push after kill-clear/bomb stays in `0x62`. ROM L7–9
door bytes: `0x62` north = wall, `0x52` south = wall. **Retarget** to the
stairs-drop into `0x52`. See [LEVEL9_ROUTE.md](LEVEL9_ROUTE.md).

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

- Survival spine residual: L1 `0x45` Wallmaster key (`rr-4d53.1`); L2/L3
  continuous Survival tapes (`rr-4d53.2` / `rr-4d53.3`); compose power-on → L5 TF (`rr-4d53.4`)
- Attach East Key → Recorder pin to the proven `0x04`→TF suffix (not one session yet)
- L6–L8 under assist (explicitly out of this pass)
- Clean residual after full-game assist pass
- Natural-entry continuous power-on chain (deferred under assist-first)
- Broader overworld bomb / white-sword inventory buys (NamedRoutes exist; shop residual)
- Continuous multi-dungeon dry run (M6–M8)

## Dual track (2026-08-06)

| Track | Tooling | STATUS role |
|-------|---------|-------------|
| Clean | default runners | only path that promotes verified Clean gates |
| Survival-assisted | `--infinite-life` / `assist.UnlimitedHealthAssist` | first-pass geometry; contract `ASSIST_CONTRACT.md` |

Infinite life is implemented and CLI-wired; **no assisted end-to-end segment
is STATUS-promoted yet**. Work: `bd ready -l zelda_i`.

## Next

1. **Active Survival tip:** `--through level6-clear09` is green (`0x09`
   `(112,173)`, TF `0x1F`, keys=4, v1 1/1). Left 0x68 **pushes**
   `(96,144)→(96,136)` then gone. `--through level6-stairs09` red
   (v1 occupancy boxed leftover; v2 vacated idle tile 119; v3 x=96 UP
   solid y=133 tile 179; v4 idle `(96,137)` tile 118). Next south-around
   remaining block then NE hole idle (`l6_stairs09_continuous_v5`).
   Do not grant Map/Rod. `rr-tne2` in progress. Isolated BFS is still
   not a spine path. See `docs/plan.md`. Isolated `Level3*` pins do
   not close spine beads. No seamed compose. Bomb/key count pokes are
   documented Survival shortcuts, not Clean.
2. **Parked:** L9 dest walk (`rr-yxy6`) and hygiene (`rr-ekwl`).
3. Clean residual only after a continuous assist pass (`rr-4oz`).
