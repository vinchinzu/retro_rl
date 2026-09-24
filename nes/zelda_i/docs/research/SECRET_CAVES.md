# First-quest secret rupee caves and potion shops

Measured 2026-09-23/24. The ROM and live RAM take priority over any wiki.
Grid: column A-P = `screen & 0xF`, row 1-8 = `(screen >> 4) + 1`. Start is H8 `0x77`.

## Where the evidence comes from

- **rom**: a static decode of the room layouts. It is a Python port of `LayoutRoomOW` / `CheckTileObject` (aldonunez `Z_05.asm`) and runs over the iNES file:
  - room layout pointer at PRG `0x15F9C`
  - ColumnDirectoryOW at PRG `0x19D0F`
  - PrimarySquaresOW at PRG `0x1697C`
  - unique room id = AttrsD (PRG `0x18580`) `& 0x7F`
  - the cave table (raw file `0x18610` items / `0x1864C` prices)
  - `SecretArmosRoomIds` (`Z_04`)

  A tile object is primary square `$E5..$EA` → type `$62 + i` at `(col*16, row*16+0x40)`.
- **ram**: the slot-11 object read live. Checked two ways:
  - Every play-mode overworld save state in the integration dir: 1357 states on 74 screens. The ROM decode matched 1180 of them. The other 177 are all explained: 174 had `$067F+screen` bit `$80` set (secret already revealed, so no object), and 3 had the raft dock (`0x61`) in slot 11 on 0x55. None are unexplained.
  - All 128 screens loaded by `scratch/secret_teleport_scan.py`. Every one matches.
- **live**: opened, entered and paid out this session with `stage_replay.py` and the probes in `scratch/secret_walks.py`. Bombs, the candle and teleports are what-if RAM writes: this is measurement, not a route result.
- **parent**: stands measured earlier by the gather lane (`SECRET_RUPEE_CAVES`).
- **wiki**: the RPGClassics Q1 overworld map ([tartarus.rpgclassics.com](https://tartarus.rpgclassics.com/zelda1/1stquest/overworldhighyes.shtml)). It gives location and type only. Zelda Dungeon, StrategyWiki, Zeldapedia, GameFAQs and cheatcc all returned 403/402 to the fetcher.

## Mechanics (ROM, cited routine)

- **Tile objects.** A burnable tree is `0x64` and a bombable rock wall is `0x63`. Both sit in slot 11 from the frame the screen loads. A revealed room (`$80`) lays out stairs or a doorway instead, so slot 11 is empty. `0x62` is a rock that needs the Power Bracelet (`UpdateRockOrGravestone` checks `InvBracelet`) and pushes only vertically, with Link's x equal to the rock's x. `0x65` is a gravestone and needs no item.
- **Hit test** (`CheckTileObjWeaponCollision`). The weapon centre and the object centre must both be within 16 px per axis: `|wx - X| < 16` and `|wy - Y| < 16`, with both centres at +8.
- **Bomb** (`WieldBomb` → `PlaceWeapon`). The bomb lands 16 px ahead of Link. It must be *detonating* (state `$13`) while it overlaps. For a top-row rock at `(X, 80)`, Link at `x ∈ {X-8, X, X+8}`, `y ∈ 85..109`, facing UP works. Measured: `(X, 85)` UP on 0x13, 0x67, 0x71, 0x0D, 0x27, 0x33 and 0x2D.
- **Candle** (`WieldCandle`, `UpdateFire`):
  - The flame spawns 16 px ahead and walks 16 px more, so it stands 32 px ahead of Link.
  - It reveals once its stand timer is below 2.
  - It needs a free 16 px path: on 0x4B, `(144,93)` RIGHT never opened, because a rock block covers x 152..191.
  - Blue candle: one flame per screen.
  - Measured DOWN stands:
    - `tree_y-19` opened 0x51 and 0x78.
    - `tree_y-27` is the parent's measured stand on 0x28, 0x56, 0x5B and 0x6B (it found `tree_y-19` too late on 0x28).
- **Armos secrets** (`InitArmosOrFlyingGhini`, table `SecretArmosRoomIds` / `SecretArmosXs`, `y == $80`). The rooms are 0x24, 0x0B, 0x1C, 0x22, 0x34, 0x3D and 0x4E, with armos x = E0, B0, B0, 30, 40, 90, A0.
  - Touching that armos wakes it, and stairs replace it at `(x, 128)`.
  - This reveal does **not** set `$80`. Only `$10` (taken) is set after the payout.
- **Payouts.** Cave `0x21` pays 30R, `0x22` 100R, `0x23` 10R. The keeper is cave type + `0x5A` (`0x7B`/`0x7C`/`0x7D`).
- **Potion shop.** Cave `0x1A`, keeper `0x74`. Wares are blue potion `0x1F` for 40 and red potion `0xE0` for 68, measured live in all seven shops. The shop only draws wares once `InvLetter == 2` (`UpdateCavePerson`, `Z_01`). Before that, Link must select the letter from `0x0E` and press B inside a potion shop.
- **Door repair.** Cave `0x17`, keeper `0x71`. It posts a 20R charge on entry (`Z_01`: "Post 20 rupees to subtract"). A reveal is permanent (`$80`), so any later walk onto that doorway or stairs pays.
- **Room `0x1F` false wall** (`GetCollidableTile`). Link at `x == $80` with `y < $56`, moving vertically, reads walkable. This is the only way into P1 `0x0F`: UP at x = 128 exactly.
- **Mazes** (`CheckMazes`). From `0x61` (Lost Woods) only RIGHT exits freely. From `0x1B` (Lost Hills) only LEFT does.

## Secret rupee caves (all 14 Q1 moblin caves)

"Connects" lists the neighbours whose shared edge joins the stand's walkable region, from the ROM lattice (`ow_walkable_nodes` on the live `$6530`). The span given is the edge x or y range. "From 0x77" is the shortest on-foot screen chain with no items.

| Screen | Grid | Cave | Pay | Open | Secret object | Stand + facing | Connects | From 0x77 | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| 0x0F | P1 | 0x22 | 100R | **open** doorway (centre arch) | doorway (128,128) | (128,141) UP | only 0x1F, UP at x=128 (false wall) | 67 68 58 48 38 28 29 2A 2B 2C 2D 1D 1E 1F 0F (~2100f walk) | rom, parent live, wiki |
| 0x13 | D2 | 0x21 | 30R | bomb | 0x63 (32,80) | (32,85) UP | 0x12 W y85-189, 0x14 E y85-189, 0x03 N x160 | not on foot (west region, see below) | rom, ram, live +30 (idle 20 and 45; idle 0 lost the bomb to a Lynel hit) |
| 0x28 | I3 | 0x21 | 30R | burn | 0x64 (208,160) | (208,133) DOWN | 0x38 S x112-128, 0x29 E, 0x27 W y85-141 | 67 68 58 48 38 28 | rom, ram, parent live, wiki |
| 0x2D | N3 | 0x21 | 30R | bomb | 0x63 (80,80) | (80,85) UP | 0x3D S x112-128, 0x1D N, 0x2C W / 0x2E E y133-189 | 67 68 58 48 38 28 29 2A 2B 2C 2D | rom, ram, parent live, wiki |
| 0x3D | N4 | 0x21 | 30R | **armos**, the right one | armos (144,128), no slot-11 object | (128,125) RIGHT, then onto the stairs at (144,128) | 0x4D S x16-224, 0x2D N x112-128 | 67 68 58 59 5A 5B 5C 5D 4D 3D (~1500f) | rom, live +30 (BFS_3D, 536f), wiki ("armos to the right") |
| 0x48 | I5 | 0x21 | 30R | burn | 0x64 (208,96) | (188,93) RIGHT | 0x38 N, 0x58 S x112-128, 0x47 W y133-157 | 67 68 58 48 | rom, ram, parent live, wiki |
| 0x4E | O5 | 0x23 | 10R | **armos**, the right one | armos (160,128) | (144,125) RIGHT | only 0x4D W y117-157 | 67 68 58 59 5A 5B 5C 5D 4D 4E | rom, live +10, wiki |
| 0x51 | B6 | 0x23 | 10R | burn | 0x64 (144,160) | (144,141) DOWN (measured); (144,133) is the tree_y-27 alternative | 0x52 E y117-157; 0x61 S is into the woods only | 76 66 65 64 63 53 52 51 | rom, ram, live +10 |
| 0x56 | G6 | 0x23 | 10R | burn | 0x64 (160,160) | (160,133) DOWN | 0x57 E, 0x46 N x112-128, 0x55 W y133-141 | 67 68 58 57 56 | rom, ram, parent live, wiki |
| 0x5B | L6 | 0x23 | 10R | burn | 0x64 (32,160) | (32,133) DOWN | 0x5A W, 0x5C E, 0x4B N, 0x6B S | 67 68 58 59 5A 5B | rom, ram, parent live, wiki |
| 0x62 | C7 | 0x22 | 100R | burn | 0x64 (128,96) | (148,93) LEFT (east half) | 0x63 E y101-173, 0x52 N x144-208 | 76 66 65 64 63 62 | rom, ram, parent live, wiki |
| 0x67 | H7 | 0x21 | 30R | **bomb** | 0x63 (112,80) | (112,85) UP | 0x77 S x112-128, 0x66 W y117-157, 0x68 E y117-157; **not** 0x57 | 67 (one hop) | rom, ram, live +30 (807f from `BlueRingFull17_bomb_walk`) |
| 0x6B | L7 | 0x22 | 100R | burn | 0x64 (128,160) | (128,133) DOWN | 0x6A W, 0x6C E y117-157, 0x5B N, 0x7B S x176-208 | 78 68 69 6A 6B | rom, ram, parent live, wiki |
| 0x71 | B8 | 0x21 | 30R | bomb | 0x63 (80,80) | (80,85) UP | 0x72 E y117-157, 0x70 W y85-189; 0x61 N is into the woods only | 76 66 65 64 63 73 72 71 | rom, ram, live +30 |

## Potion shops (all 7 Q1): blue 40 / red 68, letter first

| Screen | Grid | Open | Secret object | Stand + facing | Connects | From 0x77 | Evidence |
|---|---|---|---|---|---|---|---|
| 0x04 | E1 | **open** doorway | doorway (192,80) | (192,93) UP | only 0x05 E y101-173 | not on foot (west region) | rom, ram, live entered |
| 0x0D | N1 | **bomb** | 0x63 (144,80) | (144,85) UP | 0x0C W y85-189, 0x1D S x208 | …2C 1C 0C 0D | rom, ram, live entered |
| 0x27 | H3 | bomb | 0x63 (224,80) | (224,85) UP | 0x28 E y85-141, 0x17 N x112-160 | 67 68 58 48 38 28 27 | rom, ram, live entered |
| 0x33 | D4 | bomb | 0x63 (160,80) | (160,85) UP | 0x32 W y133-141, 0x23 N x96/208 | not on foot (west region) | rom, ram, live entered |
| 0x4B | L5 | burn | 0x64 (176,96) | **(208,93) LEFT**, in the east corridor x 192-208 | the corridor: 0x3B N / 0x5B S at x 192-208; the west half cannot burn it | 67 68 58 59 5A 5B 4B | rom, ram, live entered ((144,93) RIGHT failed) |
| 0x64 | E7 | **open** doorway | doorway (112,80) | (112,93) UP | 0x65 E, 0x63 W y117-157, 0x54 N | 76 66 65 64 | rom, ram, live entered |
| 0x78 | I8 | burn | 0x64 (64,160) | (64,141) DOWN | 0x77 W y133-141, 0x68 N, 0x79 E | 78 (one hop) | rom, ram, live entered |

## Door-repair caves (pay 20R on entry): keep flames and bombs off these

| Screen | Grid | Open | Object | Sits next to |
|---|---|---|---|---|
| 0x68 | I7 | burn | tree (32,160) | east of 0x67, north of 0x78; on the L1 walk 0x78→0x68→0x58 |
| 0x63 | D7 | burn | tree (96,160) | between 0x64 and 0x62 (the 0x62 approach) |
| 0x6A | K7 | burn | tree (192,160) | west of 0x6B, on 0x69→0x6A→0x6B |
| 0x14 | E2 | bomb | rock (192,80) | east of 0x13, on the 0x24→0x14→0x13 walk |
| 0x03 | D1 | bomb | rock (112,128) | north of 0x13 |
| 0x1E | O2 | bomb | rock (192,80) | on the NE walk 0x1D→0x1E→0x1F→0x0F |
| 0x7D | N8 | bomb | rock (96,80) | east of 0x7C |
| 0x01 | B1 | bomb | rock (144,80) | |
| 0x07 | H1 | bomb | rock (160,128) | |

A reveal needs a weapon within 15 px per axis. A stray flame or bomb elsewhere on the screen is harmless.

## Route notes

- **0x67 cannot pay for the first bomb.** Its wall needs a bomb. Once bombs are bought (the `0x1D` shop sells them at 20R), one bomb there nets +30R, one hop north of start. Walk it UP from 0x77 at x 112-128. 0x66 and 0x68 connect too (y 117-157); 0x57 does not.
- **Rupee caves that need no item:**
  - 0x3D, 30R armos, 10 screens from start.
  - 0x4E, 10R armos, from 0x4D.
  - 0x0F, 100R open, far NE.

  The armos wakes as an enemy, and its stairs stay open while it walks away.
- **Candle only, near start:** 0x48 30R (4 hops), 0x56 10R (5 hops), 0x28 30R, 0x5B 10R, 0x6B 100R, 0x62 100R, 0x51 10R.
- **West region needs the stepladder.** It covers 0x00-0x09, 0x10-0x16, 0x20-0x26, 0x30-0x36, 0x40, 0x41, 0x50 and 0x60, which includes 0x13, 0x04 and 0x33. The ROM-lattice graph cannot reach it on foot from 0x77, because the river on 0x17 (x 88-111) cuts it off. 0x2F and 0x45 need the raft.

## Catalog errors (`overworld/locations.py` as of 2026-09-24)

Cave ids: every Q1 row matches the ROM, with no row missing or extra. The screens whose AttrsF has bit 7 set are Q2-only and correctly absent. The kind labels (`0x21/22/23`, `0x1C`) are now right.

Wrong open method (the ROM value is live-checked where marked *):

| Screen | Catalog | ROM | Note |
|---|---|---|---|
| 0x3D rupees_n4 | burn | **armos** (144,128)* | This is why BFS_3D had no slot-11 object: it is an armos secret, not a tile object |
| 0x4E rupees_10_o5 | secret | **armos** (160,128)* | |
| 0x0D potion_n1 | open | **bomb** (144,80)* | |
| 0x04 potion_e1 | secret | **open** (192,80)* | |
| 0x64 potion_e7 | secret | **open** (112,80)* | |
| 0x0F rupees_100_p1 | secret | **open** (128,128) | Reached only through the 0x1F false wall |
| 0x66 shop_g7 | bomb | **open** (112,80) | |
| 0x26 shop_g3 | open | **bomb** (48,80) | |
| 0x16 gamble_g2, 0x76 gamble_g8, 0x7C gamble_m8 | open | **bomb** (96,80) each | 0x76 is the screen west of start |
| 0x1C hint_m2 | open | **armos** (176,128) | |
| 0x1D / 0x23 / 0x49 warps | open | **push rock** (48,144) | Needs the Power Bracelet |
| 0x79 warp_j8 | open | **push rock** (128,144) | Needs the Power Bracelet |

The rows marked `secret` should name their method:

- bomb: 0x01, 0x03, 0x07, 0x10, 0x13*, 0x14, 0x1E, 0x27*, 0x33*, 0x67*, 0x71*
- burn: 0x51*, 0x63, 0x68, 0x6A, 0x78*

These rows can move to `verified`: 0x13, 0x3D, 0x4E, 0x51, 0x67, 0x71, 0x04, 0x0D, 0x27, 0x33, 0x4B, 0x64, 0x78.

Unmeasured oddity: `SecretArmosRoomIds` also lists 0x0B (armos x=176) and 0x22 (armos x=48) beside their open dungeon doorways. That is probably a second stairs. `open` stays correct for routing.

0x1A paid_hint_k2 `open` is right: the entrance is behind the waterfall column at x=96 (primary `0x89`), walked UP from (96,125).

## Wiki cross-check

The RPGClassics Q1 map agrees with the ROM on every location and type:

- rupees 1P 100, 2D, 3I, 3N, 4N, 5I 30; 5O, 6B, 6G, 6L 10; 7C, 7L 100; 7H, 8B 30
- potion shops 1E, 1N, 3H, 4D, 5L, 7E, 8I
- door repair 1B, 1D, 1H, 2E, 2O, 7D, 7I, 7K, 8N
- games 2A, 2G, 2P, 8G, 8M
- warps 2N, 3D, 5J, 8J
- information caves 2K, 2M, **8A**, 8F. This confirms that 0x70 is a hint, not a shop.

Two search-result snippets also agree:

- A walkthrough snippet says N4 is "30 hidden rupees under the Armos Statue to the right".
- A Zelda Dungeon snippet says warp stairs sit under rocks that need the Power Bracelet.

The pages that give open methods (Zelda Dungeon "Secret Rupees", StrategyWiki Quest 1) could not be fetched, so no disagreement with them could be checked.

## Reproduce

```bash
# all 128 screens: slot-11 object, flag, lattice, tiles (what-if teleport)
QT_QPA_PLATFORM=offscreen uv run python nes/zelda_i/scratch/secret_teleport_scan.py OW_78 /tmp/tp.json
# live opens (what-if bombs 0x0658/0x0656=1, candle 0x065B=1/0x0656=4; teleport 0x00EB/0x0070/0x0084)
uv run python nes/zelda_i/scripts/stage_replay.py BlueRingFull17_bomb_walk zelda_i.scratch.secret_walks:probe_67 --assist --set 0x0658=1 --set 0x0656=1
uv run python nes/zelda_i/scripts/stage_replay.py OW_78 zelda_i.scratch.secret_walks:probe_71 --assist --set 0x00EB=0x72 --set 0x0070=0 --set 0x0084=141 --set 0x0658=1 --set 0x0656=1
uv run python nes/zelda_i/scripts/stage_replay.py BFS_3D zelda_i.scratch.secret_walks:probe_3d --assist
uv run python nes/zelda_i/scripts/stage_replay.py OW_78 zelda_i.scratch.secret_walks:probe_4b_east --assist --set 0x00EB=0x3B --set 0x0070=200 --set 0x0084=221 --set 0x065B=1 --set 0x0656=4
```

Other probes in `scratch/secret_walks.py`: `probe_13`, `probe_51`, `probe_4e`, `probe_78`, `probe_0d`, `probe_27`, `probe_33`, `probe_64` and `probe_04`. Each docstring gives its pin and the `--set` teleport.
