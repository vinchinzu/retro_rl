# Pre-L1 loadout

Primary walkthrough: [Zelda Dungeon — The Gathering](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/the-gathering/)
(2015-01-23). Secondary: [IGN Preparation](https://www.ign.com/wikis/the-legend-of-zelda/Preparation)
(2025-07-16). First quest only. Grid: `screen = (row << 4) | col`, start
`0x77` = H8.

M5 Clean is still power-on → L1 Triforce on 3 containers and the wooden sword
(19416f, TF `0x01`). This prefix is the combat-budget answer: gather **before**
the 0x37 mouth, then re-enter L1 with 6 containers and the White Sword. Do
not overwrite the 19416f claim.

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
| 1.1 | `0x6F` `shop_p7` | `CAVE_SHOP_ARROWS` | open | **bypass 0x79** via `0x78`→`0x68`; **0x68 east from x=48 is dead** (bush / occupancy). Next: `0x58`/`0x59` SOUTH onto `0x69` then east | Bombs 20R. Same shop family as `0x4A`. Farm drops on the way, not the 0x4A tektite wave |
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
M5 19416f stays the wooden 3HC oracle. Re-measure L1 after this prefix greens.

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
