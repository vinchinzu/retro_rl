# ALTTP RAM map

stable-retro `get_ram()`:

- index `0..8191`: WRAM `$7E0000-$7E1FFF`
- index `16384+offset`: full WRAM for offsets `>= 0x2000`

| Field | WRAM | Index | Notes |
|-------|------|-------|-------|
| module | `$10` | 16 | `0x00` intro, `0x01` file select, `0x07`/`0x09` play, `0x0E` dialog. `0x02` was not the file select on the census run |
| submodule | `$11` | 17 | `0` when controllable |
| indoors | `$1B` | 27 | `0` outdoor |
| link Y/X | `$20`/`$22` | 32/34 | u16 |
| screen id | `$8A` | 138 | overworld screen |
| room id | `$A0` | 160 | u16 dungeon room |
| dark world | `$0FFF` | 4095 | nonzero in DW |
| sword | `$F359` | 78713 via WRAM_IDX | high WRAM (0=none, 1=fighter) |
| dungeon keys | `$F36F` | 78735 via WRAM_IDX | `0xFF` = blank HUD / no dungeon keys yet |
| follower | `$F3CC` | 78828 via WRAM_IDX | tagalong; `1` = Zelda |
| secret passage room | `$A0` base | 160 | opening hole drop-in `0x55` |
| secret hole approach | world X/Y | `$22`/`$20` | near Yaze `0x7D` ~(2432,1696) on screen `0x1B` |
| link direction | `$2F` | 47 | 0=up, 2=down, 4=left, 6=right |
| link action | `$5D` | 93 | `21` = hold-up-item after sword get |

## Progress and follower

Scanned every `custom_integrations/Zelda3-Snes/*.state` on 2026-09-29.
No RAM writes. Names below are the values those pins actually held.
`$F3C5` did not move when Zelda started following: `CastleZeldaFollower`
is still progress `0x01`. Progress `0x02` was not on any pin. Progress
`0x03` is only on the late dungeon-entry pins; this scan did not watch
that byte change in play, so it stays unnamed. Follower `5` is on the
pre-sword pins; its name is not verified.

| Field | WRAM | Disassembly | Value | Pins | Meaning |
|-------|------|-------------|-------|------|---------|
| progress | `$F3C5` | `sram_progress_indicator` | `0x00` | `LinksHouseWake`, `FirstAction`, `HyruleCastleGrounds` | no fighter sword |
| progress | `$F3C5` | `sram_progress_indicator` | `0x01` | `CourtyardSecretPocket`, `FighterSword`, `CastleZeldaFollower`, other sword-route castle pins | sword received; still `0x01` while Zelda follows |
| progress | `$F3C5` | `sram_progress_indicator` | `0x03` | `IcePalaceEntry`, `PalaceOfDarknessEntry`, `SwampPalaceEntry`, `ThievesTownEntry`, `YazeSlot000` | seen; name not verified here |
| flags | `$F3C6` | `sram_progress_flags` | `0x10` | `LinksHouseWake`, `FirstAction`, `HyruleCastleGrounds` | bit `0x10` already set before the sword; clear state not on these pins |
| flags | `$F3C6` | `sram_progress_flags` | `0x11` | `CourtyardSecretPocket`, `FighterSword`, `CastleZeldaFollower` | bit `0x01` uncle gear, added with the sword and shield |
| follower | `$F3CC` | `follower_indicator` | `0` | `CourtyardSecretPocket`, `FighterSword`, castle pins without Zelda | none |
| follower | `$F3CC` | `follower_indicator` | `1` | `CastleZeldaFollower` (also `CastleMainZeldaReady`, `CastleMantleZelda`, `CastleRoom51Zelda`, `CastleZeldaB1West` and the other `*Zelda*` pins) | Zelda |
| follower | `$F3CC` | `follower_indicator` | `5` | `LinksHouseWake`, `FirstAction`, `HyruleCastleGrounds` | seen; name not verified |

Old man, Kiki, maiden, super bomb: to verify at AP-091 / AP-131 / AP-158 / AP-098.

## Save and quit spawns

`save_and_quit` in `primitives.py`. Buttons only, 2026-09-29. The box is
SELECT during play. START opens the item menu and does not quit. Progress
`$F3C5` = 0 never opens the box (message stays off, Link stays in play).
Quit is DOWN to choice 1, then B, after `$1CE9` rises and falls. Continue
is two STARTs on intro module `0x00` and two A's on file select `0x01`.

Frames are from the call, including 8 idle frames before SELECT, to the
first frame with control. Each pin was reloaded with `set_state`. `$F3C8`
(`which_starting_point`) is still the saved value when control returns.
Spawn `0x01` was not on any pin. Mirror `$F353` was only `0x00` or `0x01`
across the 76 states, so message `0x0185` (the mirror==2 menu) did not
appear. On message `0x0184`, UP from row 0 stays on row 0.

`$F3CA` (`savegame_is_darkworld`) was `0x00` on every pin except
`PalaceOfDarknessEntry`, where it was `0x40`. Nonzero skips the spawn menu.

| Progress | `$F3CA` | Menu | Choice | Landing | Frames | Pin |
|----------|---------|------|--------|---------|--------|-----|
| `0x00` | `0x00` | box refused | — | stays in play, room `0x0104` | 91 | `LinksHouseWake` |
| `0x01` | `0x00` | none (spawn `0x00`) | ignored | Link's house, room `0x0104` (2424,8568) | 487 | `EasternPalaceEntry` |
| `0x01` | `0x00` | none (spawn `0x02`) | ignored | Zelda cell, room `0x0080` (352,4161) | 512 | `CastleMainZeldaBoomerang` |
| `0x01` | `0x00` | none (spawn `0x03`) | ignored | secret entrance, room `0x0055` (2822,2672) | 487 | `FighterSwordLamp` |
| `0x03` | `0x00` | `0x0184` row 0 | 0 | Link's house, room `0x0104` (2424,8568) | 538 | `IcePalaceEntry` |
| `0x03` | `0x00` | `0x0184` row 1 | 1 | Sanctuary, room `0x0012` (1272,668) | 669 | `IcePalaceEntry` |
| `0x03` | `0x00` | `0x0184` row 0 | 0 | Link's house, room `0x0104` (2424,8568) | 539 | `SwampPalaceEntry` (saved spawn was `0x03`) |
| `0x03` | `0x40` | none | ignored | pyramid, screen `0x5B` room `0x0020` (2032,1628), dark world | 490 | `PalaceOfDarknessEntry` |

Mountain cave was not measured. No pin reached a third menu row.

## Module census

One `module_census.py` pass of `run_to_verified_tip.py` on 2026-09-29.
Report went to `/tmp`, not `recordings/verified_tip_run.json`. The run
stopped red in `pocket_to_main_hall` (`missed door approach`), so castle
rooms `0x61` / `0x60` / `0x50` are not in this table. Frame is the step
count after reset. Byte `0x55` at frame 0 is the pre-boot fill, not a
module. Menu (`$10=0x0E` `$11=0x01`) was not entered. Module `0x06`
(`Module_PreDungeon`) was not entered; the bush hole used module `0x11`.

Names are snesrev `kMainRouting` / submodule tables for the pair that
actually appeared. `play` is submodule `0` only.

| Module | Sub | Name | First frame | Room | Screen |
|--------|-----|------|-------------|------|--------|
| `0x00` | `0x00` | intro | 78 | `0x0000` | `0x00` |
| `0x00` | `0x01` | intro | 79 | `0x0000` | `0x00` |
| `0x00` | `0x02` | intro | 219 | `0x0000` | `0x00` |
| `0x00` | `0x03` | intro | 222 | `0x0000` | `0x00` |
| `0x00` | `0x04` | intro | 309 | `0x0000` | `0x00` |
| `0x00` | `0x05` | intro | 750 | `0x0000` | `0x00` |
| `0x00` | `0x06` | intro | 843 | `0x0000` | `0x00` |
| `0x00` | `0x07` | intro | 886 | `0x0000` | `0x00` |
| `0x01` | `0x00` | file select | 973 | `0x0000` | `0x00` |
| `0x01` | `0x01` | file select | 974 | `0x0000` | `0x00` |
| `0x01` | `0x02` | file select | 1031 | `0x0000` | `0x00` |
| `0x01` | `0x03` | file select | 1033 | `0x0000` | `0x00` |
| `0x01` | `0x04` | file select | 1034 | `0x0000` | `0x00` |
| `0x01` | `0x05` | file select | 1035 | `0x0000` | `0x00` |
| `0x04` | `0x00` | name file | 1038 | `0x0000` | `0x00` |
| `0x04` | `0x01` | name file | 1039 | `0x0000` | `0x00` |
| `0x04` | `0x02` | name file | 1041 | `0x0000` | `0x00` |
| `0x04` | `0x03` | name file | 1042 | `0x0000` | `0x00` |
| `0x05` | `0x00` | load file | 1363 | `0x0000` | `0x00` |
| `0x07` | `0x0F` | landing wipe | 1441 | `0x0104` | `0x00` |
| `0x0E` | `0x02` | dialog | 1477 | `0x0104` | `0x00` |
| `0x07` | `0x00` | play | 2836 | `0x0104` | `0x00` |
| `0x0F` | `0x00` | spotlight close | 3905 | `0x0104` | `0x00` |
| `0x0F` | `0x01` | spotlight close | 3907 | `0x0104` | `0x00` |
| `0x08` | `0x00` | OW load | 3940 | `0x0104` | `0x00` |
| `0x08` | `0x01` | OW load | 3982 | `0x0104` | `0x2C` |
| `0x08` | `0x02` | OW load | 3989 | `0x0104` | `0x2C` |
| `0x10` | `0x00` | spotlight open | 4007 | `0x0104` | `0x2C` |
| `0x10` | `0x01` | spotlight open | 4008 | `0x0104` | `0x2C` |
| `0x09` | `0x0A` | exit walk | 4043 | `0x0104` | `0x2C` |
| `0x09` | `0x00` | play | 4059 | `0x0104` | `0x2C` |
| `0x09` | `0x01` | OW load | 4634 | `0x0104` | `0x1B` |
| `0x09` | `0x02` | OW load | 4650 | `0x0104` | `0x1B` |
| `0x09` | `0x03` | OW load | 4651 | `0x0104` | `0x1B` |
| `0x09` | `0x04` | OW load | 4665 | `0x0104` | `0x1B` |
| `0x09` | `0x05` | room scroll | 4672 | `0x0104` | `0x1B` |
| `0x09` | `0x06` | room scroll | 4673 | `0x0104` | `0x1B` |
| `0x09` | `0x07` | room scroll | 4701 | `0x0104` | `0x1B` |
| `0x09` | `0x08` | room scroll | 4710 | `0x0104` | `0x1B` |
| `0x11` | `0x00` | dungeon load | 7371 | `0x010F` | `0x1B` |
| `0x11` | `0x07` | dungeon load | 7490 | `0x0055` | `0x00` |
| `0x07` | `0x01` | room scroll | 8206 | `0x0055` | `0x00` |
| `0x07` | `0x05` | shutters | 8338 | `0x0055` | `0x00` |
| `0x07` | `0x10` | stairs | 8401 | `0x0055` | `0x00` |

## Replay sync window

Two `run_to_verified_tip.py` runs in one process, then a third in a fresh
process, on 2026-09-29. Tapes `/tmp/ap020_a.tape.npz`, `ap020_b`, `ap020_c`
(boot `YazeSlot000`). Buttons matched, shape `(15366, 12)`. Every
`get_ram()` byte matched on every frame (`147456` bytes, `0` indexes
differed). All three stopped on the same red `pocket_to_main_hall` door
miss, so the path was one path.

Sync window: the full `get_ram()` image. WRAM `$0000-$1FFF` is index
`0..8191`. WRAM at `offset >= $2000` is index `16384 + offset`.

Excluded bytes: none. `$0000-$000F` and everything above `$07FF` also
matched, so this is not the Zelda `$0010-$07FF` cut.

## Inventory (`$F340`–`$F37B`)

Candidate addresses are snesrev/zelda3 `variables.h` (`link_item_*` and the
words that follow). `verified_on` is a pin whose live RAM showed a distinctive
value on 2026-09-29, or `unverified` when the scanned pins only read 0.
Scanned: `HyruleCastleGrounds`, `FighterSword`, `FighterSwordLamp`,
`CastleB1Key`, `CastleB3Boomerang`, `CastleMainZeldaBoomerang`. No RAM writes.
Index for these offsets is `16384 + offset`.

`$F34F` (`link_item_bottle_index`) is not a bottle slot. `$F358` is unused in
that header. Rupee goal is `$F360` (`link_rupees_goal`); `rupees` is the
actual count at `$F362`. `$F379` (`ability`) is the A-press mask. Dash is
bit `0x04`: `$07:9C2F` loads `$9BA2,X` and ANDs this byte, and action 2
uses mask `0x04`. Scanned pins read `0xF8` (that bit clear).

| Field | WRAM | Disassembly | verified_on | Notes |
|-------|------|-------------|-------------|-------|
| bow | `$F340` | `link_item_bow` | unverified | 0 none, 1 bow, 2 bow+arrows, 3 silver |
| boomerang | `$F341` | `link_item_boomerang` | CastleB3Boomerang | 1 there and on `CastleMainZeldaBoomerang`; 0 on `FighterSword` |
| hookshot | `$F342` | `link_item_hookshot` | unverified | 0 on the scanned pins |
| bombs | `$F343` | `link_item_bombs` | unverified | count; 0 on the scanned pins |
| powder | `$F344` | `link_item_mushroom` | unverified | 0 none, 1 mushroom, 2 powder |
| fire_rod | `$F345` | `link_item_fire_rod` | unverified | 0 on the scanned pins |
| ice_rod | `$F346` | `link_item_ice_rod` | unverified | 0 on the scanned pins |
| bombos | `$F347` | `link_item_bombos_medallion` | unverified | medallion; 0 on the scanned pins |
| ether | `$F348` | `link_item_ether_medallion` | unverified | medallion; 0 on the scanned pins |
| quake | `$F349` | `link_item_quake_medallion` | unverified | medallion; 0 on the scanned pins |
| lamp | `$F34A` | `link_item_torch` | FighterSwordLamp | 1 there; 0 on `FighterSword` |
| hammer | `$F34B` | `link_item_hammer` | unverified | 0 on the scanned pins |
| flute | `$F34C` | `link_item_flute` | unverified | 0 none, 1 shovel, 2 flute, 3 bird |
| net | `$F34D` | `link_item_bug_net` | unverified | 0 on the scanned pins |
| book | `$F34E` | `link_item_book_of_mudora` | unverified | 0 on the scanned pins |
| somaria | `$F350` | `link_item_cane_somaria` | unverified | 0 on the scanned pins |
| byrna | `$F351` | `link_item_cane_byrna` | unverified | 0 on the scanned pins |
| cape | `$F352` | `link_item_cape` | unverified | 0 on the scanned pins |
| mirror | `$F353` | `link_item_mirror` | unverified | 0 none, 1 wrong gfx, 2 mirror |
| gloves | `$F354` | `link_item_gloves` | unverified | 0 none, 1 power, 2 titan |
| boots | `$F355` | `link_item_boots` | unverified | 0 on the scanned pins |
| flippers | `$F356` | `link_item_flippers` | unverified | 0 on the scanned pins |
| pearl | `$F357` | `link_item_moon_pearl` | unverified | 0 on the scanned pins |
| sword | `$F359` | `link_sword_type` | FighterSword | 1 fighter. 0 on `HyruleCastleGrounds` |
| shield | `$F35A` | `link_shield_type` | FighterSword | 1. 0 on `HyruleCastleGrounds` |
| armor | `$F35B` | `link_armor` | unverified | 0 green mail is also the empty value |
| bottle_1 | `$F35C` | `link_bottle_info[0]` | unverified | 0 on the scanned pins |
| bottle_2 | `$F35D` | `link_bottle_info[1]` | unverified | 0 on the scanned pins |
| bottle_3 | `$F35E` | `link_bottle_info[2]` | unverified | 0 on the scanned pins |
| bottle_4 | `$F35F` | `link_bottle_info[3]` | unverified | 0 on the scanned pins |
| rupees | `$F362` | `link_rupees_actual` | unverified | u16 actual; goal is `$F360` |
| compass | `$F364` | `link_compass` | unverified | u16 dungeon bits; 0 on the scanned pins |
| big_key | `$F366` | `link_bigkey` | CastleMainZeldaBoomerang | u16 reads `0x4000` (`$F367` bit 6, Hyrule Castle candidate) |
| dungeon_map | `$F368` | `link_dungeon_map` | unverified | u16; same bit positions as `big_key`; 0 on the scanned pins |
| heart_pieces | `$F36B` | `link_heart_pieces` | unverified | count toward the next container; 0 on the scanned pins |
| max_hp | `$F36C` | `link_health_capacity` | CastleB1Key | `0x18` (3 hearts) while `hp` is `0x14` |
| hp | `$F36D` | `link_health_current` | CastleB1Key | `0x14`; full `0x18` on `FighterSword` |
| magic | `$F36E` | `link_magic_power` | IcePalaceEntry | `0x80` there. Meter is `0..0x80`. Costs are below |
| keys | `$F36F` | `link_num_keys` | CastleB1Key | 1 there; `0xFF` blank on `FighterSword` |
| pendants | `$F374` | `link_which_pendants` | unverified | bits 0 courage, 1 wisdom, 2 power (candidate) |
| arrows | `$F377` | `link_num_arrows` | unverified | count; 0 on the scanned pins |
| ability | `$F379` | `link_ability_flags` | unverified | bit `0x04` is dash (`$07:9C2F`, `$9BA2` index 2). Scanned pins read `0xF8` |
| crystals | `$F37A` | `link_has_crystals` | unverified | bitfield; 0 on the scanned pins |

## Magic costs

Measured 2026-09-29. `$F37B` (`link_magic_consumption`) was 0 on every pin,
so these are the normal column. Half and quarter were not measured. Y
while `$006C` is nonzero spends nothing; each use below is indoors,
module `0x07`, submodule 0, doorway 0. The ledger spend field already
includes `magic` (`spine/ledger.py`).

`$F373` (`link_magic_filler`) moves one point onto `$F36E` per frame.
A small magic sprite (`0xDF`) adds `0x10` to the filler. A full magic
sprite (`0xE0`) sets the filler to `0x80`. The meter stops at `0x80`
and the leftover filler is cleared. Both drops were spawned on Link
(state `0x09`) on `EasternPalaceEntry` and `IcePalaceEntry` with the
meter poked to `0x20`: small ended at `0x30`, full ended at `0x80`.

Byrna's spark was still taking magic at frame 183. The cape arms only
after `$02E2` counts down to 0 (it was `0x0F` and fell once per frame
while the cape was the Y item). The first cape point left 4 frames
after it armed. No chunk cost.

| Use | `$F36E` | Pin |
|-----|---------|-----|
| lamp | 4 on Y | `EasternPalaceEntry` room `0xC9`, owned lamp, meter poked to `0x80` |
| fire rod | 16 on Y | `IcePalaceEntry` room `0x0E`, owned rod, meter already `0x80` |
| ice rod | 16 on Y | `SwampPalaceEntry` room `0x28`, item byte poked, menu |
| somaria | 8 on Y | `SwampPalaceEntry` room `0x28`, item byte poked, menu |
| byrna | 16 on Y, then 4 every 24 frames while the spark is up | `SwampPalaceEntry` room `0x28`, item byte poked, menu |
| cape | 1 every 4 frames while `$0055` is on | `SwampPalaceEntry` room `0x28`, item byte poked, menu |
| bombos | 32 on Y | `SwampPalaceEntry` room `0x28`, item byte poked, menu |
| ether | 32 on Y | `SwampPalaceEntry` room `0x28`, item byte poked, menu |
| quake | 32 on Y | `SwampPalaceEntry` room `0x28`, item byte poked, menu |
| small magic `0xDF` | +16 | Eastern and Ice, meter `0x20` to `0x30` |
| full magic `0xE0` | fill to `0x80` | Eastern and Ice, meter `0x20` to `0x80` |

## Overworld flags

`ow_flags(ram, screen)` is `save_ow_event_info` at `$F280 + screen`
(screens `0x00`–`0x7F`). Disassembly bits: `0x02` bombable secondary
overlay, `0x20` event overlay. Neither is the castle bush hole.

`$F280+0x1B` was `0x00` on `HyruleCastleGrounds` and on
`CourtyardSecretPocket`. The pin diff in the save block is sword, shield,
`$F3C5`, `$F3C6`, `$F3C8`, and `$F3CC`. Those are not the hole.
`num_memorized_tiles` (`$4AC`) is `0` on both pins.

The opened hole is one memorized map16 record
(`Overworld_LiftingSmallObj` → `Overworld_RevealSecret` →
`Overworld_Memorize_Map16_Change`). Secret byte `0x80` is Hole;
`kTileBelow[0]` is `0x0DCC`. Warp is secret `0x82` / tile `0x212`, and
this bush is not that. The ROM secret list for screen `0x1B` has position
`0x0570`, data `0x80`.

Live lift 2026-09-29, no RAM writes, from `HyruleCastleGrounds` parked at
(2429, 1704) facing up. On the A press: `$4AC` `0→2`, `$F800` `0x0570`,
`$FA00` `0x0DCC`, map16 `$2000+0x0570` `0x0036→0x0DCC`. `$F29B` stayed
`0x00`. Walking north entered module `0x11` (room low byte `0x0F`, the
same hole fall the census later lands in room `0x55`). The record does
not survive `CourtyardSecretPocket`.

| Field | WRAM | Disassembly | Live value |
|-------|------|-------------|------------|
| event byte | `$F280+screen` | `save_ow_event_info` | screen `0x1B` stayed `0x00` |
| count | `$4AC` | `num_memorized_tiles` | `2` (one tile; +2 per record) |
| position | `$F800` u16 | `memorized_tile_addr` | `0x0570` |
| tile | `$FA00` u16 | `memorized_tile_value` | `0x0DCC` hole |

## Doors

Live scan of every `custom_integrations/Zelda3-Snes/*.state` on 2026-09-29.
No RAM writes. One `set_state` per pin. Addresses are snesrev `variables.h`;
`get_ram()` index equals the offset (all of these are below `$2000`).

Room load (`RoomDraw_FlagDoorsAndGetFinalType`) writes one slot per door.
`$0460` (`dung_cur_door_idx`) is the next free slot times 2. Slots 8–15 are
the other layer of the same door when the ROM position enum is >= 6. That
pair is not a second exit. A slot is occupied when any of the three words
is nonzero. Type `0x00` is a real regular door, not an empty slot.

| Field | WRAM | Disassembly | Notes |
|-------|------|-------------|-------|
| type and slot | `$1980` + 2×slot | `door_type_and_slot[16]` | low byte type, high byte slot |
| position | `$19A0` + 2×slot | `dung_door_tilemap_address[16]` | tilemap byte offset (`dsto * 2`), not Link xy |
| direction | `$19C0` + 2×slot | `dung_door_direction[16]` | 0 north, 1 south, 2 west, 3 east |
| next slot | `$0460` | `dung_cur_door_idx` | units of 2 |

Type byte is the even value `RoomDraw_Door_*` stores. Names below are
snesrev `kDoorType` for codes this scan actually saw.

| Code | Name | Where |
|------|------|-------|
| `0x00` | regular | `CastleMain` slot 3, west, pos `$0784` |
| `0x02` | regular (other graphic) | `CastleMain` slots 0–2: west `$0F8A`, east `$0FEE`, south `$1BBC` |
| `0x0A` | entrance, large | south door on the `*Entry` pins (`EasternPalaceEntry` room `0xC9`, Desert, Ice, PoD, Swamp, Thieves) |
| `0x0C` | entrance, large 2 | `HeraEntry` room `0x77` south |
| `0x18` | two-way shutter | `EasternPalaceEntry` north slots. Not the cell-lock object |
| `0x1C` | small key | `CastleB1Key` room `0x72` slot 0 north pos `$123C`. Slot 8 is the lower-layer south pair, pos `$0D3C`, same type |
| `0x22` | stair mask | `IcePalaceEntry` north, `SwampPalaceEntry` north |
| `0x26` | stair mask | `HeraEntry` north |
| `0x40` | regular, high range | every room `0x81` pin, slot 0 north pos `$239C` only |
| `0x44` | shutter | `CastleB1SecondKey` room `0x71` south and west (slot 10 is the east pair) |
| `0x46` | warp-room door | `CastleRoom50` east, `CastleRoom01` west/east |

`kDoorType_1E` (`0x1E`) is the big-key door (the open check reads
`link_bigkey`). No scanned pin held `0x1E`, including every room `0x81` pin.

Room `0x81` (`CastleZeldaB1West`, `CastleB1FarDoor`, `CastleB1GreenRoom`,
and the other `0x81` states) does not keep the west gold door in `$1980`.
Those pins have only the north `0x40` door. Door-attribute tiles
(`$12000` high nibble `$F`) are absent there too. The small-key door on
`CastleB1Key` does show `$F0` / `$F8` in that table, matching slots 0 and 8.

The gold cell lock is dungeon object `0x18`, not door type `0x18`. It
shares the chest table. `$0496` is the chest count times 2.
`$0498` is that count plus cell locks. Words at `$06E0` are
`dung_chest_locations`. Entries before the chest count are chests.
Entries from the chest count up to `$0498` are cell locks. The attribute
pass sets bit `0x8000` on a closed cell lock. Room `0x80`
(`CastleB3`, `CastleB3Boomerang`, `CastleZeldaFollower`) has chests=2,
locks=4, cell lock `$8758` (tile 44,14), and an empty door array.
`CastleB3` save bits `$0402` are `0`. `CastleZeldaFollower` bits are
`$4200` and the location word is still `$8758`. Room `0x81` counts are
both 0. A leftover `$8758` in that room's `$06E0` is the previous room,
not a live lock.
