# Dungeon route — Tower of Hera

Copy of `docs/routes/DUNGEON_TEMPLATE.md`. A save-state pin is not continuous. Facts below are from `dungeon/route_data/hera.yaml` or a live scan on `HeraEntry`.

## Entry

| Field | Value |
|-------|--------|
| Pin | HeraEntry |
| Room | 0x77 |
| First hop | `room_77` edge `west_down_to_0x87` (map door `west_down_to_0x87`, UP, to room 0x87). verification=isolated. Not natural. |

`HeraEntry` is room 0x77 at (3832,4032), layer 1, floor 0x01, module 0x07, dungeon 0x14. The pin holds keys=0, big_key word `$F366`=0, bombs=0, lamp `$F34A`=1, magic `$F36E`=0. Crystal (3840,3968) toggles `$C172` in play: 00 keeps the west alcove on raised pegs (attribute 0x67), and 01 lowers those cells to 0x66. Holding tile 0x5E at (17,43) on peg 01 crosses to room 0x87 at (3721,4424), layer 0, floor 0x00. That hop is the isolated AP-115 edge. This scan walked it again and saved the landing.

Assist was armed for a last-heart refill of `$F36D` only. Writes=0, hearts=0, first_frame none. Saved pins stay at hp 24. A later unsaved stand in 0x87 dropped hp 24 to 16, still above the last heart, and wrote nothing. This scan is a development scan. It is not a Clean claim. Lamp stayed 1. Magic stayed 0. `$C172`, keys, the big-key bit, rupees, and `$F3CC` were not written.

## Requirements

Items the route needs, taken from z3 logic (`hera.yaml`, z3 id 114).

| Item | Why | z3 source |
|------|-----|-----------|
| KeyP3 | Big key chest, and the Basement Cage contents | locations Big Key Chest, Basement Cage |
| canLightTorches | Big key chest | location Big Key Chest |
| BigKeyP3 | Compass chest, big chest, Moldorm | locations Compass Chest, Big Chest, Moldorm |
| MoonPearl | Big chest contents | location Big Chest item MoonPearl |
| L1Sword | Moldorm sword strat | strat Moldorm - Sword |
| Hammer | Moldorm hammer strat | strat Moldorm - Hammer |

Map Chest (MapP3, address 0xE9AD) and Basement Cage (KeyP3, address 0x180162) have empty requires. Prize location Tower of Hera - Prize has empty requires. Moldorm requires BigKeyP3.

## Room table

One row per room reached with the keys on the scan pins (keys=0, big_key=0). Geometry stays in `alttp/maps/` json. Do not copy approach xy into Python. Dumps: `Scan_hera_0x77` (`$C172=00`), `Scan_hera_0x87` (`$C172=01`).

| Room | Doors + types | Chests | Sprites | Notes |
|------|---------------|--------|---------|-------|
| 0x77 | slot 0 north type 0x26 stairs pos $239C dest 0x87, walked as tile 0x5E (17,43) at `$C172=01`; slot 1 south type 0x0C entrance pos $3BBC to overworld screen 0x03 | chest 0 MapP3 z3=0xE9AD tile (31,15) layer 1 opened=0. Same 36-cell pocket as stair byte 0x22 at (31,18). No path from the entrance floor at either peg. | mini moldorm 0x18 (3760,3728) hp 3; crystals 0x1E (3840,3968), (3728,4000), (3936,4000); kodongo 0x86 (3700,3739), (3956,3739) hp 0 | Pin Scan_hera_0x77. `$C172=00` at (3832,4032) layer 1 floor 0x01 hp 24 keys=0 lamp=1 magic=0. Crystal (3840,3968) toggles 00→01 and 01→00 in play. Peg 01 opens the west alcove (0x66) and the west 0x5E stair. East 0x5E (45,43) has no body path at either peg. |
| 0x87 | stair back to 0x77 (the landing mouth); slot 0 west type 0x00 pos $07C4; slot 1 north type 0x00 pos $125C; slot 8 east type 0x18 shutter pos $07B4 other layer; slot 9 south type 0x00 pos $0D5C other layer | chest 0 BigKeyP3 z3=0xE9E6 tile (47,52) layer 0 opened=0. East component. No path from the landing at either peg. | mini moldorm 0x18 hp 3 at (3904,4176), (4000,4208), (3889,4272), (3681,4496); crystals 0x1E (3952,4160), (3632,4288), (3648,4432); stalfos 0xA7 hp 4 at (3760,4464), (3984,4480), (3648,4496), (3920,4544); small key 0xE4 (3712,4512) | Pin Scan_hera_0x87. `$C172=01` at (3721,4424) layer 0 floor 0x00 hp 24 keys=0 big_key=0 lamp=1 magic=0. Crystal (3648,4432) toggles 01→00 and 00→01 in play. Peg 00 puts tile (16,52) on the landing component. |

Stopping locks. The pin holds keys=0, big_key=0, bombs=0. The far side of each stayed unentered. A lock is the scan result.

| Lock | Live door | Why it stops |
|------|-----------|--------------|
| 0x77 south | slot 1 type 0x0C pos $3BBC | Overworld exit. Both pegs land screen 0x03 at (2288,123). Death Mountain was not scanned. |
| 0x77 east stair | tile 0x5E (45,40), also (45,43) and (46,43) | Entrance floor at either peg has no body path. The north component on `Whatif_HeraEntry_parallel` at peg 00 does: stand (45,41) stairs into room 0x31. |
| 0x77 map pocket | MapP3 tile (31,15) layer 1, stair byte 0x22 at (31,18) | Entrance floor has no path at either peg. After the remote arrow toggle, tile (31,20) is on the north component and the chest stands still have no path. Holding UP on 0x22 did not leave the room. |
| 0x87 west | slot 0 type 0x00 pos $07C4 col 34 row 15 | No floor path at peg 00 or 01. Inward cells are wall. At peg 00 the last inward cell is 0x66 and still disconnected. |
| 0x87 north | slot 1 type 0x00 pos $125C col 46 row 36 | No path at either peg. The floor past the mouth is the east component. |
| 0x87 east shutter | slot 8 type 0x18 pos $07B4 | Other layer. Landing is layer 0. Far side unentered. |
| 0x87 south | slot 9 type 0x00 pos $0D5C | Other layer. Landing is layer 0. Far side unentered. |
| 0x87 big key chest | BigKeyP3 tile (47,52) layer 0 opened=0 | East component. No path from the landing at either peg. yaml requires KeyP3 and canLightTorches. Pin keys=0 and magic=0. |
| 0x31 big-key door | slot 0 type 0x1E pos `$121C` tile (14,36) | Landing flood does not touch it. Zero peg gates on that flood. Shutter slot 1 type 0x18 stayed shut after a sword hit (beetle hp 60→56). |

The walked mouth is tile 0x5E (17,43) in 0x77, door slot 0 type 0x26. It is the way into 0x87. No type 0x1C door was live. Room 0x31, reached only from the isolated poke pin below, has slot 0 north type 0x1E.

### Isolated poke pin `Whatif_HeraEntry_parallel`

Not natural, not continuous, not a STATUS claim. Kind poke. Room 0x77, module 0x07. Writes already on the pin: big key `$F366=0x0020`, keys `$F36F=1`, magic `$F36E=128`. This lane did not write those, the pearl, or `$F3CC`. `$F3CC` stayed 0.

From the lobby crystal (3840,3968), a sword hit sets `$C172` to 01. Sidestep to x=3800, then stand at y=3910, x=3840, and fire an arrow south. The arrow flips peg 01→00 while Link stays north of the blue bar. Walk to (3838,3744), tile (31,20), still room 0x77, peg 00, hp 24, pearl 0. MapP3 stands (31,17), (29,15), (33,15), and (31,13) have no path. Holding UP on stair byte 0x22 at (31,18) does not change rooms.

From that north component, stand (45,41) and hold UP on tile 0x5E (45,40). `transition_kind` is stairs. Landing is room 0x31 at (872,1858), layer 0, floor 0x02, peg 00, hp 24, pearl 0, big key still the poked `0x0020`. Doors: slot 0 north type 0x1E pos `$121C` tile (14,36); slot 1 west type 0x18 shutter pos `$17C4` tile (34,47); slots 8 and 9 are the other layer. No chests. Flood from the landing is 238 cells. No 0x66 or 0x67 cell borders that flood, so a peg toggle on this component does not open the big-key door. Stands (14,38), (14,39), (15,38), and (13,38) have no path. The door graphic at cols 15–16 rows 37–38 is a separate pocket.

Pit tile (47,44), stand (48,44), hold LEFT. `transition_kind` is fall (submodule 0x07). Landing room 0x77 (3952,3933), layer 1, floor 0x01, peg 00. That is the AP-052 fall primitive, live on this pin. It returns to the lobby.

Ten hardhat beetles (0x26, 6 hp each) share the landing. Hammer `$F34B` is 0. Twenty-four sword taps plus one spin dropped the hp sum 60→56. The shutter stayed type 0x18 and tiles (35,48) and (36,48) stayed 0xF1. Two last-heart refills wrote `$F36D` 8→24. Writes=2. Not a Clean claim. Moon Pearl `$F357` stayed 0. The big chest was not entered. Report: `recordings/probe_AP-115/ap118_north.json`.

### Isolated what-if pin `Whatif_HeraEntry_pearl`

Not natural, not continuous, not a STATUS claim. Kind whatif. Source `Whatif_HeraEntry_parallel`. The only write on the copy is pearl `$F357=1`. Inherited bytes stayed: big key `$F366=0x0020`, keys `$F36F=1`, magic `$F36E=128`. `$F3CC` read 0. Read-only pose is room 0x77 at (3832,4032), layer 1, floor 0x01, module 0x07, hp 24, sword 1, hammer 0. No sprite 0x09 (mini moldorm 0x18 is not the boss). Live doors: slot 0 north type 0x26 stairs to room 0x87, slot 1 south type 0x0C to screen 0x03. Moldorm's room is 0x07 (`kBossRooms` value 7). The door toward it is room 0x31 slot 0 type 0x1E, which AP-118 already calls no path from the stair landing. The west shutter type 0x18 and pit (47,44) stay those recorded failures. This read did not rewalk the chest, the peg floor, or that pit. The arena pin is the later poke `Whatif_Hera_moldorm`. Report: `recordings/probe_AP-119/pearl_pin.json`.

## Keys and big key

Where each small key is taken, and where the big key is taken. Do not poke keys on a quoted run.

`hera.yaml` Basement Cage is KeyP3 at address 0x180162 with empty requires. Live sprite 0xE4 sits at (3712,4512), tile (16,52), in room 0x87. At `$C172=00` a stand on tile (16,51) reached it: keys 0→1, hp 24→16, pose (3707,4491), layer 0. `Scan_hera_0x87` was not overwritten. The pin still has keys=0. The count was not spent. SRAM 0x180162 was not read. No type 0x1C door was on the pins, so the yaml small-key door was not reached.

Big key chest is BigKeyP3, address 0xE9E6, live chest 0 in room 0x87, tile (47,52), layer 0, opened=0. yaml requires KeyP3 and canLightTorches. The landing has no path to that tile at either peg. big_key stayed 0. Lamp stayed 1 and magic stayed 0, so this lane did not light a torch.

## Big chest

Big Chest item is MoonPearl, address 0xE9F8, needs BigKeyP3. The scan did not enter that room. On `Whatif_HeraEntry_parallel` the poked big key stayed `0x0020` and `$F357` stayed 0. The chest was not opened. A poke of the pearl byte would not pass this step.

## Mechanics

Crystal switches are sprite 0x1E. A sword hit in play toggles `$C172`. A RAM write of the peg byte was not used. Attribute 0x66 is a down peg and is floor. Attribute 0x67 is a raised peg and is wall. On `HeraEntry` the west alcove row 49, cols 24–25, is 0x67 at peg 00 and 0x66 at peg 01. Room 0x87 crystal (3648,4432) toggles the same byte: the scan pin is peg 01, and peg 00 is the state that reaches the small-key tile. Compass chest (CompassP3, 0xE9FB), the big chest, and Moldorm are the Top region and require BigKeyP3. They were not entered.

## Boss

| Field | Value |
|-------|--------|
| Sprite ids | 0x09 Moldorm, one slot, room 0x07. Mini moldorm 0x18 is not this boss. The heart container sprite is 0xEA. |
| HP byte | `$0E50` = 12 on `Whatif_Hera_moldorm`. Fighter sword damage in the z3 table is 2. |
| Phase byte | No phase byte. Tail index is `$0E80`. After a hit, `$0E10` (aux2) skips the tail check while it is nonzero. Idle samples read 0. |
| What hurts it | Sword on the tail. History lookback 0x30 in the `$1FC00` ring. A B-only `strike_verdict` dropped HP 12 to 10. Lookback 0, the head, did not. |
| When | Placed window: 8–16 px east facing LEFT (east edge starts at 6 only below 3 HP), or 8–16 px south facing UP, and `$0E10` is 0. `dx` 16 with `|dy| >= 2` whiffs. Walked `rel=(-4, 0)` facing UP whiffs. |

z3 name is Moldorm. Location `Tower of Hera - Moldorm` is address 0x180152 and needs BigKeyP3. Recorded strats are Moldorm - Sword (L1Sword) and Moldorm - Hammer. `Whatif_Hera_moldorm` is kind poke, indoors in room 0x07. It is not natural, not continuous, and not a STATUS claim. There is no `Scan_hera_boss` pin. At (3880, 256) Link is 48 px from the nearest pit and 54 px chebyshev from sprite 0x09. Floor is 0x01, hp 24, `$F3CC` is 0. Room 0x31 slot 0 type 0x1E and the pit at (47, 44) were not rewalked. A rollout that changes `$A4` or the room is a floor hit: one north-rim probe loaded room 0x17. `bosses/moldorm.py` walks under `PolicyGuard` and commits a swing only through `strike_if_safe` from that 8–16 px window. `eval_offsets` on this pin with `bosses/moldorm:fight` passed 12 of 12 hearts, each hp_lost 0 and hits empty. Frames: 982, 820, 855, 858, 849, 856, 855, 929, 853, 850, 828, 979. Offsets 3, 5, 6, and 8 branch on `$001A` idle (`0x30+N`) and refuse rim poses. Offset 0 stayed at frame 982. Log `recordings/probe_AP-120/eval_rim.log`. The earlier `eval_restored.log` score of 7 of 12 is not the closed result. Isolated poke. Not natural, not continuous, not a STATUS claim. `$F3CC` was not written. AP-120 is done.

## Prize and exit

Prize is PendantOfPower (location Tower of Hera - Prize; addresses 0x120A5, 0x53F0A, 0x53F0B, 0x18005A, 0x18007A, 0xC706). Leave back to the overworld is the step-8 chain. The south mouth of 0x77 already exits to screen 0x03 at (2288,123). That outdoor screen was not scanned. Leave proof is a RAM glance, not an MP4.

## Save points

Power-on save points only. Name the pin and its predecessor. A `*Entry` pin is development only.

| Save point | Predecessor | What it proves |
|------------|-------------|----------------|
| | | |

`HeraEntry` is a development pin for the isolated lobby hop. It is not a power-on save point. `Scan_hera_0x77` and `Scan_hera_0x87` are provenance kind `scan` from `ap116-hera-scan`. They are not power-on saves.

## Open reds

What is still red. Do not weaken a check to close a row.

| Red | Evidence | Blocker |
|-----|----------|---------|
| 0x77 south type 0x0C | Both pegs exit to screen 0x03 at (2288,123). | Overworld. Death Mountain unentered. |
| 0x77 map pocket | MapP3 tile (31,15) unopened. Stair 0x22 at (31,18) had no hop. Ledge and 0x5F stands failed. | No path from the entrance at either peg. |
| 0x77 east 0x5E | Tiles (45,40), (45,43), and (46,43). | Spawn floor has no path. North component at peg 00 stairs to room 0x31. |
| 0x87 west and north | Slot 0 type 0x00 and slot 1 type 0x00. No path at either peg. | Stopping locks. Far sides unentered. |
| 0x87 other layer | Slot 8 type 0x18 and slot 9 type 0x00. | Landing is layer 0. Far sides unentered. |
| Big key chest | BigKeyP3 tile (47,52) opened=0. Requires KeyP3 and canLightTorches. magic=0. Pin keys=0. | No path from the landing. big_key stayed 0. |
| Small key on the pin | Live stand at peg 00 counted keys 0→1 and was not saved. Pin keys stay 0. No 0x1C door was live. | Key was not taken onto the scan pin and was not spent. |
| Moon Pearl and Moldorm door | `$F357` stayed 0. Room 0x31 big-key door type 0x1E has no path from the stair landing. Rooms 0x17, 0x27, and 0xA7 unentered. | Boss door not isolated. Big chest did not open in play. The room 0x07 fight on `Whatif_Hera_moldorm` is done at 12 of 12. |
