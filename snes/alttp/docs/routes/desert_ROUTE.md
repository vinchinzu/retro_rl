# Dungeon route — Desert Palace

Copy of `docs/routes/DUNGEON_TEMPLATE.md`. A save-state pin is not continuous. Facts below are from `dungeon/route_data/desert.yaml` or the isolated hop on `DesertPalaceEntry`. Geometry stays in `maps/room_84.json`.

## Entry

| Field | Value |
|-------|--------|
| Pin | DesertPalaceEntry |
| Room | 0x84 |
| First hop | `room_84` edge `north_to_0x74` (map door `north_to_0x74`, UP, to room 0x74). verification=isolated. Not natural. |

`room_engine.py run room_84 --edge north_to_0x74 --state DesertPalaceEntry` was green 2/2. Both runs: `ok=True`, `frames=638`, landing room 0x74 xy=(2272,4052), indoors. `audit_pins.py --pin DesertPalaceEntry`: hp=0x18/0x18, module=0x07, room=0x0084, dungeon=0x06, 0 incoherent. Straight UP from the spawn wedges on the alcove north wall; the map path goes west around that wall, then back to the hall. This pin is a development load, not a power-on entry.

## Requirements

Items the route needs, taken from z3 logic (`desert.yaml`).

| Item | Why | z3 source |
|------|-----|-----------|
| BigKeyP2 | Big chest and Lanmolas | locations Big Chest, Lanmolas |
| KeyP2 | Compass chest, big key chest, Lanmolas | locations Compass Chest, Big Key Chest, Lanmolas |
| PowerGlove | Big chest contents | location Big Chest item PowerGlove |
| canLiftRocks | Lanmolas | location Lanmolas |
| canLightTorches | Lanmolas | location Lanmolas |
| canDash | Torch key | location Torch item KeyP2 |

## Room table

Rooms 0x84 and 0x74 are the keys=0 scan pins. Room 0x73 is the isolated poke pin. Geometry stays in `maps/`. Dumps: `Scan_desert_0x84`, `Scan_desert_0x74`, `Scan_desert_ow_0x30`.

| Room | Doors + types | Chests | Sprites | Notes |
|------|---------------|--------|---------|-------|
| 0x84 | south slot 0 type 0x0A entrance to overworld screen 0x30 at (296,3240). North edge to 0x74 is the isolated hop, not a second typed slot. | none | leevers 0x71, beamos 0x61 | Pin Scan_desert_0x84, spawn (2296,4568), lamp=1 boots=1 gloves=0 magic=0 keys=0. West pots are bytes 0x70–0x73. One lift at (2148,4213) let the west aisle cross north into 0x74 at (2095,4052). Keys stayed 0. |
| 0x74 | slot 1 north type 0x00 pos $121C and slot 0 north type 0x00 pos $125C. Both stay in 0x74. Slots 8 and 9 are the other layer of those mouths. | chest 0 MapP2 z3=0xE9B6 tile (31,15) layer 0, opened=0 | debirando pit, debirando, green eyegore, leever | Pin Scan_desert_0x74 is the center landing (2272,4052), hp 20, keys=0. 0x27 statues at cols 19–20 and 43–44 seal that pocket from both mouths. From the west aisle, UP at x=2168 goes through the left mouth and stops on the north wall at (2168,3632), still room 0x74. A pot there lifted; magic went 0→16. Keys stayed 0. Right mouth not stood in. |
| 0x73 | slot 0 north type 0x18 shutter pos $121C tile (14,36). Slot 1 west type 0x00 pos $17C4 tile (34,47). Slot 2 north type 0x00 pos $125C tile (46,36). Slots 8–10 are the other layer. | chest 0 big PowerGlove z3=0xE98F tile (14,14) layer 0 loc=$071C opened=0 | debirando pit 0x63, debirando 0x64, beamos 0x61, leever 0x71, dash item 0x3B, blue soldier 0x41 | Whatif_DesertPalaceEntry_parallel isolated poke. East landing xy=(2020,3906) tile (60,40). West stand xy=(1672,3908) tile (17,40) via the row 48 seam. Center wall cols 27–36 is 0x01. Shutter face tile (15,37)=0xF0. |

Stopping points, keys held 0 and big_key 0.

| Lock | What was measured | Why it stops |
|------|-------------------|--------------|
| Outdoor back | 0x84 south type 0x0A to screen 0x30 (296,3240). Whatif_DesertPalaceEntry_glove (gloves=1, what-if, not play) lands (296,3259). Only loaded rock pile is map16 0x36D/0x36E/0x374/0x375 at (704,3168), attr 0x55. Pad BFS reachable=2. DOWN sticks at (296,3432); that floor's BFS is 354 cells and still has no path (closest (368,3384)). LEFT/RIGHT stay on the pad. UP re-enters 0x84. | Back entrance not walked. The rock was not stood on, so it was not lifted. |
| 0x74 center pocket | 0x27 statues cols 19–20 and 43–44, rows 53–58. A 48-frame push into the east face did not move them. | No layer-0 path from (2272,4052) to either mouth. The west aisle is the path that was walked. |

## Keys and big key

Where each small key is taken, and where the big key is taken. Do not poke keys on a quoted run.

No type 0x1C or 0x1E door was live in 0x84 or 0x74. Keys stayed 0. big_key stayed 0. The 0x84 pot and the 0x74 north-lip pot were not keys. The north-lip pot raised magic from 0 to 16.

Torch location is KeyP2 and needs canDash. Sprite 0x3B (dash item) is at (1856,3680) in room 0x73, north of the open type 0x00 mouth at col 46. Boots $F355=1. Ability $F379=0xF8, dash bit clear. A+UP from (1866,3715) ended (1872,3638) with link_state 0 and keys 0. A+RIGHT from (1850,3664) ended (1935,3664), link_state 0, keys 0. KeyP2 did not rise. Big key chest BigKeyP2 address 0xE9C2 is chest 0 in room 0x75, tile (47,10), opened=0. The west landing there is (2572,3915), tile (1,41). No floor path reaches the chest: center wall cols 27-36. Keys stayed 0. Last-heart refill twice, hp 4 to 24, writes=2. Not a Clean claim. Gloves are 0.

## Big chest

Big Chest item is PowerGlove, address 0xE98F, needs BigKeyP2. Live chest is room 0x73 tile (14,14), loc=$071C, opened=0.

Whatif_DesertPalaceEntry_parallel is an isolated poke (room 0x84, module 0x07). Its inventory writes are big_key $F366=0x1000, keys $F36F=1, and magic $F36E=128. Those bits did not rise in play. On that pin $F354 stayed 0. The chest was not pressed: find_path from the west stand xy=(1672,3908) has no path to tile (14,14).

The separator is slot 0, a north shutter, type 0x18, pos $121C. Tile (15,37) stayed 0xF0. Sword contact on sprites 0x41, 0x61, 0x64, and 0x71 left their hp unchanged and the door type unchanged. Five lifts in the south-west pot block (bytes 0x70–0x77) also left the door at 0x18. Row 48, the seam between the east band and that stand, is bytes 0xF9, 0x01, and 0x81. Assist last-heart refill on the pot pass: writes=2, hearts=4.0. Not a Clean claim. No edge to a boss door was isolated. `light_torch` has no live check.

## Mechanics

Book entrance is the pin's mouth (ROM entrance index 0x09 → room 0x84). The two 0x74 mouths are internal: the left one was walked to the north wall without a room change. The outdoor link between the front and the back was not walked. Front landing is screen 0x30. gloves=0 on DesertPalaceEntry and on Whatif_DesertPalaceEntry_parallel. Whatif_DesertPalaceEntry_glove is a what-if copy with gloves=1 (pin_probe read $F354=1). That byte did not rise in play. The only rock pile on the loaded screen-0x30 area is not connected to the south-door floor. The parallel pin is an isolated poke, and the glove copy is an isolated what-if, so the glove, the rocks, and the torches were not taken in play.

## Boss

| Field | Value |
|-------|--------|
| Sprite ids | `0x54` Lanmolas, three slots, room `0x33`. Rocks are `0xC2`. Heart container is `0xEA`. |
| HP byte | `$0E50` per slot, starts at 16. Idle 1200 frames left all three at 16. |
| Phase byte | `$0D80` per slot. 0 wait, 1 telegraph, 2 flight, 3 dive, 4 buried. Slot 0: 0→1 at frame 123, 1→2 at 250, 2→3 at 251, 3→4 at 299, 4→0 at 427. Later flights hold state 2 for about 40 frames. `$0EF0` stayed 0 on the idle pin. |
| What hurts it | Fighter sword drops `$0E50` by 2 while `$0D80` is 2 or 3 and `$0F70` is low. z3 `dmgFromLink` class 1 is 2. State 4 does not take that swing. |
| When | Measured commits on the poke: slot 1 AI 3 `$0F70`=11 dist 20 facing DOWN (16→14); slot 2 AI 3 `$0F70`=15 dist 28 facing RIGHT (16→14); slot 0 AI 2 `$0F70`=0 dist 27 facing RIGHT (16→14). Point-blank swings are hurt or whiff. A swing on the sprite origin during state 4 left HP at 16. |

z3 name is Lanmolas. Location gate needs canLiftRocks, canLightTorches, BigKeyP2, and KeyP2. `Whatif_Desert_lanmolas` is kind poke, indoors room 0x33, module 0x07, dungeon 0x06. Audit: 0 incoherent, hp 0x18/0x18. Source pin `Whatif_DesertPalaceEntry_parallel` was room 0x84. Warp submodule 0x15 kept the previous room and set `$A0` to 0x33. `$F3CC` stayed 0. Link (1677, 1871) is floor tile (17, 41). Three sprite 0x54, state 0x09, HP 16, AI 0, at (1632, 1904), (1680, 1904), and (1648, 1936). Isolated poke. Not natural, not continuous, and not a STATUS claim. There is no `Scan_desert_boss` pin.

Idle 1200 frames on that pin: Link HP 24→16 at frame 329 and 16→12 at frame 919. Rocks `0xC2` born at frame 250 (slots 12–15) and frame 1070 (slots 10–11). Worm HP stayed 16. Recordings: `recordings/probe_AP-111/load_lanmolas.json`, `recordings/probe_AP-111/boss_idle.json`.

Indoor `0xC2` on that idle is four diagonals, not eight rays. `$0D50` is X and `$0D40` is Y; a byte of 28 moves 28px in 16 frames. The box is `link_x` in `[x-11, x+4]` and `link_y` in `[y-15, y-4]` (north-edge hits were HP 24→20). Cardinals are the gap. A head at origin chebyshev 17 with `$0F70` 12 dealt 8.

Entrance 0x0C and screen 0x30 were not rewalked. Those earlier loads stayed room 0x63 or short of the rock at (704, 3168). `recordings/probe_AP-111/load_lair.json` and `back_entrance.json` did not save this pin. Cardinal-gap offset 0 of `bosses/lanmolas:fight` died at frame 2894, pose (1683, 1918) facing LEFT, hp 24→0, hits `boulder:2,lanmolas:2`, after 5 cardinal commits (cross-axis 0–4, long axis 21–24, `$0F70` 1–9). Hurts: f=1028 (1672, 1910) UP 24→16 on a diagonal; f=1134 (1663, 1907) UP 16→8 on a diagonal; f=2164 (1631, 1974) RIGHT 8→4 with a rock 7px west; f=2894 (1683, 1918) LEFT 4→0. The earlier chebyshev fight died at frame 1653 (`boulder:1,lanmolas:2`). An earlier offset-0 sample died at frame 1117 (`lanmolas:3`, 0 commits) and another at frame 6362 (`boulder:3,lanmolas:1`). Not 10 of 12. Log: `recordings/probe_AP-112/offset0.log`.

## Prize and exit

Prize is PendantOfWisdom (location Desert Palace - Prize). Leave back to the overworld is the step-8 chain. Leave proof is a RAM glance, not an MP4.

## Save points

Power-on save points only. Name the pin and its predecessor. A `*Entry` pin is development only.

| Save point | Predecessor | What it proves |
|------------|-------------|----------------|
| | | |

`DesertPalaceEntry` is a development pin for the isolated lobby hop. It is not a power-on save point.

## Open reds

What is still red. Do not weaken a check to close a row.

| Red | Evidence | Blocker |
|-----|----------|---------|
| Outdoor back | Whatif_DesertPalaceEntry_glove gloves=1. Rock (704,3168) attr 0x55 has no path from (296,3259) or from the south stair (296,3432). | Back entrance not walked. |
| Big key | No 0x1C or 0x1E in 0x84 or 0x74. big_key stayed 0. | AP-109 |
| Power Glove | Whatif_DesertPalaceEntry_glove sets $F354=1 as a what-if, not in play. Chest 0xE98F was not opened. Rock (704,3168) was not lifted. | AP-110 |
| Lanmolas | Whatif_Desert_lanmolas is an isolated poke in room 0x33, three sprite 0x54. Cardinal-gap offset 0 died at frame 2894, pose (1683, 1918) facing LEFT, hp 24→0, hits boulder:2,lanmolas:2, after 5 cardinal commits. Entrance 0x0C still loads 0x63. Not natural. | AP-112 |
| Approach | Outdoor control road.state (3216,2344) standing 0x26, book 0, mode 0x09, follower 5. Eastern exits only south. West from 0x2C and 0x34 wraps link_x to 65533 in mode 0x09 submodule 6. 0x2D to 0x25 works; the west step lands a 41-cell ledge (2528,2216) whose exits only return to 0x25, stair portals 0. book_play.state (3643,8504) mode 0x07 book 1 quadrant (0, 2). Reachable local box x 40..200 y 304..488, 196 cells, stair counts 0. The only cells with local y>=476 are two 0x8E iris tiles. room+=0x10 needs a non-0x8E south tile and a west room decrement needs local x<8; neither is reachable. Doors are south type 0x00 at $1D1C and south type 0x2E at $1D5C. DesertPalaceEntry DOWN reaches screen 0x30 pad (296,3259) book 1, then sand (296,3432) mode 0x09. Nearest 0x46 is (296,3464) map16 1802, (304,3464) map16 1803, (296,3472) map16 1804, and (304,3472), with 0x02 at (296,3456) and (304,3456). North hold stays (296,3432). South UP stops at (304,3472); $02ED stays 0 and the (x+8, y+20) sample stays attr 0x00. From that pose, UP does not enter (304,3464). LEFT then UP stops at (296,3472) and does not enter (296,3464). The east step stops at (312,3464) and the west step stays at (280,3464); both leave $02ED at 0 with feet attr 0x00. LEFT from (312,3464) stays there, book 1 mode 0x09 control, $02ED=0 feet attr 0x00, and does not enter (304,3464). RIGHT from (280,3464) stays there, $02ED=0 feet attr 0x00, and does not enter (296,3464). Log recordings/probe_AP-113/side_step.json. Plaque not stood. No book poke. Isolated pin. | AP-113 |
