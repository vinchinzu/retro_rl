# Item route

Authority: `refs/z3-json-data` pin `1eb7a785bda0d671136316c24f223c7ce12257e6`
(the refs checkout `alttp.z3_json_data.resolve_data_root` selects). Mode assumed
below: `state.standard`, `mods.ohko` off, `swords.swordless` off, `isLink`
true. A requires list is a conjunction unless the JSON object is `or`, `and`,
or `not`. Overworld connections in this pin carry no item requires. A door
whose node has no `requires` has no item gate in the pin.

AP-006 played the Tompa movie to frame 2800 with the name menu still up and
no Sanctuary. That movie is not an order source. Where two legs have no
encoded edge, the order below is the phase map (rungs C2–C14 and AP-089–098).

Each Phase H approach task has a row here. The row id is the task id:
AP-105, AP-113, AP-121, AP-129, AP-137, AP-145, AP-153, AP-161, AP-169,
AP-177, AP-185, AP-193.

## Helpers used below

From `helpers.json` and `tech.json` at this pin.

| Token | Expands to |
|---|---|
| `canRead` | BookOfMudora |
| `canDash` | PegasusBoots |
| `canSwim` | Flippers |
| `canMirror` | MagicMirror |
| `canFly` | OcarinaActive |
| `canGrapple` | Hookshot |
| `canShootArrows` | Bow |
| `canHammerThings` | Hammer |
| `canBlockLasers` | MirrorShield |
| `canLightTorches` | FireRod or Lamp |
| `hasLightCone` | Lamp (`canDarkNav` requires nothing) |
| `canLiftDarkRocks` | TitansMitt |
| `canLiftRocks` | TitansMitt or PowerGlove (or ProgressiveGlove) |
| `canActivateMedallions` | a sword (`hasSword`) |
| `canMeltThings` | FireRod, or a sword and Bombos |
| `hasSword` | L1Sword, L1SwordAndShield, or `hasMagicalSword` |
| `hasMagicalSword` | L2Sword, L3Sword, L4Sword, or MasterSword |
| `canSlashCurtains` | a sword, under the mode above |
| `hasProperMedallion` | a sword and Bombos and Ether and Quake |
| `isDarkLink` | MoonPearl, in `state.standard` |
| `isLightLink` | holds in `state.standard` with no Moon Pearl |
| `canInvul` | Cape and CaneOfByrna |

`hasProperMedallion` carries the pin's own note: "FIXME: This is an
unintelligent function at the moment."

`RescueZelda` is required on the Desert, PoD, Swamp, Skull, Thieves, Ice,
Mire, and Turtle Rock nodes listed below. The only `yields` in this pin are
`DefeatAgahnim1` (one Agahnim strat), `DrainedFloodgate` (Dam / Floodgate),
and `RescueBlacksmith` (Missing Smith). `RescueZelda` has no yield.

## Prize bytes

`nodeItem` is the pin's vanilla item string. The last address on each prize
node was read as a file offset in `roms/zelda3.sfc` (1,048,576 bytes, no
header). `items.json` `dungeonPrizes` entries end in Courage `0x01`, Power
`0x02`, Wisdom `0x03`, and every crystal `0x06`.

| Prize node | nodeItem | File offset | Byte |
|---|---|---|---|
| Eastern | PendantOfCourage | `0xC6FE` | `0x01` |
| Desert | PendantOfWisdom | `0xC6FF` | `0x02` |
| Swamp | Crystal2 | `0xC701` | `0x06` |
| PoD | Crystal1 | `0xC702` | `0x06` |
| Mire | Crystal6 | `0xC703` | `0x06` |
| Skull | Crystal3 | `0xC704` | `0x06` |
| Ice | Crystal5 | `0xC705` | `0x06` |
| Hera | PendantOfPower | `0xC706` | `0x03` |
| Thieves | Crystal4 | `0xC707` | `0x06` |
| Turtle Rock | Crystal7 | `0xC708` | `0x06` |

Eastern's byte matches Courage. Desert's byte matches the Power entry's last
byte, and Hera's matches the Wisdom entry's last byte. Crystal bytes match
the shared crystal trailing byte. Neighboring file bytes include `0x60` then
`0x00 0x00` before `0xC6FE`, a `0x00` at `0xC700`, and `0xAD` at `0xC709`.
These are the bytes at the listed addresses. The other addresses on the same
nodes (`0x53EF8`, `0x53F1C`, `0x53F0A`, and the `0x1800xx` pair, which sits
past the end of this ROM) are a separate measurement and are not this table.

## Legs

### Escape

Predecessor of the approach rows. Link's Uncle grants L1Sword and requires
nothing. Secret Passage grants Lamp and requires nothing. Link's House grants
Lamp and requires nothing. Sanctuary grants SanctuaryHeart and requires
`KeyH2`. The sewer chest ThreeHundredRupees requires `KeyH2`.

### AP-089 · Pegasus Boots

Sahasrahla grants PegasusBoots and requires nothing. The hut chests are
FiftyRupees, FiftyRupees, and ThreeBombs, each with no requires. The phase
map slots this leg after Eastern's pendant. The pin has no edge from that
pendant to this node. Desert's torch key requires `canDash`, so the boots
sit before AP-113 on this route.

Isolated, not continuous. In room 0x05, with pendant `$F374` already 0x04
and boots 0, the elder receipt raised `$F355` 0 to 1 and ability `$F379`
0xF8 to 0xFC on one play step (ledger frame 962, source=play, no write).
From (2678, 8617), A plus a direction sets `$5D` to 17. The first 4px step
is frame 31; frames 2-30 hold with `$5E` 16 and no motion. The fast step
is 4px/frame. RIGHT into attribute 0x27 stops at (2688, 8617), the same x
a walk stops, then `$5D` 2 rests at (2672, 8617).

### AP-090 · Book of Mudora

Library grants BookOfMudora and requires nothing. The phase-map hint is a
dash into the shelf. The node does not list boots.

Isolated from `probe_AP-089/boots.state` (`$F355` already 1, `$F379` 0xFC,
`$F34E` 0, follower 0x05). Both dashes stayed outdoors and `$F34E` stayed 0.
Screen 0x2B: LEFT from (1720, 2992), 4px/frame from frame 31, bonk state 2
at (1650, 2992), front attribute 0x00, zero west exits, still standing 0x2B.
Screen 0x1B y=1848: LEFT from (1696, 1848), 4px/frame from frame 34, text
stop at (1573, 1848), rest (1574, 1848) state 0, front attribute 0x40, still
standing 0x1B. Room base stayed 0x04 and 0x05.

### AP-105 · Eastern

Approach row for Sanctuary to the Eastern entrance (tracked under epic `rr-eog9`).

The screen graph gives cardinal adjacency, not tile-level walkability.
Sanctuary heart, south exit, and traversal (`rr-eog9.1–.3`) now have replay
evidence. The direct east exits from the `0x1D` north lip remain inaccessible.
A detour through `0x25 → 0x2D → 0x2E → 0x26`, followed by outdoor stairs,
reaches northwest `0x1E` at `(3328, 2047)`. The power-on Survival replay is
41305f with zero desync/loads/deaths and seven disclosed opening HP refills.
`Play_eog9_3` is the next predecessor; hut, hedge, and lobby actions remain
unrun. See [`eastern_ROUTE.md`](eastern_ROUTE.md) and
`recordings/eog9_crossing_20261001/power_on_replay.json`.

1. **Sanctuary interior (room `0x12`):** Collect the Sanctuary Heart Container
   chest (`0xEA79`), raising max HP from 24 to 32 and refilling HP. This child
   is closed on the recorded heart replay.
2. **Sanctuary exit and overworld:** The south exit reaches Screen `0x13` on the
   recorded exit replay. The proved southern detour reaches Screen `0x1E`;
   do not replace it with the direct topological graph path. Roadside cuts
   produced no drops; gathering remains `rr-yx26`.
3. **Sahasrahla's hut (Screen `0x1E`):** Planned only. Enter hut (room `0x05`),
   talk to the Elder, and loot resources only after a predecessor route reaches
   `0x1E` and those actions are verified.
4. **Eastern Palace outdoor approach and entrance:** Planned only. Hedge-maze
   navigation and the `eastern_palace_door_to_0xc9` transition have not run from
   the Sanctuary predecessor. `EasternPalaceEntry` remains a development pin
   for the isolated lobby hop, not proof of natural entry.

Light World door Eastern Palace requires nothing. The big key chest requires
`hasLightCone` (Lamp). The big chest grants Bow and requires `BigKeyP1`. The
Armos Knights node requires `canShootArrows`, `hasLightCone`, and `BigKeyP1`.
Strats on that fight also collect the heart and the prize with L1Sword,
`hasSwordSparkles`, L3Sword, L4Sword, a cane, a rod, or the Hammer, and those
strats do not list the bow. The node require and the strats both stand.

Gives Bow and PendantOfCourage. The prize node requires nothing. The
cannonball chest is OneHundredRupees and requires nothing.

### AP-113 · Desert

Approach row for the book at the entrance tablet.

Overworld Desert Palace doors require nothing. Interior nodes Desert Palace
East Wing, Pots (Outer), Exit (West), and Exit (East) require `RescueZelda`
and (`canRead` or (`canLiftDarkRocks` and `canFly` and `canMirror`)). This
route takes `canRead` (the book). Exit (South) and Exit (North) require
nothing.

The torch chest grants KeyP2 and adds `canDash` to that same book branch.
The big key chest requires `KeyP2`. The big chest grants PowerGlove on that
same branch. The Lanmolas node requires `canLiftRocks`, `canLightTorches`,
`BigKeyP2`, and `KeyP2`. The glove is inside the palace, and the boss node
wants `canLiftRocks` after that chest. Lamp satisfies `canLightTorches`.
Lanmolas strats list a sword, canes, a rod, the bow, the hammer, and bombs.

Gives PowerGlove. The prize node requires nothing. Its `nodeItem` is
PendantOfWisdom. The byte at `0xC6FF` is `0x02`.

### AP-091 · Magic Mirror

Old Man grants MagicMirror and requires nothing. The phase map walks the
Death Mountain old-man cave before Hera. The pin has no edge from the glove
or the lamp to this node, and the Tower of Hera door does not list the mirror.

### AP-121 · Hera

Approach row for Death Mountain to the Tower of Hera entrance.

Death Mountain (Top) door Tower of Hera requires nothing. The basement cage
grants KeyP3 and requires nothing. The big key chest requires
`canLightTorches` and `KeyP3`. The big chest grants MoonPearl and requires
`BigKeyP3`. The Moldorm node requires `BigKeyP3`. Moldorm strats are L1Sword
or Hammer.

Gives MoonPearl, which makes `isDarkLink` true in `state.standard`. The
prize node requires nothing. Its `nodeItem` is PendantOfPower. The byte at
`0xC706` is `0x03`.

### AP-092 · Master Sword

Master Sword Pedestal grants L2Sword and requires nothing. The phase map
collects three pendants first. The node does not list them. L2Sword is one
`hasMagicalSword` alternative.

### AP-129 · Tower

Approach row for Agahnim's Tower.

Hyrule Castle Ledge door Agahnims Tower requires nothing. The Agahnim 1 door
and both KeyA1 chests (Room 03 and Dark Maze) require `hasLightCone`. A
second Agahnim 1 node requires nothing. The only strat named Agahnim 1 yields
`DefeatAgahnim1` and requires `swords.swordless` and (Hammer or
BugCatchingNet), plus `canSlashCurtains`, `hasLightCone`, and `KeyA1:2`.
Under the mode above, that strat's swordless term is false. The phase-map
hint puts the Master Sword on the tower barrier. That sword is absent from
this strat.

### AP-093 · Flippers

King Zora grants Flippers and requires nothing. The pin stores no price.
Swamp's indoor exit requires `canSwim`, so the flippers sit before AP-145.
The charge is in the budget table.

### AP-095 · Ether, Bombos, Quake

Ether Tablet grants Ether and requires nothing. Bombos Tablet grants Bombos
and requires nothing. `canActivateTablets` (book and `hasMagicalSword`) is
defined in `helpers.json` and is not a require on either tablet node.

Catfish grants Quake and requires `isDarkLink` and `canLiftRocks`.
PowerGlove satisfies `canLiftRocks`. MoonPearl satisfies `isDarkLink` in
this mode.

Mire and Turtle Rock call `hasProperMedallion`, so this route reads all three
tablets before those gaps. The helper still requires every medallion at once.
The phase-map hints name Ether at Mire and Quake at Turtle Rock.

### AP-137 · PoD

Approach row for the Pyramid, the paid follower, and the Palace of Darkness
entrance.

East Dark World door Palace of Darkness requires nothing. Indoor nodes Bridge
Room, Bonk Wall, Exit, and the Shooter Room key require `RescueZelda` and
`isDarkLink`. The string `Kiki` appears nowhere in this pin. The payment is
in the budget table.

The big key chest requires `canShootArrows`, `hasLightCone`, and Hammer. The
big chest's `nodeItem` is Hammer and its requires are `BigKeyD1`, Hammer,
`canShootArrows`, and `hasLightCone`. That chest is the only Hammer
`nodeItem` in the pin, and it lists Hammer. The Helmasaur node requires
`canShootArrows`, `hasLightCone`, `BigKeyD1`, `KeyD1:6`, and Hammer.

Gives Crystal1. The prize node requires nothing. Later rows that list Hammer
(Swamp, Thieves' mitt, Ice's prize and big key, Turtle Rock's gap, Ganon's
Tower's hookshot path, the Pyramid Fairy hammer branch) are using this same
token.

### AP-145 · Swamp

Approach row for the Light World dam, then the mirror.

South Dark World door Swamp Palace requires `DrainedFloodgate`. Dam /
Floodgate yields `DrainedFloodgate` and requires nothing. The indoor moat and
the indoor exit require `RescueZelda`, `canMirror`, `canSwim`, and
`isDarkLink`.

The big key chest requires Hammer and `KeyD2`. The big chest grants Hookshot
and requires Hammer, `BigKeyD2`, and `KeyD2`. The Arrghus node requires
`canGrapple`, Hammer, and `KeyD2`.

Gives Hookshot and Crystal2. The prize node requires nothing.

### AP-153 · Skull

Approach row into Skull Woods.

Skull Woods First Section exit, bomb jump, south door, and west door require
`RescueZelda` and `isDarkLink`. The overworld door Skull Woods Final Section
requires nothing. The big key chest requires nothing. The big chest grants
FireRod and requires the token `BigKeyD3 ` with the trailing space. The big
key chest's `nodeItem` is `BigKeyD3` with no trailing space. The bridge-room
key requires `isDarkLink` and FireRod. The Mothula node requires
`canSlashCurtains`, `isDarkLink`, FireRod, and `KeyD3:3`.

Gives FireRod and Crystal3. The prize node requires nothing. FireRod is
inside the dungeon, then the bridge key and the boss list it. The final
section door does not.

### AP-161 · Thieves

Approach row into Thieves' Town.

The exit and the big key chest require `RescueZelda` and `isDarkLink`. The
big chest grants TitansMitt and requires Hammer, `BigKeyD4`, and `KeyD4`.
Blind's cell grants KeyD4 and requires `BigKeyD4`. The Blind node requires
`BigKeyD4` and `KeyD4`.

Gives TitansMitt and Crystal4. The prize node requires nothing.

Missing Smith, on the Light World side of this item, yields
`RescueBlacksmith` and requires `canLiftDarkRocks`. The Purple Chest grants
a Bottle and requires `RescueBlacksmith` and `isLightLink`. No Phase H row
lists that bottle.

### AP-169 · Ice

Approach row to Ice Palace.

Dark Lake Hylia door Ice Palace requires nothing. The indoor entrance room
and the indoor exit require `canLiftDarkRocks`, `canMeltThings`, `canSwim`,
and `RescueZelda`. TitansMitt, Flippers, and a melt source (FireRod from
Skull, or a sword and Bombos) are the external items. The node does not list
`isDarkLink`.

The big key chest requires Hammer, `canLiftRocks`, and (`not mods.ohko` or
`canGrapple` or `canInvul`). With `mods.ohko` off, the `not` term holds, so
the hookshot and `canInvul` are spare. The big chest grants BlueMail and
requires `BigKeyD5`. The Kholdstare node requires nothing. The prize node
grants Crystal5 and requires Hammer, `canMeltThings`, `canLiftRocks`,
`BigKeyD5`, and ((CaneOfSomaria and `KeyD5`) or `KeyD5:2`).

### AP-094 · Flute

Stumpy grants Shovel and requires nothing. That node sits in South Dark
World; the pin does not encode the walk there. Flute Spot grants
OcarinaInactive and requires nothing. `canFly` requires OcarinaActive.
`OcarinaActive` occurs on `canFly` and in `items.json`. No node grants it.
The Mire gap does not list the flute.

### AP-177 · Mire

Approach row for the flute, the portal, and Ether.

Dark Desert door Misery Mire requires nothing. The entrance gap and the exit
require `RescueZelda`, MoonPearl, (`canDash` or `canGrapple`), and
`hasProperMedallion`. This route uses the boots for the dash term. The
medallion term is a sword plus Bombos plus Ether plus Quake, which is the
helper's FIXME above. The gap does not list TitansMitt, the flute, or Ether
alone.

The big key chest requires `canLightTorches`. The big chest grants
CaneOfSomaria and requires `BigKeyD6`. The Vitreous node requires
`hasLightCone`, CaneOfSomaria, and `BigKeyD6`.

Gives CaneOfSomaria and Crystal6. The prize node requires nothing.

### AP-098 · Silver arrows

Side leg the phase map places after Ice and Mire. The Pyramid Fairy room
requires `hasSword`, Crystal5, Crystal6, `isDarkLink`, and (Hammer or
(`canMirror` and `DefeatAgahnim1`)). `DefeatAgahnim1` is the yield of the
swordless Agahnim strat in the Tower row. The left node grants L4Sword and
requires nothing. The right node grants SilverArrowUpgrade and requires
nothing. The pin has no super-bomb token. The charge is in the budget table.

### AP-185 · Turtle Rock

Approach row for the Quake entrance.

Dark Death Mountain door Turtle Rock requires nothing. The entrance gap and
the front exit require `RescueZelda`, `canLiftDarkRocks`,
`hasProperMedallion`, `isDarkLink`, CaneOfSomaria, and Hammer. Turtle Rock
Entrance Gap Reverse requires nothing. The big key chest requires nothing.
The big chest grants MirrorShield and requires (`canGrapple` or
CaneOfSomaria), `BigKeyD7`, and `KeyD7:2`. The Trinexx node requires
`hasLightCone`, `BigKeyD7`, `KeyD7:4`, and CaneOfSomaria. The eye-bridge
bottom-left key requires (`canBlockLasers` or `canInvul`), `hasLightCone`,
CaneOfSomaria, `BigKeyD7`, and `KeyD7:3`.

Gives MirrorShield and Crystal7. The prize node requires nothing. Quake
reaches this gap as one term inside `hasProperMedallion`, together with
Bombos, Ether, and a sword.

### AP-193 · Ganon's Tower

Approach row for the tower opening.

Dark Death Mountain door Ganons Tower requires nothing. No GT node lists a
Crystal token. Bob's Torch grants a KeyA2 and requires `canDash`. The tile
room grants a KeyA2 and requires CaneOfSomaria. The big key chest, Bob's
chest, and the big-key-room chests require (`canGrapple` and Hammer) or
(FireRod and CaneOfSomaria), plus `KeyA2:3`. The big chest adds `BigKeyA2`
and grants RedMail. The mini-helmasaur chests and the pre-Moldorm key require
`canShootArrows`, `canLightTorches`, `BigKeyA2`, and `KeyA2:3`. The Agahnim 2
node and the validation chest require `canGrapple`, `canShootArrows`,
`canLightTorches`, `BigKeyA2`, and `KeyA2:4`. The Moldorm Gap obstacle
requires nothing.

The Pyramid node Ganon grants Triforce and requires nothing. Ganon Drop
requires nothing.

## Budget

Prices are the immediates in `roms/zelda3.sfc` that these routines subtract
from the SRAM wallet `$7EF360`. `ram.py` records rupees at `$F362`, the
display word. Shop prices go through `ShopItem_HandleCost` at file
`0xF739E`: store A and Y as a 16-bit cost, `LDA.l $7EF360`, compare, `SBC`,
`STA.l $7EF360`. Callers pass the cost as `LDA` low, `LDY` high, then
`JSR` that routine. Direct charges are `REP #$20` / `LDA.l $7EF360` /
`SEC` / `SBC.w #imm` / `STA.l $7EF360`. One such site exists for each row
below. z3 stores none of these prices. The King Zora, Kiki, shop, and
super-bomb nodes are either missing or have empty requires.

| Route cost | Rupees | Measurement in this ROM |
|---|---:|---|
| Kiki, agree to follow | 10 | `0xF63F4` loads choice `$1CE8`, then `LDA #$0A` / `LDY #$00` into HandleCost. On success, `ORA #$03` into `$02F2`. |
| Kiki, open the Palace | 100 | `0xF6509` loads choice `$1CE8`, then `LDA #$64` / `LDY #$00` at `0xF651C`. On success, `INC $02E4`. This is the AP-137 payment. |
| King Zora, flippers | 500 | `0x29A94` is the only `SBC.w #$01F4` of `$7EF360`. The bytes before it load choice `$1CE8`. z3's King Zora node grants Flippers. |
| Super bomb | 100 | `0xF61EE` is `LDA #$64` / `LDY #$00` into HandleCost, then `LDA #$0D` / `STA.l $7EF3CC`. The other bomb-shop charge, at `0xF61B0`, is also 100 and stores `#$1B` to bomb count `$7EF375`. |
| Shop type `$07` | 150 | `LDA #$96` at `0xF7183` into HandleCost. Spawn index 7. Empty-bottle check, then receipt id `$2E`. |
| Shop type `$0A` | 10 | `LDA #$0A` at `0xF7296`. Spawn index `$0A`. Refuses when heart count `$7EF36C` equals max `$7EF36D`. Receipt id `$42`. |
| Shop type `$0C` | 50 | `LDA #$32` at `0xF7310`. Spawn index `$0C`. Compares bomb count `$7EF343` with the capacity lookup. Receipt id `$31`. |
| Shop type `$08` | 50 | `LDA #$32` at `0xF720A`. Spawn index 8. Refuses when shield byte `$7EF35A` is already nonzero. Receipt id `$04`. |
| Shop type `$09` | 500 | `LDA #$F4` / `LDY #$01` at `0xF724A`. Spawn index 9. Refuses when `$7EF35A` is 2 or more. Receipt id `$05`. |
| Shop type `$0B` | 30 | `LDA #$1E` at `0xF72CF`. Spawn index `$0B`. Compares arrow count `$7EF377` with the capacity lookup. Receipt id `$44`. |
| Shop type `$0D` | 10 | `LDA #$0A` at `0xF7337`. Spawn index `$0D`. Empty-bottle check, then receipt id `$0E`. |
| Green cauldron | 60 | `0x2F6DA` subtracts `#$003C` from `$7EF360`. The compare just above uses `#$003C`. Receipt id `$2F`. |
| Red cauldron | 120 | `0x2F81C` subtracts `#$0078`. The compare just above uses `#$0078`. Receipt id `$2E` (same receipt id as shop type `$07`, different site and different amount). |
| Blue cauldron | 160 | `0x2F777` subtracts `#$00A0`. The compare just above uses `#$00A0`. Receipt id `$30`. |
| Bottle merchant | 100 | `0x2EB2E` subtracts `#$0064` after `ORA #$02` into `$7EF3C9`. Receipt id `$16`. The offer path also compares `$7EF360` with `#$0064`. z3's Bottle Merchant node grants Bottle and has no price. |

The shopkeeper prep's room-id list is the 13 bytes at file `0x30BE4`:
`0F 10 00 06 18 12 1E FF 1F 23 24 25 27`. The jump table that follows
selects the stock:

| Room id | Stock spawned | Prices |
|---|---|---|
| `$12`, `$FF`, `$1F` | types `$07`, `$0A`, `$0C` (file `0x30C72`) | 150, 10, 50 |
| `$0F` | types `$07`, `$08`, `$0C` (file `0x30C37`, shared tail) | 150, 50, 50 |
| `$10` | types `$09`, `$0D`, `$0B` (file `0x30C43`) | 500, 10, 30 |

z3 names the shop doors (Kakariko Shop, Cave Shop, Dark World Potion Shop,
Village of Outcasts Shop, Big Bomb Shop, and the others) and does not store
these room ids. Dark World Potion Shop's room requires `isDarkLink` and
(`canLiftRocks` or `canSwim` or Hammer). Village of Outcasts Shop's room
requires Hammer. Big Bomb Shop's door requires nothing. The Powder node in
the Light World Potion Shop requires nothing and is a separate grant from
the three cauldron charges.

Chest-game preps in that same room list store a shop-sprite type, and that
type's HandleCost immediate is the price: room `$00` type 4 charges 100
(`0xF7125`), room `$06` type 1 charges 30 (`0xF6FDA`), room `$18` type 3
charges 20 (`0xF70B7`).

Three further wallet subtracts are in this ROM and are not assigned to a
seller here: 20 at `0x282EB`, a 20-rupee compare-and-subtract that returns
carry at `0x32EB6`, and 10 at `0x334D9`.

### Chests near the route

These are z3 `nodeItem` values, the contents of a chest, not a price the
shop routine charges.

| Where | What the node grants | Requires |
|---|---|---|
| Sewer secret room, middle | ThreeHundredRupees | `KeyH2` (escape leg) |
| Eastern cannonball chest | OneHundredRupees | nothing |
| Sahasrahla's hut, left and right | FiftyRupees each | nothing |
| Kakariko Well, three chests | TwentyRupees each | nothing |
| Blind's Hideout, four chests | TwentyRupees each | nothing |
| Mini Moldorm Cave, left and right | TwentyRupees each | nothing |
| Mini Moldorm Cave, generous | ThreeHundredRupees | nothing |
| Paradox Cave, five lower chests | TwentyRupees each | nothing |
| Spiral Cave | FiftyRupees | nothing |
| Brewery | ThreeHundredRupees | nothing |
| C-Shaped House | ThreeHundredRupees | nothing |
| Hype Cave, four chests | TwentyRupees each | room requires `canBombThings`; the chest nodes require nothing |
| Hype Cave, generous | ThreeHundredRupees | same room require |

On the way to King Zora's 500, the escape sewer (300), Eastern (100),
Sahasrahla (100), the Kakariko well (60), and Blind's Hideout (80) are 640
rupees of chest contents in those nodes. Kiki's palace payment is 100.
The super bomb is 100. The standard shop trio is 150 + 10 + 50 when those
three types are bought once.
