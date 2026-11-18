# Full-clear order

Pin `refs/z3-json-data` `1eb7a785bda0d671136316c24f223c7ce12257e6`. Mode `state.standard`, `mods.ohko` off, `swords.swordless` off. A requires list is a conjunction unless the object is `or`, `and`, or `not`. Helpers below are expanded from `helpers.json`. `canDarkNav` (`tech.json`) requires nothing, so `hasLightCone` is Lamp. This file does not boot a ROM.

| Helper | This route uses |
| --- | --- |
| canShootArrows | Bow |
| canDash | PegasusBoots |
| canRead | BookOfMudora |
| canLiftRocks | PowerGlove (or TitansMitt, or ProgressiveGlove) |
| canLiftDarkRocks | TitansMitt |
| canLightTorches | Lamp (or FireRod) |
| canMeltThings | FireRod (or a sword and Bombos) |
| canSlashCurtains | L1Sword via hasSword (`swords.swordless` is off) |
| hasSword | L1Sword |
| isDarkLink | MoonPearl |
| isLightLink | holds, no item |
| canGrapple | Hookshot |
| canMirror | MagicMirror |
| canBlockLasers | MirrorShield |
| canSwim | Flippers (no census row lists it) |
| canFly | OcarinaActive (no node grants it; desert takes the book) |

Leg 0 is the Clean power-on Sanctuary tape in `recordings/clean_escape_20261002/`.
Rows 1–4 remain the only true rows. The current development segment crosses
room `0xB9` from `Play_eog9_7` to the lower channel's clear area; its 396-frame
forward replay has zero desync, loads or writes and HP stays 28. It has not
entered `0xA9` or earned C2. The current endpoint, evidence, invariants and
next blocker live only in [the residual](../../tasks/residual.md).

## Pyramid gate

Ganon's nodeItem is Triforce. The node has no requires list:

```json
{
  "id": 2,
  "name": "Ganon",
  "area": "Pyramid",
  "nodeType": "item",
  "nodeItem": "Triforce"
}
```

`connections/main.json` has no `requires` key. No connection is a crystal gate. Dungeon prize nodes Crystal1, Crystal2, Crystal3, Crystal4, Crystal6, Crystal7, PendantOfCourage, PendantOfWisdom, and PendantOfPower have no requires list. The only prize node with a gate is Ice Palace, and it is not seven crystals:

```json
"requires": [
  "Hammer",
  "canMeltThings",
  "canLiftRocks",
  "BigKeyD5",
  {"or": [
    {"and": ["CaneOfSomaria", "KeyD5"]},
    "KeyD5:2"
  ]}
]
```

The only Crystal tokens on a room are Pyramid Fairy's Crystal5 and Crystal6, not a seven-crystal pyramid gate:

```json
"requires": [
  "hasSword",
  "Crystal5",
  "Crystal6",
  "isDarkLink",
  {"or": [
    "Hammer",
    {"and": ["canMirror", "DefeatAgahnim1"]}
  ]}
]
```

The pin does not encode seven crystals on the pyramid. This order does not invent that gate. No Ganon's Tower census row lists a Crystal token.

## Tokens with no expansion

These are not nodeItems. The census line stays.

| Token | Why it stays |
| --- | --- |
| RescueZelda | No yield in the pin. |
| RescueBlacksmith | Yield of Missing Smith, not a nodeItem. Purple Chest stays. |
| DefeatAgahnim1 | Yield of the Agahnim 1 strat. That strat also requires `swords.swordless`, which is off. Pyramid Fairy takes Hammer instead. |
| DrainedFloodgate | Yield of the Floodgate event, not Floodgate Chest. No census row requires it. |
| `BigKeyD3 ` | Trailing space. FireRod stays. The nodeItem BigKeyD3 does not match. |
| KeyD1:6, KeyD3:3, KeyD5:2, KeyD7:2, KeyD7:3, KeyD7:4, KeyA2:3, KeyA2:4 | Counts, not nodeItems. The Key lines sit above the count. |
| mods.ohko | Inside `not`. Mode is off, so that term holds. |
| QuarterMagic | `canExtendMagic` is HalfMagic and QuarterMagic and Bottle. Spike Cave's or does not clear. |
| CaneOfByrna | Spike Cave's other branch requires the chest's own item. |
| OcarinaActive | `canFly`. No node grants it. Desert uses BookOfMudora, not the fly branch. |
| ProgressiveGlove | Unused `canLiftRocks` alternative. |
| swords.swordless, BugCatchingNet | Swordless Agahnim strat is off. BugCatchingNet is not a nodeItem. |
| MasterSword, L1SwordAndShield | Unused sword alternatives. Not census nodeItems. |

Hammer and BigKeyD1 require each other. Palace of Darkness - Big Chest is the only Hammer nodeItem, and it lists Hammer. Both lines stay. Later Hammer uses sit after that chest line.

## Leg 0, true

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 1 | Link's Uncle | L1Sword | none | true |
| 2 | Hyrule Castle - Zelda's Chest | Lamp | none | true |
| 3 | Sewers - Dark Cross | KeyH2 | none | true |
| 4 | Sanctuary | SanctuaryHeart | KeyH2 | true |

## Frontier

The west drop in room `0xB9` lands on layer 1 at (4808, 6006). Not a census row. The north door is not runnable. Nothing below is true on the Clean tape or on `Play_eog9_7`.

## Eastern Palace

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 5 | Eastern Palace - Compass Chest | CompassP1 | none | |
| 6 | Eastern Palace - Map Chest | MapP1 | none | |
| 7 | Eastern Palace - Cannonball Chest | OneHundredRupees | none | paid earlier; do not press A |
| 8 | Eastern Palace - Big Key Chest | BigKeyP1 | Lamp (hasLightCone) | not-runnable |
| 9 | Eastern Palace - Big Chest | Bow | BigKeyP1 | not-runnable |
| 10 | Eastern Palace - Armos Knights | BossHeartContainer | Bow (canShootArrows), Lamp (hasLightCone), BigKeyP1 | not-runnable |
| 11 | Eastern Palace - Prize | PendantOfCourage | none on the node | |

Big Key Chest is the blocked walk: room `0xB8` not entered, big key not raised in play. Bow and Armos need that key. An isolated Armos 12/12 is not this tape.

## Book, boots, then Desert

AP-113 stays claimed. Lanmolas is not 10 of 12. That fight is not runnable.

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 12 | Library | BookOfMudora | none | |
| 13 | Sahasrahla | PegasusBoots | none | |
| 14 | Sahasrahla's Hut - Left | FiftyRupees | none | |
| 15 | Sahasrahla's Hut - Middle | ThreeBombs | none | |
| 16 | Sahasrahla's Hut - Right | FiftyRupees | none | |
| 17 | Desert Palace - Map Chest | MapP2 | BookOfMudora (canRead), RescueZelda | |
| 18 | Desert Palace - Torch | KeyP2 | PegasusBoots (canDash), BookOfMudora (canRead), RescueZelda | |
| 19 | Desert Palace - Compass Chest | CompassP2 | KeyP2 | |
| 20 | Desert Palace - Big Key Chest | BigKeyP2 | KeyP2 | |
| 21 | Desert Palace - Big Chest | PowerGlove | BigKeyP2, BookOfMudora (canRead), RescueZelda | |
| 22 | Desert Palace - Lanmolas | BossHeartContainer | PowerGlove (canLiftRocks), Lamp (canLightTorches), BigKeyP2, KeyP2 | not-runnable |
| 23 | Desert Palace - Prize | PendantOfWisdom | none on the node | |

## Light Death Mountain

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 24 | Old Man | MagicMirror | none | |
| 25 | Ether Tablet | Ether | none | |
| 26 | Spectacle Rock Cave | PieceOfHeart | none | |
| 27 | Spectacle Rock | PieceOfHeart | none | |
| 28 | Paradox Cave Lower - Far Left | TwentyRupees | none | |
| 29 | Paradox Cave Lower - Left | TwentyRupees | none | |
| 30 | Paradox Cave Lower - Right | TwentyRupees | none | |
| 31 | Paradox Cave Lower - Far Right | TwentyRupees | none | |
| 32 | Paradox Cave Lower - Middle | TwentyRupees | none | |
| 33 | Paradox Cave Upper - Left | ThreeBombs | none | |
| 34 | Paradox Cave Upper - Right | TenArrows | none | |
| 35 | Spiral Cave | FiftyRupees | none | |
| 36 | Mimic Cave | PieceOfHeart | none | |
| 37 | Floating Island | PieceOfHeart | none | |

## Tower of Hera

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 38 | Tower of Hera - Basement Cage | KeyP3 | none | |
| 39 | Tower of Hera - Map Chest | MapP3 | none | |
| 40 | Tower of Hera - Big Key Chest | BigKeyP3 | Lamp (canLightTorches), KeyP3 | |
| 41 | Tower of Hera - Compass Chest | CompassP3 | BigKeyP3 | |
| 42 | Tower of Hera - Big Chest | MoonPearl | BigKeyP3 | |
| 43 | Tower of Hera - Moldorm | BossHeartContainer | BigKeyP3 | |
| 44 | Tower of Hera - Prize | PendantOfPower | none on the node | |

## Pedestal, tower keys, Zora

Master Sword Pedestal lists no pendant require. The two KeyA1 chests are the only Castle Tower census rows. There is no Agahnim nodeItem.

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 45 | Master Sword Pedestal | L2Sword | none | |
| 46 | Castle Tower - Room 03 | KeyA1 | Lamp (hasLightCone) | |
| 47 | Castle Tower - Dark Maze | KeyA1 | Lamp (hasLightCone) | |
| 48 | King Zora | Flippers | none | |
| 49 | Zora's Ledge | PieceOfHeart | none | |

## Light World rows the tapes do not already hold

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 50 | Hyrule Castle - Boomerang Chest | BlueBoomerang | none | |
| 51 | Hyrule Castle - Map Chest | MapH2 | none | |
| 52 | Sewers - Secret Room - Left | ThreeBombs | KeyH2 | |
| 53 | Sewers - Secret Room - Middle | ThreeHundredRupees | KeyH2 | |
| 54 | Sewers - Secret Room - Right | TenArrows | KeyH2 | |
| 55 | Secret Passage | Lamp | none | |
| 56 | Link's House | Lamp | none | |
| 57 | Mushroom | Mushroom | none | |
| 58 | Bottle Merchant | Bottle | none | |
| 59 | Flute Spot | OcarinaInactive | none | |
| 60 | Sunken Treasure | PieceOfHeart | none | |
| 61 | Potion Shop | Powder | none | |
| 62 | Blind's Hideout - Top | PieceOfHeart | none | |
| 63 | Blind's Hideout - Left | TwentyRupees | none | |
| 64 | Blind's Hideout - Right | TwentyRupees | none | |
| 65 | Blind's Hideout - Far Left | TwentyRupees | none | |
| 66 | Blind's Hideout - Far Right | TwentyRupees | none | |
| 67 | King's Tomb | Cape | none | |
| 68 | Kakariko Tavern | Bottle | none | |
| 69 | Chicken House | BlueBoomerang | none | |
| 70 | Kakariko Well - Top | PieceOfHeart | none | |
| 71 | Kakariko Well - Left | TwentyRupees | none | |
| 72 | Kakariko Well - Middle | TwentyRupees | none | |
| 73 | Kakariko Well - Right | TwentyRupees | none | |
| 74 | Kakariko Well - Bottom | ThreeBombs | none | |
| 75 | Magic Bat | HalfMagic | none | |
| 76 | Sick Kid | Bottle | none | |
| 77 | Lost Woods Hideout | PieceOfHeart | none | |
| 78 | Lumberjack Tree | PieceOfHeart | none | |
| 79 | Graveyard Cave | PieceOfHeart | none | |
| 80 | Bonk Rock Cave | PieceOfHeart | none | |
| 81 | Aginah's Cave | PieceOfHeart | none | |
| 82 | Cave 45 | PieceOfHeart | none | |
| 83 | Checkerboard Cave | PieceOfHeart | none | |
| 84 | Mini Moldorm Cave - Far Left | ThreeBombs | none | |
| 85 | Mini Moldorm Cave - Left | TwentyRupees | none | |
| 86 | Mini Moldorm Cave - Right | TwentyRupees | none | |
| 87 | Mini Moldorm Cave - Far Right | TenArrows | none | |
| 88 | Mini Moldorm Cave - Generous Guy | ThreeHundredRupees | none | |
| 89 | Ice Rod Cave | IceRod | none | |
| 90 | Hobo | Bottle | none | |
| 91 | Lake Hylia Island | PieceOfHeart | none | |
| 92 | Maze Race | PieceOfHeart | none | |
| 93 | Desert Ledge | PieceOfHeart | none | |
| 94 | Waterfall Fairy - Left | RedBoomerang | none | |
| 95 | Waterfall Fairy - Right | FireShield | none | |
| 96 | Floodgate Chest | ThreeBombs | none | |

## Palace of Darkness

Shooter Room's KeyD1 satisfies the later `or` of KeyD1 versus Hammer and Bow. Six KeyD1 rows sit above Helmasaur. The count token KeyD1:6 is not itself a nodeItem.

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 97 | Palace of Darkness - Shooter Room | KeyD1 | MoonPearl (isDarkLink), RescueZelda | |
| 98 | Palace of Darkness - The Arena - Ledge | KeyD1 | Bow (canShootArrows) | |
| 99 | Palace of Darkness - Map Chest | MapD1 | Bow (canShootArrows) | |
| 100 | Palace of Darkness - The Arena - Bridge | KeyD1 | KeyD1 | |
| 101 | Palace of Darkness - Stalfos Basement | KeyD1 | KeyD1 | |
| 102 | Palace of Darkness - Big Key Chest | BigKeyD1 | Bow (canShootArrows), Lamp (hasLightCone), Hammer | cycle |
| 103 | Palace of Darkness - Big Chest | Hammer | BigKeyD1, Hammer, Bow (canShootArrows), Lamp (hasLightCone) | cycle |
| 104 | Palace of Darkness - Compass Chest | CompassD1 | Bow (canShootArrows), Lamp (hasLightCone), Hammer | |
| 105 | Palace of Darkness - Dark Basement - Left | Arrow | Bow (canShootArrows), Lamp (hasLightCone), Hammer | |
| 106 | Palace of Darkness - Dark Basement - Right | KeyD1 | Bow (canShootArrows), Lamp (hasLightCone), Hammer | |
| 107 | Palace of Darkness - Dark Maze - Top | ThreeBombs | Bow (canShootArrows), Lamp (hasLightCone), Hammer | |
| 108 | Palace of Darkness - Dark Maze - Bottom | KeyD1 | Bow (canShootArrows), Lamp (hasLightCone), Hammer | |
| 109 | Palace of Darkness - Harmless Hellway | FiveRupees | Bow (canShootArrows), Lamp (hasLightCone), Hammer | |
| 110 | Palace of Darkness - Helmasaur | BossHeartContainer | Bow (canShootArrows), Lamp (hasLightCone), BigKeyD1, Hammer, KeyD1:6 | |
| 111 | Palace of Darkness - Prize | Crystal1 | none | |

## Medallions, then Swamp

Quake is the medallion row that has a require. Ether is already above. Swamp chest rows do not list Flippers or DrainedFloodgate.

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 112 | Bombos Tablet | Bombos | none | |
| 113 | Catfish | Quake | MoonPearl (isDarkLink), PowerGlove (canLiftRocks) | |
| 114 | Swamp Palace - Entrance | KeyD2 | none | |
| 115 | Swamp Palace - Map Chest | MapD2 | KeyD2 | |
| 116 | Swamp Palace - Big Key Chest | BigKeyD2 | Hammer, KeyD2 | |
| 117 | Swamp Palace - Compass Chest | CompassD2 | Hammer, KeyD2 | |
| 118 | Swamp Palace - West Chest | TwentyRupees | Hammer, KeyD2 | |
| 119 | Swamp Palace - Big Chest | Hookshot | Hammer, BigKeyD2, KeyD2 | |
| 120 | Swamp Palace - Flooded Room - Left | TwentyRupees | Hookshot (canGrapple), Hammer, KeyD2 | |
| 121 | Swamp Palace - Flooded Room - Right | TwentyRupees | Hookshot (canGrapple), Hammer, KeyD2 | |
| 122 | Swamp Palace - Waterfall Room | TwentyRupees | Hookshot (canGrapple), Hammer, KeyD2 | |
| 123 | Swamp Palace - Arrghus | BossHeartContainer | Hookshot (canGrapple), Hammer, KeyD2 | |
| 124 | Swamp Palace - Prize | Crystal2 | none | |

## Skull Woods

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 125 | Skull Woods - Pinball Room | KeyD3 | none | |
| 126 | Skull Woods - Compass Chest | CompassD3 | none | |
| 127 | Skull Woods - Pot Prison | KeyD3 | none | |
| 128 | Skull Woods - Big Key Chest | BigKeyD3 | none | |
| 129 | Skull Woods - Big Chest | FireRod | `BigKeyD3 ` | |
| 130 | Skull Woods - Map Chest | MapD3 | MoonPearl (isDarkLink), RescueZelda | |
| 131 | Skull Woods - Bridge Room | KeyD3 | MoonPearl (isDarkLink), FireRod | |
| 132 | Skull Woods - Mothula | BossHeartContainer | L1Sword (canSlashCurtains), MoonPearl (isDarkLink), FireRod, KeyD3:3 | |
| 133 | Skull Woods - Prize | Crystal3 | none | |

## Thieves, then the smith yield

Blacksmith's nodeItem row lists no requires. Missing Smith, same hut, yields RescueBlacksmith and requires TitansMitt. Purple Chest stays on that yield.

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 134 | Thieves' Town - Big Key Chest | BigKeyD4 | MoonPearl (isDarkLink), RescueZelda | |
| 135 | Thieves' Town - Map Chest | MapD4 | MoonPearl (isDarkLink), RescueZelda | |
| 136 | Thieves' Town - Compass Chest | CompassD4 | MoonPearl (isDarkLink), RescueZelda | |
| 137 | Thieves' Town - Ambush Chest | TwentyRupees | MoonPearl (isDarkLink), RescueZelda | |
| 138 | Thieves' Town - Blind's Cell | KeyD4 | BigKeyD4 | |
| 139 | Thieves' Town - Attic | ThreeBombs | BigKeyD4, KeyD4 | |
| 140 | Thieves' Town - Big Chest | TitansMitt | Hammer, BigKeyD4, KeyD4 | |
| 141 | Thieves Town - Blind | BossHeartContainer | BigKeyD4, KeyD4 | |
| 142 | Thieves Town - Prize | Crystal4 | none | |
| 143 | Blacksmith | L3Sword | none on the node | |
| 144 | Purple Chest | Bottle | RescueBlacksmith, isLightLink | |

## Ice Palace

AP-168 stays claimed. Kholdstare's heart was not taken. The big-key bit did not rise in play. Those are not runnable. `not mods.ohko` holds, so the hookshot and `canInvul` are spare. Crystal5 uses KeyD5:2, not the CaneOfSomaria branch.

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 145 | Ice Palace - Compass Chest | CompassD5 | none | |
| 146 | Ice Palace - Iced T Room | KeyD5 | none | |
| 147 | Ice Palace - Spike Room | KeyD5 | not mods.ohko | |
| 148 | Ice Palace - Freezor Chest | ThreeBombs | FireRod (canMeltThings) | |
| 149 | Ice Palace - Big Key Chest | BigKeyD5 | Hammer, PowerGlove (canLiftRocks), not mods.ohko | not-runnable |
| 150 | Ice Palace - Map Chest | MapD5 | Hammer, PowerGlove (canLiftRocks), not mods.ohko | |
| 151 | Ice Palace - Big Chest | BlueMail | BigKeyD5 | not-runnable |
| 152 | Ice Palace - Kholdstare | BossHeartContainer | none | not-runnable |
| 153 | Ice Palace - Prize | Crystal5 | Hammer, FireRod (canMeltThings), PowerGlove (canLiftRocks), BigKeyD5, KeyD5:2 | not-runnable |

## Dark World side rows

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 154 | Pyramid | PieceOfHeart | none | |
| 155 | Hype Cave - Top | TwentyRupees | none | |
| 156 | Hype Cave - Middle Right | TwentyRupees | none | |
| 157 | Hype Cave - Middle Left | TwentyRupees | none | |
| 158 | Hype Cave - Bottom | TwentyRupees | none | |
| 159 | Hype Cave - Generous Guy | ThreeHundredRupees | none | |
| 160 | Stumpy | Shovel | none | |
| 161 | Digging Game | PieceOfHeart | none | |
| 162 | Brewery | ThreeHundredRupees | none | |
| 163 | C-Shaped House | ThreeHundredRupees | none | |
| 164 | Chest Game | PieceOfHeart | none | |
| 165 | Mire Shed - Left | PieceOfHeart | MoonPearl (isDarkLink) | |
| 166 | Mire Shed - Right | TwentyRupees | MoonPearl (isDarkLink) | |
| 167 | Bumper Cave Ledge | PieceOfHeart | PowerGlove (canLiftRocks), Cape | |
| 168 | Peg Cave | PieceOfHeart | TitansMitt (canLiftDarkRocks), Hammer | |
| 169 | Superbunny Cave - Top | ThreeBombs | MoonPearl (isDarkLink) | |
| 170 | Superbunny Cave - Bottom | TwentyRupees | MoonPearl (isDarkLink) | |
| 171 | Hookshot Cave - Top Right | FiftyRupees | MoonPearl (isDarkLink), Hookshot | |
| 172 | Hookshot Cave - Top Left | FiftyRupees | MoonPearl (isDarkLink), Hookshot | |
| 173 | Hookshot Cave - Bottom Right | FiftyRupees | MoonPearl (isDarkLink), PegasusBoots (canDash) | |
| 174 | Hookshot Cave - Bottom Left | FiftyRupees | MoonPearl (isDarkLink), Hookshot | |
| 175 | Spike Cave | CaneOfByrna | PowerGlove (canLiftRocks), MoonPearl, Hammer; or(canExtendMagic and Cape, (mods.ohko or canExtendMagic) and CaneOfByrna). QuarterMagic blocks canExtendMagic | kept |

## Misery Mire

Entrance-gap `hasProperMedallion` is not on these rows. Ether, Bombos, and Quake are already above. Bridge Chest supplies KeyD6 before the lobby row, which also lists KeyD6. Vitreous is not 10 of 12. Not runnable.

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 176 | Misery Mire - Bridge Chest | KeyD6 | none | |
| 177 | Misery Mire - Spike Chest | KeyD6 | not mods.ohko | |
| 178 | Misery Mire - Compass Chest | CompassD6 | Lamp (canLightTorches) | |
| 179 | Misery Mire - Big Key Chest | BigKeyD6 | Lamp (canLightTorches) | |
| 180 | Misery Mire - Big Chest | CaneOfSomaria | BigKeyD6 | |
| 181 | Misery Mire - Map Chest | MapD6 | KeyD6, BigKeyD6 | |
| 182 | Misery Mire - Main Lobby | KeyD6 | KeyD6, BigKeyD6 | |
| 183 | Misery Mire - Vitreous | BossHeartContainer | Lamp (hasLightCone), CaneOfSomaria, BigKeyD6 | not-runnable |
| 184 | Misery Mire - Prize | Crystal6 | none | |

## Pyramid Fairy

Hammer side of the or. DefeatAgahnim1 is not used.

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 185 | Pyramid Fairy - Left | L4Sword | L1Sword (hasSword), Crystal5, Crystal6, MoonPearl (isDarkLink), Hammer | |
| 186 | Pyramid Fairy - Right | SilverArrowUpgrade | L1Sword (hasSword), Crystal5, Crystal6, MoonPearl (isDarkLink), Hammer | |

## Turtle Rock

Two KeyD7 rows sit above KeyD7:2. Crystaroller is the third, above KeyD7:3. Eye Bridge - Bottom Left is the fourth, above KeyD7:4. `canBlockLasers` is MirrorShield, taken before those eyes. Trinexx's heart did not rise on the isolated probe. Not runnable.

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 187 | Turtle Rock - Big Key Chest | BigKeyD7 | none | |
| 188 | Turtle Rock - Compass Chest | CompassD7 | CaneOfSomaria | |
| 189 | Turtle Rock - Roller Room - Left | MapD7 | CaneOfSomaria, FireRod | |
| 190 | Turtle Rock - Roller Room - Right | KeyD7 | CaneOfSomaria, FireRod | |
| 191 | Turtle Rock - Chain Chomps | KeyD7 | KeyD7 | |
| 192 | Turtle Rock - Big Chest | MirrorShield | Hookshot (canGrapple), BigKeyD7, KeyD7:2 | |
| 193 | Turtle Rock - Crystaroller Room | KeyD7 | BigKeyD7, KeyD7:2 | |
| 194 | Turtle Rock - Eye Bridge - Bottom Left | KeyD7 | MirrorShield (canBlockLasers), Lamp (hasLightCone), CaneOfSomaria, BigKeyD7, KeyD7:3 | |
| 195 | Turtle Rock - Eye Bridge - Bottom Right | TwentyRupees | MirrorShield (canBlockLasers), Lamp (hasLightCone), CaneOfSomaria, BigKeyD7, KeyD7:3 | |
| 196 | Turtle Rock - Eye Bridge - Top Left | FiveRupees | MirrorShield (canBlockLasers), Lamp (hasLightCone), CaneOfSomaria, BigKeyD7, KeyD7:3 | |
| 197 | Turtle Rock - Eye Bridge - Top Right | OneRupee | MirrorShield (canBlockLasers), Lamp (hasLightCone), CaneOfSomaria, BigKeyD7, KeyD7:3 | |
| 198 | Turtle Rock - Trinexx | BossHeartContainer | Lamp (hasLightCone), BigKeyD7, KeyD7:4, CaneOfSomaria | not-runnable |
| 199 | Turtle Rock - Prize | Crystal7 | none | |

## Ganon's Tower

Three KeyA2 rows (Firesnake, Bob's Torch, Tile Room) sit above KeyA2:3. Pre-Moldorm is the fourth, above KeyA2:4. The big-key walk stayed blocked (big key 0, room not entered). That walk and every row that lists BigKeyA2 are not runnable. The or used here is Hookshot and Hammer. FireRod and CaneOfSomaria would also satisfy it.

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 200 | Ganons Tower - Hope Room - Left | TenArrows | none | |
| 201 | Ganons Tower - Hope Room - Right | ThreeBombs | none | |
| 202 | Ganons Tower - Compass Room - Top Left | CompassA2 | none | |
| 203 | Ganons Tower - Compass Room - Top Right | OneRupee | none | |
| 204 | Ganons Tower - Compass Room - Bottom Left | TwentyRupees | none | |
| 205 | Ganons Tower - Compass Room - Bottom Right | TenArrows | none | |
| 206 | Ganons Tower - DMs Room - Top Left | ThreeBombs | none | |
| 207 | Ganons Tower - DMs Room - Top Right | TenArrows | none | |
| 208 | Ganons Tower - DMs Room - Bottom Left | TwentyRupees | none | |
| 209 | Ganons Tower - DMs Room - Bottom Right | TwentyRupees | none | |
| 210 | Ganons Tower - Map Chest | MapA2 | none | |
| 211 | Ganons Tower - Firesnake Room | KeyA2 | none | |
| 212 | Ganons Tower - Randomizer Room - Top Left | TenArrows | none | |
| 213 | Ganons Tower - Randomizer Room - Top Right | TenArrows | none | |
| 214 | Ganons Tower - Randomizer Room - Bottom Left | ThreeBombs | none | |
| 215 | Ganons Tower - Randomizer Room - Bottom Right | ThreeBombs | none | |
| 216 | Ganons Tower - Bob's Torch | KeyA2 | PegasusBoots (canDash) | |
| 217 | Ganons Tower - Tile Room | KeyA2 | CaneOfSomaria | |
| 218 | Ganons Tower - Bob's Chest | TenArrows | Hookshot (canGrapple), Hammer, KeyA2:3 | |
| 219 | Ganons Tower - Big Key Room - Left | TenArrows | Hookshot (canGrapple), Hammer, KeyA2:3 | |
| 220 | Ganons Tower - Big Key Room - Right | ThreeBombs | Hookshot (canGrapple), Hammer, KeyA2:3 | |
| 221 | Ganons Tower - Big Key Chest | BigKeyA2 | Hookshot (canGrapple), Hammer, KeyA2:3 | not-runnable |
| 222 | Ganons Tower - Big Chest | RedMail | Hookshot (canGrapple), Hammer, BigKeyA2, KeyA2:3 | not-runnable |
| 223 | Ganons Tower - Mini Helmasaur Room - Left | ThreeBombs | Bow (canShootArrows), Lamp (canLightTorches), BigKeyA2, KeyA2:3 | not-runnable |
| 224 | Ganons Tower - Mini Helmasaur Room - Right | ThreeBombs | Bow (canShootArrows), Lamp (canLightTorches), BigKeyA2, KeyA2:3 | not-runnable |
| 225 | Ganons Tower - Pre-Moldorm Chest | KeyA2 | Bow (canShootArrows), Lamp (canLightTorches), BigKeyA2, KeyA2:3 | not-runnable |
| 226 | Ganons Tower - Validation Chest | TwentyRupees | Hookshot (canGrapple), Bow (canShootArrows), Lamp (canLightTorches), BigKeyA2, KeyA2:4 | not-runnable |

## Triforce, then credits

| n | node | item | requires, expanded | status |
| --- | --- | --- | --- | --- |
| 227 | Ganon | Triforce | none | |

The last leg is the credits. AP-197 asks for a Triforce-room stop plus the credits module. Neither is implemented. There is no credits tape.

227 census rows. Rows 1–4 are true. The frontier is the layer-1 landing after the west drop in room `0xB9`. The north door has no approach.
