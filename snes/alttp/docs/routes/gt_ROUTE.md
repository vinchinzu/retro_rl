# Dungeon route — Ganon's Tower

Isolated development route. A save-state pin is not continuous and not natural. Facts are from `refs/z3-json-data` (pin `1eb7a785`) or the live scan in `recordings/probe_AP-187/`. Hint lines are not authority. `spine/hops.py` was not edited.

## Entry

| Field | Value |
|-------|--------|
| Pin | `GanonsTowerEntry` (kind poke, room `0x000C`, module `0x07`, dungeon `0x1A`). Not natural. |
| Room | `0x0C`, floor 1, layer 0, spawn `(6392, 472)`. Dark-world flag 0. |
| First hop | `north_stair_to_0x6b` isolated 2/2. Not natural. |

ROM `$02C813` dungeon entrance table, headerless file map used by `dungeon/rom_scan.py`. Index `0x09` reads room word `0x0084` (the desert check). Yaml Hope Room chests `0xEAD9` and `0xEADC` join the chest table to room `0x008C`. That word is not in the entrance table. Palace `0x1A` in `audit_pins.py` (usdasm Ganon's Tower) starts at room `0x0C`. The only index whose room word is `0x000C` is `0x37`. Overworld door `0x36` is a different table: the same file offset's index `0x36` is room word `0x0010`.

`trigger_entrance` wrote `$010E=0x37`, `$010F=0`, `$010C=0x06`, submodule 0, `$00B0=0`, and module `0x06`. It did not write `$A0` or `$F3CC`. Module `0x06` reached indoors, module `0x07`, room `0x000C`, dungeon `0x1A`, control, in 120 frames. Module `0x0F` was not needed. Kit pokes on `HyruleCastleGrounds` (sword 0, shield 0, progress 0, follower 5): sword `$F359=1`, shield `$F35A=1`, progress `$F3C5=1`. Follower stayed 5. Keys stayed 0. Big key stayed 0. Crystals stayed 0.

`pin_probe GanonsTowerEntry --doors --chests`: module `0x07`, room `0x000C`, dungeon `0x1A`, floor `0x01`, link `(6392,472)`, no sprites, no chests, slot 0 south type `0x0A` entrance. `audit_pins.py --pin GanonsTowerEntry`: hp `0x18/0x18`, kind poke, 0 incoherent.

`room_engine.py run room_0c --edge north_stair_to_0x6b --state GanonsTowerEntry` was green twice. Both runs: 613 frames, room `0x6B`, `(5880,3123)`, indoors, module `0x07`, dungeon `0x1A`, floor 2, dark-world flag 0. Logs: `recordings/probe_AP-187/hop1.json` and `hop2.json`. The north stair has no door slot. Slot 0 is the south entrance. DOWN from the spawn settles outdoors at `(2296, 59)`, module `0x09`, dungeon `0xFF`. The screen id was not read. That exit is not the forward hop.

## Requirements

From `dungeon/route_data/gt.yaml`. Entrance doors require nothing. These items are on later nodes. None of them were poked onto `GanonsTowerEntry`.

| Item | Why | z3 source |
|------|-----|-----------|
| `canDash` | Bob's Torch key | Ganons Tower - Bob's Torch `0x180161` |
| CaneOfSomaria | Tile-room key, and one big-key path | Tile Room; Bottom chests |
| Hammer | One big-key path, with `canGrapple` | Bottom chests |
| FireRod | The other big-key path, with CaneOfSomaria | Bottom chests |
| `canGrapple` | One big-key path, validation chest, Agahnim 2 | Bottom; Agahnim 2 |
| `canShootArrows` | Mini-helmasaur chests, pre-Moldorm key, Agahnim 2 | Before Moldorm; Agahnim 2 |
| `canLightTorches` | Same upper rooms | Before Moldorm; Agahnim 2 |
| `KeyA2` | Count 4 on the prize | Agahnim 2 |
| `BigKeyA2` | Big chest, upper chests, Agahnim 2 | Big Chest `0xEAD6`; Agahnim 2 |

Chest-table rooms already joined, not yet walked: Hope `0x8C`, tile `0x8D`, compass `0x9D`, DMs `0x7B`, map `0x8B`, firesnake `0x7D`, randomizer `0x7C`, big chest `0x8C` (big bit), big key `0x1C`, helmasaur and pre-Moldorm `0x3D`, validation `0x4D`.

## Room table

Geometry stays in `maps/room_0c.json` and `maps/room_6b.json`. Do not copy approach xy into Python. Keys held on the scan pins: 0. Big key field: 0. These are the rooms that walk reached. `Scan_gt_*` pins are isolated, not natural.

| Room | Doors + types | Chests | Sprites | Notes |
|------|---------------|--------|---------|-------|
| `0x0C` | south slot 0 type `0x0A` entrance; north stair has no slot | none | none | Floor 1. `pin_0x0c.txt`. |
| `0x6B` | slot 0 north `0x1E` big_key; slots 1 and 3 north `0x18` shutter; slot 2 west `0x18` shutter; slot 9 south `0x00` regular and slots 10–11 shutters on the other layer | none | two crystal switches, green eyegore, red eyegores, antifairy, statue, two spike blocks, two beamos | Floor 2. `pin_0x6b.txt`. |

Stopping locks, keys 0 and big key 0:

- `0x6B` slot 0 big key. The `0xF0` graphic at `(47, 5)` is walled off from the landing corridor. Not entered.
- `0x6B` slots 1 and 3, north shutters. Door-family tiles sit at `(47, 27)` and `(15, 27)`, across the row 27 wall. Not entered.
- `0x6B` slot 2, west shutter, and the other-layer slots 9–11. Landing is layer 0. Not entered.

The lobby side passages are not locks and not new rooms. West `0x3E` `(21, 47)` runs submodule `0x10` and stops on wall `0x04` at `(21, 9)`, `(6312, 72)`, still `0x0C`. East stops at `(41, 9)`, `(6472, 72)`. Center UP crosses the row 49 ledge and reaches tile `0x1E` at `(31, 15)`, still `0x0C`. The known north stair is the only lobby exit into the tower.

## Keys and big key

Not taken. Both scan pins keep keys `0` and big key `0`. No chest in `0x0C` or `0x6B`.

Two tries to make the big key rise, both on the keys and sword already on the pin:

1. Walk the lobby sides, the center, and the `0x6B` corridor. No new room. Big key stays 0.
2. `hit_switch` on the crystal at `(5808, 3136)` changes `$C172` from `00` to `01` in play (60 frames). Door slots stay the same types. `0x66` and `0x67` counts stay 56 and 84. The south walk then dies: module `0x12` at `(5824, 3224)`, hp 0. Big key stays 0.

`fight_nearby` on eyegores `0x83` and `0x84` from `Scan_gt_0x6b` dies at 132 frames, module `0x12`, defeated none. Door slots stay `0x1E` and `0x18`. Keys stay 0. Big key stays 0.

Room `0x1C` (yaml big key chest `0xEAF1`) was not entered. The bit was not poked.

## Big chest

Yaml `Ganons Tower - Big Chest` item `RedMail` address `0xEAD6`, region Ganons Tower (Bottom). The chest table joins that offset to room `0x008C` with the big-chest bit set. The same room holds Hope chests `0xEAD9` and `0xEADC`. Not entered. Armor on the entry pin is 0.

Step 4 needs edges to the boss door. z3 has one connection into Ganons Tower (Top): Ganons Tower Big Key Door. From Top the only continuation is Torch Rooms, Before Moldorm, Moldorm Door, Moldorm Gap, then Agahnim 2. Room `0x6B` slot 0 is type `0x1E` and was not entered. Big key stayed 0. Those edges were not isolated. No walk.

## Mechanics

North stair from the lobby changes the room to `0x6B` without a door slot. South slot `0x0A` leaves indoors. The lobby has no shutter and no key door.

`0x6B` has a crystal switch. Swording it toggles `$C172` and does not open the big-key door or the shutters. The antifairy at `(5828, 3228)` drains hp on the south walk. That walk ends in module `0x12` before a shutter mouth.

## Boss

Yaml prize node `Agahnim 2` has no room word and no address. z3 enemy `Agahnim` is sprite type `0x7A`. The `$09:D62E` list has that type in two rooms, each one record: `0x0D` and `0x20`, both at list `(y, x) = (0x15, 0x07)`. Those rooms share one object pointer. Palace `0x1A` lists neither. `audit_pins` files `0x0D` under palace `0x08`. `0x20` is in no palace list.

`GanonsTowerEntry` is the source pin (kind poke, room `0x000C`, dungeon `0x1A`, follower 5). Warp submodule `0x15` kept `$A2` on the lobby and set `$A0` to the candidate. `$F3CC` was not written. Dungeon stayed `0x1A`. Isolated poke. Not natural, not continuous, and not a STATUS claim. No `custom_integrations` pin was written. `bosses/agahnim.py` was not edited.

Text on the warp is message `0x013F`, module `0x0E`, submodule 2. Two settled A taps (`$1CD4==3`, `$1CE9==0`) leave that box. A held A never leaves it (`load_agahnim_hold_a.json`).

Two fight tries, 1800 frames each, both rooms, module `0x07` after the taps. Log: `recordings/probe_AP-191/load_agahnim.json`.

1. Idle. One sprite `0x7A` slot 0, `$0E50` 96. `$0D80` cycles 2, 3, 4, 5. Five `0x7B` balls. No second `0x7A`. No lightning.
2. B taps and a step toward that body. Same one `0x7A`. `$0E50` stays 96. No second `0x7A`.

No sprite field separated a clone. A second body was not on screen. Room `0x0D` audit: room belongs to dungeon `0x08` while the pin dungeon is `0x1A`. Room `0x20` audit issues were empty. Follower stayed 5.

## Prize and exit

Not taken. Agahnim 2 requires `canGrapple`, `canShootArrows`, `canLightTorches`, `BigKeyA2`, and `KeyA2:4`.

## Save points

| Save point | Predecessor | What it proves |
|------------|-------------|----------------|
| `GanonsTowerEntry` | `HyruleCastleGrounds` via `$010E=0x37` module `0x06` | Isolated lobby. Kind poke. Not a power-on save. |
| `Scan_gt_0x0c` | `GanonsTowerEntry` | Isolated scan copy of the lobby. Not natural. |
| `Scan_gt_0x6b` | north stair from that lobby | Isolated scan of room `0x6B`. Not natural. |

## Open reds

| Red | Evidence | Blocker |
|-----|----------|---------|
| Big key | Field stays 0. Room `0x1C` not entered. | Zero-key walks and the crystal toggle. |
| Boss door | Edges not isolated. | `0x6B` slot 0 type `0x1E`. Only z3 path into Top. |
| Natural entry | Pin is a module poke | AP-193 needs AP-186 and seven crystals in play |
| Agahnim clones | Sprite-list rooms `0x0D` and `0x20`, idle and sword. One `0x7A`. | Second body did not spawn. Tower lane owns `bosses/agahnim.py` |
