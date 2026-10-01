# Dungeon route — Eastern Palace

Copy of `docs/routes/DUNGEON_TEMPLATE.md`. A save-state pin is not continuous. Facts below are from `dungeon/route_data/eastern.yaml` or a live scan on `EasternPalaceEntry`.

## Entry

| Field | Value |
|-------|--------|
| Pin | EasternPalaceEntry |
| Room | 0xC9 |
| First hop | `room_c9_north_to_0xb9` (map door `north_to_0xb9`, UP, to room 0xB9). verification=isolated. Not natural. |
| Predecessor approach | `rr-eog9.1–.3` closed. A southern detour through `0x25 → 0x2D → 0x2E → 0x26` and outdoor stairs reaches northwest `0x1E` at `(3328, 2047)`. `Play_eog9_3` has power-on Survival replay provenance: 41305f, zero desync/loads/deaths, seven disclosed opening HP refills. Direct east exits from the `0x1D` north lip remain inaccessible. Hut, hedge, and lobby actions have not run from this predecessor. |

The forward edge is the north hall, not `south_to_overworld`. Live door slot 1 is a north shutter (type 0x18) toward 0xB9. The center pot (attribute 0x71) lifts to a 0x23 switch; walking off it opens both mouths (bytes become 0x82). Crossing the center shutter closes the north mouth again until the three 0x4F popos are dead. The push then starts near y=6248 with link x in 4840–4872. A push that is already at y≈6204 wedges. Left and right 0x80 mouths do not change room.

## Requirements

Items the route needs, taken from z3 logic (`eastern.yaml`, region id 78).

| Item | Why | z3 source |
|------|-----|-----------|
| BigKeyP1 | Big chest and Armos Knights | locations Big Chest, Armos Knights |
| hasLightCone | Big key chest and Armos Knights | locations Big Key Chest, Armos Knights |
| canShootArrows | Armos Knights location gate | location Armos Knights; helpers.json name canShootArrows requires Bow |
| Bow | Big chest contents, and the canShootArrows gate | location Big Chest item Bow |

z3 also lists sword, cane, rod, and hammer strats for Armos Knights. The location gate is still canShootArrows, which is Bow. Red eyegore sprite 132 has sword give-damage 0 and arrow 4, and it does not ignore damage, so the bow is the table hit there. Green eyegore sprite 131 takes sword damage 2. Armos Knight sprite 83 has sword byte 4 and arrow byte 16 but `ignores_damage` is set, so that row does not prove a hit. The boss probe measures the fight.

## Room table

One row per room reached with the keys on the scan pins (keys=0, big_key=0). Geometry stays in `alttp/maps/` json. Do not copy approach xy into Python. Dumps: `Scan_eastern_0xc9`, `Scan_eastern_0xb9`, `Scan_eastern_0xa9`.

| Room | Doors + types | Chests | Sprites | Notes |
|------|---------------|--------|---------|-------|
| 0xC9 | south entrance type 0x0A to overworld screen 0x1E; north shutters type 0x18 (center and slot 1); side slots type 0x00 | none | three popo slow 0x4F | Center pot opens the shutter. Side mouths do not change room. Pin Scan_eastern_0xc9. lamp=1 keys=0. |
| 0xB9 | north type 0x00 to 0xA9; south type 0x00 to 0xC9; west slot 3 type 0x00 pos $0784, no layer-0 floor path, not entered; east stopping lock slot 0 type 0x1C pos $07F4 tile (58,15), keys=0, not walked | chest 0 OneHundredRupees z3=0xE9B3 tile (50,36) layer 0. A from tile (50,37) xy=(5009,5928) layer 0 set opened=1 and rupees 0→100. Keys stayed 0. A from tile (48,37) xy=(4992,5928) on a reloaded closed chest left opened=0. Do not open it again. | cannonball 0x50 | Pin Scan_eastern_0xb9 lamp=1 keys=0. Layer-1 stair at cols 10-11 lands on the chest balcony (tile (10,37) xy=(4688,5935) layer 0). Four cannonballs on the layer-1 lane. |
| 0xA9 | south type 0x00 from 0xB9; west shutter stopping lock slot 0 type 0x18; east shutter stopping lock slot 1 type 0x18; north stopping lock slot 2 type 0x1E big key, big_key=0; west slot 3 and east slot 4 type 0x40, no layer-0 floor stand | chest 0 Bow z3=0xE97D layer 1 tile (30,26), unopened, no layer-0 floor path | two green eyegores, two stalfos | Pin Scan_eastern_0xa9 landing (4856,5592) lamp=1 keys=0 big_key=0. Four layer-0 pots lifted; keys stayed 0; door types stayed put. Shutter pushes stayed in 0xA9. |

Stopping locks, keys held 0 and big_key 0. The far side of each was not entered.

| Lock | Live door | Why it stops |
|------|-----------|--------------|
| 0xB9 east | slot 0 type 0x1C pos $07F4 tile (58,15) | Engine small-key door. `eastern.yaml` has no small key and no KeyP1. Not walked. |
| 0xA9 north | slot 2 type 0x1E pos $023C | Big-key door. big_key=0. Not walked. |
| 0xA9 west | slot 0 type 0x18 shutter | Push stayed in 0xA9 at the west mouth. |
| 0xA9 east | slot 1 type 0x18 shutter | Push stayed in 0xA9 at the east mouth. |

## Keys and big key

Where each small key is taken, and where the big key is taken. Do not poke keys on a quoted run.

`eastern.yaml` has no small key and no KeyP1 location. The 0xB9 cannonball chest is OneHundredRupees at 0xE9B3; opening it counted rupees 0 to 100 and left keys at 0. The live east door is still type 0x1C. That is a stopping lock, not a key to hunt.

Big key chest is BigKeyP1, address 0xE9B9, ROM chest-table room 0xB8, requires hasLightCone. The region file does not name that room. The scan did not enter 0xB8, and play has not raised big_key. EasternPalaceEntry has `$F34A` lamp=1 and `$F36E` magic=0x00. A lamp use on that pin costs 4, so the cone cannot light.

`Whatif_EasternPalaceEntry_parallel` is an isolated poke (kind poke, room 0xC9, module 0x07). It already holds big_key `$F366=0x2000` and magic `$F36E=128`. That bit did not rise in play. On 2026-09-30 the pin entered 0xA9 at (4856,5592), layer 0, bow 0, arrows `$F377` 0, big_key still 0x2000, hp 24 to 12. North slot 2 stayed type 0x1E. The door was not contacted: no layer-0 path from that landing, and 0xA9 had no reachable two-tile floor pinch. Ledger writes were 0. Not a Clean claim and not an isolated edge.

## Big chest

Big Chest item is Bow, address 0xE97D, needs BigKeyP1. Live chest 0 in 0xA9 is that item on layer 1, tile (30,26). It was not opened. On the what-if pin the layer-0 byte at that tile was 0x1C (stairs) and the layer-1 byte was 0x58 (unknown). Four stands around the chest had no path on layer 0, before a fight and again after. Bow stayed 0. Arrows stayed 0. That chest was not rewalked.

## Mechanics

Lobby shutter: lift the center pot, step on the revealed switch, then the north door opens. Fight the popos only after that shutter is open; they sit beyond the closed door's clear distance. Stalfos bones, if met on the way to the big key, are a projectile note for the arrow work and do not retick AP-064.

`primitives.shoot_arrow` selects the bow with `select_y_item(env, "bow")` (hud slot 0x01, no RAM write), refuses when bow `$F340` is 0, arrow count `$F377` is 0, or doorway `$006C` is set, presses Y once, and accepts only a new ancilla 0x09 or 0x0A plus a lower `$F377`. The draw is 3, 3, then 8 frames. z3 `canShootArrows` requires Bow, so the Armos Knights location is where it is for, and red eyegore sprite 132 takes arrow damage and not sword damage.

`Whatif_EasternPalaceEntry_bow` is an isolated poke of that parallel pin: bow `$F340=1`, arrows `$F377=30`, kind poke, room 0xC9. The bytes read back as bow=1 and arrows=30. The spawn (4856,6616) is doorway `$006C=1`, and Y there does not start the draw. Seventeen UP frames stand at (4856,6590), still room 0xC9, door 0. `shoot_arrow` then made ancilla 0x09 at (4860,6594) and arrows 30→29. The menu set bow 1→2 in play. `$F3CC` stayed 5. Not a chest prize and not a boss-door edge.

## Boss

| Field | Value |
|-------|--------|
| Sprite ids | `0x53` Armos Knight, six slots, room `0xC8` |
| HP byte | `$0E50` per slot, starts at 48 |
| Phase byte | `$0D80` per slot, starts at 0. `$0D90` becomes 1 at idle frame 242. |
| What hurts it | Arrow drops `$0E50` by 16. A `strike_if_safe` sword swing drops it by 4. |
| When | Still spawn xy: an aligned arrow turns ancilla `0x09` into halted `0x0A` and HP stays 48. Sword swings from the closest stand (slot 5, Link y=6549) are whiffs. After the knight leaves that xy, the arrow drop lands and `$0EF0` is set (sampled 23). The first committed sword hit was approach frame 238. |

`Whatif_Eastern_armos` is an isolated poke (kind poke) indoors in `0xC8`, module `0x07`, dungeon 4, bow `$F340=1`, arrows `$F377=30`, sword 1, bottles 0, `$F3CC` stayed 5. Not a Scan pin and not continuous. The pose (4448, 6560) is floor tile (44, 52) and doorway `$006C=1`; a short move clears the doorway. Idle 1800 frames on that pin: Link HP 24→20, all six knight HP stayed 48, no new sprite type. Five slots took `$0D80` from 0 to 1 between frames 293 and 327. Slot 4 stayed 0 for the whole idle window.

z3 name is Armos Knights. Location gate needs canShootArrows, hasLightCone, and BigKeyP1. Prize is PendantOfCourage. The table row sets `ignores_damage`, so the sword byte 4 and arrow byte 16 are not the proof. The drops above are the probe.

On that same isolated pin, `eval_offsets` of `bosses/armos_knights:fight` passed 12/12 (frames 1708–1756, hp_lost 4–12). Each offset took the heart container (`$F36C` 0x18→0x20). Shots wait until the knight is off its spawn xy. When one knight remained, slot 0 HP returned 16→48 and the arrow still dropped 16. Not continuous.

## Prize and exit

Prize is Pendant of Courage (location Eastern Palace - Prize). Leave back to the overworld is the step-8 chain. Leave proof is a RAM glance, not an MP4.

On `Whatif_Eastern_armos`, one run of the existing fight reached the heart (`$F36C` 0x18→0x20, ledger frame 1718, source=play) with `$F374` still 0x00. Falling-prize ancilla 0x29 was at (4507, 6536). 118 frames later receipt ancilla 0x22 was up, Link was in the hold-up pose, and `$F374` read 0x04. Ledger frame 1835, 0→4, source=play, writes=0. `$F3CC` stayed 5. Room stayed 0xC8. Isolated poke, not the Sanctuary chain, not continuous, and not a STATUS claim.

## Save points

Power-on save points only. Name the pin and its predecessor. A `*Entry` pin is development only.

| Save point | Predecessor | What it proves |
|------------|-------------|----------------|
| | | |

`EasternPalaceEntry` is a development pin for the isolated lobby hop. It is not a power-on save point.

## Open reds

What is still red. Do not weaken a check to close a row.

| Red | Evidence | Blocker |
|-----|----------|---------|
| 0xB9 east type 0x1C | Scan_eastern_0xb9 slot 0 type 0x1C pos $07F4. Keys stayed 0. Cannonball chest is OneHundredRupees, not a key. `eastern.yaml` has no small key. | Stopping lock. Far side not entered. |
| 0xA9 north type 0x1E | Scan_eastern_0xa9 slot 2 type 0x1E. On Whatif_EasternPalaceEntry_parallel the same slot stayed 0x1E. The landing had no path to it. | Stopping lock. Door not contacted. Poke bit 0x2000 is not a key that rose in play. |
| Bow chest | Layer-1 chest (30,26) still unopened. No layer-0 path. Not rewalked. | Prize did not rise in play. |
| 0xA9 shutters type 0x18 | West slot 0 and east slot 1 stayed 0x18. Pushes stayed in 0xA9. | Stopping locks. Far sides not entered. |
| Big key chest room 0xB8 | ROM chest table joins 0xE9B9 to room 0xB8. Requires hasLightCone. Not entered. | Not on a scan pin. |
