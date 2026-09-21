# B1 key pin census — residual (rr-ccxt.10)

State-load RAM glance only. Did not walk rooms. Did not poke `$F3CC` / sword / keys.
Did not STATUS-promote. Did not claim continuous. Did not edit maps or `escape_graph`.

Dump: `recordings/probe_key_pins/key_pin_census.json`
Probe: `scratch/probe_key_pins.py` (every work-queue pin → `settle_control` → leftover).

West-wall band: room `0x81`, `x<=640`, `|y-4168|<=32` (`maps/room_81.json` `west_to_0x80_approach`).
`0x72` north ledge `y<=3776`; lower floor `y>=3900`. Keys `$F36F==0xFF` is HUD blank, not held.

## Honest leftover

**None on 0x81 west wall with keys>=1.** No keys>=1 pin in `0x81` / `0x80` / `0x82` / `0x72` lower floor.

`west_to_0x80` still needs a small key at the cell-door y-band. This sitting did not produce that pin.

## Pins (settled)

61 Zelda3-Snes work-queue pins. Ranked catalog order. Two hold a dungeon key:

| State | Room | xy | keys `$F36F` | follower | hp | band |
|-------|------|----|--------------|----------|----|------|
| `CastleB1Key` | `0x72` | 1320, 3656 | **1** | 0 | 20 | north ledge |
| `CastleB1SecondKey` | `0x71` | 904, 3988 | **1** | 0 | 24 | east pocket |

Eight pins are keys blank (`0xFF`, not dungeon held): `FighterSwordLamp`, `FighterSword`, `Castle_55`, `FirstAction`, `LinksHouseWake`, `HyruleCastleGrounds`, `YazeSlot000`, `CourtyardSecretPocket`. Opening / secret-entrance HUD.

## keys>=1 locations

- **`CastleB1Key`** — same north `0x72` ledge as `CastleB1Guard` (1273, 3665, keys=0). Only physical exit is the F1 stair. Cannot reach `0x81`.
- **`CastleB1SecondKey`** — `0x71` (904, 3988). Last sitting: stuck x≈832–944, cannot walk to the `0x81` door column.

No keys>=1 pin already in `0x81`, especially not west wall ~(608, 4168).

## 0x81 west wall (keys=0)

Only pin in the west-wall band: **`CastleZeldaB1West`** (632, 4152) keys=0, `$F3CC==1` as loaded. Already post-follower; LEFT would still need a key if this were pre-rescue.

`CastleB1FarDoor` is `0x81` (632, 4416) keys=0 — west-ish x, **south** of the cell y-band, so LEFT stays in `0x81`.

Other `0x81` pins (all keys=0): `CastleB1GreenRoom` / `Cleared` / `Done`, `CastleB1FarWest`, `CastleB1WestRoom`, `CastleB1Shutter` / `ShutterGuard` / `ShutterRoom`. Shutter pair is near y=4192 but x≈776–804, not the west wall.

## 0x80 / 0x82 / 0x72 lower

| Room | keys>=1 | Notes |
|------|---------|-------|
| `0x80` | **none** | `CastleZeldaFollower` (352, 4168) keys=0 `$F3CC==1`. B3 pins also load in `0x80` west of the cell bbox, keys=0. |
| `0x82` | **none** | Bridge / south / Zelda-east / island / west-landing all keys=0. |
| `0x72` lower | **none** | `CastleB1Pit` (1272, 4032) keys=0 hp=4; `PitGuardCleared` (1230, 3989) keys=0; `CastleZeldaB1Pit` (1184, 4016) keys=0 `$F3CC==1`. |
| `0x72` mid | **none** | `CastleB1PitFull` (1272, 3872) keys=0 hp=12; `PitCleared` same xy, dead (`$10==0x12`, hp=0). |

## Next

Need a **keys>=1** leftover on the `0x81` west wall (or a measured walk off the `0x72` north ledge **with** `CastleB1Key`, or out of the `0x71` east pocket **with** `CastleB1SecondKey`). Until then `west_to_0x80` stays locked.

## Non-claims

Did not STATUS. Did not poke `$F3CC` / sword / keys. Did not treat a pin as power-on. Did not walk rooms. Did not git commit.
