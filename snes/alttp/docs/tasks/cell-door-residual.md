# 0x81 west_to_0x80 cell door residual (rr-ccxt.13)

Planner owns `docs/STATUS.md`. Isolated only. Not natural_entry, not
continuous, not STATUS. Did not poke `$F3CC` / sword / keys. Did not add
a graph hop.

**Door still locked. No keys>=1 pin on the 0x81 west wall.** `west_to_0x80`
is **not** isolated.

## Best leftover (at the cell door)

`CastleZeldaB1West` is already on the west column. Walk to map approach
`(619,4168)` then LEFT, y-sweep 4144–4216 along x=608. Stays `0x81`.
`$F3CC==1` **as loaded**, not a rescue.

| Field | Value |
|-------|--------|
| Pin | `CastleZeldaB1West` |
| Room | **0x81** (608, 4208) after LEFT |
| Approach | (619, 4168) then LEFT |
| Module | `$10==0x07` `$11==0` ctrl=1 |
| Sword | `$F359==1` |
| Lamp | `$F34A==0` |
| Keys | **`$F36F==0`** |
| Follower | **`$F3CC==1` as loaded** |

## Census (no keys on 0x81)

Every row: module `0x07` / sub `0`, ctrl=1, sword=1, lamp=0, no floor key.

| State | Room | xy | keys | F3CC |
|-------|------|-----|------|------|
| `CastleZeldaB1West` | **0x81** | 632, 4152 | 0 | **1 as loaded** |
| `CastleB1FarDoor` | 0x81 | 632, 4416 | 0 | 0 |
| `CastleB1WestRoom` | 0x81 | 912, 4496 | 0 | 0 |
| `CastleB1FarWest` | 0x81 | 996, 4496 | 0 | 0 |
| `CastleB1GreenRoom` | 0x81 | 776, 4316 | 0 | 0 |
| `CastleB1GreenRoomCleared` | 0x81 | 880, 4256 | 0 | 0 |
| `CastleB1Shutter` | 0x81 | 804, 4256 | 0 | 0 |
| `CastleB1West` | **0x82** | 1184, 4108 | 0 | 0 |
| `CastleB1SecondKey` | **0x71** | 904, 3988 | **1** | 0 |

`CastleB1West` is the north corridor of 0x82 (water maze is another bead).
Did not drop the 0x72 north ledge (`CastleB1Key` keys=1 stays there).

## CastleB1SecondKey (keys=1, cannot walk here)

East pocket of 0x71. Cardinals + detours stay in 0x71. LEFT hits a **wall**
at x=832 (keys stay 1 — not a key door). Box ≈ x 832–944, y 3976–4008.

| Dir / detour | leftover xy | room | keys |
|--------------|-------------|------|------|
| DOWN | 904, 4008 | 0x71 | 1 |
| LEFT | 832, 3984 | 0x71 | 1 |
| RIGHT | 944, 3984 | 0x71 | 1 |
| UP | 904, 3976 | 0x71 | 1 |
| west_column / nw / sw / mid_west | 832, 3984 | 0x71 | 1 |

Never entered 0x81. Did not spend the key.

## Map

`maps/room_81.json` door `west_to_0x80`: `landingXy` still `null`. Notes
record this sitting. Do not claim isolated.

## Files

- `docs/tasks/cell-door-residual.md` (this file)
- `scratch/probe_cell_door.py`
- `recordings/probe_cell_door/` (`cell_door.json`, `pin_census.json`,
  `CastleZeldaB1West.json`, `second_key.json`)
- `maps/room_81.json` locked-door notes

## Non-claims

Did not STATUS. Did not poke `$F3CC` / sword / keys. Did not treat a pin as
power-on. Did not copy approach xy into Python. Did not add a graph hop.
Did not claim Zelda rescue — follower 1 only as loaded on `CastleZeldaB1West`.

## Next

Need a **keys>=1** pin that can stand on the 0x81 west wall (0x72 ledge drop
with `CastleB1Key`, or a path out of the 0x71 east pocket that does not spend
the SecondKey). Then LEFT into 0x80 and measure `landingXy`.
