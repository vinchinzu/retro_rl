# Zelda-cell pin census — residual (rr-ccxt.3)

State-load RAM glance only. Did not walk rooms. Did not poke `$F3CC` / sword / keys.
Did not STATUS-promote. Did not claim continuous. Did not edit maps or `escape_graph`.

Dump: `recordings/probe_zelda_pins/zelda_pin_census.json`
Probe: `scratch/probe_zelda_pins.py` (load → `settle_control` → leftover RAM).

Follower is true only when `$F3CC==1` as loaded. All ten `*Zelda*` / follower pins
on disk were loaded; no extra `*Zelda*` files beyond the work-queue set.

## Pins (settled)

Every pin: `module=0x07` `submodule=0` `indoors=true` `screen=0x00` `has_control=true`
`sword $F359=1` `lamp=0` `keys $F36F=0` `follower $F3CC=1`. Settle 0f.

| State | Room | xy | in_zelda_cell |
|-------|------|----|---------------|
| `CastleMainZeldaBoomerang` | `0x61` main hall | 960, 3320 | no |
| `CastleMainZeldaReady` | `0x61` main hall | 943, 3320 | no |
| `CastleZeldaFollower` | `0x80` zelda_cell | 352, 4168 | **yes** |
| `CastleRoom51Zelda` | `0x51` throne | 760, 2984 | no |
| `CastleRoom52ZeldaBoomerang` | `0x52` NE | 1142, 2982 | no |
| `CastleZeldaB1East` | `0x82` B1 east | 1036, 4492 | no |
| `CastleZeldaB1Island` | `0x82` B1 island | 1275, 4253 | no |
| `CastleZeldaB1Pit` | `0x72` B1 pit | 1184, 4016 | no |
| `CastleZeldaB1West` | `0x81` B1 west | 632, 4152 | no |
| `CastleMantleZelda` | `0x51` mantle | 760, 2631 | no |

## Leftover

- **Only `CastleZeldaFollower` is in the cell** (`in_zelda_cell`, room `0x80`, spawn matches `maps/room_80.json` `entry_spawn`).
- There is **no pre-rescue cell pin** (`$F3CC==0` in `0x80`). Every Zelda-named pin already has follower 1 as loaded.
- Lamp is 0 on every pin, including `CastleMantleZelda` (mantle escort still needs lamp inventory).
- Nearest B1 neighbor is `CastleZeldaB1West` in `0x81` (632, 4152), already post-follower, not a cell leftover.

## Next

Use `CastleZeldaFollower` as the Zelda-cell leftover for a future hop (isolate `east_to_0x81` from that pin; map notes the door is not isolated).
