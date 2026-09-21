# 0x71 east-pocket wrap residual (`rr-ccxt.18`)

Planner owns `docs/STATUS.md`. Isolated state-load only. Not natural_entry, not
continuous, not STATUS. Did not poke `$F3CC` / sword / keys. Did not add a
graph hop.

## Result: **ok** — pocket opens; leftover is 0x81 with keys=1

`CastleB1SecondKey` (904, 3988) keys=1 walks out the east alcove **without
spending the key**, then `south_to_0x81` DOWN settles in **0x81 (632, 4155)
keys=1** `$F3CC==0`.

Cardinals from the pin stay boxed (sibling `rr-ccxt.13`). The west door is
real; it is **north** of the pin y. LEFT at y=3988 is the south jamb (keys
stay 1, not a key door).

## Path (beyond cardinals)

| Step | xy | Note |
|------|-----|------|
| pin | (904, 3988) | east alcove; chest + 4 pots (tiles, not sprites); bombs=0 boots=0 |
| east_pocket_nw | (835, 3982) | walk to NW lip (target 832,3976) |
| UP only | (835, 3944) | leaves the 32px bbox north; still x=835 (east of corridor) |
| LEFT only | (832, 3982) | jamb; keys stay 1 |
| UP+LEFT | (688, 3960) | through west door at **y=3960**; x=east_extent keys=1 |
| south_door_approach | (632, 4100) | map approach |
| DOWN | **0x81 (632, 4155)** | landing (632, 4084) sub=2; settle sub=0 keys=1 |

No lift / bomb / dash. Overlay door is the west alcove door, not a bomb wall.

## 0x81 leftover (RAM glance)

| Field | Value |
|-------|--------|
| Pin chain | `CastleB1SecondKey` → NW lip UP+LEFT → `south_to_0x81` DOWN |
| Room | **0x81** (632, 4155) |
| Module / sub | `$10==0x07` `$11==0` ctrl=1 |
| Sword | `$F359==1` |
| Lamp | `$F34A==0` |
| Keys | **`$F36F==1`** (not spent) |
| Follower | `$F3CC==0` |
| HP | `$F36D==24` / `$F36C==24` |
| Layer | `$00EE==1` (same as pin) |

This is the west-corridor north door band (`zelda_west` 632,4152 / cell
approach 608,4168). Overlay shows the locked 0x80 door on the left.

## Pocket bbox (cardinals still true)

x **832–944**, y **3976–4008**. LEFT at x=832 is a wall at pin y. Wrap is
y=3960 (4–16px north of that box's north lip while sliding through the door).

## Files

- `docs/tasks/second-key-residual.md` (this file)
- `scratch/probe_second_key.py`
- `recordings/probe_second_key/` (`leftover.json`, `pin.png`, `opened.png`, `room_81.png`)
- `maps/room_71.json` waypoints `second_key_pin` / `east_pocket_nw` /
  `east_pocket_west_door` / `wrap_to_corridor` (door landing unchanged)

## Non-claims

Did not STATUS. Did not poke `$F3CC` / sword / keys. Did not treat a pin as
power-on. Did not copy approach xy into Python hops. Did not add a graph hop.
Did not LEFT into `west_to_0x80`.

## Next

0x81 leftover is keys>=1 on the west-corridor north door, one screen from
`west_to_0x80_approach` (608, 4168). Cell-door sitting can LEFT from here.
