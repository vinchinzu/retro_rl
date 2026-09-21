# 0x81 west_to_0x80 from SecondKey leftover (`rr-ccxt.19`)

Planner owns `docs/STATUS.md`. Isolated state-load only. Not natural_entry, not
continuous, not STATUS. Did not poke `$F3CC` / sword / keys. Did not add a
graph hop. Did not mash Zelda dialogue.

## Result: **red** — dest stays 0x81; small key does not open the cell

Replayed `CastleB1SecondKey` wrap (NW lip UP+LEFT → `south_to_0x81`) onto
**0x81 (632, 4155) keys=1** `$F3CC==0`. Walked to `west_to_0x80_approach`
~(608, 4168) and held LEFT. Dest **stays 0x81**. Keys stay **1**. `$F366==0`
`$F367==0` (no big key). Overlay shows the gold jail door on the west wall.

`west_to_0x80` is **not** isolated. `landingXy` still `null`.

## Path

| Step | xy | Note |
|------|-----|------|
| pin | (904, 3988) | `CastleB1SecondKey` 0x71 keys=1 `$F3CC==0` |
| east_pocket_nw | (835, 3982) | walk to NW lip (target 832,3976) |
| UP+LEFT | (688, 3960) | through west door at y=3960; keys=1 |
| south_door_approach | (632, 4100) | map approach |
| DOWN | **0x81 (632, 4155)** | sibling leftover; keys=1 `$F3CC==0` |
| west_to_0x80_approach | (611, 4169) | map ~(608, 4168) |
| LEFT | **0x81 (608, 4169)** | keys still 1; idle on the west wall |
| y-sweep 4144–4216 | **0x81 (608, 4212)** | LEFT at each y; never spends the key |

## 0x81 leftover (RAM glance)

Door attempt leftover (LEFT at map approach):

| Field | Value |
|-------|--------|
| Pin chain | `CastleB1SecondKey` → NW lip UP+LEFT → `south_to_0x81` DOWN → LEFT |
| Room | **0x81** (608, 4169) |
| Module / sub | `$10==0x07` `$11==0` ctrl=1 |
| Sword | `$F359==1` |
| Lamp | `$F34A==0` |
| Keys | **`$F36F==1`** (not spent) |
| Big key | **`$F366==0` `$F367==0`** |
| Follower | **`$F3CC==0`** |
| HP | `$F36D==24` / `$F36C==24` |
| Layer | `$00EE==1` |
| Dungeon | `$040C==2` |

Y-sweep leftover is the same room/keys/follower at (608, 4212).

## Map

`maps/room_81.json` door `west_to_0x80`: `landingXy` still `null`. Notes record
this sitting. Do not claim isolated.

## Files

- `docs/tasks/cell-key-residual.md` (this file)
- `scratch/probe_cell_key.py`
- `recordings/probe_cell_key/` (`leftover.json`, `pin.png`, `wrap.png`,
  `room_81.png`, `approach.png`, `locked.png`)
- `maps/room_81.json` locked-door notes (`landingXy` null)

## Non-claims

Did not STATUS. Did not poke `$F3CC` / sword / keys. Did not treat a pin as
power-on. Did not copy approach xy into Python hops. Did not add a graph hop.
Did not claim Zelda rescue. Did not write `$F3CC`.

## Next

Small key on the 0x81 west wall is not enough. Vanilla cell door is the **big
key** jail (Ball and Chain drop). Need a pin with `$F367` castle/sewer big-key
bit, then LEFT here. `CastleZeldaFollower` is already inside 0x80 with
`$F3CC==1` as loaded — not this leftover.
