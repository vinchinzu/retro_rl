# B1 Guard → Zelda cell residual (rr-ccxt.8)

Planner owns `docs/STATUS.md`. Isolated only. Not natural_entry, not
continuous, not STATUS. Did not poke `$F3CC`.

## Honest blocker from CastleB1Guard

**CastleB1Guard cannot walk to 0x80.** It sits on the **north ledge** of 0x72
at the F1 stair column. The only physical exit from that ledge is **UP** the
already-measured F1 stair → 0x01. A pit + railing blocks south. Grab-around
the north block reaches the star tile / south lip (y≈3776) but does not drop
to the lower floor. Soldiers on the lower floor are not reachable without
falling; walking into them from the ledge is a death.

| Field | Guard leftover |
|-------|----------------|
| Pin | `CastleB1Guard` |
| Room | **0x72** (1273, 3665) |
| Module | `$10==0x07` `$11==0` ctrl=1 |
| HP | `$F36D==24` / `$F36C==24` |
| Sword | `$F359==1` |
| Lamp | `$F34A==0` |
| Keys | `$F36F==0` |
| Follower | **`$F3CC==0`** |

`CastleB1Key` is the same ledge, 48px east (1320, 3656), **keys=1**, still
cannot leave except F1 stairs. Key is not on the floor (already collected).

## Measured B1 doors (not a full Guard→cell chain)

Intended vanilla chain is **0x72 south → 0x82 west → 0x81 west → 0x80**.
Pieces:

| Edge | Pin | Result | RAM |
|------|-----|--------|-----|
| `room_72` `south_to_0x82` | `CastleB1PitGuardCleared` (1230,3989) hp=24 keys=0 `$F3CC==0` | **room_engine ok** 159f → 0x82 (1198,4108) | `$F3CC==0` |
| `room_82` `west_to_0x81` | `CastleZeldaB1East` (1036,4492) | **room_engine ok** 103f → 0x81 (996,4492) | **`$F3CC==1` as loaded** |
| reverse 0x81 RIGHT | `CastleB1FarWest` (996,4496) | RIGHT → 0x82 (1040,4496) ~108f | `$F3CC==0` keys=0 |
| `room_81` `west_to_0x80` | `CastleB1FarDoor` walk to (608,4178) | LEFT **stays 0x81** | keys=0 `$F3CC==0` — **locked** |

**0x82 water maze:** the 0x72 south landing is the **north corridor** of 0x82.
The west door to 0x81 is on the **south-west floor**. Direct walk hits water;
pit fall (submodule 20) respawns at the north door. So 0x72 south and 0x82 west
are both isolated, but **not chained from one pin**.

**0x81 cell door:** west wall at Zelda-cell y-band ~(608, 4168). Needs a small
key. The key on this dungeon is on the **north 0x72 ledge** (`CastleB1Key`)
which cannot reach 0x81. `CastleB1SecondKey` (0x71 904,3988 keys=1) is in an
east pocket of 0x71 and cannot walk to the 0x81 door column (stuck x≈832–944).

`maps/room_70.json` `west_north_to_0x80` remains the B2-landing stair into 0x80
(state-local from `CastleB2Landing`), not on the Guard/0x72 path.

## Graph (isolated, off Sanctuary plan)

- `room_72_south_to_0x82` map_id=`room_72` door=`south_to_0x82`
- `room_82_west_to_0x81` map_id=`room_82` door=`west_to_0x81`
- `room_81` `west_to_0x80` is map-only (locked). **Not** an isolated hop.
- `room_01_to_zelda_cell` stays **planned**.

## Files

- `docs/tasks/b1-to-zelda-residual.md` (this file)
- `scratch/probe_b1_to_zelda.py`
- `recordings/probe_b1_to_zelda/`
- `maps/room_72.json` south path from `pit_guard_cleared`
- `maps/room_82.json` west door notes
- `maps/room_81.json` locked `west_to_0x80` approach
- `opening_route/escape_graph.py` isolated hops 72→82 and 82→81

## Non-claims

Did not STATUS. Did not poke `$F3CC` / sword / keys. Did not copy xy into
Python hops. Did not edit `room_01.json`. Did not claim Zelda rescue.
Follower 1 only where the pin already had `$F3CC==1` (`CastleZeldaB1East`,
`CastleZeldaFollower`).

## Next action

Need a **keys>=1** pin on the 0x81 west wall (or a way off the 0x72 north
ledge **with** the key) to isolate `west_to_0x80`. Then chain 0x72-south
landing across 0x82 water to `east_b1`. Until then Guard→cell is blocked.
