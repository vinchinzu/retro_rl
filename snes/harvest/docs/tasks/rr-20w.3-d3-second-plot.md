## Residual — rr-20w.3 D3 spring second plot + daily grape

**Status:** one clipped mountain-grape run is the only berry path. D3 grape
ship + shop is ROM GREEN. Second same-day grape is not. Do not STATUS.

### Canonical run

`MountainGrapeShipTask` wrapping `MountainBerryTask`. Farm-bush
`BerryShipTask` / `SHIP_BERRY_*` / `GET_BERRIES_AND_SHIP` are gone.
Count is 1 on D2, 2 on D3+; a second loop is best-effort and bails at
10:00 so the seed shop still happens.

Route clips from the live pose (`farm_to_west_gate_waypoints`). Farm→path
ends at the plaza with `force_run` left. Do not BFS from the east landing
(232,128) or leaked (10,422) toward the farm gate.

### ROM (Y1_D3_Morning, Clean)

| Hop | Frames | Clock | Result |
|-----|--------|-------|--------|
| 1 grape house→bin | 3470 | 06:00→10:00 | shipping 0→150 |
| 2nd grape | — | bailed at 10:00 at the bin | stand empty / inbound dies if forced |
| shop from `Y1_D3_PostGrape` | 2391 | after 10:00 | 0x1C, potato 0→1, money 250→50, back on farm |

One grape + shop fits. Two full mountain loops do not: the first lands at
10:00 and a second loop is another ~4h. Forcing it soaks until 17:00 in
the carpenter corridor and misses the shop.

### Still open

- Second-plot `CROP_ESTABLISH` tiles beside the D2 rows.
- Water all 16 (rr-3ae8 refill).
- Same-day 2nd grape spawn/route. Not proven; do not soak the shop window
  looking for it.

### Non-claims

- No STATUS. Did not start from `Y1_D2_Morning_After_D1`.
- Did not record a BFS-closable walk.
- 2-grape daily income not ROM-verified.
