## Residual — rr-20w.3.1 mountain grape return strand

**Status:** unit GREEN + ROM GREEN for D2/D3 ship. Do not STATUS.

ROM:
- `Y1_Inside_House --ship`: 3099f, shipping 0→150, farm `(135,456)`, Clean path. Bench 3154→3099 (Δ=-55).
- `Y1_D3_Morning --ship --count 2`: 3472f, 1 grape (shop-bail hour 10), farm bin. No mountain strand.

### Why it crashed

Inbound `MountainBerryTask` is reactive. Return was not.

Run6 D15: grape kept (`held=0x03`), `return_to_bin` `soft_solid pin` at
carpenter `(474,630)` toward `(520,632)`, `entities=10`. Optional
`MOUNTAIN_BERRY` then ran farm-only phases on tilemap `0x10`.
`ExitToFarmTask` used spa `mountain_to_farm`, which slices back onto the
same terrace and times out.

Rain D4/D7/D8: farm→path lands on unregistered `0x57` with leaked
`(10,422)`. `path_settle` treated "not path+leaked" as settled and armed
`path_to_mountain`, which failed `expected 0x0C, got 0x57`.

`$700` grape gate skipped the hop in run7. That hides the crash.

### Fix

- `mountain_downhill_escape`: hops with `y > py+16`. Carpenter pin drops
  to `(520,712)` `force_run` down, not the terrace.
- `MountainGrapeShipTask` retries that suffix instead of failing on `0x10`.
- `ExitToFarmTask` / `ReturnHomeTask` use the grape exit when `y>=380`.
- Unknown tilemaps settle (walk left) instead of clipping the next hop.
- `MultiMapNavTask` waits 90f on unregistered maps; known mismatches
  still fail closed.
- Day plan fail-fasts `required_maps` farm phases when not on farm.

### Non-claims

No STATUS. Did not start from `Y1_D2_Morning_After_D1`. Did not claim
rr-20w.3.2 (establish-nav after harvest).
