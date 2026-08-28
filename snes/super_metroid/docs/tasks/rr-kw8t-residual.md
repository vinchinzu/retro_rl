## Residual — rr-kw8t Gravity on the Phantoon tip

**Continue (rr-1xc2.3):** Halt-3 is out. After a red, keep going from Next
below (a). Three of the same miss class → dump a phase pin at the last
held seat, or replace the takeoff (b). A new sitting (c) only because
context died; it still reads this file. Gravity epic continues from this
residual. `attic_door` is the Attic dual.

**Miss class:** Take02 two-hop. Short A facing LEFT at ~(1166,1979) peaked
`(1177, 1843)` p81 f59 (hatch-column, not over the lip wall) then leftover
stairs `(1111, 1899)` p157. Fire slope was not seated. 3692f ×2. Stairs
leftover is pit-takeoff recovery. Closed knobs: pocket X / walk-RIGHT
leftover `(1181, 1883)`, shoulder-R to stairs, first-jump air peak
`(1194, 1836)` then fall-back.

**Status:** Hop 1 dual-green. Hop 2 fire-slope `grate_seat` is RED. Pocket
`(1177, 1883)` p2 was a 189f PhaseStop, not the fire slope. Living tip
`--to phantoon` **195,336f** ×2 (STATUS). Hop GREEN is Attic gs=8 only.
Phase dumps are named scratch pins.

**Pin in:** `scratch/post_ws_basement_to_main.state` (`0xCAF6` ~(1173,1979)
p1 gs=8)
**Goal:** Attic `0xCA52` gs=8. Full hop GREEN is that leave only.
**Living checkbox:** `west_super` (**RED**). Gate in front is a held
fire-slope pin.

### Already green (do not re-prove)

| Layer | Dual | Leave |
|-------|-----:|-------|
| Power-on Phantoon | **195,336f** ×2 | `0xCC6F` (1240,139) p10 gs=8 |
| Hop 1 Basement → Main | **1,579f** ×2 | `0xCAF6` (1173,1979) p1 gs=8 |

Hop 2 has no green phase row until a dual glances the fire slope
`~(1223, 1860) p3` (or take04 `~(1195, 1883)`).

### Hop 2 seams

Controller: `routes/kpdr/k6/` geometry + overlay + play. Unpowered
`ws_main.py` is a different hop. Phase ladder:
[`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md).

| File | Role |
|------|------|
| `ws_main_geometry.py` | bands + region |
| `ws_main_actions.py` | one action per region |
| `ws_main_shaft.py` | overlay loop |
| `ws_main_ice.py` | ice overlay |
| `ws_main_climb.py` | play |

| Phase | From | Held exit | Status |
|-------|------|-----------|--------|
| 1 pit_shot | pin (1173,1979) | 3-shot, still Main, not Basement | PARTIAL — Wave+Spazer **opens the grate** |
| 2 grate_seat | pin | take02/04 fire slope ~(1223, 1860) p3 | **RED** — pocket `(1177, 1883)` is not this seat |
| 3 west_super | fire slope | y~1675 in shaft, not 0xCDA8 | **RED** living |
| 4 mid_climb | 1675 | y~680 in shaft | not started |
| 5 attic_seat | 680 | ~(1135, ≤160) stand | not started |
| 6 attic_door | door | Attic `0xCA52` gs=8 | hop GREEN only |

west_super greens before mid_climb opens. Phase dumps are not hop GREEN.

### Tape scan (take02 policy; 03/04/05 agree)

| Take | Lip fire | First `0xD080` | Morph (not lip) | First y≤1675 |
|------|----------|----------------|-----------------|--------------|
| 02 | (1223,1860) p3 UP+X ~7f then UP | f306 | (1189,1785) p56 | f717 (1099,1675) |
| 03 | (1227,1856) p3 UP+X | f139 | (1189,1785) p56 | f547 |
| 04 | **(1195,1883) p3 UP+X** then RIGHT | f215 | (1214,1801) p56 | f1039 |
| 05 | (1221,1862) p4 UP+X | f296 | (1189,1785) p56 | f974 |

No take fires LEFT+X from the hatch-lip pocket ~(1177, 1883). Take02
two-hop: short A from ~(1166,1979) that **fails**, land, walk LEFT to
1156, committed A, RIGHT+A at y~1920, land (1208,1875) p9, walk to
(1223,1860). Human never hops LEFT off 1166 toward the stairs.

### Next — one seam

1. Land the take02 two-hop onto the fire slope from the hop-1 pin.
2. When a dual PhaseStops on `grate_seat` **and** glances the fire-slope
   spec (`~(1223, 1860) p3`, or take04 `~(1195, 1883)`), the probe writes
   `scratch/post_ws_main_grate_seat.state`.
3. Dual `--start-phase grate_seat --stop-at west_super --source` that pin.

Until that pin exists, `--stop-at west_super` from the hop-1 pit pin is
the diagnostic. Stairs leftover `(1111, 1899)` is pit-takeoff recovery
(classifier). Compose after Attic is dual-green. Living tip stays
`--to phantoon`.

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  snes/super_metroid/scripts/probe/ws_main_climb.py --stop-at west_super --no-video --dual
# Once scratch/post_ws_main_grate_seat.state exists:
# QT_QPA_PLATFORM=offscreen uv run python \
#   snes/super_metroid/scripts/probe/ws_main_climb.py \
#   --start-phase grate_seat --stop-at west_super --no-video --dual \
#   --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ws_main_grate_seat.state
uv run python snes/super_metroid/scripts/probe/kpdr.py pure ws-main-to-attic \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ws_basement_to_main.state \
  --headed
```

### Non-claims

- Did not STATUS-promote Gravity
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Did not wire hops 1–2 onto `--to phantoon`
- Did not dual the full hop this wrap
- Did not pin out `post_ws_main_to_attic.state`
- Did not treat a phase dump as hop GREEN
- Did not treat pocket `(1177, 1883)` as fire-slope green
