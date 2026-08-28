## Residual — rr-kw8t Gravity on the Phantoon tip

**Continue (rr-1xc2.3):** Halt-3 is out. After a red, keep going from Next
below (a). Three of the same miss class → dump a phase pin at the last
held seat, or replace the takeoff (b). A new sitting (c) only because
context died; it still reads this file. Gravity epic continues from this
residual. `attic_door` is the Attic dual. Farm plan: `rr-1xc2.8`.

**Miss class:** Observable grate land is not a usable outgoing pin. Dual
`--stop-at grate_seat` 92f ×2 seated `(1189, 1883) p2` gs=8 (sha256
`e2e8ec9c7c53a7b5094aa79ac8588fb764d1cd55743cf69448343151e10242fd`).
Copy: `scratch/post_ws_main_grate_land.state`. Glance now rejects that
still. LEFT+A from it, Ice suppressed, peaks `y=1843` and falls back.
Human take02 LEFT+A is from x≈1221–1227.

**Status:** Hop 1 dual-green. Hop 2 usable `grate_seat` is RED. Observable
land `(1189, 1883) p2` is not the living checkbox. Living tip `--to
phantoon` **195,336f** ×2 (STATUS). Hop GREEN is Attic gs=8 only.
Phase dumps are named scratch pins.

**Pin in:** `scratch/post_ws_basement_to_main.state` (`0xCAF6` ~(1173,1979)
p1 gs=8)
**Goal:** Attic `0xCA52` gs=8. Full hop GREEN is that leave only.
**Living checkbox:** `west_super` (**RED**). Gate in front is a *usable*
take02 fire-slope pin, not observable land.

### Already green (do not re-prove)

| Layer | Dual | Leave |
|-------|-----:|-------|
| Power-on Phantoon | **195,336f** ×2 | `0xCC6F` (1240,139) p10 gs=8 |
| Hop 1 Basement → Main | **1,579f** ×2 | `0xCAF6` (1173,1979) p1 gs=8 |

Hop 2 has no green phase row until a dual glances the usable fire slope
`~(1223, 1860) p3`. Observable `(1189, 1883) p2` and take04
`~(1195, 1883)` are not that pin.

### Hop 2 seams

Controller: `routes/kpdr/k6/` geometry + overlay + play. Unpowered
`ws_main.py` is a different hop. Phase ladder:
[`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md). Beads: `rr-1xc2.8`.

| File | Role |
|------|------|
| `leave_specs.py` | usable outgoing pin (`WS_MAIN_GRATE_SEAT`) |
| `ws_main_geometry.py` | observable land band (`GRATE_LAND_*`) + region |
| `ws_main_actions.py` | one action per region |
| `ws_main_shaft.py` | overlay loop |
| `ws_main_ice.py` | ice overlay |
| `ws_main_climb.py` | play |

| Phase | From | Held exit | Status |
|-------|------|-----------|--------|
| 1 pit_shot | pin (1173,1979) | 3-shot, still Main, not Basement | PARTIAL — Wave+Spazer **opens the grate** |
| 2 grate_seat | pin | usable take02 ~(1223, 1860) p3 | **RED** — land `(1189, 1883) p2` is observable, not usable |
| 3 west_super | usable fire slope | y~1675 in shaft, not 0xCDA8 | **RED** living |
| 4 mid_climb | 1675 | y~680 in shaft | not started |
| 5 attic_seat | 680 | ~(1135, ≤160) stand | not started |
| 6 attic_door | door | Attic `0xCA52` gs=8 | hop GREEN only |

west_super greens before mid_climb opens. Phase dumps are not hop GREEN.
Do not `--source` `post_ws_main_grate_land.state` (or the 1189 hash) as
if it were grate_seat.

### Tape scan (take02 policy; 03/04/05 agree on lip fire, not departure)

| Take | Lip fire | First `0xD080` | Morph (not lip) | First y≤1675 |
|------|----------|----------------|-----------------|--------------|
| 02 | (1223,1860) p3 UP+X ~7f then UP | f306 | (1189,1785) p56 | f717 (1099,1675) |
| 03 | (1227,1856) p3 UP+X | f139 | (1189,1785) p56 | f547 |
| 04 | **(1195,1883) p3 UP+X** then RIGHT | f215 | (1214,1801) p56 | f1039 |
| 05 | (1221,1862) p4 UP+X | f296 | (1189,1785) p56 | f974 |

Living departure is take02 LEFT+A from x≈1221–1227. Take04 first travels
right from ~(1195,1883) — a different policy; `rr-1xc2.8.2` owns the
window lock. No take fires LEFT+X from the hatch-lip pocket ~(1177, 1883).
Take02 two-hop: short A from ~(1166,1979) that **fails**, land, walk LEFT
to 1156, committed A, RIGHT+A at y~1920, land (1208,1875) p9, walk to
(1223,1860). Human never hops LEFT off 1166 toward the stairs.

### Next — one seam

`rr-1xc2.8.1` (this sitting): usable glance is `(1216, 1232) × (1852, 1868)`.
Observable land stays `GRATE_LAND` `(1188, 1232) × (1852, 1888)`. Probe
writes `post_ws_main_grate_seat.state` only on usable glance.

Ready in parallel: `rr-1xc2.8.2` tape recon (take02 LEFT+A vs take04
walk-right). Serial after both: `rr-1xc2.8.3` land the usable seat from
the hop-1 pin. Do not 2nd-dual LEFT+A from 1189. Do not `--start-phase
grate_seat` from the 1189 hash.

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  snes/super_metroid/scripts/probe/ws_main_climb.py --stop-at grate_seat --no-video --dual
# Usable pin only (not the 1189 land hash):
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
- Did not treat pocket `(1177, 1883)` or land `(1189, 1883)` as fire-slope green
- Did not open mid_climb
