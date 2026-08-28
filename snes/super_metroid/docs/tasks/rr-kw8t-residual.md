## Residual — rr-kw8t Gravity on the Phantoon tip

**Continue (rr-1xc2.3):** Halt-3 is out. After a red, keep going from Next
below (a). Three of the same miss class → dump a phase pin at the last
held seat, or replace the takeoff (b). A new sitting (c) only because
context died; it still reads this file. Gravity epic continues from this
residual. `attic_door` is the Attic dual. Farm plan: `rr-1xc2.8`.

**Miss class:** west_super LEFT+A from the usable pin still launches at
x~1217 (not `SLOPE_LEFT_A` 1227–1231). Dual `--start-phase grate_seat
--stop-at west_super` 3618f timeout leftover `(1209, 1787) p2`. Peak
`y=1769` at `(1217, 1769) p26` f107 — same class as the prior two reds
(peaks 1762 then 1769, fall to ~1209,1787). Halt-3: do not repeat this
dual. Next is a new trajectory or a dump at a held seat that is not
this leftover.

**Status:** Hop 1 dual-green. Hop 2 usable `grate_seat` **118f** ×2
`(1217, 1867) p9` gs=8 (sha256
`a19a2078cc0c844ace7927cb2bfc72725500406b8fe5ed5c11e1f6903354315f`).
Living checkbox is `west_super` (**RED**). Living tip `--to phantoon`
**195,336f** ×2 (STATUS). Hop GREEN is Attic gs=8 only.
Phase dumps are named scratch pins.

**Pin in:** `scratch/post_ws_basement_to_main.state` (`0xCAF6` ~(1173,1979)
p1 gs=8)
**Goal:** Attic `0xCA52` gs=8. Full hop GREEN is that leave only.
**Living checkbox:** `west_super` (**RED**). Usable grate_seat pin is
`scratch/post_ws_main_grate_seat.state` (1217, 1867) p9, not the 1189 land.

### Already green (do not re-prove)

| Layer | Dual | Leave |
|-------|-----:|-------|
| Power-on Phantoon | **195,336f** ×2 | `0xCC6F` (1240,139) p10 gs=8 |
| Hop 1 Basement → Main | **1,579f** ×2 | `0xCAF6` (1173,1979) p1 gs=8 |
| Hop 2 grate_seat (usable) | **118f** ×2 | `0xCAF6` (1217,1867) p9 gs=8 |

Observable `(1189, 1883) p2` and take04 `~(1195, 1883)` are not that pin.

### Hop 2 seams

Controller: `routes/kpdr/k6/` geometry + overlay + play. Unpowered
`ws_main.py` is a different hop. Phase ladder:
[`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md). Beads: `rr-1xc2.8`.

| File | Role |
|------|------|
| `leave_specs.py` | usable outgoing pin (`WS_MAIN_GRATE_SEAT`) |
| `ws_main_geometry.py` | observable land band (`GRATE_LAND_*`) + region |
| `ws_main_departure.py` | take02 LEFT+A vs take04 walk-right (data) |
| `ws_main_actions.py` | one action per region |
| `ws_main_shaft.py` | overlay loop |
| `ws_main_ice.py` | ice overlay |
| `ws_main_climb.py` | play |

| Phase | From | Held exit | Status |
|-------|------|-----------|--------|
| 1 pit_shot | pin (1173,1979) | 3-shot, still Main, not Basement | PARTIAL — Wave+Spazer **opens the grate** |
| 2 grate_seat | pin | usable take02 fire slope | **118f** ×2 (1217,1867) p9 |
| 3 west_super | usable fire slope | y~1675 in shaft, not 0xCDA8 | **RED** — 3× peak y=1762–1769 from x~1217 |
| 4 mid_climb | 1675 | y~680 in shaft | not started |
| 5 attic_seat | 680 | ~(1135, ≤160) stand | not started |
| 6 attic_door | door | Attic `0xCA52` gs=8 | hop GREEN only |

west_super greens before mid_climb opens. Phase dumps are not hop GREEN.
Do not `--source` `post_ws_main_grate_land.state` (or the 1189 hash) as
if it were grate_seat.

### Departure windows (rr-1xc2.8.2 locked)

Living policy is take02 slope LEFT+A. Window: `SLOPE_LEFT_A` x
(1227, 1231) × y (1852, 1856). Data:
`routes/kpdr/k6/ws_main_departure.py`. Not `climb_action`.

| Take | Policy | Lip fire | Grounded LEFT+A | Peak y |
|------|--------|----------|-----------------|--------|
| 02 | **slope_left_a (living)** | (1223,1860) p3 UP+X, 0xD080 f306 | (1231,1852) p3, then LEFT+A | 1763 |
| 03 | slope_left_a (agrees) | (1227,1856) p3 UP+X, f139 | (1227,1856) p1, A then LEFT+A | 1763 |
| 04 | walk_right_alcove | **(1195,1883) p3 UP+X then RIGHT**, f215 | (1242,1851) p1 save ledge | 1795 |
| 05 | walk_right_alcove | (1243,1851) p6 X+R, f296 | (1243,1851) p2 | 1795 |

take02 after fire walks RIGHT 8px on the slope to 1231 (still inside
`WS_MAIN_GRATE_SEAT`). x≈1221–1227 during that LEFT+A is airborne
y~1800, not the takeoff. take05 (1221,1862) p4 UP+X is a later fire
(f539), not the first 0xD080. Take04/05 alcove x≥1242 is outside the
glance spec. No take fires LEFT+X from the hatch-lip pocket ~(1177, 1883).
Take02 two-hop: short A from ~(1166,1979) that **fails**, land, walk LEFT
to 1156, committed A, RIGHT+A at y~1920, land (1208,1875) p9, walk to
(1223,1860). Human never hops LEFT off 1166 toward the stairs.

### Next — one seam

`rr-1xc2.8.4` (halt-3): LEFT+A still leaves from x~1217, not
`SLOPE_LEFT_A`. Do not repeat `--start-phase grate_seat --stop-at
west_super` from `post_ws_main_grate_seat.state` without a new
trajectory (walk to 1227–1231 before A, or dump a held takeoff pin).
Ice stays off this seat. Do not open mid_climb. Natural-entry is
`rr-1xc2.8.5`.

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  snes/super_metroid/scripts/probe/ws_main_climb.py \
  --start-phase grate_seat --stop-at west_super --no-video --dual \
  --source snes/super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_ws_main_grate_seat.state
# Natural-entry (rr-1xc2.8.5, after 8.4):
# QT_QPA_PLATFORM=offscreen uv run python \
#   snes/super_metroid/scripts/probe/ws_main_climb.py --stop-at west_super --no-video --dual
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
- Did not treat take04 alcove as the living handoff
- Did not treat leftover `(1209, 1787)` as west_super
- Did not open mid_climb, compose, or STATUS
