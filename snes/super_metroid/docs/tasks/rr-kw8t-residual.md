## Residual — rr-kw8t Gravity on the Phantoon tip

**Continue (rr-1xc2.3):** Halt-3 is out. After a red, keep going from Next
below (a). Three of the same miss class → new probe angle, same sitting (b).
A new sitting (c) only because context died; it still reads this file.
Do not dest-hop dual this checkbox. Do not reboot the hop because the
count hit three.

**Miss class:** Hatch-lip pocket `(1181, 1883)` p1. Wall is RIGHT; Wave
blocks are LEFT; UP hits the pocket ceiling (`min_y` 1843). Take02 fire
column `(1223, 1860)` is not reachable by walk/jump-RIGHT from the green
grate_seat. Three duals RED × same leftover. New angle (ROM-free, not
dualed): LEFT+X from the pocket until `0xD080` spawn, then LEFT+A.
Morph only after that spawn, at ~(1189, 1785) — not on the lip.

**Status:** Hop 1 dual-green. Hop 2 **grate_seat PHASE 189f ×2**. Living
tip is `--to phantoon` **195,336f** ×2 (STATUS). Do not STATUS from a pin.
Do not wire hops 1–2 onto `--to phantoon`. Phase dumps are not hop GREEN.

**Pin in:** `scratch/post_ws_basement_to_main.state` (`0xCAF6` ~(1173,1979)
p1 gs=8)
**Goal:** Attic `0xCA52` gs=8. Full hop GREEN is that leave only.

### Already green (do not re-prove)

| Layer | Dual | Leave |
|-------|-----:|-------|
| Power-on Phantoon | **195,336f** ×2 | `0xCC6F` (1240,139) p10 gs=8 |
| Hop 1 Basement → Main | **1,579f** ×2 | `0xCAF6` (1173,1979) p1 gs=8 |
| Hop 2 grate_seat (phase) | **189f** ×2 | `0xCAF6` (1177,1883) p2 gs=8 |

### Hop 2 seams (HARD_ROOM_SPLITS)

Controller kept: `ws_main_climb.py` / `ws_main_shaft.py` / `ws_main_actions.py`
/ `ws_main_ice.py` / `ws_main_grate.py`. Seats: `ws_main_phases.py`.
Six chunks, not one climb.

| Phase | From | Held exit | Status |
|-------|------|-----------|--------|
| 1 pit_shot | pin (1173,1979) | 3-shot, still Main, not Basement | PARTIAL — Wave+Spazer **opens the grate** |
| 2 grate_seat | pin | right hatch-lip ~(1177, 1883) p2 | **PHASE 189f ×2** |
| 3 west_super | lip | y~1675 in shaft, not 0xCDA8 | **RED ×3** living |
| 4 mid_climb | 1675 | y~680 in shaft | not started |
| 5 attic_seat | 680 | ~(1135, ≤160) stand | not started |
| 6 attic_door | door | Attic `0xCA52` gs=8 | hop GREEN only |

Do not open mid_climb while west_super is red. Phase dumps are not hop GREEN.

### west_super duals this sitting (stop dualing this arc)

All from hop-1 pin `--stop-at west_super`. Leftover always
`0xCAF6` (1181, 1883). `min_y` 1843 @ f61 (pit jump). 3691f timeout.

| # | Knob | Leftover | Why |
|---|------|----------|-----|
| 1 | hold UP+X until spawn | p3 charge 120 | X held; never releases; UP misses pocket blocks |
| 2 | walk RIGHT to take02 column | p1 charge 0 | wall; x stuck 1181 |
| 3 | RIGHT+A onto save-ledge | p1 charge 0 | same leftover; ceiling/wall bonk |

PNG: Samus on the hatch-lip, Wave blocks LEFT, save-column wall RIGHT.

### Tape scan (take02 policy; 03/04/05 agree; 01 is the retry)

Five human takes, all Attic `0xCA52`. Series `tasks/ws_main_attic_v1/`.

| Take | Frames | Attic f | Lip UP+X seat | First `0xD080` | Morph (not lip) | First y≤1675 |
|------|-------:|--------:|---------------|----------------|-----------------|--------------|
| 02 | **2175** | 2053 | (1223,1860) p3, 20f | f306 | (1189,1785) p56 | f717 (1099,1675) |
| 03 | 2235 | 2120 | (1227,1856) p3, 13f | f139 | (1189,1785) p56 | f547 (1099,1675) |
| 04 | 2452 | 2383 | (1194,1883) p15, 14f | f215 | (1214,1801) p56 | f1039 (1093,1675) |
| 05 | 2633 | 2560 | (1221,1862) p4 + R cubby | f296 | (1189,1785) p56 | f974 (1093,1672) |
| 01 | 10338 | 9651 | (1215,1868) p3 @f7640 | f7648 | (1189,1785) p56 | f8021 |

Take02/03 fire from the save-ledge ~(1223,1860) then LEFT+A to
~(1189,1771) and morph at ~(1189,1785). The bot never reaches that
column from grate_seat. Take04 did fire UP from y=1883 at x=1195 — after
a different first-jump. Do not retouch grate_seat.

### Taught this sitting (ROM-free; 30 passed)

`ws_main_grate.py`: hatch-lip x<1216 faces LEFT and fires (`LEFT+X`,
release at CHARGE_FULL), then LEFT+A. Take02 column still UP+X.
`grate_morph_action` DOWN-morphs at ~(1189,1785) only after spawn.
Alcove x≥1224 stays out. No 24f DOWN on the lip.

### Next — one seam only

New trajectory, not a 4th dual of walk/jump-RIGHT. Dual `--stop-at
west_super` from the hop-1 pin with the LEFT-shot pocket skill. Morph
only after a 0xD080-family spawn, and not on the lip. Do not dest-hop.
Do not STATUS. Do not wire `--to phantoon`.

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  snes/super_metroid/scripts/probe/ws_main_climb.py --stop-at west_super --no-video --dual
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
- Did not 4th-dual the RIGHT-into-wall arc
