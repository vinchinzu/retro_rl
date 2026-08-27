## Residual — rr-kw8t Gravity on the Phantoon tip

**Continue (rr-1xc2.3):** Halt-3 is out. After a red, keep going from Next
below (a). Three of the same miss class → new probe angle, same sitting (b).
A new sitting (c) only because context died; it still reads this file.
Do not dest-hop dual this checkbox. Do not reboot the hop because the
count hit three.

**Miss class:** Two-hop takeoff (take02 floor recipe). Short A facing LEFT
at ~(1166,1979) peaked `(1177, 1843)` p81 f59 — hatch-column, not over
the lip wall — then leftover stairs `(1111, 1899)` p157. Fire slope was
not seated. One dual this sitting; do not 2nd-dual this LEFT-facing short
hop. Previous sitting's first-jump air peaked `(1194, 1836)` and fell
back — do not restore that peak tweak.

**Status:** Hop 1 dual-green. Hop 2 **grate_seat PHASE 189f ×2** (old
pocket seat; fire-slope land is not PHASE-green). Living tip is
`--to phantoon` **195,336f** ×2 (STATUS). Do not STATUS from a pin.
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

Controller kept: `ws_main_climb.py` / `ws_main_shaft.py` / `ws_main_pit.py`
/ `ws_main_actions.py` / `ws_main_ice.py` / `ws_main_grate.py`. Seats:
`ws_main_phases.py`. Split: `ws_main_actions.py` **401**, `ws_main_pit.py`
**202** (was 524 before the two-hop knob). `ws_main_shaft.py` still **500**.

| Phase | From | Held exit | Status |
|-------|------|-----------|--------|
| 1 pit_shot | pin (1173,1979) | 3-shot, still Main, not Basement | PARTIAL — Wave+Spazer **opens the grate** |
| 2 grate_seat | pin | take02/04 fire slope ~(1223, 1860) p3 | old PHASE 189f ×2 was the **pocket**; fire-slope land RED |
| 3 west_super | fire slope | y~1675 in shaft, not 0xCDA8 | **RED** living |
| 4 mid_climb | 1675 | y~680 in shaft | not started |
| 5 attic_seat | 680 | ~(1135, ≤160) stand | not started |
| 6 attic_door | door | Attic `0xCA52` gs=8 | hop GREEN only |

Do not open mid_climb while west_super is red. Phase dumps are not hop GREEN.

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

### west_super duals — previous sittings (stop)

Pocket walk/jump-RIGHT leftover always `(1181, 1883)`. Shoulder-R fell
to stairs `(1113, 1899)` p156. Pocket X-tap leftover `(1179, 1883)` p2
crystals still up. First-jump air-steer best peak `(1194, 1836)` then
fell back; ceiling-release reverted. Do not 4th-dual those.

### west_super duals — this sitting

From hop-1 pin `--stop-at west_super`. One two-hop dual, then stop.

| # | Knob | Leftover | Why |
|---|------|----------|-----|
| 1 | take02 two-hop: short A facing LEFT at 1166, committed at 1156 | **(1111, 1899) p157** stairs face R mov=14 | 3692f ×2; min_y **(1177, 1843) p81** f59. LEFT-facing short A never cleared the wall |

PNG: Samus on the left stairs west of the hatch, Wave crystals still up
over the pit. Not the fire slope.

### Taught this sitting (kept)

`ws_main_pit.py` (split out of `ws_main_actions.py` **before** the knob):

- Take02 two-hop floor recipe, not another first-jump peak tweak.
- Short A at `SHORT_HOP_X` (1163, 1171) on the floor y≥1960, facing LEFT.
- Facing RIGHT in that band walks LEFT to 1156 (land, do not hop again).
- Committed takeoff target x=1156. RIGHT+A at y~1920 unchanged.
- Lip leftover y~1883 is **not** the short hop.
- Do not revert the two-hop because the dual was RED.

ROM-free: `uv run pytest snes/super_metroid/tests/test_ws_main_climb.py snes/super_metroid/tests/test_practice_takes.py -q` → 31 passed.

### Next — one seam only

Do **not** 2nd-dual LEFT-facing short A. Do not 4th-dual first-jump
air-steer / land. Do not dual pocket X. Do not dual shoulder-R. New
angle, still the two-hop floor recipe: take02's short A at ~1166
**facing RIGHT** (fail into the wall), land, walk LEFT to 1156, committed
A. This sitting hopped LEFT and stole to the stairs. Compose only after
Attic is dual-green. Do not dest-hop. Do not STATUS. Do not wire
`--to phantoon`.

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
- Did not 4th-dual first-jump land / air-steer
- Did not 2nd-dual the LEFT-facing short hop
- Did not compose hop 1 + hop 2 onto `--to phantoon`
