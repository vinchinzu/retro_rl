## Residual — rr-kw8t Gravity on the Phantoon tip

**Continue (rr-1xc2.3):** Halt-3 is out. After a red, keep going from Next
below (a). Three of the same miss class → new probe angle, same sitting (b).
A new sitting (c) only because context died; it still reads this file.
Do not dest-hop dual this checkbox. Do not reboot the hop because the
count hit three.

**Miss class:** morph-tunnel from the right lip, then spin. Peak **(1101, 1820)
p25** f217; leftover pit (1149, 1979) p1. 155px short of y=1675. Living
checkbox is `west_super`.

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

Controller kept: `ws_main_climb.py` / `ws_main_actions.py` / `ws_main_ice.py`.
Seats: `ws_main_phases.py`. Six chunks, not one climb:

| Phase | From | Held exit | Status |
|-------|------|-----------|--------|
| 1 pit_shot | pin (1173,1979) | 3-shot, still Main, not Basement | PARTIAL — Wave+Spazer **opens the grate** |
| 2 grate_seat | pin | right hatch-lip ~(1177, 1883) p2 | **PHASE 189f ×2** |
| 3 west_super | lip | y~1675 in shaft, not 0xCDA8 | **RED** living |
| 4 mid_climb | 1675 | y~680 in shaft | not started |
| 5 attic_seat | 680 | ~(1135, ≤160) stand | not started |
| 6 attic_door | door | Attic `0xCA52` gs=8 | hop GREEN only |

Do not open mid_climb while west_super is red. Phase dumps are not hop GREEN.

Measured (do not re-guess): pin x=1173 is under the right lip (bonk y~1940);
hatch-column A from ~x=1150 lands (1177,1883) p2 (pose 3 aiming-up is also
the seat); floor HiJump peaks ~1868 so left (1075,1845) is above it; DOWN
on the **hatch** → Basement, DOWN on the **lip** is morph; Wave UP from 1845
breaks the floor you stand on; cubby spin is a leftover, not a takeoff.
Hatch-column HiJump (no ceiling) clears y=1850 so `_three_shot_tunnel`
returns early; climb must finish the lip landing facing RIGHT. Ice
jump-shot from the lip is a vertical bonk at y~1843. Spin-jump LEFT off
the lip falls through the hole. Remaining Wave blocks are an AFS morph
tunnel, not a standing hole. Morph + roll LEFT + mid-ledge spin RIGHT
peaks **(1101, 1820) p25**. Do not farm another spin window from that
ledge.

### Next — one seam only

`--stop-at west_super` from the hop-1 pin. New angle, not another mid-ledge
spin: land the 1820 peak on the left platform ~(1075, 1845) / stairs, or
open a standing-height hole, or wall-jump the gap. First shaft hop to
y~1675 in Main, not 0xCDA8. Do not dual the cubby. Do not farm mid_climb.

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  snes/super_metroid/scripts/probe/ws_main_climb.py --stop-at west_super --no-video
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
