## Residual — rr-kw8t Gravity on the Phantoon tip

**Continue (rr-1xc2.3):** Halt-3 is out. After a red, keep going from Next
below (a). Three of the same miss class → new probe angle, same sitting (b).
A new sitting (c) only because context died; it still reads this file.
Do not dest-hop dual this checkbox. Do not reboot the hop because the
count hit three.

**Miss class:** standing hole is open; jump through is Covern knockback.
Peak **(1126, 1840) p83** f2583; leftover hole (1112, 1900) p47 falling.
165px short of y=1675. Living checkbox is `west_super`.

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
the seat); floor HiJump peaks ~1868 so left (1075,1845) is **air over the
gap**, not a standing seat. Standing shelf is **(1082, 1878) p10**. DOWN
on the **hatch** → Basement, DOWN on the **lip** is morph; Wave UP from 1845
breaks the floor you stand on; cubby spin is a leftover, not a takeoff.
Hatch-column HiJump (no ceiling) clears y=1850 so `_three_shot_tunnel`
returns early; climb must finish the lip landing facing RIGHT. Ice
jump-shot from the lip is a vertical bonk at y~1843. Spin-jump LEFT off
the lip falls through the hole. Morph + roll LEFT + mid-ledge spin RIGHT
peaks **(1101, 1820) p25** — do not farm that spin. Unmorph A-settle idles
over the gap; UP-only unmorph **lands the shelf**. From the shelf, stand
and Wave **opens a standing hole** (leftover still: blocks gone). Gun-jump
A through it is Covern knockback **(1126, 1840) p83**; leftover falls in
the hole (1112, 1900) p47. HiJump from shelf y~1885 peaks ~1774 even
through the hole — short of 1675. Ice-first without a y-filter locks onto
the stairs enemy (1048, 1928) and crouch-charges LEFT 3600f. RIGHT+A off
the shelf pose-38 turn-locks.

### Next — one seam only

`--stop-at west_super` from the hop-1 pin. Hole is open. Wall-jump the
gap, or ice **only** the shelf Covern (1129, 1818) then jump. Do not ice
the stairs enemy. Do not farm mid-ledge spin. Do not dual the cubby. Do
not farm mid_climb.

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
