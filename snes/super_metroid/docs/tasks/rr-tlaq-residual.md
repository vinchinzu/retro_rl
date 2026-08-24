## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** window 1 charge 300 GREEN. Window 2 rain park **(48, 96)** charge
300 GREEN (2200→1900). Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 2 --weapon beam --wait 4000`
**Report:** `scratch/phantoon_window_beam_w2_rain48.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start a 16k until more windows chip. Do not fire x=219. Do not
jump under (128, 96). Do not charge (88, 64) from the left seat (crosses
the body).

### Public policy (wiki)

https://wiki.supermetroid.run/Phantoon
https://wiki.supermetroid.run/Phantoon#Phantoon_First

KPDR beginner 4-round: charge when the eye opens, two more, repeat.
Super **enrages**. Flame rain: this pin's **(48, 96)** is a charge window
from the living left seat; (128, 96) is jump-under; (88, 64) crossed.

### What works (verified this pass)

  | Probe | Park | Spend | Shots | HP | Health |
  |-------|------|-------|------:|----|-------:|
  | w1 | (120, 108) fig-8 | (104, 149) p43 UP | 1 charge | 2500→2200 | 239 |
  | w2 rain48 | (48, 96) `$D767` | (37, 132) p44 UP | 1 charge | 2200→1900 | 59→39 |

- **Hit rule:** `in_release_band` dy 28–56, `$0CD0` ≥60, airborne UP.
  W1 dy=41. W2 dy=36 `|dx|=11` jump-in-place (eye sits on the seat).
- Skip right fig-8 (x=219 is the body). Skip rain parks x>64.
- Assist off.

### What fails

1. **(88, 64) from (37, 187)** dashed through the body: p83 at (101, 127),
   min_y 117 `|dx|=25` not close, charge dumped on land, HP 2200.
2. **(128, 96)** jump-under already missed. **x=219** is the wall/body.
3. **Full fight RED.** Two chips (600) leave 1900. Rain continues. Health
   39 after W2. Dual-green kill still needs HP 0 + boss bit ×2.

### Next actions (do not start a 16k first)

1. `--windows 3` (or 4) skip right + skip (128, 96) + skip (88, 64); charge
   the next **(48, 96)** or a later left fig-8. Halt at first miss or
   health≤20.
2. Dual-green `scratch/post_phantoon_poweron.state` only after a kill
   (do **not** clobber `post_phantoon_defeated.state`).

```bash
uv run python snes/super_metroid/scripts/probe/phantoon_combat.py window --windows 2 \
  --weapon beam --wait 4000 --report snes/super_metroid/scratch/phantoon_window.json
```

### Non-claims

- Did not STATUS-promote past Ice
- Did not change `DEFAULT_CONTINUOUS_TIP`
- Did not write `recordings/ws.json`
- Did not append to `POST_ICE_SPINE` / `WS_ONLY_HOPS` / `--to ws`
- Did not close `rr-g3nj`
- Did not rewrite `play_ws_entrance_to_main` / `play_ws_main_to_basement` /
  `play_ws_basement_to_phantoon`
- Did not close `rr-tlaq` (full kill still RED)
