## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** W1–W4 charge 300 GREEN each (2500→**1300**). Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 4 --weapon beam --wait 4000`
**Report:** `scratch/phantoon_window_beam_w4.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start a 16k. Do not fire x=219 / (128, 96) / (88, 64).

### What works (this pass)

  | W | Park | Spend | HP | Health |
  |---|------|-------|----|-------:|
  | 1 | (120, 108) fig-8 | (104, 149) p43 | 2500→2200 | 239 |
  | farm | W1→W2 tap-snipe 2000f | — | — | 239→184 (`health_up=5`) |
  | 2 | (48, 96) `$D767` | (37, 148) p21 | 2200→1900 | 164 |
  | 3 | (48, 96) `$D767` | (37, 148) p21 | 1900→1600 | 44 |
  | farm | W3→W4 tap-snipe 2000f | — | — | 44→24 (`health_up=20`) |
  | 4 | (48, 96) `$D767` | (37, 147) p43 | 1600→1300 | **4** |

- Rain fire dy 48–56 (y=144–152) p21/p43 jump-in-place. Assist off.

### What fails

1. **Full fight RED.** Four chips leave **1300**. Health **4** after W4 —
   one contact kills. Farm W3→W4 did not reach 100 (end 24). Dual-green
   still needs HP 0 + boss bit ×2.
2. Skip forever: x=219, (128, 96), (88, 64) from the left seat, morph-tank.

### Next actions (do not start a 16k first)

1. Farm harder after W4 or accept 4 HP cannot take another window. Halt at
   death. Do not chase right parks.
2. Dual-green `scratch/post_phantoon_poweron.state` only after a kill
   (do **not** clobber `post_phantoon_defeated.state`).

```bash
uv run python snes/super_metroid/scripts/probe/phantoon_combat.py window --windows 4 \
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
