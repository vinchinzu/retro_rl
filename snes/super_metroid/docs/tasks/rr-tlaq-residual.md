## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** W1–W4 charge 300 GREEN each (2500→**1300**). Skip fig-8 (53, 82).
Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 4 --weapon beam --wait 4000`
**Report:** `scratch/phantoon_window_beam_w4_skip53.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start a 16k. Do not fire x=219 / (128, 96) / (88, 64) / **(53, 82)**.
Do not sit-charge rain. Do not 2k farm.

### Health in/out (snipe-wait, skip (53, 82))

  | W | Park | Spend | HP | Health in→out |
  |---|------|-------|----|-------:|
  | 1 | (120, 108) fig-8 | (104, 149) p43 | 2500→2200 | 279→239 |
  | 2 | (48, 96) rain | (37, 148) p21 | 2200→1900 | 199→179 |
  | 3 | (48, 96) rain | (37, 148) p21 | 1900→1600 | 164→144 |
  | 4 | (48, 96) rain | (37, 148) p21 | 1600→**1300** | 104→**109** |

W4 skipped the (53, 82) fig-8 (snipe-wait) then cheap-jumped the next
(48, 96). Health **109 > 20**. Assist off.

### What fails

1. **Full fight RED.** Four chips leave **1300**. Need ~5 more 300s.
   Dual-green still needs HP 0 + boss bit ×2.
2. Skip forever: x=219, (128, 96), (88, 64), (53, 82) from the left seat.

### Next actions (do not start a 16k first)

1. `--windows 5` same skip set + snipe-wait. Halt at miss or health≤20.
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
