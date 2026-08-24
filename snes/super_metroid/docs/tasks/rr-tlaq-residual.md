## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** Snipe-wait W2 starts **199**. W1–W3 GREEN. W4 fig-8 (53, 82)
W1-style close **miss** (p83 at y=160 before dy band). Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 4 --weapon beam --wait 4000`
**Report:** `scratch/phantoon_window_beam_w4_fig8.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start a 16k. Do not fire x=219 / (128, 96) / (88, 64).
Do not sit-charge rain. Do not 2k farm.

### Health in/out

  | W | Park | Spend | HP | Health in→out |
  |---|------|-------|----|-------:|
  | 1 | (120, 108) | (104, 149) p43 | 2500→2200 | 279→239 |
  | 2 | (48, 96) rain | (37, 148) p21 | 2200→1900 | **199→179** |
  | 3 | (48, 96) rain | (37, 148) p21 | 1900→1600 | 164→144 |
  | 4 | fig-8 **(53, 82)** `$D4A8` | p83 (37, **160**) dy=78 | 1600 | 124→104 **miss** |

W4: charged 120, jump-in-place from x=37. p83 at y=160 after 5f (never
reached fire y=110–138). Charge kept; window closed `$D82A`. min_y 150.

### What fails

1. **Cannot reach dy 28–56 vs (53, 82) from the living seat.** Body contact
   at y=160 (dy=78) is above the band (y=110–138). Same as jump-under.
2. Full fight RED (1600 HP). Dual-green still needs HP 0 + boss bit ×2.

### Next actions (do not start a 16k first)

1. Skip the (53, 82) fig-8 (like x=219) and wait for the next **(48, 96)**
   rain close. Halt at miss / health≤20.
2. Dual-green `scratch/post_phantoon_poweron.state` only after a kill
   (do **not** clobber `post_phantoon_defeated.state`).

```bash
uv run python snes/super_metroid/scripts/probe/phantoon_combat.py window --windows 3 \
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
