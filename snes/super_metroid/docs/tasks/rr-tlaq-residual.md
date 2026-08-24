## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** W1–W3 charge 300 GREEN (2500→1600). Farm-after-W2 did **not**
lift W3 start to ≥140. Farm before W4 **died**. Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 4 --weapon beam --wait 4000`
**Report:** `scratch/phantoon_window_beam_w4_farmw2.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start a 16k. Do not fire x=219 / (128, 96) / (88, 64).
Do **not** attempt W5 at 4 HP.

### Health per window (this pass)

  | W | Park | Spend | HP | Health in→out | Farm after |
  |---|------|-------|----|-------:|-------------|------------|
  | 1 | (120, 108) | (104, 149) p43 | 2500→2200 | 239 | 2000f 239→184 (`+5`) |
  | 2 | (48, 96) | (37, 148) p21 | 2200→1900 | 164→164 | 2000f 164→**109** (`+5`) |
  | 3 | (48, 96) | (37, 148) p21 | 1900→1600 | **69→69** | 1702f 69→**0** (died `$D82A` (52, 107)) |
  | 4 | — | — | 1600 | — | never fired |

W3 started at **69** (not ≥140). Wait after W2 farm ate 109→69. Farm at 69
cannot support W4. Tap-snipe net **−55 / 2k** (same as W1→W2). Pickup
table empty.

### What fails

1. **No-assist energy cannot support 5 more 300s from this seat.** Farm
   after W2 does not raise W3 start to 140. Farm after W3 at 69 dies.
   Previous W4-at-4-HP path is also a dead end.
2. Skip forever: x=219, (128, 96), (88, 64), morph-tank.

### Next actions (do not start a 16k first)

1. A **real energy farm** (drops that stick) or a **cheaper W3 wait** so
   164 does not fall to 69. Halt at death. Do not W5 at 4 HP.
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
