## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** Snipe-wait to first (48, 96): W2 starts **199**. W1–W3 GREEN.
W4 left fig-8 (53, 82) **miss**. Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 4 --weapon beam --wait 4000` (no 2k farm)
**Report:** `scratch/phantoon_window_beam_w2_snipewait.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start a 16k. Do not fire x=219 / (128, 96) / (88, 64).
Do **not** sit-charge through rain. Do **not** 2k farm (net −55).

### Health in/out (snipe-wait, no farm)

  | W | Park | Spend | HP | Health in→out |
  |---|------|-------|----|-------:|
  | 1 | (120, 108) fig-8 | (104, 149) p43 | 2500→2200 | 279→239 |
  | 2 | (48, 96) `$D767` | (37, 148) p21 | 2200→1900 | **199→179** |
  | 3 | (48, 96) `$D767` | (37, 148) p21 | 1900→1600 | 164→144 |
  | 4 | **fig-8 (53, 82)** `$D4A8`/`$D60D` | jump p83 (37, 160) | 1600 | 124→104 **miss** |

W2 start **199 ≥ 150** (sit-charge was 59). Pose 3 UP+tap X until (48, 96),
then floor-charge and cheap jump dy 48–56.

W4 was **not** rain (48, 96). After W3 hide `$D6B9` → `$D72D` (48, 96)
`$D5E7` → open **(53, 82)** fig-8. Jump p83 at y=160, never spent. Miss
dump `$D82A` (48, 96) health 104 charge dumped.

### What fails

1. **W4 halt — left fig-8 (53, 82) is not the rain cheap-jump.** Treat it
   as W1 (dy 28–56 vs y=82 → fire y=110–138), not rain dy 48–56.
2. Full fight RED (1600 HP). Dual-green still needs HP 0 + boss bit ×2.

### Next actions (do not start a 16k first)

1. W4 = W1-style close vs (53, 82). Halt at miss. Do not sit-charge rain.
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
