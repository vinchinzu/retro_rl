## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** No-farm consecutive (48, 96): W1+W2 GREEN, W3 **died** at 39 HP.
Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 4 --weapon beam --wait 4000` (farm_frames=0)
**Report:** `scratch/phantoon_window_beam_w4_nofarm.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start a 16k. Do not fire x=219 / (128, 96) / (88, 64).
Do **not** W5 at 4 HP. Do **not** 2k farm (net −55 / 2k).

### Health in/out (no farm)

  | W | Park | Spend | HP | Health in→out |
  |---|------|-------|----|-------:|
  | 1 | (120, 108) | (104, 149) p43 | 2500→2200 | 279→239 |
  | wait | skip-right + rain to first (48, 96) | — | — | 239→59 |
  | 2 | (48, 96) `$D767` | (37, 147) p44 | 2200→1900 | 59→**39** |
  | wait | hide `$D6B9`/`$D6D4` → `$D72D` (208, 96) `$D5E7` → `$D4A8`/`$D60D` **(167, 128)** skip → `$D82A` | — | — | 39→19→**0** |
  | 3 | — | — | 1900 | **died** f4518 `$D82A` (196, 116) p1 (37, 187) |

W2 chip **hides** Phantoon (`$D6B9`). Next cycle is a **right fig-8**, not
another (48, 96). `$D5E7` tear 39→19; `$D82A` (196, 116) kills.

### What fails

1. **Must skip damaging parks between (48, 96) cycles.** Sitting through
   `$D5E7` / `$D82A` (52, 107) / (196, 116) / (167, 128) one-shots 39 HP.
   Same class as skip (207, 101).
2. 2k tap-snipe farm is net −55 and cannot lift energy.
3. Full fight RED. Dual-green still needs HP 0 + boss bit ×2.

### Next actions (do not start a 16k first)

1. Skip `$D82A` / `$D5E7` flame parks the way we skip (207, 101) — live
   through hide+right-fig-8 without eating the tear. Halt at death.
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
