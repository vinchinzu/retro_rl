## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** W1+W2 charge 300 GREEN each. W3 died waiting for the next
**(48, 96)** / left fig-8 (health 39→0). Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 3 --weapon beam --wait 4000`
**Report:** `scratch/phantoon_window_beam_w3.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start a 16k. Do not fire x=219. Do not jump under (128, 96).
Do not charge (88, 64) from the left seat.

### What works (verified this pass)

  | Probe | Park | Spend | Shots | HP | Health |
  |-------|------|-------|------:|----|-------:|
  | w1 | (120, 108) fig-8 | (104, 149) p43 | 1 charge | 2500→2200 | 239 |
  | w2 rain48 | (48, 96) `$D767` | (37, 132) p44 | 1 charge | 2200→1900 | 59→39 |

- **Hit rule:** dy 28–56, `$0CD0` ≥60, airborne UP. W2 jump-in-place `|dx|=11`.
- Assist off.

### What fails

1. **W3 halt — died at 39 HP before the next legal park.** After W2, tap-snipe
   farm 589f: `health_up=20` (39→59 then back; pickup table empty), farm
   end 39. W3 wait: `$D82A`→`$D788` **(207, 101)** skipped (right), health
   39→19→**0** at f4617 pose 1 `(37, 187)`. Never a (48, 96) or left fig-8.
   One contact at ≤39 kills.
2. Skip forever: x=219, (128, 96), (88, 64) from the left seat.
3. Full fight RED. Two chips leave **1900**. Dual-green still needs HP 0
   + boss bit ×2.

### Next actions (do not start a 16k first)

1. Survive at 39 until the next **(48, 96)** without eating the (207, 101)
   rain — or farm a real energy drop **before** leaving i-frames. Halt at
   first miss / death. Do not chase right parks.
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
