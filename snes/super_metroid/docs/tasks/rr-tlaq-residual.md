## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** W1–W6 charge 300 GREEN (2500→**700**). W7 **halt** health 14
during `$D82A` (did not jump (56, 113)). Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 7 --weapon beam --wait 4000`
**Report:** `scratch/phantoon_window_beam_w7_y96.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start a 16k. Do not fire x=219 / (128, 96) / (88, 64) / **(53, 82)**
/ **(56, 113)**. Do not sit-charge. Do not 2k farm. Do not jump at health≤20.

### Health in/out

  | W | Park | Spend | HP | Health in→out |
  |---|------|-------|----|-------:|
  | 1–6 | (48, 96) / W1 fig-8 | (37, 148) p21 | 2500→**700** | **54** |
  | 7 wait | skip (53, 82) `$D4A8`/`$D60D` | p83 −20 | 700 | 54→34 |
  | 7 wait | `$D82A` | — | 700 | 34→**14** halt |

`rain_charge_ok` is now **x≤56 and y 88–104**. (56, 113) skipped. Wait
aborts at health≤20. Never a legal (48, 96) `$D767` while HP>20.

### What fails

1. **54 HP cannot survive skip-(53, 82) tear + `$D82A`** to the next
   (48, 96). Dual-green still needs HP 0 + boss bit ×2.
2. Skip forever: x=219, (128, 96), (88, 64), (53, 82), (56, 113).

### Next actions (do not start a 16k first)

1. Eat fewer −20s on the (53, 82) skip, or get a (48, 96) **before**
   `$D82A` while HP>20. Halt at death.
2. Dual-green `scratch/post_phantoon_poweron.state` only after a kill
   (do **not** clobber `post_phantoon_defeated.state`).

```bash
uv run python snes/super_metroid/scripts/probe/phantoon_combat.py window --windows 6 \
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
