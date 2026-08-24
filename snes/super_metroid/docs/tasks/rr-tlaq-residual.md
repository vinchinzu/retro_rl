## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** W1–W6 charge 300 GREEN (2500→**700**). W7 **died** at 19 HP
jumping rain park **(56, 113)** (not (48, 96)). Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 7 --weapon beam --wait 4000`
**Report:** `scratch/phantoon_window_beam_w7.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start a 16k. Do not fire x=219 / (128, 96) / (88, 64) / **(53, 82)**.
Do not sit-charge rain. Do not 2k farm.

### Health in/out

  | W | Park | Spend | HP | Health in→out |
  |---|------|-------|----|-------:|
  | 1 | (120, 108) | (104, 149) p43 | 2500→2200 | 279→239 |
  | 2–6 | (48, 96) rain | (37, 148) p21 | 2200→**700** | 199→**54** |
  | 7 wait | skip (53, 82) `$D4A8`/`$D60D` | p83 −20 | 700 | 54→34 |
  | 7 wait | `$D82A` | — | 700 | 34→19 |
  | 7 | rain **(56, 113)** `$D767` | p83 (37, 187) | 700 | 19→**0** died f12861 `$D788` |

`rain_charge_ok` is x≤64 only — **(56, 113)** matched and we jumped. That
park is not (48, 96). Probe now aborts wait at health≤20.

### What fails

1. **W7 halt.** Need rain park **y≈96**, not any x≤64. (56, 113) is skip.
2. Skip (53, 82) still ate −20 sitting (p83 at D4A8). 54 HP cannot eat
   skip-tear + D82A + a bad jump.
3. Full fight RED (700 HP). Dual-green still needs HP 0 + boss bit ×2.

### Next actions (do not start a 16k first)

1. Rain close only if **x≤64 and y≈96** (not (56, 113)). Abort at
   health≤20. Halt at miss.
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
