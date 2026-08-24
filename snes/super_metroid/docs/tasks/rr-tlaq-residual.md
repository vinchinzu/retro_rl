## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** Jump only rain (48, 96). W1–W4 GREEN. W5 **halt** health 19 on
`$D82A` (no skip-jump). Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 7 --weapon beam --wait 4000`
**Report:** `scratch/phantoon_window_beam_w7_snipeonly.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start a 16k. Jump **only** rain (48, 96) y 88–104. Skip (53, 82),
(83, 64), x=219, (128, 96), (88, 64), (56, 113). No sit-charge. No 2k farm.

### Health in/out

  | W | Park | Spend | HP | Health in→out |
  |---|------|-------|----|-------:|
  | 1 | (120, 108) | (104, 149) p43 | 2500→2200 | 279→239 |
  | 2 | (48, 96) | (37, 148) p21 | 2200→1900 | 219→199 |
  | 3 | (48, 96) | (37, 148) p21 | 1900→1600 | 139→119 |
  | 4 | (48, 96) | (37, 148) p21 | 1600→1300 | 79→59 |
  | 5 wait | `$D5E7` (208, 96) pose 3 | — | 1300 | 59→39 |
  | 5 wait | skip fig-8 (167, 128) | — | 1300 | 39 |
  | 5 wait | `$D82A` pose 3 | — | 1300 | 39→**19** halt |

No A on skip (pose 3 throughout). `$D82A` still 39→19. 59 HP cannot tank
`$D5E7` + `$D82A` to the next (48, 96).

### What fails

1. **54–59 HP cannot tank `$D82A` between (48, 96) cycles** even with
   snipe-only skip (no jump). Dual-green still needs HP 0 + boss bit ×2.
2. Skip forever: x=219, (128, 96), (88, 64), (53, 82), (83, 64), (56, 113).

### Next actions (do not start a 16k first)

1. Get a (48, 96) **before** `$D82A` while HP>20, or accept 7 windows is
   past the energy budget. Halt at death.
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
