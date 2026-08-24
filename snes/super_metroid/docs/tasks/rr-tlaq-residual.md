## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** window 1 charge 300 GREEN. Window 2 right fig-8 **miss** (higher
jump, W1 dy band, body contact on the way up). Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 2 --weapon beam --wait 4000`
**Report:** `scratch/phantoon_window_beam_w2_high.json` (x≥230 retry:
`scratch/phantoon_window_beam_w2_x230.json`)

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start another 16k until a measured W2 chip.

### Public policy (wiki)

https://wiki.supermetroid.run/Phantoon
https://wiki.supermetroid.run/Phantoon#Phantoon_First

KPDR beginner 4-round: charge when the eye opens, two more, repeat.
Charge+Missiles: 2 + 2 + charge × 4. Super **enrages**. Pin boots with
supers selected — `select_weapon` to missiles/beam before the first open.

Eye is **slot 1** (not enemy0 spritemap). Live flags:
- body function `$0FB2` `0xD60D` / `0xD767` / `0xD788` (open windows)
- eye IL `$0FD2` in `0xCC53–0xCC7A` (open anim) or `0xCC9D–0xCCD6` (look)
- intro is `$0FB2=0xD4A9…D5E7`; figure-8 starts ~f793; first open ~f1513–1594

### What works (verified this pass)

- **Seat** from the enter pin: idle fall to `(39, 187)` p1.
- **Weapon:** pin selected=2 (supers). Cycle to beam (0) during `_go_to_seat`.
- **Window 1 park** always `(120, 108)` this seed.

  | Probe | Park | Spend | Shots | HP | Health |
  |-------|------|-------|------:|----|-------:|
  | beam v2 | (120, 108) | (104, 149) p43 UP | 1 charge | 2500→2200 | 239 |
  | beam w2_high w1 | (120, 108) | (104, 149) p43 | 1 charge | 2500→2200 | 239 |
  | beam w2_high w2 | (203, 83) RIGHT | (219, 126) p44 after p84 @138 | 1 charge | 2200 miss | 219 |

- **Hit rule (W1):** dash RIGHT+B to `|dx|≤16`, jump, release only in
  `in_release_band` (dy 28–56 below the eye; W1 was 41). Keep `$0CD0` ≥60
  through the dash. **A on the approach spin-dumps charge (p77).** Chip
  **300**. Jump only once close.
- **Park x at func change**, not live `enemy_x`.
- Assist off: `energy_writes 0`, `missile_writes 0`.

### What fails

1. **W2 higher jump (W1 dy band) — halt.** Eye at y=83 → target y=111–139
   (not 148). Floor seat `(219, 187)` charge 120. Face LEFT, longer A:
   p84 **hurt at (219, 138)** dy=55 (239→219), charge still 120, then
   fire `(219, 126)` p44 dy=43 `|dx|=16` — charge dumps, **HP 2200**.
   min_y 123. Same air column as the y=148 miss; higher is still the body.
2. **x≥230 is the right wall.** Samus sticks at x=219 p137. Wait-for-230
   skipped the open, then rain death. Do not require x≥230.
3. **Standing rain wait dies** (239→0). Morph left `(51, 201)` p29 was
   alive at 179 through `$D767`/`$D788` in a 30f dump (not a full cycle).
4. **Super still unused** (correct). Do not spray.

### Next actions (do not start another 16k first)

1. Skip the right open and **morph a full rain cycle** (80f health + `$0FB2`
   + xy), **or** a different W2 angle that is not the x=219 column.
   Halt at first miss.
2. Dual-green `scratch/post_phantoon_poweron.state` only after a kill
   (do **not** clobber `post_phantoon_defeated.state`).

```bash
# From repo root. Window 1 should still chip 300:
uv run python snes/super_metroid/scripts/probe/phantoon_combat.py window --windows 1 \
  --weapon beam --report snes/super_metroid/scratch/phantoon_window.json
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
