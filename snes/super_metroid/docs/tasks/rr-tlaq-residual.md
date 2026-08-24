## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** window 1 charge 300 GREEN. Skip-rain wait: **no second fig-8
left open in ~4k** — next fig-8 is RIGHT (203, 83), then rain until death.
Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 2 --weapon beam --wait 4000`
**Report:** `scratch/phantoon_window_beam_w2_skiprain.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start another 16k until the seat survives flames.

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

- **Seat** from the enter pin: idle fall to `(39, 187)` p1. `seated()` is
  standing/crouch in `x=16–56`, `y=160–210`, not morph, not airborne.
- **Weapon:** pin selected=2 (supers). Cycle to missiles (1) or beam (0)
  during `_go_to_seat`. Charge RAM `$0CD0` only increments on **beam**.
- **Window 1 park** always `(120, 108)` this seed (fast/mid/slow is the
  *timer*, this pin rolled a mid/slow ~f1513).

  | Probe | Park | Spend | Shots | HP | Health |
  |-------|------|-------|------:|----|-------:|
  | ms v11 | (120, 108) | (123, 149) p21 UP | 1 missile | 2500→2400 | 259 |
  | beam v2 | (120, 108) | (104, 149) p43 UP release | 1 charge | 2500→2200 | 239 |
  | beam w2b w1 | (120, 108) | (104, 149) p43 | 1 charge | 2500→2200 | 239 |
  | beam w2_rain w2 | (128, 96) rain | (116, 187) p3 floor UP | 1 charge | 2200 miss | 59 |
  | beam w2_corner w2 | (128, 96) rain | (55, 187) p3 UP+R | 2 charges | 2200 miss | 99 |

- **Hit rule (W1):** dash RIGHT+B to `|dx|≤16`, jump, release only in
  `in_release_band` (dy 28–56 below the eye; W1 was 41). Keep `$0CD0` ≥60
  through the dash. **A on the approach spin-dumps charge (p77).** Chip
  **300**. A on the approach is a regression — jump only once close.
- Assist off: `energy_writes 0`, `missile_writes 0`.

### What fails

1. **Full fight dies ~4.5–4.8k** with HP 1900–2100 left. Varia 20 dmg
   flames at the seat; one 40-dmg frame (double flame / contact) also
   seen. Samus 299 cannot tank a 16k. Do not chase right-side parks
   (`enemy0_x≥155`) — that was death at `x≈210`.
2. **One connecting missile per left-side window** in the fight loop
   (window v8's second hit was a swoop chase). 20 missiles × 100 = 2000
   even if every shot hits — still 500 short of 2500 without charge/farm.
3. **No second fig-8 left open (halt).** After W1 hide `$D6D4`, `$0FB2`:
   `D6E2` → `D72D` place **(208, 96)** → `D5E7` fig-8 right → `D4A8`/`D60D`
   open **(203, 83)** (skipped, `enemy_x≥155`) → rain `D82A` then
   `D73F`/`D767`/`D788`/`D7D5`/`D7F7` cycling parks (128,96), (88,64),
   (48,96), (168,64) until Samus dies (239→0) sitting left. `charge_window_ok`
   skips rain + right. Never a second `(120,108)`-style left charge.
4. **Seat flames.** Figure-8 tear ~f980 costs 20 even on beam/morph/crouch.
   Opening spiral costs 20–40 unless the dash leaves the SW corner before
   ~f1552. Horizontal charge from the seat can eat some flames (drop
   `$F337` seen once at (121, 186)) but does **not** chip the eye.
5. **Super still unused** (correct). Do not spray.

### Next actions (do not start another 16k first)

1. Survive sitting through rain (morph / flame snipe) so a later fig-8
   left open is reachable — this seed's **next** fig-8 after W1 is RIGHT
   (203, 83). Do not chase it.
2. Snipe the ~f980 figure-8 tear so W1 starts at 299 not 279.
3. Dual-green `scratch/post_phantoon_poweron.state` only after a kill
   (do **not** clobber `post_phantoon_defeated.state`).

```bash
# From repo root. Window 1 should still chip 100:
uv run python snes/super_metroid/scripts/probe/phantoon_combat.py window --windows 1 \
  --report snes/super_metroid/scratch/phantoon_window.json
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
