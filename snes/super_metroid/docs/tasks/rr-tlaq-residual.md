## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** window 1 charge 300 GREEN. Window 2 right fig-8 **miss** (body
contact at the W1-mirror pose). Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 2 --weapon beam --wait 4000`
**Report:** `scratch/phantoon_window_beam_w2_right.json`

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
  | beam v2 | (120, 108) | (104, 149) p43 UP release | 1 charge | 2500→2200 | 239 |
  | beam w2_right w1 | (120, 108) | (104, 149) p43 | 1 charge | 2500→2200 | 239 |
  | beam w2_right w2 | (203, 83) RIGHT | (219, 148) p84 | 1 charge | 2200 miss | 219 |

- **Hit rule (W1):** dash RIGHT+B to `|dx|≤16`, jump, release only in
  `in_release_band` (dy 28–56 below the eye; W1 was 41). Keep `$0CD0` ≥60
  through the dash. **A on the approach spin-dumps charge (p77).** Chip
  **300**. Jump only once close.
- **Park x at func change**, not live `enemy_x`. Live x crosses 155 mid
  left fig-8 then opens at (120, 108). Walking on live x is death under the
  body (this bump, first probe: pre-fire `(119, 187)` p12, W1 miss).
- Assist off: `energy_writes 0`, `missile_writes 0`.

### What fails

1. **W2 right fig-8 (203, 83) — halt.** After W1 hide `$D6D4`, `$0FB2`:
   `D6E2` → `D72D` place **(208, 96)** → `D5E7` fig-8 → `D4A8`/`D60D` open
   **(203, 83)** ~f2473. Floor dash to right seat `(219, 187)` charge 120
   (do not jump while crossing). Mirror W1: face LEFT, jump in place, UP
   release at y≈149 `|dx|=16`. Spend `(219, 148)` **p84 hurt**, health
   239→219, charge dumps, **HP stays 2200**. Shallower `(214, 159)` p22
   did not contact and did not chip. Do not fire from the left seat.
   Do not dash from the left under the body.
2. **Standing rain wait dies** (239→0). Morph left corner after the W2 miss
   was alive at 179 p29 `(51, 201)` through `$D767`/`$D788` in a 30f dump
   (not a full rain cycle). `rain_phase` is `D82A`/`D73F`/`D767`/`D788`/
   `D7D5`/`D7F7` — standing through `D82A` is what killed the skip-rain wait.
3. **Unmorph while charged dumps the shot** (`unmorph` holds A). Do not
   unmorph pose 137 at the right seat if `$0CD0` ≥60.
4. **Full fight dies ~4.5–4.8k** with HP 1900–2100 left if it never chips
   W2. Samus 299 cannot tank a 16k.
5. **Super still unused** (correct). Do not spray.

### Next actions (do not start another 16k first)

1. W2 from **further right** (outside the (203, 83) hurtbox at y≈149) with
   airborne UP+LEFT, **or** skip the right open and morph through a full
   rain cycle (80f health + `$0FB2` + xy). Halt at first miss.
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
