## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** window 1 chips (missile 100 **or** charge 300); window 2 miss
(center park too low). Full fight RED. Seat wait still eats 20 from a
figure-8 tear; fire eats another 20–40 (round-2 health 239, not 0).
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
Room `0xCD13` `(39,124)` p81 gs=8 dt=0 items `0x3105` beams `0x1007`
selected=2 missiles 20/20 supers 5/5 energy 299/299.
**Seat:** land to `(39, 187)` pose 1, then left-corner floor `x≤56 y≈187`.
**Probe:** `uv run python snes/super_metroid/scripts/probe/phantoon_combat.py window --windows 1 --weapon beam`
**Reports:** `scratch/phantoon_window_ms_v11.json` (missile 100),
`scratch/phantoon_window_beam_v2.json` (charge 300),
`scratch/phantoon_window_beam_w4b.json` (w1 300, w2 miss halt).

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
  | beam w4b w1 | (120, 108) | (104, 149) p43 | 1 charge | 2500→2200 | 239 |
  | beam w4b w2 | (128, 96) center | (120, 174) p21 too low | 0 | 2200 miss | 199 |

- **Hit rule:** dash RIGHT+B to `|dx|≤16`, jump, **UP only** on the spend
  (no LEFT/RIGHT). Floor spends miss. Missiles: count ammo delta, chip
  **100**. Charge: keep `$0CD0` ≥60 through the dash, release airborne at
  y≈149; chip **300**. Seat-only charge release misses the eye.
- Assist off: `energy_writes 0`, `missile_writes 0`.

### What fails

1. **Full fight dies ~4.5–4.8k** with HP 1900–2100 left. Varia 20 dmg
   flames at the seat; one 40-dmg frame (double flame / contact) also
   seen. Samus 299 cannot tank a 16k. Do not chase right-side parks
   (`enemy0_x≥155`) — that was death at `x≈210`.
2. **One connecting missile per left-side window** in the fight loop
   (window v8's second hit was a swoop chase). 20 missiles × 100 = 2000
   even if every shot hits — still 500 short of 2500 without charge/farm.
3. **Window 2 miss (halt).** After a 300 charge he hides (`0xD6D4`). Next
   left-ish open is a **center park (128, 96)** — close at x=120 but y=174
   is still `on_floor`, so the charge dumps too low. Do not chase
   `enemy_x≥155` (w4 first attempt was (203, 83) skip).
4. **Seat flames.** Figure-8 tear ~f980 costs 20 even on beam/morph/crouch.
   Opening spiral costs 20–40 unless the dash leaves the SW corner before
   ~f1552. Horizontal charge from the seat can eat some flames (drop
   `$F337` seen once at (121, 186)) but does **not** chip the eye.
5. **Super still unused** (correct). Do not spray.

### Next actions (do not start another 16k first)

1. Jump higher on the (128, 96) park so the charge releases at y≈149, not
   174 — same UP-only spend as window 1. Halt if that misses.
2. Snipe the ~f980 figure-8 tear (horizontal Wave when a `0x9C29` flame
   has x<90, y>140) so window 1 starts at 299 not 279.
3. Four **left-side** charge windows only after (1) chips. Then dual-green
   `scratch/post_phantoon_poweron.state` (do **not** clobber
   `post_phantoon_defeated.state`).

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
