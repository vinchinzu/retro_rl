## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** window 1 charge 300 GREEN. Skip-right + morph rain: **died** in
`$D788` before a left fig-8. Full fight RED.
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
**Probe:** `window --windows 2 --weapon beam --wait 6000`
**Report:** `scratch/phantoon_window_beam_w2_morphrain.json`

Do **not** STATUS-promote. Default CLI stays `ice`. Super-spray is not a hit.
Do **not** start another 16k until a measured W2 chip. Do not fire the
right fig-8 at x=219 (the body).

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
- **Window 1 park** always `(120, 108)` this seed. Charge **300** at
  `(104, 149)` p43 vs `(120, 108)` dy=41.
- `charge_window_ok` skips rain **and** `enemy_x≥155` (park x at func
  change, so a left fig-8 that crosses 155 still counts).
- Assist off: `energy_writes 0`, `missile_writes 0`.

### What fails

1. **Morph rain dies — halt.** Skip right open `(203, 83)` ~f2473. Morph
   left `(53, 201)` p29/65 from `D72D`. Health 239→0 by ~f4100 in `$D788`
   vs (168, 64), pose 65 `(52, 201)`. No left fig-8. Rolling to the other
   corner when the body parks on the left (`enemy_x≤80`) walks through the
   wave (−20 each pass: 239→219→199→179→159→139→119→99→79→59→39→19→0).
2. **Right fig-8 at x=219 is the body.** Do not retry x≥230 (wall, p137).
   Higher jump p84 at (219, 138).
3. **Standing rain wait dies** 239→0. Super unused (correct).

### Rain dump (every ~30f)

`D72D` place (208, 96) → `D5E7` fig-8 (skip) → `D4A8`/`D60D` open (203, 83)
skipped in morph → `D82A` then `D73F`/`D767`/`D788`/`D7D5`/`D7F7` parks
(200,114)→(128,96)→(88,64)→(48,96)→(168,64). Death sitting morph-left
during (168, 64) rain.

### Next actions (do not start another 16k first)

1. Morph rain that **does not roll across the room** when the body parks
   on the left seat (stay left, or swap without crossing the wave). Halt
   at first miss. Do not fire x=219.
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
