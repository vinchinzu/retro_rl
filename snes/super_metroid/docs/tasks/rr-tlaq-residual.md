## Residual — rr-tlaq Phantoon fight pure (0xCD13)

**Status:** window 1 chips HP from the natural pin (1 missile → 100, or 2/2
airborne); full fight still RED (died ~1900–2000 HP left).
**Pin:** `scratch/post_ws_basement_to_phantoon.state`
Room `0xCD13` `(39,124)` p81 gs=8 dt=0 items `0x3105` beams `0x1007`
selected=2 missiles 20/20 supers 5/5 energy 299/299.
**Seat:** land to `(39, 187)` pose 1, then left-corner floor `x≤56 y≈187`.
**Probe:** `uv run python snes/super_metroid/scripts/probe/phantoon_combat.py window --windows 1`
**Reports:** `scratch/phantoon_window_v10.json` (1/1 100 HP),
`scratch/phantoon_window_v8.json` (2/2 200 HP),
`scratch/phantoon_fight_v5.json` (6 shots, 2500→1900, died).

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

  | Probe | Park | Spend | Shots | HP |
  |-------|------|-------|------:|----|
  | window v8 | (120, 108) then swoop (69, 100) | (123, 149) p21, (64, 153) p22 | 2/2 | 2500→2300 |
  | window v10 | (120, 108) | (123, 149) p21 UP | 1/1 | 2500→2400 |

- **Hit rule:** dash RIGHT+B on the floor to `|dx|≤16`, then jump, **UP+X
  only** on the spend (no LEFT/RIGHT — diagonal misses the eye-only
  hitbox). Floor spends miss. Count by `missiles` decreasing. Expected
  chip is **100** per connecting missile.
- Assist off: `energy_writes 0`, `missile_writes 0`.

### What fails

1. **Full fight dies ~4.5–4.8k** with HP 1900–2100 left. Varia 20 dmg
   flames at the seat; one 40-dmg frame (double flame / contact) also
   seen. Samus 299 cannot tank a 16k. Do not chase right-side parks
   (`enemy0_x≥155`) — that was death at `x≈210`.
2. **One connecting missile per left-side window** in the fight loop
   (window v8's second hit was a swoop chase). 20 missiles × 100 = 2000
   even if every shot hits — still 500 short of 2500 without charge/farm.
3. **Charge from the seat missed.** Ice+Wave+Spazer is equipped
   (`0x1007`). One full-charge release at `(130, 143)` vs eye `(120, 108)`
   did not chip HP. Ice hide behavior not measured.
4. **Super still unused** (correct). Do not spray.

### Next actions (do not start another 16k first)

1. Survive the seat: morph or shoot the eight opening flames with **beam**
   (then select missiles). Crouch-wait still died at 1900 HP.
2. From `scratch/phantoon_window_v10` geometry, land a **charged** shot
   at the `(123, 149)` vs `(120, 108)` pose (or 2 missiles + charge per
   round) so 20 missiles are not the cap.
3. `strategy --max-frames 16000` only after a kill from the enter pin
   with `energy_writes 0`, `missile_writes 0`, not dead. Then dual-green
   and `scratch/post_phantoon_poweron.state` (do **not** clobber
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
