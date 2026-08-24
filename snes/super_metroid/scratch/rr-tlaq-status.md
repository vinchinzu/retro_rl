# rr-tlaq Phantoon fight — status

**Bead:** rr-tlaq IN PROGRESS (full kill RED). Do not STATUS-promote.

## Window table

| W | Park | Spend | HP | Health | Result |
|---|------|-------|----|-------:|--------|
| 1 | (120, 108) fig-8 | (104, 149) p43 | 2500→2200 | 239 | GREEN **300** |
| 2 | (203, 83) RIGHT | (219, 148) p84 | 2200 | 219 | **miss** (body contact) |

## `$0FB2` after W1

`D6D4` hide → `D6E2`/`D72D` place (208, 96) → `D5E7` fig-8 right →
`D4A8`/`D60D` open **(203, 83)** ~f2473. Floor seat `(219, 187)` charge 120.
Mirror W1 jump-in-place UP at y≈149 `|dx|=16` → p84 −20, charge dump, HP 2200.
Then rain `D82A`/`D73F`/`D767`/`D788`…

Park x at **func change**, not live `enemy_x` (left fig-8 crosses 155).
`charge_window_ok` skips `rain_phase` only. No 16k. No Super.

## Rain (30f post-miss, not a full cycle)

Morph left `(51, 201)` p29, health 179, `$D767`/`$D788` vs (186, 123).
Standing wait previously died 239→0. Full rain survival still unverified.

## Next

W2 from further right (outside the hurtbox) **or** skip and morph a full
rain. Halt at first miss. `rr-tlaq` open. Dual-green kill still needs HP 0
+ boss bit ×2 → `scratch/post_phantoon_poweron.state`.
