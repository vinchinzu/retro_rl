> Historical lane note. The live sitting is the gathering prefix in [PRE_L1.md](../PRE_L1.md). This file stays because the clean-tip ladder or a route doc still cites it. It is not the current plan.

# rr-npv.2 residual — Clean L5 Entrance→TF

Stopped at fixture-live. Not a STATUS claim. `route_eligible=false`.

## Landed

- `Level5PolsVoiceController`: hop = displacement (`vx/vy != 0` or
  `state==1`). ObjState is not flyer-state.
- Strike only when still (`state==0` and not airborne) and aligned 12–24.
- Do not chase a leaping body. Projected landing feeds leap-evade (≤40) and
  aborts a backstep if a hopper is already inside 20. No path → stand.
- Unit: leftover pose `(133,157)` with a north-hopping PV must `evade_leap_*`,
  not `A` / DOWN.
- 0x66: `contact_backstep=16` + `Level5Room66Controller` NW pocket peel.
  ROM 0x66 still green 1379f.
- `Level5PolsSouthController`: waist DOWN; SW pocket RIGHT; slash only
  y>=173. 2x3 islands stay `_is_solid`. Lip peel `_ROOM77_LIP_Y = 177`.
- Hold y=173. Hopper in leap range peels LEFT/RIGHT **before** center-seek,
  even at x==120 and at x=116/124 — do not walk back under the landing.
  Slash only landed (state==0, zero displacement, cheb<=24). x in [96,132],
  no y>=177, no RIGHT at x>=132.
  Units: lip `(142,181)` / `(136,179)`; hopper `(136,173)` L/R; `(139,173)`
  LEFT not RIGHT; `(131,173)` LEFT toward 120; leftover `(120,173)` leaping
  Pols at x≈120 L/R not A/idle; (116,173) LEFT not RIGHT; (124,173) RIGHT
  not LEFT; landed still `77_hold_slash`.
  `dungeon.py` 994. `path.py` 641.

`dungeon.py` 994 LOC. `path.py` 641. `spine.py` 503.

## Unit

`QT_QPA_PLATFORM=offscreen uv run pytest nes/zelda_i/tests/test_level5*.py -q`
→ 38 passed.

## One ROM glance (this sitting)

`Level5EntranceFromL4` start: play 0x76 `(120,205)` mode 5, raft=1 ladder=1
bombs=7 keys=0 TF=`0x0c` whistle=0 health=`0x7F`.

Hypothesis: at hold y=173, hopper peels L/R even when x==120 (and when
x=116/124) — do not recenter under the landing. Slash only landed.

| stage | frames | ok |
|-------|--------|----|
| `level5_clear_0x66` | 1379 | **yes** |
| `level5_east_key_0x77` | 652 | **yes** |
| `level5_clear_0x77` | 4250 | **no** (death) |

Leftover: **room 0x77 mode 17 (death) `(120,173)`**, deaths=1, TF=`0x0c`
(no 0x10), keys=1, bombs=8, whistle=0, health=`0x70` (lo=0 hi=7, not full).
0x66 still green.

PNG: `recordings/l5_entrance_tf_clean_final.png` (mode-17 CONTINUE, 0x77
green floor, puff south-aisle center between the 2x3 islands).
JSON: `recordings/l5_entrance_tf_clean.json` (6281f,
`failed_stage=level5_clear_0x77`).

Same pose **3rd** time (136→139→131→120, then 120, then 120). 0x77 frames
4250 vs 1510 (last) / 1538 — peel-before-recenter lived longer, still died
on the stand cell. Occupancy: miss → block cell → replan; no path → stand.
Map from `$6530` not `$049E`.

## Blocked

**hold-hopper 3/3 blocked.** Do not resume L/R peel / center-seek on y=173
x≈120. New class required (leave the stand cell, or extra-block hopper
landing occupancy). `route_eligible` stays false.
