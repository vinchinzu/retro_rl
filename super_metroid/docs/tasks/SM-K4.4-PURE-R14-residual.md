## Residual — SM-K4.4-PURE-R14

### Result
PARTIAL (save-door runway launch shipped from human demo; top / door still red)

Human `tasks/bubble_jump_try.json` (2026-08-03, no beams / no Ice) proves the
maprando **left climb**: run from Save-door outer platform → spin-glide →
walljump. Exact replay peaks **min_y≈158** at ~(240,160). Ceiling pocket at
~y142 x~240–250 blocks “up and over” into Phase D (y≤200 x≥300). Falling right
under the overhang lands floor (~y523) short of place-proven right shelves.

### Files changed
- `routes/kpdr/bubble_mountain_params.py` — `SAVE_RUNWAY_*` / save-run timings
- `routes/kpdr/bubble_mountain.py` — `bubble_on_save_runway`; avoid-left allows
  runway x down to ~35 at save height
- `routes/kpdr/bubble_mountain_mid.py` — R14 save-door runway open-loop before
  lip fallback; human-prefix run/spin/WJ + period-8 follow
- `routes/kpdr/guide_paths.py` — Bubble guide marks save-runway / wj-peak
- `tests/test_k4_norfair_scaffold.py` — save-runway seat unit test
- This residual; tip boards (BUBBLE_MOUNTAIN_TODO / phase ladder) as needed

### Human evidence (load-bearing)

| Fact | Detail |
|------|--------|
| Source | `scratch/post_rising_tide_to_bubble_pure.state` |
| Recording | `tasks/bubble_jump_try.json` (1398f, no Bat) |
| Attempt 2 runway | f1135 `(27,395)` pose 2 → run 21f + spin 83f |
| Wall contact | ~(264,297) LEFT+A |
| Peak | `(237,160)` then fall — **not** Phase D (need x≥300) |
| Exact replay | min_y=158 max_x=264; one extra WJ → min_y=142 still max_x≤264 |
| Ceiling | y~142 x~240–251 wall ends; cannot walk right at height |
| Fall under overhang | x only ~279 @ y380 — misses shelf band x≥360 y≤390 |

Maprando: room 97 left climb (Hard walljump / Medium running WJ) — **no Ice**.
Wiki: “running jump across into wall jumps” from save-door platform.

### Acceptance

- [x] Named trajectory from human + maprando (save-door runway)
- [x] Unit green (save-runway seat)
- [ ] Full pure min_y≤280 — expect hold/improve vs R13 once pure re-run
- [ ] Full pure phase_c_hit — may still come from R13 floor-reclimb fallback
- [ ] Full pure top_reached — **red** (lip clear)
- [ ] Ordinary `0xB07A` — **red**

### Why top is still red

1. Natural height class from runway is excellent (y≈158) but **wrong column**
   (mid cavity shaft, not right-structure shelf/air).
2. “Up and over” needs clearing a ceiling lip at ~y140–160 into x≥300 while
   still high — open-loop WJ thrash and morph roll under do not yet clear it.
3. Place finish still holds: air `(360,y≤370)` WJ8; grounded
   `(360–388,331–363)` LEFT HJ → top. Natural path does not seat those.

### Rejected this session (do not re-ship without new pin)

| Attempt | Why |
|---------|-----|
| Place open-loop from (40,395) without human timing | min_y regress to ~228; timing-sensitive |
| Finish scripts from free-air peak (245,163) | fall; never x≥300 @ y≤200 |
| Morph/ceil crawl from y142 | max_x high only after deep fall |
| Floor sprint after fall | lands y523; no shelf contact |

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R15`
- **One change:** Clear the **mid-cavity ceiling lip** after save-runway WJ
  (natural seat into Phase D `x≥300 y≤200` or right air/shelf), **or** convert
  runway height into a **velocity-matched** handoff onto place-proven right
  structure without losing height class. Prefer human re-record that finishes
  ordinary `0xB07A` as the target open-loop.
- Keep R14 runway + R5/R6/R13 as regression.
- **Source:** `scratch/post_rising_tide_to_bubble_pure.state`
- **Human refs:** `tasks/bubble_jump_try.json`,
  `scratch/bubble_human_runway.state` / `bubble_human_peak.state` (dev)

### Non-claims

- Did not STATUS-promote / continuous tip advance.
- Did not close SM-K4.4-PURE pure GREEN to Bat.
- Human recording is not hop GREEN.
- Place finish ≠ natural proof.

### Probe pin (post R14 ship)

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
# 23 passed

uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin_r14.json --no-red-diag
# success=false
# max_x=408 min_y=260 phase_c_hit=True top_reached=False launched=True
# frames≈30806  (R13 envelope held; lip not stolen by walk-to-save)
```
