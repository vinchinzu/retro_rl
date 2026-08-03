## Residual — SM-K4.4-PURE-R9

### Result
PARTIAL (honest progress — min_y restored, thrash cut; top still red)

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — R9 open-loop after height
  class: (1) lip HJ extension (`ol_extend`) so mid-nub micro-land does not
  eat the peak, (2) period-8 WJ (into 2 / bounce 2) from x≥250 while y≤400,
  (3) no-A shelf_drop in right band, (4) grounded right shelf → **LEFT**
  charged HJ (RIGHT hits SC outer-wall trap). Mid reseat no longer
  charge/run-walks off the ~5px nub. Constants `_BUBBLE_RIGHT_WJ_*`,
  shelf y band widened to 390.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R9.md` — living R9 card.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R9-residual.md` — this residual.

No committed pure-green claim; no STATUS / continuous promote.

### Verify paste

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
exit 0
12 passed in 0.17s

# Full CATH-04 source
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json --no-red-diag
exit 1
success=false
error: bubble_to_bat_cave: Bat Cave Super door missed before room 0xB07A;
  room=0xACB3 pose=47 xy=(328,519) door_transition=0
  max_x=389 min_y=260 mid_reached=True top_reached=False door_reached=False
  standing_mid_pinned=True launched=True supers=5 selected=0
frames=7331 controllerOnly=true
```

### Acceptance

- [x] Source loads at `0xACB3` — source OK
- [x] Full pure `standing_mid_pinned=True` — no regression
- [x] Full pure min_y≤280 — **min_y=260** (better than R7/R8 min_y=270)
- [ ] Top band y≤200 / x≥300 — **not achieved**
- [ ] Ordinary `0xB07A` — **not achieved pure**
- [ ] Successor state only if pure GREEN — **no successor**
- [x] Unit/registration green (12 passed)
- [x] Residual PROCESS fields; no continuous/STATUS claim

### Residual risks / recon facts (load-bearing)

1. **R9 one-change shipped:** open-loop peak retention + period-8 WJ + LEFT
   shelf→top hop. Full pure **min_y=260** (R6 height class restored),
   **max_x=389** (slightly above R7 387), **frames≈7.3k** (vs R7/R8 ≈25k
   thrash). Top still red.

2. **Place-proven still holds.** Shelves `(360–380,331–363)` LEFT charged HJ
   → top in isolation. Air place `(360,320)` period-8 WJ → top. Gap remains
   **natural approach at height**.

3. **One-shot lip→right shelf is too far at fall rate.** Lip peak ~(130–165,
   260) then ~1.5 px/f right; first x≥250 lands at y≈400+ (below shelf
   band y≤390). No open-loop timing from lip alone hits x≥300 while y≤360.
   Mid-nub reseat charge/run walks off (~5px).

4. **Cavity floor cannot climb to shelves.** Place/hop from y~520–530 only
   reaches min_y≈435 (ceiling geometry). Right structure must be met from
   **above/side while still high**, not from floor thrash.

5. **R5 lower + R6 lip remain load-bearing.** Wrong-door avoid + cavity x
   cap remain load-bearing. Lip run-up (r4) helps place-isolation cross but
   **regressed pure height** when applied on natural path — do not re-add
   without re-verify.

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R10`
- **One change:** Intermediate re-seat / multi-hop that puts Samus into the
  place-proven right air band `(x≥340, y∈[280,340])` **before** falling past
  shelf height — e.g. mid-cavity solid chain, left-wall height gain then
  cross, or single bomb-boost from a grounded mid land (not infinite BJ).
  Then reuse R9 shelf LEFT HJ / period-8 WJ. Do not change R5 lower or R6
  lip pad unless height regresses.
- **Source state:**
  `scratch/post_rising_tide_to_bubble_pure.state` (acceptance).

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
- Did not close SM-K4.4-PURE as pure GREEN to Bat.
- Continuous tip remains Frog Save (114,923f).

### Probe pin (if pure/geometry) — mandatory metrics

```text
# Full CATH-04 source
room=0xACB3 pose=47 x=328 y=519 door_transition=0
frames=7331 max_x=389 min_y=260
mid_reached=True top_reached=False door_reached=False
standing_mid_pinned=True launched=True
supers=5 selected=0
```
