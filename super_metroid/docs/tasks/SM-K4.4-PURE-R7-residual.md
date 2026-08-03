## Residual — SM-K4.4-PURE-R7

### Result
PARTIAL (honest progress — peak-cross reaches right structure; top still red)

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — R7 phase-2 climb after lip
  launch height class (y≤280): stop left-column thrash; mid reseat second
  hop right; right-shelf class (x≥300 y≤370) charged HJ to top; air
  peak-cross WJ-up on right structure (x≥250). Constants
  `_BUBBLE_HEIGHT_CLASS_Y`, `_BUBBLE_MID_RESEAT_Y`, `_BUBBLE_RIGHT_SHELF_*`.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R7.md` — living R7 card.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R7-residual.md` — this residual.

No committed pure-green claim; no STATUS / continuous promote.

### Verify paste

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
exit 0
12 passed in 0.16s

# Full CATH-04 source
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json --no-red-diag
exit 1
success=false
error: bubble_to_bat_cave: Bat Cave Super door missed before room 0xB07A;
  room=0xACB3 pose=26 xy=(326,484) door_transition=0
  max_x=387 min_y=270 mid_reached=True top_reached=False door_reached=False
  standing_mid_pinned=True launched=True supers=5 selected=0
frames=25323 controllerOnly=true
```

### Acceptance

- [x] Source loads at `0xACB3` — source OK
- [x] Full pure `standing_mid_pinned=True` — no R5 regression
- [x] Full pure min_y≤280 height class — **min_y=270** (shy of R6 min_y=260;
      still height class; max_x **387** vs R6 **332**)
- [ ] Top band y≤200 / x≥300 — **not achieved**
- [ ] Ordinary `0xB07A` — **not achieved pure**
- [ ] Successor state only if pure GREEN — **no successor**
- [x] Unit/registration green (12 passed)
- [x] Residual PROCESS fields; no continuous/STATUS claim

### Residual risks / recon facts (load-bearing)

1. **R7 one-change shipped:** after lip HJ height class, climb no longer
   left-column thrash. Mid reseat (y≤320) second-hops right; right-shelf
   band (x≥300 y≤370) aims top; air uses right-structure WJ-up when x≥250.
   Full pure now **max_x=387** (right-structure x class) with pin + launched.

2. **Right-shelf → top is place-proven.** Dense/place grid solids
   `(384,363) (368,331) (352,283) (336,219)` each one charged HJ → top band
   in isolation. Gap is **landing** those shelves naturally after lip peak.

3. **Mid reseat nubs are tiny.** Uncharged run-jump from lip can ground
   ~(140–175, 270–295) pose 25, but ledge is ~5px wide (walk falls in ~4f).
   Second hop from there often falls to cavity floor before right shelf.

4. **R5 lower + R6 lip remain load-bearing.** Wrong-door avoid + cavity x
   cap remain load-bearing. min_y slightly worse than R6 (270 vs 260) —
   peak-cross bias trades a few height px for far more max_x.

5. **Frame budget rose** (~25k vs R6 ~7k) from right-structure WJ thrash
   without shelf re-seat — next knob must land shelf or abort thrash.

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R8`
- **One change:** After peak-cross reaches right-structure x (x≥300) at mid
  height, **re-seat on right shelf class** (y≤370, preferably y≤340 at
  x∈[320,390]) then fire the proven shelf→top charged HJ. Do not expand
  left-column or lower path. Avoid far-right SC height trap (x cap).
- **Source state:**
  `scratch/post_rising_tide_to_bubble_pure.state` (acceptance).

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
- Did not close SM-K4.4-PURE as pure GREEN to Bat.
- Pure tip remains first Bubble; Bat still blocks K4 advance.

### Probe pin (if pure/geometry) — mandatory metrics

```text
# Full CATH-04 source
room=0xACB3 pose=26 x=326 y=484 door_transition=0
frames=25323 max_x=387 min_y=270
mid_reached=True top_reached=False door_reached=False
standing_mid_pinned=True launched=True
supers=5 selected=0
```
