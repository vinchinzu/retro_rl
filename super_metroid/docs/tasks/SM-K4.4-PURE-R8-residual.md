## Residual — SM-K4.4-PURE-R8

### Result
PARTIAL (no metric advance over R7 — top still red)

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — R8 air: when height class +
  right-structure x≥300 falling through y 300–400, release A and track
  horizontally (`shelf_drop`) to re-seat on place-proven shelves before
  shelf→top HJ. Builds on R7 peak-cross / right WJ-up.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R8.md` — living R8 card.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R8-residual.md` — this residual.

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
- [x] Full pure `standing_mid_pinned=True` — no regression
- [x] Full pure min_y≤280 — **min_y=270** (same class as R7)
- [ ] Top band y≤200 / x≥300 — **not achieved** (same as R7)
- [ ] Ordinary `0xB07A` — **not achieved pure**
- [ ] Successor state only if pure GREEN — **no successor**
- [x] Unit/registration green (12 passed)
- [x] Residual PROCESS fields; no continuous/STATUS claim

### Residual risks / recon facts (load-bearing)

1. **R8 shelf_drop did not change pure pin vs R7.** Same max_x=387
   min_y=270 frames≈25k end ~(326,484). Either the fall path does not
   satisfy the drop window long enough, or drop still misses solid shelf
   collision.

2. **R7 peak-cross remains load-bearing** for right-structure x (387 vs R6
   332). Place-proven shelves still one-hop to top in isolation.

3. **Likely need open-loop or place-assisted trajectory recon** from a
   captured mid-right air state (x≈330–380, y≈300–380) rather than more
   reactive dir bias. Uncharged lip→mid-nub reseat is too narrow to chain.

4. **R5 lower + R6 lip + height-class gate remain load-bearing.**

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R9`
- **One change:** Scripted open-loop (or short place-grid-informed hop
  sequence) from lip height class that **lands grounded** on a right shelf
  solid `(384,363)` / `(368,331)` / `(352,283)` class, then reuse shelf→top
  HJ. Prefer offline recon first; one controller knob. Do not change lower
  path or lip pad.
- **Source state:**
  `scratch/post_rising_tide_to_bubble_pure.state` (acceptance).

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
- Did not close SM-K4.4-PURE as pure GREEN to Bat.
- R8 did not beat R7 pure metrics.

### Probe pin (if pure/geometry) — mandatory metrics

```text
# Full CATH-04 source (matches R7 pin)
room=0xACB3 pose=26 x=326 y=484 door_transition=0
frames=25323 max_x=387 min_y=270
mid_reached=True top_reached=False door_reached=False
standing_mid_pinned=True launched=True
supers=5 selected=0
```
