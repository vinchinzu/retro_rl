## Residual — SM-K4.4-PURE-R2

### Result
RED (honest geometry gap — mid open-loop lands mid; top band still not pure-green)

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — R2 mid phase open-loop for
  `play_bubble_to_bat_cave`: standing launch (run-up + charged HJ) → G-style
  alternating walljump climb (period 12) → peak-cross fresh-A right WJ;
  cavity x hard-cap; constants `_BUBBLE_CAVITY_X_MAX` / peak cross band.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R2.md` — living R2 card.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R2-residual.md` — this residual.
- Probe/debug under `super_metroid/debug/bubble_*` / red_diag (not route-ready).

No committed pure-green claim; no STATUS / continuous promote.

### Verify paste

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
exit 0
12 passed in 0.16s

# Full CATH-04 source (acceptance path)
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json
exit 1
success=false
error: bubble_to_bat_cave: Bat Cave Super door missed before room 0xB07A;
  room=0xACB3 pose=26 xy=(327,490) door_transition=0
  max_x=340 min_y=388 mid_reached=True top_reached=False door_reached=False
  supers=5 selected=0
frames=63821 controllerOnly=true

# Mid isolation (dev; not acceptance)
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_mid_climb_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_mid_pin.json --no-red-diag
exit 1
success=false
  max_x=333 min_y=260 mid_reached=True top_reached=False
frames=33170
```

### Acceptance

- [x] Source loads at `0xACB3` (CATH-04 pin band) — source OK
- [ ] Top band y≤200 / x≥300 still in `0xACB3` — **not achieved pure**
- [ ] Ordinary `0xB07A` without warp / item grants — **not achieved pure**
- [ ] Successor state only if pure GREEN — **no successor written**
- [x] Unit/registration green (12 passed)
- [x] Residual PROCESS fields; no continuous/STATUS claim

### Residual risks / recon facts (load-bearing)

1. **R2 one-change shipped:** mid phase is now scripted open-loop
   (launch → alt-WJ climb → peak cross) rather than R1’s frame-reactive
   delayed WJ. Full pure still tops out at **min_y≈388** (same class as R1).

2. **Mid isolation improves height.** From
   `post_bubble_mid_climb_pure.state`, pure probe reaches **min_y≈260**
   (`mid_reached=True`, `top_reached=False`). G-style alt WJ is real on a
   standing mid pin; full pure does not reliably hand off that pin after
   lower climb.

3. **Top band still needs simultaneous x≥300 and y≤200.** Offline place at
   door-1 runway (≈x55–100, y139) reaches min_y≈60 and max_x≈380–480 but
   **never co-occurring** (joint y≤200 ∧ x≥250 count was 0 across edge-jump
   sweeps). Gap cross over junction 9 remains the hard composition after
   high-left.

4. **Cavity right-wall pure WJ from mid still weak.** Grid wall-find at
   mid-cavity x=150–320 gained height only on **left** faces (toward save
   door), not on strat-154 “cavity right wall.” Far-right SC wall remains a
   height trap (x hard-cap kept).

5. **Door compose still not the pure blocker.** Place `(420,130)` Super
   path remains a prior diagnostic only.

6. **Wrong-door hard-avoid + cavity x cap remain load-bearing.**

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R3`
- **One change:** After lower climb, **force a standing mid re-pin**
  (pose 1/2, vy=0, x∈[90,160], y≤400) with a short settle/HJ retry loop
  before R2 open-loop launch — so full pure matches mid-isolation handoff
  (target: reproduce min_y≤260 on CATH-04 source, then peak-cross). Do not
  change door phase or CATH geometry.
- **Source state:**
  `scratch/post_rising_tide_to_bubble_pure.state` (acceptance);
  keep mid isolation for knob tests.

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
- Place/door success and mid isolation are **development diagnostics**, not
  pure-green acceptance.
- Did not close SM-K4.4-PURE or R2 as green.
- Pure tip remains first Bubble (`post_rising_tide_to_bubble_pure`); Bat is
  still the blocker for more of K4.

### Probe pin (if pure/geometry) — mandatory metrics

```text
# Full CATH-04 source
room=0xACB3 pose=26 x=327 y=490 door_transition=0
frames=63821 max_x=340 min_y=388
mid_reached=True top_reached=False door_reached=False
supers=5 selected=0
# No Bat Cave ordinary settle. No successor state written.

# Mid isolation (dev)
room=0xACB3 pose=47 x=326 y=474
frames=33170 max_x=333 min_y=260
mid_reached=True top_reached=False
```
