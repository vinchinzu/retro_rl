## Residual — SM-K4.4-PURE-R3

### Result
RED (honest geometry gap — standing mid re-pin shipped; full pure still not mid-iso height class)

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — R3 phase 1.5 standing mid
  re-pin before R2 open-loop: walk/HJ into mid-iso handoff band
  (`x∈[77,160]`, `y≤400`, poses `{1,2,9,10,25–28}`, `|vy|≤2`); launch pose
  set widened to mid-iso class (25/26 turn); R2 grounded-mid launch fallback
  kept if pin misses; error text adds `standing_mid_pinned` / `launched`.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R3.md` — living R3 card.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R3-residual.md` — this residual.
- Probe/debug under `super_metroid/debug/bubble_*` (not route-ready).

No committed pure-green claim; no STATUS / continuous promote.

### Verify paste

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
exit 0
12 passed in 0.18s

# Full CATH-04 source (acceptance path)
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json --no-red-diag
exit 1
success=false
error: bubble_to_bat_cave: Bat Cave Super door missed before room 0xB07A;
  room=0xACB3 pose=47 xy=(324,474) door_transition=0
  max_x=332 min_y=364 mid_reached=True top_reached=False door_reached=False
  standing_mid_pinned=False launched=True supers=5 selected=0
frames=64642 controllerOnly=true

# Mid isolation (dev; not acceptance)
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_mid_climb_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_mid_pin.json --no-red-diag
exit 1
success=false
  max_x=332 min_y=260 mid_reached=True top_reached=False
  standing_mid_pinned=True launched=True
frames=6811
```

### Acceptance

- [x] Source loads at `0xACB3` (CATH-04 pin band) — source OK
- [ ] Full pure min_y≤260 (mid-iso class) — **not achieved** (best **364**)
- [ ] Top band y≤200 / x≥300 still in `0xACB3` — **not achieved pure**
- [ ] Ordinary `0xB07A` without warp / item grants — **not achieved pure**
- [ ] Successor state only if pure GREEN — **no successor written**
- [x] Unit/registration green (12 passed)
- [x] Residual PROCESS fields; no continuous/STATUS claim

### Residual risks / recon facts (load-bearing)

1. **R3 one-change shipped:** phase 1.5 re-pin loop after lower climb,
   before R2 open-loop. Mid-iso source still reports
   `standing_mid_pinned=True` and **min_y≈260** (same class as R2 isolation).

2. **Full pure still does not pin.** CATH-04 path:
   `standing_mid_pinned=False`, `launched=True`, **min_y≈364** (modest gain
   vs R2 full pure **≈388**). Lower climb still exits on broad mid
   (`x≤320`); re-pin budget does not reliably walk/HJ onto the save-door
   platform before open-loop starts from cavity-right mid.

3. **Mid-iso pin class is not pose 1/2 + vy=0.** Empirical source
   `post_bubble_mid_climb_pure.state` loads **pose=26 x≈98 y≈374 vy≈1**.
   Re-pin / launch accept `{1,2,9,10,25–28}` with `|vy|≤2`. Strict pose 1/2
   alone never matched the working isolation handoff.

4. **Launch pose widen is load-bearing.** Gate on only `_BUBBLE_GROUND`
   `{1,2,9,10}` left `launched=False` on live mid exits that land pose 25/26;
   open-loop never started. Stand-pin set restores launch.

5. **Top band / door still not the pure blocker.** Place door path remains a
   prior diagnostic only. Gap is still mid→high walljump after a true left
   save-door pin on the full pure chain.

6. **Wrong-door hard-avoid + cavity x cap remain load-bearing.**

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R4`
- **One change:** Make **lower climb exit target the save-door platform**
  (`x∈[77,160]`, `y≤400`, stand-pin poses) instead of broad mid
  `100≤x≤320` — so phase 1.5 re-pin starts already on (or one hop from)
  the mid-iso pin and full pure can reproduce `standing_mid_pinned=True` +
  min_y≤260. Do not change door phase or CATH geometry; keep R2 open-loop
  once pinned.
- **Source state:**
  `scratch/post_rising_tide_to_bubble_pure.state` (acceptance);
  keep mid isolation for knob tests.

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
- Mid isolation pin/height are **development diagnostics**, not pure-green
  acceptance for Bat Cave.
- Did not close SM-K4.4-PURE or R3 as green.
- Pure tip remains first Bubble (`post_rising_tide_to_bubble_pure`); Bat is
  still the blocker for more of K4.

### Probe pin (if pure/geometry) — mandatory metrics

```text
# Full CATH-04 source
room=0xACB3 pose=47 x=324 y=474 door_transition=0
frames=64642 max_x=332 min_y=364
mid_reached=True top_reached=False door_reached=False
standing_mid_pinned=False launched=True
supers=5 selected=0
# No Bat Cave ordinary settle. No successor state written.

# Mid isolation (dev)
room=0xACB3 pose=25 x=315 y=484
frames=6811 max_x=332 min_y=260
mid_reached=True top_reached=False
standing_mid_pinned=True launched=True
```
