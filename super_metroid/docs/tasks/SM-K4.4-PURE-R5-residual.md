## Residual — SM-K4.4-PURE-R5

### Result
PARTIAL (honest progress — full pure **standing_mid_pinned=True**; top still red)

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — R5 lower phase: dedicated
  left-column ledge multi-hop (`_BUBBLE_LOWER_SHELVES` waypoints from
  place-grid recon) replacing HJ dir-bias lower climb; walk-to-floor shelf
  then charged hops to save-door pin band.
- `super_metroid/scripts/probe/bubble_lower_left_recon.py` — place/grid +
  natural recon (diagnostic only).
- `super_metroid/docs/tasks/SM-K4.4-PURE-R5.md` — living R5 card.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R5-residual.md` — this residual.
- Probe/debug: `debug/bubble_lower_left_recon.json`, pin JSON refresh.

No committed pure-green claim; no STATUS / continuous promote.

### Verify paste

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
exit 0
12 passed in 0.19s

# Full CATH-04 source
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --output super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_to_bat_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json --no-red-diag
exit 1
success=false
error: bubble_to_bat_cave: Bat Cave Super door missed before room 0xB07A;
  room=0xACB3 pose=47 xy=(324,474) door_transition=0
  max_x=332 min_y=364 mid_reached=True top_reached=False door_reached=False
  standing_mid_pinned=True launched=True supers=5 selected=0
frames=6845 controllerOnly=true

# Mid isolation (dev; land break preserves pin)
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_mid_climb_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_mid_pin.json --no-red-diag
exit 1
  max_x=323 min_y=292 mid_reached=True top_reached=False
  standing_mid_pinned=True launched=True
frames=6514
```

### Acceptance

- [x] Source loads at `0xACB3` (CATH-04 pin band) — source OK
- [x] Full pure `standing_mid_pinned=True` — **True** (R5 one-change goal)
- [ ] Full pure min_y≤260 — **not achieved** (min_y=364; pin lands, peak-cross does not)
- [ ] Ordinary `0xB07A` — **not achieved pure**
- [ ] Successor state only if pure GREEN — **no successor**
- [x] Unit/registration green (12 passed)
- [x] Residual PROCESS fields; no continuous/STATUS claim

### Residual risks / recon facts (load-bearing)

1. **R5 one-change shipped:** lower phase is a **scripted left-column
   multi-hop** along place-grid solid shelves
   `(120,560)→(110,515)→(100,475)→(90,450)→(95,430)→(105,370)` after walk
   onto floor shelf ~x108 y651. Offline controller recon pinned **5/5** in
   ~66f after setup; full pure now reports **`standing_mid_pinned=True`**.

2. **Dir bias cannot substitute.** R3/R4 left-column HJ bias still exited
   cavity mid-right and never reconstructed mid-iso pin. Place climbs from
   floor shelves pinned; long single HJ from natural walk desynced (enemy /
   morph class) — multi-hop shelf path is the reliable natural path.

3. **Top band still the pure blocker.** Full pure: pin True, launched True,
   **min_y=364**, `top_reached=False`, end cavity mid-right (~324,474).
   Mid-iso after land fix: pin True, **min_y=292** (better than unbroken
   land drift min_y≈388; still shy of prior R2/R3 isolation min_y≈260).
   Next knob: R2 open-loop retune / peak-cross from the live full-pure pin,
   not more lower path work.

4. **Land settle is load-bearing for isolation.** Idle land from mid-iso
   (vy≈1) drifted to ~x69 y427 off pin; land now breaks on mid-iso pin class
   with |vy|≤2. Lower path also skips when already pinned.

5. **Wrong-door hard-avoid + cavity x cap remain load-bearing.**

6. **Frame budget dropped** on full-pure fail (~6845f vs R4 ~64642f) because
   lower phase no longer burns the full lower budget before open-loop — good
   signal the pin path is short.

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R6`
- **One change:** Retune **mid open-loop / peak-cross** from the live full
  pure standing save-door pin (`standing_mid_pinned=True`) so full pure
  reaches mid-iso height class (**min_y≤260**) and preferably top band
  (y≤200, x≥300) still in `0xACB3`. Do not change door phase or lower
  ledge path unless pin regresses.
- **Source state:**
  `scratch/post_rising_tide_to_bubble_pure.state` (acceptance).

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
- Did not close SM-K4.4-PURE or R5 as full pure GREEN to Bat.
- Pure tip remains first Bubble; Bat still blocks K4 advance.
- Place/grid recon is diagnostic only (not pure proof).

### Probe pin (if pure/geometry) — mandatory metrics

```text
# Full CATH-04 source
room=0xACB3 pose=47 x=324 y=474 door_transition=0
frames=6845 max_x=332 min_y=364
mid_reached=True top_reached=False door_reached=False
standing_mid_pinned=True launched=True
supers=5 selected=0

# Mid isolation (dev)
room=0xACB3 pose=48 x=318 y=519 door_transition=0
frames=6514 max_x=323 min_y=292
mid_reached=True top_reached=False
standing_mid_pinned=True launched=True
```
