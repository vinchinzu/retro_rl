## Residual — SM-K4.4-PURE-R6

### Result
PARTIAL (honest progress — full pure **min_y=260** + pin; top still red)

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — R6 phase 2: launch only from
  **solid save-door lip** (`x∈[65,100]`, `y∈[410,450]`) then shelf charged-HJ
  climb (left column → high cross). Drop from unstable mid float (y~370)
  onto lip before launch. Constants `_BUBBLE_LIP_*`.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R6.md` — living R6 card.
- `super_metroid/docs/tasks/SM-K4.4-PURE-R6-residual.md` — this residual.

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
  room=0xACB3 pose=47 xy=(324,474) door_transition=0
  max_x=332 min_y=260 mid_reached=True top_reached=False door_reached=False
  standing_mid_pinned=True launched=True supers=5 selected=0
frames=6993 controllerOnly=true

# Mid isolation
uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_bubble_mid_climb_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_mid_pin.json --no-red-diag
exit 1
  max_x=332 min_y=260 standing_mid_pinned=True launched=True top_reached=False
frames=6614
```

### Acceptance

- [x] Source loads at `0xACB3` — source OK
- [x] Full pure `standing_mid_pinned=True` — no R5 regression
- [x] Full pure min_y≤260 — **min_y=260** (mid-iso height class)
- [ ] Top band y≤200 / x≥300 — **not achieved**
- [ ] Ordinary `0xB07A` — **not achieved pure**
- [ ] Successor state only if pure GREEN — **no successor**
- [x] Unit/registration green (12 passed)
- [x] Residual PROCESS fields; no continuous/STATUS claim

### Residual risks / recon facts (load-bearing)

1. **R6 one-change shipped:** mid open-loop launches only from **solid lip**
   ~(70,427), not mid-iso float at y~370. Recon: A from float gains **no**
   height; place/idle solid lip charged HJ reaches min_y≈228–260. Full pure
   now **min_y=260** with pin + launched.

2. **Mid float is not jumpable.** Idle mid-iso drifts to (69,427) pose 1/2
   in ~30f. R2/R5 launch from pin band was launching from air-lock / slide
   and fell into cavity without climb height.

3. **Top band still open.** Full pure ends cavity mid-right (~324,474) with
   `top_reached=False` after gaining height class. Next knob: second-hop /
   peak-cross retention after first HJ to y≤260 (often one landing then fall
   without re-seat for higher shelf / right wall WJ).

4. **R5 lower path + lip drop remain load-bearing.** Wrong-door avoid +
   cavity x cap remain load-bearing.

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R7`
- **One change:** After lip HJ reaches mid-iso height (y≤280), **re-seat and
  second-hop / peak-cross** so full pure hits top band (y≤200, x≥300) still
  in `0xACB3`, then door phase can fire. Do not change R5 lower path or lip
  launch pad unless height regresses.
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
room=0xACB3 pose=47 x=324 y=474 door_transition=0
frames=6993 max_x=332 min_y=260
mid_reached=True top_reached=False door_reached=False
standing_mid_pinned=True launched=True
supers=5 selected=0

# Mid isolation
room=0xACB3 pose=47 x=324 y=474 door_transition=0
frames=6614 max_x=332 min_y=260
standing_mid_pinned=True launched=True top_reached=False
```
