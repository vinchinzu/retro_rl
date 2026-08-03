## Residual — SM-K4.4-PURE-R10

### Result
PARTIAL (mid-high open-loop band expanded; top still red; height class held)

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — R10 mid-high approach:
  open-loop air WJ / shelf_drop / ol_cross now engage while
  `y ≤ _BUBBLE_MIDHIGH_Y` (**450**, was R9 **400**). Natural pure first
  reaches `x≥250` only near `y≈400+`, so R9 skipped open-loop on the
  peak fall. Constants `_BUBBLE_MIDHIGH_Y`, `_BUBBLE_RIGHT_AIR_X`,
  `_BUBBLE_RIGHT_AIR_Y`; helper `_in_right_air_band`. R5 lower + R6 lip
  + R9 period-8 WJ / LEFT shelf HJ unchanged. **No lip run-up** (pure
  recheck: run-up regressed `min_y` to ~365).
- `super_metroid/scripts/probe/bubble_r10_midhigh_recon.py` — offline
  place/WJ recon (not pure proof).
- `super_metroid/docs/tasks/SM-K4.4-PURE-R10-residual.md` — this residual.

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
  room=0xACB3 pose=25 xy=(313,484) door_transition=0
  max_x=349 min_y=260 mid_reached=True top_reached=False door_reached=False
  standing_mid_pinned=True launched=True supers=5 selected=0
frames=7227 controllerOnly=true
```

### Acceptance

- [x] Source loads at `0xACB3` — source OK
- [x] Full pure `standing_mid_pinned=True` — no regression
- [x] Full pure min_y≤280 — **min_y=260** (matches R9 height class)
- [ ] Top band y≤200 / x≥300 — **not achieved**
- [ ] Ordinary `0xB07A` — **not achieved pure**
- [ ] Successor state only if pure GREEN — **no successor**
- [x] Unit/registration green (12 passed)
- [x] Residual PROCESS fields; no continuous/STATUS claim

### Residual risks / recon facts (load-bearing)

1. **R10 one-change shipped:** mid-high open-loop window `y≤450` (was
   400). Full pure **min_y=260** (no height regress), **max_x=349**
   (shy of R9 389 thrash max), frames≈7.2k. Top still red. Trace: after
   ol_extend, climb now runs `ol_cross` while still mid-high
   (~y430–440) instead of only post-fall recovery.

2. **Place-proven still holds, approach gap remains.**
   - Air place `(360, y≤370)` period-8 WJ → **top**.
   - Grounded shelves `(360–388,331–363)` LEFT charged HJ → **top**.
   - Natural lip peak still ~(150,260); at shelf height y≈360 pure is
     only ~x211; first `x≥250` still ~y500+; first `x≥340` ~y467.
   - **One-shot lip→right air band is still too far at fall rate.**

3. **Tried and rejected this session (not shipped):**
   - Lip run-up (place +x@height) → pure **min_y≈365** height regress.
   - Peak morph-bomb → no height gain on natural arc.
   - Floor WJ climb → flaky; best fluke min_y≈364 at x~280, not stable.
   - Left wall to top platforms (y~140) → free-air peak ~260 only; **no
     solids** left column between y~174 and y~394 (place grid).
   - Upper mid-island solids `(256–320, y~140–176)` **are** top band if
     reached — hop from left-top gap ~176px not closed this session.

4. **R5 lower + R6 lip + R9 period-8 / LEFT shelf HJ remain load-bearing.**
   Wrong-door avoid + cavity x cap remain load-bearing.

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R11`
- **One change:** Follow hard-room ladder
  ([`SM-K4.4-PHASE-LADDER.md`](SM-K4.4-PHASE-LADDER.md) /
  [`HARD_ROOM_SPLITS.md`](HARD_ROOM_SPLITS.md)): capture Phase C
  (`--dump-phase-c --stop-at-phase-c`), then one named climb change from
  handoff / velocity-matched place (right-wall WJ or mid-nub chain) —
  not lip run-up, not mid-high window-only. Full pure recheck for GREEN.
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
room=0xACB3 pose=25 x=313 y=484 door_transition=0
frames=7227 max_x=349 min_y=260
mid_reached=True top_reached=False door_reached=False
standing_mid_pinned=True launched=True
supers=5 selected=0
```
