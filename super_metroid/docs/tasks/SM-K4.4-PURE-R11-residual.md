## Residual — SM-K4.4-PURE-R11

### Result
PARTIAL (spin-apex false-land bug fixed; Phase C / top still red; room not closed)

### Files changed
- `super_metroid/routes/kpdr/k4_norfair.py` — R11 climb grounded / mid-nub
  reseat use **true ground poses only** (`_BUBBLE_GROUND` = 1/2/9/10).
  Spin apex (pose 25 + vy≈0) was treated as a land and interrupted
  `ol_extend` / reseat-hop at peak (~x150,y260), killing horizontal
  progress. Lip launch stays R10 charge/spin (no dash — see rejected).
- Phase ladder + hard-room docs/harness (prior session):
  `docs/tasks/HARD_ROOM_SPLITS.md`, `SM-K4.4-PHASE-LADDER.md`,
  probe `--dump-phase-c` / `--start-phase climb` / `--stop-at-phase-c`.
- `docs/tasks/SM-K4.4-PURE-R11-residual.md` — this residual.

No pure GREEN to Bat; no STATUS / continuous promote.

### Verify paste

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
exit 0
16 passed

uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json --no-red-diag
exit 1
success=false
error: … room=0xACB3 pose=25 xy=(313,484) …
  max_x=349 min_y=260 mid_reached=True top_reached=False
  standing_mid_pinned=True launched=True phase_c_hit=False
frames=7227 controllerOnly=true
```

Trace (post-fix): peak ~(150,260); best x while y≤360 still **~210**;
first x≥300 still **y≈512** (below Phase C). Same approach gap as R10.

### Acceptance

- [x] Source loads at `0xACB3`
- [x] Full pure pin + min_y≤280 — **min_y=260** (height held)
- [ ] Phase C usable right contact — **phase_c_hit=False**
- [ ] Top band y≤200 x≥300 — **not achieved**
- [ ] Ordinary `0xB07A` — **not achieved**
- [x] Unit/registration green
- [x] Residual PROCESS fields; no continuous/STATUS claim
- [x] Climb-only / phase-capture not presented as hop GREEN

### Residual risks / recon facts (load-bearing)

1. **Place finish still holds:** air `(360,y≤370)` period-8 WJ → top;
   grounded shelves LEFT HJ → top. Right-structure WJ TOP band from place
   starts ≈ `(330–380, 250–370)`.

2. **Natural envelope unchanged (approach gap):**
   - Lip charge HJ peak ~(150,260), ~1.3–1.5 px/f right while falling.
   - max x @ y≤360 ≈ **210**; first x≥300 @ y≈**512**.
   - Phase C predicate never fires on full pure.

3. **Rejected this session (do not re-ship without new pin):**
   - Lip walk-left + dash (wl16 r16 c4): place Phase C once; **pure
     min_y≈365** height regress / thrash.
   - Mid-iso pin dash launch: isolation Phase C; **natural enemy KB
     thrash** (tens of k frames mid_kb) + height regress.
   - Any pre-charge run on lip (even 4f): pure height regress.
   - Left-wall WJ: free-air ceiling **min_y≈228** — not top band (y≤200).
   - Peak morph-bomb / turbo-bomb chains: no stable right-band gain.
   - Floor runway / floor WJ: no height class.
   - Save-room run-out: door transition sticky in probe; not productized.
   - Mid-nub place hops: place does not settle on “nubs”; micro-lands are
     motion artifacts.

4. **Bug fixed (shipped):** climb `grounded` / reseat used
   `_BUBBLE_STAND_PIN` (includes spin 25/26). At jump apex vy≈0 + pose 25
   looked like a land → false reseat_hop during peak. Now `_BUBBLE_GROUND`
   only.

5. **R5 lower + R6 lip + R9/R10 open-loop remain load-bearing.**

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R12`
- **One change:** New **trajectory class** only (stagnation @ 3+ on Phase C):
  either (a) **velocity-matched** place+trace of a human/maprando dash that
  lands in right air band then ship that open-loop from natural mid pin
  **with enemy clearance**, or (b) intermediate **true-ground** chain
  proven by place solids grid (not spin micro-lands), or (c) planner
  topology rethink. **Not** period/window-only on current lip arc.
- **Source state:**
  `scratch/post_rising_tide_to_bubble_pure.state` (acceptance).

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
- Did not close SM-K4.4-PURE as pure GREEN to Bat.
- Continuous tip remains Frog Save (114,923f).
- Phase-C capture / climb-only tooling ≠ hop GREEN.

### Probe pin (if pure/geometry) — mandatory metrics

```text
# Full CATH-04 source (post R11 spin-apex fix)
room=0xACB3 pose=25 x=313 y=484 door_transition=0
frames=7227 max_x=349 min_y=260
mid_reached=True top_reached=False door_reached=False
standing_mid_pinned=True launched=True phase_c_hit=False
supers=5 selected=0
```
