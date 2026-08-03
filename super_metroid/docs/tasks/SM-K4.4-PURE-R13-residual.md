## Residual — SM-K4.4-PURE-R13

### Result
PARTIAL (Phase C green on full pure; top / door still red)

Named trajectory **floor-reclimb** after height class puts natural pure into
Phase C predicate (`phase_c_hit=True`) while holding R6 height class
(`min_y=260`). Climb from that contact does not yet reach top band / Super
door — Phase C pin is **marginal** (~y429 air), below place-proven recoverable
shelf band (grounded ~y≤390 x≥360).

### Files changed
- `routes/kpdr/bubble_mountain_params.py` — `FLOOR_RECLIMB_*` / `FLOOR_RUNWAY_*`
- `routes/kpdr/bubble_mountain_mid.py` — deep floor runway after height class;
  Phase-C sticky right-structure air/ground (no re-drop to floor)
- `routes/kpdr/bubble_mountain.py` — climb-handoff pins `phase_c_hit` before
  settle (dump falls out of y≤430 in a few idle frames)
- `docs/tasks/SM-K4.4-PURE-R13.md` + this residual
- Tip boards: AGENTS / QUEUE / phase ladder / BUBBLE_MOUNTAIN_TODO
- Dev dumps: `debug/bubble_r13_*.json`,
  `scratch/post_bubble_right_contact_pure.state` (Phase C handoff)

No pure GREEN to Bat; no STATUS / continuous promote.

### Verify paste

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
# 27 passed

uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin_r13.json --no-red-diag
# success=false
# max_x=408 min_y=260 phase_c_hit=True top_reached=False launched=True
# frames≈30806  (Phase C green; sticky thrash after contact)
```

Phase-C dump (dev handoff, not hop GREEN):

```text
scratch/post_bubble_right_contact_pure.state
# boot: pose=81 xy=(301,429) phase_c=True
# after few idle: y drifts >430 (sticky must arm before settle)
```

Climb-only from dump (sticky on):

```text
phase_c_hit=True min_y=428 max_x=368 top_reached=False
# no height recovery from marginal contact
```

### Acceptance

- [x] Full pure Phase C predicate — **phase_c_hit=True** (first time on CATH-04)
- [x] Full pure min_y≤280 — **min_y=260** (R6 height held)
- [ ] Full pure top_reached — **red**
- [ ] Ordinary `0xB07A` — **red**
- [x] Unit green
- [x] Residual PROCESS fields; no continuous/STATUS claim
- [x] Named trajectory outside rejected lip period thrash

### Trajectory (load-bearing)

1. **Keep** R5 lower + R6 lip + height class (min_y=260).
2. **After** height class, if deep (`y ≥ FLOOR_RECLIMB_Y=480`) and not yet
   Phase-C sticky: align mid-right floor runway `x∈[270,310]` at `y≥500`,
   charge 12 + spin 44 RIGHT B A, then deep-air WJ toward right structure.
3. Place matrix (`debug/bubble_r13_floor_refine.json`): runway ~(288,531)
   p8i2b2 → Phase C ~(302,428). Same class as natural pure dump.
4. Once Phase C fires: sticky right-structure WJ / grounded HJ; **do not**
   re-enter floor runway (that only re-hits marginal y≈428).

### Why top is still red

| Fact | Detail |
|------|--------|
| Natural Phase C pin | ~(301,429) air — bottom of Phase C window |
| Place climb from dump | best min_y≈427; no shelf land; no top |
| Place finish still holds | grounded shelf ~(380,390) wj8 → top; air (360,y≤370) → top |
| Gap | need **higher** right contact (shelf y≤390 x≥360) or climb script that recovers ≥30px from y429 |

R12 recovery note still holds: usable right re-ascent starts ~y≤400; natural
first Phase C is ~y429 — just below.

### Rejected this session (do not re-ship without new pin)

| Attempt | Why |
|---------|-----|
| Height-class lip + fall-gated WJ | Phase C never with min_y≤280 (env430≤253) |
| Left-high cross from y≤250 | max x@y≤430 ≈277; no Phase C |
| Floor climb alone to top | Phase C yes; never shelf/top from y428 contact |
| Period thrash on lip arc | R12 stagnation class |

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R14`
- **One change:** Convert Phase C / floor-reclimb contact into **grounded right
  shelf** `(x≥360, y≤390)` or air band `(x≥340, y∈[280,370])`, then reuse R9
  LEFT shelf HJ / period-8 WJ to top. Prefer improving floor-reclimb apex
  (land higher) over more sticky WJ on y429.
- Keep Phase C pure pin + min_y≤260 as regression.
- **Source state:**
  `scratch/post_rising_tide_to_bubble_pure.state` (acceptance);
  climb iteration may use `scratch/post_bubble_right_contact_pure.state`.

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
- Did not close SM-K4.4-PURE as pure GREEN to Bat.
- Phase C predicate green ≠ hop GREEN; climb-only ≠ full pure.
- Continuous tip remains Frog Save (114,923f).

### Probe pin (if pure/geometry) — mandatory metrics

```text
# Full CATH-04 source after R13 floor-reclimb
room=0xACB3 pose=47 x=323 y=461 door_transition=0
frames=30806 max_x=408 min_y=260
mid_reached=True top_reached=False door_reached=False
standing_mid_pinned=True launched=True phase_c_hit=True
supers=5 selected=0
```
