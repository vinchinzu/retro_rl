# SM-TIGHTEN-01 Report — `business_to_warehouse` dwell (2,257f)

**Source:** `recordings/start_to_varia.json` (outcome=`varia_collected`, total=101,954f)
**Split:** `business_to_warehouse` @92,241 (room 0xA6A1 Warehouse)
**Window:** [`hj_shaft_to_business` @89,984 → `business_to_warehouse` @92,241] = 2,257f
**Controller:** `routes/kpdr/business_climb.py` → `play_business_to_warehouse`

---

## Function structure map

```
Window [89,984 → 92,241]  (2,257f)
│
├─ hj_shaft_to_business_settle (124f)  ── settle from HJ shaft exit into Business
├─ hj_return_business_floor (35+4+20=59f)  ── walk to climb anchor (x≥88)
│
├─ SETUP JUMPS (460f)  ── always runs on return (not skipped)
│  ├─ 4× (business_climb_release 12f + business_climb_setup 85f + business_climb_setup_land 30f)
│  └─ only skipped if already standing on y=1339 (never true on floor return)
│
├─ 1339→1227 HOP (71f)  ── walk-left-to-x≤84, standing gate, charge jump
│  ├─ 1339_position (26f) + 1339_brake (4f) + 1339_release (8f) + 1339_settle (20f)
│  └─ 1339_replant (13f) ── retry if walk-left steps off platform lip
│
├─ 1227→1147 HOP (134f)
│  ├─ 1227_back (16f) + 1227_release (15f) + 1227_runup (8f) + 1227_settle (20f)
│  └─ to_1147 (90f) ── charge jump arc
│
├─ 1147→1067 HOP (129f)
│  ├─ 1147_release (16f) + 1147_settle (20f)
│  └─ to_1067 (124f) ── longest charge jump arc
│
├─ 1067→987 HOP (201f)
│  ├─ 1067_position (23f) + 1067_release (12f) + 1067_jump_release (8f) + 1067_settle (30f)
│  └─ to_987 (36f) ── short hop, overhead platform edge
│
├─ 987→907 HOP (86f)  ── the 14f-runup hop (continuous-hardened)
│  ├─ 987_runup (14f) + 987_release (12f) + 987_settle (20f)
│  └─ to_907 (43f) + 907_back (22f) + 907_brake (17f) ── overshoot correction
│
├─ 907→843 HOP (130f)
│  ├─ 907_release (12f) + 907_runup (8f) + 907_brake2 (3f) + 907_settle (20f)
│  ├─ 907_back (22f) ── walk right to x≥205
│  └─ to_843 (48f)
│
├─ 843→779 HOP (112f)
│  ├─ 843_position (21f) + 843_release (12f) + 843_jump_release (6f) + 843_settle (20f)
│  └─ to_779 (43f)
│
├─ 779→ELEVATOR HOP (112f+115f)
│  ├─ 779_position (26f) + 779_release (12f) + 779_settle (20f)
│  ├─ 779_prejump gate (6f+3f+30f in wait_standing_y)
│  ├─ to_elevator (68f) + elevator_brake (2f) + elevator_settle (20f)
│  └─ elevator_center (variable, up to 40f) ── only if x∉[95,160]
│
├─ ELEVATOR UP (501f)  ── fixed: hold UP until room_id==ROOM_WAREHOUSE
│
└─ WAREHOUSE EXIT (317f)
   ├─ to_warehouse_settle (165f) ── wait_ordinary_room settle_frames=360
   ├─ elevator_top (30f) + elevator_exit walk (122f) + exit_settle (30f)
   └─ walk to x≤40,y≤145 anchor
```

---

## Top suspected waste

### 1. Fixed 20f settle at every platform (8× = 160f)

Every platform hop ends with `_hold(session, 20, reason="business_NNNN_settle")` followed by `_wait_standing_y(session, NNNN, ...)`. The 20f hold is a fixed idle wait; the subsequent `wait_standing_y` typically returns in 0f (already standing). This pattern appears at lines 132, 146, 165, 176, 196, 216, 239, 257, 287.

- **`business_climb.py:132`**: `_hold(session, 20, reason="business_1339_settle")`
- **`business_climb.py:146`**: `_hold(session, 20, reason="business_1227_settle")`
- (repeats for 1147, 1067, 987, 907, 843, 779, elevator)

**Estimated waste:** 160f (8×20f). Could be replaced by moving the `wait_standing_y` predicate inline (start polling on the first frame after landing).

### 2. Setup jumps always run (460f)

The 4-setup-jump sequence (`business_climb.py:99-102`) always runs because the return from HJ shaft enters Business at floor level (y≈1405+), so the `already` guard at line 93-97 never triggers. Each jump: 85f button-hold + 30f land = 115f × 4 = 460f.

- **`business_climb.py:93-102`**: guard check + 4-jump loop
- **`business_climb.py:99-102`**: `_hold(session, 85, direction, "B", "A")` + 30f land
- **`business_climb.py:100`**: `business_climb_release` (48f total across the run, 4×12f)

**Estimated waste:** 115-230f (1-2 jumps). The first RIGHT jump might be unnecessary if the landing spot is reachable from the floor anchor position with 3 jumps instead of 4. Risk: the setup jumps are "forgiving" for a reason — natural entry variance.

### 3. 907-hop overshoot correction (39f)

The 907 landing uses a 14f runup (`business_climb.py:207`: `runup_907=14`). The landing check is `samus_x >= 160` (line 210), then the controller walks back to `x <= 165` (lines 212-215). Because the landing x can overshoot to x≈170-180, the walk-back takes 22f (`business_907_back`). Plus 17f brake (`business_907_brake`). The pure probe uses 8f runup and apparently lands closer to the left edge.

- **`business_climb.py:207`**: `_hold(session, runup_907, "RIGHT", "B", ...)` — 14f vs 8f
- **`business_climb.py:210`**: `frame > 35 and state.samus_y == 907 and state.samus_x >= 160`
- **`business_climb.py:212-215`**: walk-left correction loop (up to 60f)
- **`business_climb.py:215`**: `_hold(session, 2, "RIGHT", ...)` — brake

**Estimated waste:** 20-40f. Tightening the x target or reducing runup could reduce the overshoot walk-back. Risk: continuous natural entry has higher variance; the 14f runup is a deliberate robustness buffer.

### 4. `wait_standing_y` timeout gates (variable)

Each `wait_standing_y` call has a timeout of 30-90f (lines 110, 127, 134, 147, 166, 177, 197, 219, 240, 258, 280, 288, 301). These timeouts are upper bounds and typically return in 0-5f (already standing after the 20f settle). But if the settle is replaced with inline polling, the timeout parameter becomes the total wait budget, which is fine.

**Not a standalone waste item** — subsumed by item 1.

### 5. Warehouse elevator exit settle (165f)

`wait_ordinary_room(session, ROOM_WAREHOUSE, settle_frames=360)` at line 359-361 takes 165f (`business_to_warehouse_settle`). The settle_frames=360 is a generous timeout; the actual settle is 165f of polling. The `wait_ordinary_room` function (controller_common.py:194-228) returns as soon as `room_id == expected` + `game_state == 8` + `door_transition == 0` + `frame > min_settle_frame` (15). So 165f is the time it takes after the elevator reaches the top for the game to settle into ordinary state 8.

**Not easily tightenable** — the 165f is the emulator's transition time, not a controller idle. Could be reduced by adjusting `min_settle_frame` but risk of settling mid-transition.

### 6. Elevator down/wait (first visit, 451f)

`business_incoming_elevator` (451f) belongs to the first Business visit (warehouse→hj_shaft), not the `business_to_warehouse` window. **Not in scope** for this tighten.

---

## Patch recipes (future cards)

### P1: Replace 20f settle-at-each-platform with inline wait-until-standing

- **Files:** `routes/kpdr/business_climb.py`
- **Change:** Remove each `_hold(session, 20, reason="business_NNNN_settle")` and fold the settle into `_wait_standing_y` (which already polls). Or change the settle to a tighter `_hold(session, 5, ...)` and let `wait_standing_y` handle the remaining frames.
- **Risk:** Very low. The `wait_standing_y` already has a timeout (30-90f) and polls every frame. Removing the 20f idle simply starts polling sooner.
- **Expected band:** ~160f (8×20f speculative, no claim without re-record).
- **Acceptance:** `uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video` — must be integrity green with <= previous dwell.

### P2: Reduce setup jumps from 4 to 3

- **Files:** `routes/kpdr/business_climb.py` lines 98-102
- **Change:** Change the setup loop to `("RIGHT", "LEFT", "LEFT")` (drop the final RIGHT) or `("LEFT", "LEFT", "RIGHT")`. The floor anchor is at x≥88; the first RIGHT jump goes right, then LEFT jumps move left toward the left wall. Test whether 3 jumps reliably land on y=1339.
- **Risk:** Medium — if the 3rd jump misses the platform, the fallback retry path (line 119-129) re-runs all 4 setup jumps, costing more than 1 jump saved. Verify with `--to kraid --no-video` for at least 3 successful runs.
- **Expected band:** 115f (1 jump) speculative.
- **Acceptance:** As above, or pure probe: `uv run python super_metroid/scripts/probe/kpdr.py pure warehouse-hijump-kraid --source <state>`.

### P3: Tighten 907-hop overshoot (reduce 14f runup to 10-12f)

- **Files:** `routes/kpdr/business_climb.py` line 207 (`runup_907=14`) and lines 212-215 (walk-back)
- **Change:** Reduce `runup_907` from 14 to 10-12 for the continuous path. This reduces the charge jump horizontal velocity, so the landing is closer to the left edge of the 907 platform, reducing the walk-back correction.
- **Risk:** Low-medium. The 14f runup was deliberately increased (from 8f pure) for continuous reliability. Reducing to 10-12f may cause the jump to fall short (miss the platform). Test with `--to kraid --no-video`.
- **Expected band:** 20-40f (reduced walk-back) speculative.
- **Acceptance:** As above.

### Combined P1+P2+P3

If all three patches land, the combined expected band is ~295-430f. Re-record `--to kraid --no-video` and `--to varia --no-video` to verify integrity and measure the actual delta.

---

## Caveats

- **No frame savings claimed without re-record.** The "expected band" is an estimate from counting action_reason frames, not a measurement.
- **The `business_to_warehouse` window (2,257f) includes the fixed elevator ride (501f) and the Warehouse settle (165f).** Roughly 666f (29%) is inherent transition time, not controllable platforming. The controllable climb portion is ~1,591f (2,257-666).
- **The `business_descend` (392f) and `business_incoming_elevator` (451f) are in the FIRST Business visit** (warehouse→hj_shaft split), not in this window. Do not confuse them when reading aggregated action_reasons.
- **Continuous-hardening was intentional:** standing gates, 14f runup, and 20f settles were added to fix prior fail-to-climb blockers. Any tightening must maintain the natural-entry reliability on `--to kraid`.

---

## Acceptance command for a future implement card

```bash
# Full tip re-record (must be integrity green, 0 state loads, 0 progression writes)
uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video

# Then verify dwell reduction
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_kraid.json --top 20

# Also verify the full chain is unaffected
uv run python super_metroid/scripts/record/continuous.py --to varia --no-video
```

Or for a pure-probe first pass:

```bash
uv run python super_metroid/scripts/probe/kpdr.py pure warehouse-hijump-kraid \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/red_to_warehouse_controller.state
```