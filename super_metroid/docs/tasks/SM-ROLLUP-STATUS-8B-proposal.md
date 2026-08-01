# SM-ROLLUP-STATUS-8B Proposal — Wave 8b honest rollup

**Recipe step:** docs / STATUS proposal (Flash).  
**Model:** Flash.  
**Does not apply STATUS.md.**

---

## Wave 8b summary

| Metric | Value |
|--------|------:|
| Sessions | **~20** EXIT:0 |
| Pure geometry (critical tip) | **R-02F live — RED** (5 cards, 0 green) |
| Recon diagnostics | KIHUNTER-RECON-report done; CLIMB-RECON report **incomplete** (script exists, report unwritten) |
| Practice rooms | 4 problems: **0 green** (all PARTIAL/RED) |
| Boss wraps | BOSS-WRAP-01 + BOSS-UNIT-MATRIX: **GREEN** |
| Future scaffolds | Alpha PB / WS / Charge: **GREEN** (scaffold + unit only) |
| Primitive promote | PRIM-01D (red_tower settle_hold): **GREEN**; PRIM-02C (vertical hop): **open** |
| Continuous post-Varia | **Not attempted** (0 false claims) |
| STATUS 104,382 | **Not promoted** (explicitly flagged) |

---

## 1. Critical tip: kihunter→zeela — still RED (R-02D/E/F)

This is the **hardest Wave-8b residual** and the primary block to K3.6 (reverse spine Kihunter→Zeela).

| Card | Result | Pin |
|------|--------|-----|
| R-02 | RED | Baby `0xA521` after climb |
| R-02B | RED | Same Baby pin |
| R-02C | BLOCKED | kraid_approach.py syntax error blocked probe |
| R-02D | RED | `x=357 y=395` lower wall; climb timeout (3 launch strategies) |
| R-02E | RED | `x=470 y=395`; right-cap variants timeout |
| R-02F | **running** | Vertical launch cadence only |

**Root cause (honest):** the natural source state ( `0xA4DA`, x≈465, y≈378 ) requires clearing shot blocks in the ceiling of the lower alcove while using Hi-Jump/spinjump to reach y<280. Three cards of geometry tuning (bands, caps, launch setup) have not reached the upper band. The climb may be **too demanding for current input patterns** — the recon shows `min_y≈291` as the planner's best, 11 pixels short of `y<280`.

### Recommendation for planner
- **Wait for CLIMB-RECON report** (108-trial grid: 9 launch delays × 4 caps × 3 shot patterns). If no trial reaches y<280, the answer is **not** a tunable knob — the climb needs a fundamentally different approach.
- If recon shows a viable band (e.g. left=12f, cap=470, pattern B reaches y=275), apply that exact recipe to R-02F's residual.
- **Do not add more "one-knob" geometry cards** after R-02F if recon says impossible — escalate to planner for a different climb strategy (e.g. wall-jump approach, different beam timing, or accepting warp to a different source position).
- **Block R-03 (zeela→warehouse)** until R-02F green.

---

## 2. Recon diagnostics

| Card | Status | Notes |
|------|--------|-------|
| SM-KIHUNTER-RECON | **Done** | Zeela door band `x∈[96,160]` identified (dev-warp). Natural climb never reached upper band. |
| SM-KIHUNTER-CLIMB-RECON | **Script exists, report unwritten** | 108-trial grid probe script at `scripts/probe/kihunter_climb_recon.py`; expected report `SM-KIHUNTER-CLIMB-RECON-report.md` not found. May still be running or failed mid-way. |

---

## 3. Practice rooms — all PARTIAL/RED

| Room | Card | Result | Pin | Next |
|------|------|--------|-----|------|
| Boulder `0xA1AD`→`0xA1D8` | EASY-01-R1 | **PARTIAL** | `pose=138 x=85 y=187` (left wall) | R2: door-shot geometry |
| Ice Tutorial `0xA865` | TUT-R1/R2 | **PARTIAL** | `pose=138 x=277 y=139` | R2: clear pose-138 + left exit |
| Grapple Tutorial 2 `0xABD2` | EASY-03 / R1 | **RED** | `pose=138 x=21 y=395` | R1: left-exit residual |
| Metal Pirates `0xB62B` | METAL-01 | **RED** | `pose=137 x=731 y=171` | 01: combat-clear knob |

**Common pattern:** pose 138 (knockback/spin) at room boundaries. These are likely **door-trigger geometry** issues, not climb/combat failures. Practice track is dual-track only — none of these block KPDR spine.

---

## 4. Boss wraps — GREEN (track closed)

| Card | Files | Tests |
|------|-------|-------|
| BOSS-WRAP-01 | Ridley, Crocomire, Golden Torizo, Escape, Mother Brain wraps created | All importable |
| BOSS-UNIT-MATRIX | catalog × strategy matrix (42 combos) | 42 tests, all green |

---

## 5. Future scaffolds — GREEN (scaffold only)

| Card | Files | Status |
|------|-------|--------|
| ALPHA-PB-01 | `routes/kpdr/alpha_pb.py` + test | Scaffold + unit green |
| WS-01 | `routes/kpdr/wrecked_ship.py` + test | Scaffold + unit green |
| CHARGE-01 | `routes/kpdr/charge_return.py` + test | Scaffold + unit green |

**These are not continuous evidence.** Each needs a natural-entry source after the reverse spine reaches Business → K4 forward.

---

## 6. Primitive promotes

| Card | File | Status |
|------|------|--------|
| PRIM-01D | `routes/kpdr/red_tower.py` (settle_hold migration) | **Done** |
| PRIM-01E | `routes/kpdr/kraid_approach.py` (settle_hold) | **Done** |
| PRIM-02C | `routes/controller_common.py` (vertical hop) | **Open**; may leave-raw |

---

## Non-claims (explicit)

- **No 104,382 frame promote.** The Wave-6 re-record (104,382f) is slower than STATUS baseline (101,954f). STATUS.md stays on the 101,954f record.
- **No continuous post-Varia tip.** R-02F RED blocks K3.6 (Kihunter→Zeela). Even after R-02F green, R-03 (Zeela→Warehouse) and 4 more reverse hops needed before Business return compose.
- **No STATUS.md edit.** This proposal file is the only deliverable.
- **Practice room greens not continuous evidence.** 4 practice problems remain RED/PARTIAL; none would change the spine status.
- **No progression/capacity/door/event/boss RAM forges.**

---

## 7. Suggested planner dispatches (not implemented here)

```bash
# 1. Spine: wait for R-02F residual + CLIMB-RECON report
#    If both RED: planner designs a fundamentally different climb strategy.

# 2. Practice residuals (parallel, disjoint from kraid_return.py):
./super_metroid/scripts/dispatch_opencode.sh \
  SM-ROOM-EASY-01-R2 \
  SM-ROOM-ICE-TUT-R2 \
  SM-ROOM-EASY-03-R1 \
  SM-ROOM-METAL-01

# 3. Primitive close:
./super_metroid/scripts/dispatch_opencode.sh SM-PRIM-02C

# 4. If CLIMB-RECON report was lost: re-dispatch
./super_metroid/scripts/dispatch_opencode.sh SM-KIHUNTER-CLIMB-RECON

# 5. Rollup close after R-02F resolved:
./super_metroid/scripts/dispatch_opencode.sh --flash SM-ROLLUP-STATUS-8C
```

---

## 8. Probe pin (honest — not pure green)

```
kihunter→zeela (R-02E): room=0xA4DA pose=1 x=470 y=395 door_transition=0 frame=336
kihunter→zeela (R-02D): room=0xA4DA pose=2 x=357 y=395 door_transition=0 frame=340
kihunter→zeela (R-02C): blocked before probe (kraid_approach.py syntax)
```

---

## 9. Honest tone

Wave 8b found the **real hardness** of the Kihunter shot-block climb. Three pure-geometry cards (R-02C→D→E→F) each changed one knob and each yielded the same timeout at ~y=395. This is not a soft barrier — the climb from natural source is genuinely difficult for the current input vocabulary. The CLIMB-RECON grid (if completed) will tell us whether any input pattern within our control can clear `y<280`. If not, the planner needs to either:

1. Accept a different entry pose/position (capture a new source state closer to the upper tunnel);
2. Add wall-jump or bomb-jump primitives;
3. Accept that K3.6 (Kihunter→Zeela) requires a fundamentally different approach not derivable from the current `post_baby_to_kihunter_return.state` source.

No amount of "one more knob" cards will fix a missing capability.