# SM-ROLLUP-STATUS-9 Proposal — Wave 9 honest rollup

**Recipe step:** docs / STATUS proposal (Flash).  
**Model:** Flash.  
**Does not apply STATUS.md.**

---

## Wave 9 summary

| Metric | Value |
|--------|------:|
| Sessions | **~10** EXIT:0 |
| Critical spine redesign (kihunter→zeela) | **GREEN** ~1716f, mid-ledge+bomb-hole |
| Next spine hop (zeela→warehouse) | **RED** — floor-left pin → R-03B redesign |
| Practice rooms | **4 problems**: 0 GREEN, 4 PARTIAL/RED (all parked or re-dispatched) |
| Continuous post-Varia | **Not attempted** (0 false claims) |
| STATUS 104,382 | **Not promoted** |
| K3.6 tracker status | Updated to `controller_dev` |

---

## 1. Critical spine: kihunter→zeela — GREEN at last (CLIMB-REDESIGN)

SM-K4-R-CLIMB-REDESIGN switched maneuver class from cadence-only (min_y=371 forever) to **mid-ledge + morph bomb-jump through x≈376 hole**. Pure green ~1716f from `post_baby_to_kihunter_return.state` → `0xA471` Zeela ordinary.

| Card | Result | Why it worked |
|------|--------|---------------|
| R-02D/E/F (5 prior) | RED | One-knob cadence tuning: min_y=371 hard wall |
| CLIMB-REDESIGN | **GREEN** ~1716f | Wall-plant → mid ledge → morph bomb-jump through hole → Zeela morph-roll |

Source captured: `scratch/post_kihunter_to_zeela_return.state` (`0xA471`).

---

## 2. Next spine hop: zeela→warehouse — RED (R-03 → R-03B)

R-03 probed from the fresh `post_kihunter_to_zeela_return.state` (Zeela bottom-right, x≈403 y≈362). Morph left-push reached floor-left band (x=19 y=395 door_transition=1) — **wrong band** (not upper-left Warehouse). Root cause: reverse must **climb the reverse of forward Zeela drops** before left Warehouse door, not just floor-push.

| Card | Result | Pin |
|------|--------|-----|
| R-03 | **RED** | `room=0xA471 pose=16 x=19 y=395 door_transition=1` |
| R-03B | **dispatched** | Maneuver-class rewrite: reverse-roll → climb mid → climb top → left Warehouse |

**R-03B is the critical tip blocker.** Until it greens, the reverse chain cannot reach Warehouse/Business.

---

## 3. Practice room residuals — all RED/PARTIAL, dual-track only

| Room | Wave 9 result | Pin | Next |
|------|---------------|-----|------|
| Ice Tutorial `0xA865` | **PARTIAL** (TUT-R3) | `x=277` same pin | **Parked** — pose-138 class, no R4 spam |
| Grapple Tutorial 2 `0xABD2` | **RED** (EASY-03-R2) | `x=21` same pin | R-03B **dispatched** |
| Crab Hole `0xCF80` | **RED** (EASY-02B) | still wrong exit `0xCF80` | C **dispatched** |
| Metal Pirates `0xB62B` | **RED** (METAL-02) | fixture `max_supers=0` | 03 **dispatched** |

None block KPDR spine.

---

## 4. Tracker / graph — partial

- K3.6 CSV updated to `controller_dev` (~1716f CLIMB-REDESIGN) — **done**
- K3.7 stays `open` (blocked on R-03B)
- SM-K4-R-GRAPH-B dispatched to promote `kihunter_to_zeela_return` edge → `controller_dev`

---

## 5. Non-claims (explicit)

- **No 104,382 frame promote.** No continuous tip total is proposed for STATUS.md.
- **No continuous post-Varia tip.** R-03B (zeela→warehouse) is the next blocker; then 3+ more reverse hops (warehouse→business→business climb) needed before compose.
- **No STATUS.md edit.** This proposal file is the only deliverable.
- **Practice room greens are not continuous evidence.** None of the 4 practice problems would change spine status even if green.
- **No progression/capacity/door/event/boss RAM forges.**
- **CLIMB-REDESIGN pure green does not imply continuous green.** Graph edge stays `controller_dev`.

---

## 6. Suggested planner dispatches (not implemented here)

```bash
# 1. Spine critical: wait for R-03B result (reverse-drop rewrite)
#    If R-03B also RED: escalate to planner — reverse geometry needs different
#    approach (e.g., different source state, or accept forward-reverse asymmetry).

# 2. Practice residuals (parallel, disjoint from kraid_return.py):
./super_metroid/scripts/dispatch_opencode.sh \
  SM-ROOM-EASY-02C \
  SM-ROOM-EASY-03-R3 \
  SM-ROOM-METAL-03 \
  SM-ROOM-BOOT-01 \
  SM-ROOM-ICE-TUT-PARK

# 3. Graph/Tracker:
./super_metroid/scripts/dispatch_opencode.sh SM-K4-R-GRAPH-B

# 4. Next rollup after R-03B resolves:
./super_metroid/scripts/dispatch_opencode.sh --flash SM-ROLLUP-STATUS-10
```

---

## 7. Next planner gates (suggested)

| Gate | Action |
|------|--------|
| R-03B green | Promote `zeela_to_warehouse_return` edge → `controller_dev`; capture source |
| Zeela→Warehouse pure green | Dispatch `warehouse-to-business` pure (SM-K4-R-04) |
| Business return pure green | Chain compose + continuous re-record `--to kraid` and `--to varia` |
| Practice rooms | Ice parked; Grapple/Crab/Metal remain dual-track only |
| STATUS | Do not promote until Business return continuous green |

---

## 8. Honest tone

Wave 9 achieved the **critical redesign** the Wave 8b/8c/8d cadence cards could not: kihunter→zeela climb switched from pure-cadence to a mid-ledge bomb-jump class that actually reaches the upper band. This is a meaningful spine unlock after ~10 sessions of RED results. However, the next hop (zeela→warehouse) immediately hit a new blocker — the reverse geometry is fundamentally different from forward drops. R-03B is scoped to rewrite the maneuver class again (reverse-roll → climb mid → climb top), not as a knob tweak.

No STATUS promotion. No continuous post-Varia. Honest RED on R-03.