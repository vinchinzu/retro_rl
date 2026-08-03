## Residual — SM-K4.4-PURE-R12

### Result
PARTIAL (structural pure-regression fixed; trajectory IMPL still red on Phase C)

Session: (1) discovered extract regression that broke full pure launch,
(2) restored R11 pure envelope, (3) extensive place/trajectory recon for a
new approach into right air band — no height-preserving Phase C path found
to ship.

### Files changed
- `routes/kpdr/bubble_mountain.py` — lip / right-shelf detection back to
  **stand_pin** poses (HEAD pre-extract); add `bubble_is_stand_pin_pose`.
  Climb reseat / mid-nub land stay **true_ground** (R11 spin-apex fix).
- `routes/kpdr/bubble_mountain_mid.py` — to-lip approach walk uses stand_pin
  (not true_ground-only).
- `tests/test_k4_norfair_scaffold.py` — regression test:
  `test_bubble_launch_lip_uses_stand_pin_not_true_ground_only`.
- `docs/tasks/SM-K4.4-PURE-R12-residual.md` — this residual.
- Dev recon dumps (not pure proof): `debug/bubble_r12_impl_recon*.json`.

No pure GREEN to Bat; no STATUS / continuous promote. R12 trajectory IMPL
into `(x≥340, y∈[280,370])` still open.

### Verify paste

```text
uv run pytest super_metroid/tests/test_k4_norfair_scaffold.py -q
# 27+ passed (incl. lip stand_pin regression)

uv run python super_metroid/scripts/probe/kpdr.py pure bubble-to-bat-cave \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/post_rising_tide_to_bubble_pure.state \
  --pin-json super_metroid/debug/bubble_to_bat_pure_pin.json --no-red-diag
# success=false
# max_x=349 min_y=260 phase_c_hit=False top_reached=False launched=True
# frames≈7227  (R11 envelope restored; extract thrash fixed)
```

### Acceptance

- [ ] Full pure Phase C or better — **not achieved** (trajectory gap holds)
- [x] Full pure min_y≤280 — **held** (min_y=260 after lip fix)
- [ ] Full pure top_reached — **red**
- [ ] Ordinary `0xB07A` — **red**
- [x] Unit green + lip stand_pin regression test
- [x] Residual PROCESS fields; no continuous/STATUS claim
- [x] Extract pure regression fixed (load-bearing for all further R-series)

### Structural regression (load-bearing — fixed this session)

Extract applied `bubble_is_true_ground` (poses 1/2/9/10 only) to:

- `bubble_on_launch_lip` (was private `_on_launch_lip`)
- to-lip approach walk
- `bubble_on_right_shelf` (was private `_on_right_shelf`)

Pre-extract HEAD used stand_pin poses (includes 25/26/27/28) for those
three. True-ground-only on lip **blocked natural pure launch**:

| Metric | Broken extract | Restored (R11 class) |
|--------|----------------|----------------------|
| launched | False | True |
| max_x | 144 | 349 |
| min_y | ~365 | 260 |
| frames | ~56054 | ~7227 |
| phase_c_hit | False | False |

R11 spin-apex rule still applies **only** to mid-air charge/reseat/land
during peak (`ol_extend` mid-nub break, climb `grounded` / reseat_hop).

### Trajectory recon (R12 IMPL — no ship)

Place + lip matrices (`debug/bubble_r12_impl_recon*.json`):

1. **Mid-cavity true-ground bridge** — reconfirmed absent (R12 RECON grid).
   Upper denser solids: top platform y~160 x∈[252,324] + prior right shelves
   only. No intermediate seats for a multi-hop chain lip→right.

2. **Lip one-shot envelope (place, no enemies):**
   - Height-preserving spin (R6 class): max_x @ y≤430 ≈ 198–254; **no Phase C**.
   - Fall-gated WJ after peak (switch_y=300, into=3): **Phase C once** at
     ~(302,428) but **min_y≈359** (height class regress).
   - Long spin 140f + WJ from x=90: Phase C, min_y≈388 (worse height).
   - **Cannot hold min_y≤280 and Phase C on the same lip arc.**

3. **Right-structure re-ascent from thrash band** — place climb from
   (x≥300, y≥450) never reaches usable shelves; floor solids y~530 cannot
   climb to Phase C. Recoverable place band starts ~y≤400 on right
   (`below_376_400` → top with shelf script). Natural first x≥300 is still
   ~y449 — **just below** recovery.

4. **Left-high / ceiling cross** — place from (80–200, y≤250) cannot reach
   x≥300 while high; free-air ceiling ~y228 left column holds (R11).

5. **Finish still place-proven only** from right air/shelf
   `(360,y≤370)` period-8 WJ or grounded shelves LEFT HJ → top.
   Top solids y~160 x≥252 also standable (Phase D seat exists).

### Rejected this session (do not re-ship without new pin)

| Attempt | Why |
|---------|-----|
| Fall-gated WJ on lip after height class | Phase C only with min_y regress (~359–388) |
| Mid-high window raise / period-only | Stagnation class; natural x≥250 only at y>450 |
| Right-wall climb from y≥450 / floor y530 | No height recovery to shelves |
| Left WJ → ceiling cross | No right band at height |
| Double-hop / mid-nub chain | No true-ground intermediate |

### Next action (required)

- **Next card ID:** `SM-K4.4-PURE-R13` (or continue R12 IMPL) with **one**
  named trajectory outside the rejected set, e.g.:
  1. Enemy-clear + mid-iso dash into right band **with** pure pin that holds
     min_y≤280 (R11 rejected mid-iso dash without clearance), or
  2. Human/maprando velocity-matched open-loop from natural mid pin dump, or
  3. Honest **BLOCKED on trajectory** + topology rethink (different door /
     order if product allows).
- Do **not** re-touch lip stand_pin vs true_ground split without pure pin.
- **Source state:**
  `scratch/post_rising_tide_to_bubble_pure.state` (acceptance).

### Non-claims

- Did not STATUS-promote.
- Did not forge progression/capacity/door/event/boss RAM.
- Not continuous evidence.
- Did not close SM-K4.4-PURE as pure GREEN to Bat.
- Place Phase C with height regress ≠ shippable R12.
- Continuous tip remains Frog Save (114,923f).

### Probe pin (if pure/geometry) — mandatory metrics

```text
# Full CATH-04 source after lip stand_pin restore
room=0xACB3 pose=25 x=313 y=484 door_transition=0
frames=7227 max_x=349 min_y=260
mid_reached=True top_reached=False door_reached=False
standing_mid_pinned=True launched=True phase_c_hit=False
supers=5 selected=0
```
