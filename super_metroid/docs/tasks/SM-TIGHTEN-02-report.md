# SM-TIGHTEN-02 Dwell Report — `hj_shaft_to_business`

**Date:** 2026-07-31
**Source:** `recordings/start_to_varia.json` (outcome=`varia_collected`, total_frames=101,954)
**Target:** `hj_shaft_to_business` split — 1,885f dwell, room 0xA7DE (Business) @89,984f
**Controller:** `routes/kpdr/hijump_return.py` → `play_hj_shaft_to_business`
**Preceding:** `hj_room_to_shaft` (403f, @88,099) → `hj_shaft_to_business` (1,885f, @89,984)
**Return chain total:** `hj_room_to_shaft` (403f) + `hj_shaft_to_business` (1,885f) = 2,288f

## Phase map

| Phase | Lines | Expected frames | Action reason |
|-------|-------|----------------:|---------------|
| Bottom settle | 80 | 50 | `hj_return_bottom_land` |
| First jump (floor → right shelf) | 83–86 | 125 + 10 + 80 = 215 | `hj_return_first_jump` + `hj_return_first_land` |
| Right shelf position | 90–96 | 50 + up to 80 + 6 + 8 = ~144 | `hj_return_shelf_*` |
| Second jump (shelf → upper-left slope) | 99–101 | 130 + 50 = 180 | `hj_return_second_jump` + `hj_return_second_land` |
| Slope → morph tunnel | 106–112 | 40 + up to 110 + 40 = ~190 | `hj_return_top_jump` + `hj_return_top_land` |
| Bomb tunnel crawl | 117–122 | **up to 1,100** | `hj_return_bomb_tunnel` |
| Sova cleanup (conditional) | 124–129 | up to 500 | `hj_return_sova_cleanup` |
| Gray door approach | 131 | 80 | `hj_return_gray_approach` |
| Gray door exit | 134–139 | up to 600 | `hj_return_gray_exit` |
| Business settle (wait_ordinary) | 141–142 | 280 | `hj_shaft_to_business` (settle) |
| Business floor fall | 144–147 | up to 120 | `hj_return_business_floor` |
| Business climb anchor | 148–152 | up to 100 | `hj_return_business_climb_anchor` |
| Brake + settle | 153–154 | 4 + 20 = 24 | brake + settle |

## Waste candidates with reason labels

### 1. Bomb tunnel (719f — 38% of split)
**Reason:** `hj_return_bomb_tunnel` (719f, #17 in global reasons)
**Analysis:** The tunnel is ~100px wide (x 250→350). Morph crawl speed is ~1px/frame, so pure traversal should be ~100f. The bomb cycle `frame % 45 < 2` means only 2 bomb frames per 45-frame cycle, creating long coasting gaps. Each bomb has a 4-frame arm + detonation + knockback, so Samus advances intermittently. The 719f actual cost suggests ~85% of the loop is spent waiting for bomb cycles instead of crawling.

The early-break condition `state.samus_x >= 350` is generous — the tunnel exit is around x=320, so Samus may overshoot.

### 2. Business settle (524f — 28% of split)
**Reasons:** `hj_shaft_to_business` (280f settle) + `hj_return_business_floor` (120f) + `hj_return_business_climb_anchor` (100f) + brake/settle (24f)
**Analysis:** Three sequential waits:
- 280f `wait_ordinary_room` — generic safety margin from `dev.common.door_warp` pattern. Could be reduced to 180–200f since the gray door exit leads directly into Business (known room, known entry door).
- 120f `hj_return_business_floor` — waits for y ≥ 1419 after entering. The room entrance is the upper-left door; Samus falls from ~y=1200 to ~y=1419 in ~40f. The 120f extra is pure idle.
- 100f `hj_return_business_climb_anchor` — waits for x ≥ 88. This is walking right from the door (~x=30) to the climb anchor. At ~1.5px/frame, this takes ~40f. The 100f is generous.

### 3. Gray door exit (estimated ~200–300f of max 600f)
**Reason:** `hj_return_gray_exit` (below 200f threshold, exact not in top reasons)
**Analysis:** The exit pattern `Right+B+X` for 4f every 30f, then `Right+B+A` for 26f. The gray door opens ~50f after the Sova is killed. The pattern is designed to keep Samus running while shooting. The actual cost is unknown but likely 200–300f based on the door-open delay. Not a top priority but could be tightened.

### 4. Sova cleanup (0–500f, conditional)
**Reason:** `hj_return_sova_cleanup` (below threshold)
**Analysis:** If the bomb tunnel kills the Sova before the x=350 break, cleanup is skipped (0f). If the Sova survives, up to 500f of extra bombing. The bomb tunnel loop already drops bombs every 45f, so the Sova is likely killed during the tunnel crawl. Actual cost is probably 0f for this run.

### 5. Room traversal (gap: ~320f from 1,885 total)
**Analysis:** After accounting for bomb tunnel (719f) + settle (524f) + gray exit (~300f estimated), the remaining ~320f covers the three jump segments (first jump ~215f, second jump ~180f, top jump ~190f for ~585f nominal). These are the actual movement segments and are reasonably tight. The jumps themselves are not bottlenecks.

## 2–3 future patch recipes

### Recipe A: Tighten bomb tunnel (target: 719f → ~400f)
**File:** `routes/kpdr/hijump_return.py`, lines 117–122
**Change:** Increase bomb frequency from `frame % 45 < 2` to `frame % 30 < 3`. This increases the duty cycle from 4.4% to 10%, reducing the coasting gaps.
**Risks:** More bombs may knock Samus backward against the tunnel ceiling. Test with 3–5 runs.
**Expected savings:** ~300f from faster tunnel traversal.

### Recipe B: Trim business settle (target: 524f → ~300f)
**File:** `routes/kpdr/hijump_return.py`, lines 141–154
**Change 1:** Reduce `wait_ordinary_room` settle from 280f to 180f (line 142).
**Change 2:** Reduce `hj_return_business_floor` loop from 120f to 60f (line 144).
**Change 3:** Reduce `hj_return_business_climb_anchor` loop from 100f to 60f (line 148).
**Risks:** If the gray door load is slow, the room may not be ordinary by the 180f mark. The earlier floor wait reduces the safety margin but the gray door exit is consistent (known room, no RNG).
**Expected savings:** ~220f from reduced idle.

### Recipe C: Gray door exit speed (target: ~300f → ~150f)
**File:** `routes/kpdr/hijump_return.py`, lines 134–139
**Change:** Replace the `Right+B+X` / `Right+B+A` pattern with `hold(session, 1, "RIGHT", "B", "X")` for 4f, then `hold(session, 1, "RIGHT", "B")` for the remainder. Use `X` only on the first 4 frames (to open the door) and `B` continuously for speed. Remove the `A` (jump) since the approach is flat.
**Risks:** Sova may be alive if bomb tunnel failed; door won't open. Add a 2-super-shot safety if needed.
**Expected savings:** ~150f from better approach speed.

## Total estimated savings (recipes A+B+C)
~300f + ~220f + ~150f = **~670f** (11s real time). Post-tighten target: **~1,215f** for the `hj_shaft_to_business` split.

## Verify command for implement card
```bash
uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video
# Then compare split dwell for hj_shaft_to_business:
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_kraid.json --top 15 | grep hj_shaft_to_business
```

## Caveat
No frame savings claimed without re-record. These are pre-tighten estimates based on action-reason accounting and code analysis. The bomb tunnel (719f) and business settle (524f) are the two dominant cost centers; other phases are movement-constrained and unlikely to yield significant savings.