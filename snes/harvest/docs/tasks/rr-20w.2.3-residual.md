## Residual — rr-20w.2.3 D2 field clearing

**Status:** IN PROGRESS. Last stump chunks vs `is_complete` are unit-green.
No live pin this sitting. Do not record `--video`.

### Verified this session

- Last leftover stumps are not D2 complete:
  - 5 remaining (`Wood_Progress` shape: nw 3 / ne 0 / sw 1 / se 1) keep
    `observe_d2_farm` `WORK_REMAINING`. Hour 18 + ship evidence + one SE
    stump is still not `is_complete`.
  - `next_d2_spec` skips empty stump chunks; live order is nw → sw → se.
  - Leftover `--section stumps --chunk se` SUCCESS is not whole-farm
    complete (4 stumps remain).
- Empty last stump chunk is a loaded-map SUCCESS no-op (`quota_satisfied`).
- Last live stump SUCCESS drains stamina and still settles `is_complete`
  without a spa trip. Mid-chain nw SUCCESS with se remaining still spas.
- Tactic remaining smash is empty once `_section_done` for this
  section/chunk. `leftover_chain_decision(..., remaining=())` is
  `continue`, not `insert_spa`.
- Units: d2_work + chunks + quota + leftover budget + progress +
  farm_clearer + stamina + fence + toss + crop splice + run_to_day2
  budget + tools + walk solids **302** passed. No STATUS. No ROM pin.

### Exact next action

Do not 400k `--section all`. Last stump completion is unit-only — bench
the 5 remaining from `Y1_D2_Wood_Progress` (chunk-scoped if a dump shows
a single live quadrant):

```bash
HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe \
  --section stumps --state Y1_D2_Wood_Progress \
  --timeout 200000 --out recordings/d2_leftover_smash.json
```

Do not start from `Y1_D2_Morning_After_D1`. Do not STATUS.

### Non-claims

- No STATUS promotion
- No natural power-on Day 2 farm-clear
- No D2 movie / `--video`
- 5 stumps remain on `Y1_D2_Wood_Progress` (not live-cleared)
- 34 boulders remain (nw/ne/sw); SE 13 are gone only if you re-run
  After_Stones with spa (no end pin that sitting)
- `--stop-after-d2-clear` is unit-wired, not live-green
