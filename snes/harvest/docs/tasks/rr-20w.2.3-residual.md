## Residual — rr-20w.2.3 D2 field clearing

**Status:** IN PROGRESS. Production now has the D2 Tactic + verifier.
Unit-green only. No live pin this sitting. Do not record `--video`.

### Verified this session

- Production gap vs leftover probe is closed in code, not live:
  - `observe_d2_farm` / `D2FarmStatus.is_complete` is the ADR contract
    (8 planted, 8 wet, ship before 17:00, zero debris, no damage boulder,
    hands clear, loaded farm, not animating). Two COMPLETE observes
    required. Hour>=17 with an empty bin is not shipping.
  - `D2FarmClearTactic` is the one post-shop row (`D2_FARM_CLEAR`).
    Live stamina spa/retry, skip empty smash chunks, mandatory plant/water
    /hammer/axe while those milestones remain. Probe is a thin adapter.
  - Empty quota chunks SUCCESS as loaded-map no-ops (`quota_satisfied`).
  - `ProgressSnapshot.signature()` ignores `step_count`.
  - `FenceClearLoopTask` has a real timeout, bounded pond-carry retries (3),
    bounded input-lock A/B (12), loaded-farm zero success, persistent
    `_skip_tiles` across dumps.
- `--stop-after-d2-shipping` unchanged. `--stop-after-d2-clear` stays on
  D2 (`include_end_day=False`) and exits success only from `is_complete`.
- Unit: d2_work + chunks + crop splice + leftover budget + quota +
  progress + fence + farm_clearer + stamina **265** passed.
- No STATUS. No ROM pin.

### Exact next action

Do not 400k `--section all`. Do not redo SE. Production Tactic is the
leftover probe now — bench SW rocks from After_Stones with spa:

```bash
HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe \
  --section rocks --chunk sw --state Y1_D2_After_Stones \
  --timeout 200000 --out recordings/d2_leftover_smash.json
```

Do not start from `Y1_D2_Morning_After_D1`. Do not STATUS.

### Non-claims

- No STATUS promotion
- No natural power-on Day 2 farm-clear
- No D2 movie / `--video`
- 5 stumps remain on `Y1_D2_Wood_Progress`
- 34 boulders remain (nw/ne/sw); SE 13 are gone only if you re-run
  After_Stones with spa (no end pin that sitting)
- `--stop-after-d2-clear` is unit-wired, not live-green
