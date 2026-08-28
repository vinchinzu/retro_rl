## Residual — rr-20w.2.3 D2 field clearing

**Status:** IN PROGRESS. 4/4 quota boulders and 2/2 quota stumps are
pin-green. CLEAR_STONES is pin-green exhaustive from leftover
continuation (not natural-entry). CLEAR_ROCKS live **47→1**.
**Natural entry:** power-on. Named states below are diagnostic pins and
do not promote STATUS.

### Verified this session

- FA-east / west-A1 trap-escapes still hold. Live 80k from
  `Y1_D2_Leftover_Checkpoint` (48,13) held stone: stones **39→1** in
  26.5k then stall-abort 24k (no 400k). `cleared_count=39`.
- Last on-map stone was **(12,55)**. Farmer sat at **(16,48)** empty-
  handed. Viewport BFS walked onto south-stream 0xFC A1 banks
  `(13/15,49–50)`. Live: DOWN at x=16 is open (8f to (16,49)); DOWN
  from (15,48) slides east back to (16,48); DOWN from (13,48) is a
  wall. `SOUTH_STREAM_FC_BANKS` is now in `FARM_NO_GO_TILES`.
- From `Y1_D2_Leftover_Partial` after that no-go: last stone **1→0**
  in **545f**, F0 toss, egress `(29,35)`. Pin `Y1_D2_After_Stones`
  (0 stones, 47 large, 36 stumps, axe+hoe, stam 76). Report
  `recordings/d2_leftover_stones.json`.
- CLEAR_ROCKS from After_Stones: hammer fetched. One-shot spa retry
  capped the first 80k at **47→33**. Cap removed: `stamina_low`
  requeues spa until timeout/stall.
- Continue from Partial: **33→1** with three successful spa soaks
  then last spa **timeout 12k** still on farm. End `(54,42)` stam 4
  hammer, last boulder **(60,51)**. Report
  `recordings/d2_leftover_rocks.json`. Partial pin is that stall.
- Do not start from `Y1_D2_Morning_After_D1`.

### Exact next action

Last boulder from `Y1_D2_Leftover_Partial` (stam 4, must spa). Spa
estimated 12k was not enough from (54,42) — diagnose `farm_to_spa`
from that tile, do not 400k. Then smash (60,51). Human inspect:

```bash
uv run python -m harvest.runtime.harvest_bot play \
  --state Y1_D2_Leftover_Partial --no-day-plan --record leftover_rocks_last
```

```bash
HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe \
  --section rocks --state Y1_D2_Leftover_Partial --timeout 80000 \
  --out recordings/d2_leftover_rocks.json
```

### Non-claims

- No STATUS promotion
- No natural power-on Day 2 completion
- CLEAR_STONES green is leftover-pin continuation, not power-on
- 1 large rock and 36 stumps remain
- Last spa from (54,42) did not reach the spring
