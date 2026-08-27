## Residual — rr-20w.2.3 D2 field clearing

**Status:** IN PROGRESS. 4/4 quota boulders and 2/2 quota stumps are
pin-green. CLEAR_STONES / CLEAR_ROCKS are exhaustive in catalog.
Live leftover from `Y1_D2_Leftover_Checkpoint` at 8k: **39→21** after
the FA-east dump hop moved. 21 stones remain. Natural-entry Day 2 is not.
**Natural entry:** power-on. Named states below are diagnostic pins and
do not promote STATUS.

### Verified this session

- Leftover stall no longer "checkpoint and continue". Unchanged debris
  for `stall_frames` (default 24k) **aborts the phase** (FAILURE
  `no debris progress`). That is why 400k cannot recur on a hug.
- FA-east lip live physics (checkpoint dump): stand `(46,16)=0x01`,
  water `(46,14)/(46,15)=0xFA`, A8 bank `(48–50,16)`, stump 2×2 at
  `(44,12)`, farmer on 0xA0 at `(48,13)` px (771,216) held=13.
  **LEFT to (47,13) works in 2f. DOWN at x=46–50 y=13 is a wall.**
  From (47,13), DOWN+B slides east back onto (48,13) — that was the
  400k oscillation, not a missing via list. First open DOWN is **x=51**
  (4f). Scripted dump (48,13)→(51,13)→(51,16)→(46,16) in 82f; toss
  empties hands. On-map count does not include the held stone.
- Carry vias now cross at `EAST_SPUR_FA_SOUTH_OPEN_X=51` when y≤13.
  A8 stays no-go from the north (repair skirts y=17). FA-east stasis
  does not temp-block the x=51 column.
- West of the spur: `(45,14)/(45,15)=0xA1` push-blocks. Live 8k end
  `(44,14)` RIGHT is blocked. Vias drop at x=44 to y=16 then east onto
  `(45,16)`. Unit-green; not a second live leftover.
- Live 8k from `Y1_D2_Leftover_Checkpoint`: stones **39→21** (18
  pond-dumped), `cleared_count=18`, left the (48,13) lip. End `(44,14)`
  still `carry to pond stand (46,16)`. Report
  `recordings/d2_leftover_stones.json`. No 400k.
- Do not start from `Y1_D2_Morning_After_D1`.

### Exact next action

Continue leftover stones from `Y1_D2_Leftover_Checkpoint` (or a fresh
checkpoint if one is saved). Default stall 24k, timeout **not** 400k
until remaining 21 move. Human inspect:

```bash
uv run python -m harvest.runtime.harvest_bot play \
  --state Y1_D2_Leftover_Checkpoint --no-day-plan --record leftover_stones_ne
```

```bash
HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe \
  --section stones --state Y1_D2_Leftover_Checkpoint --timeout 80000 \
  --out recordings/d2_leftover_stones.json
```

### Non-claims

- No STATUS promotion
- No natural power-on Day 2 completion
- No claim that remaining 21 stones or 47 boulders are gone
- 8k did not finish CLEAR_STONES and did not start CLEAR_ROCKS
