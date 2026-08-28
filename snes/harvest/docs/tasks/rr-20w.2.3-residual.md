## Residual — rr-20w.2.3 D2 field clearing

**Status:** IN PROGRESS. Leftover smash is now four farm chunks
(`nw`/`ne`/`sw`/`se`) chained stones → rocks → stumps. Stumps are
exhaustive (not quota 2). Unit chunked + full-chain tests are green.
Live last-cell stalls are still open.

### Verified this session

- Farm 64×64 partitions with no gap: `nw` (0,0)-(31,31), `ne`
  (32,0)-(63,31), `sw` (0,32)-(31,63), `se` (32,32)-(63,63). Live stall
  tiles land in named chunks: pocket `(11,29)` nw, FA-east `(48,13)` ne,
  south-stream stone `(12,55)` sw, last boulder `(60,51)` se.
- `d2_leftover_phases` emits 4 CLEAR_STONES + 4 CLEAR_ROCKS + 4
  CLEAR_STUMPS, each with `farm_bounds`. `--section stones --chunk sw`
  is one bounded phase. Stall abort still 24k per phase (no 400k hug).
- CLEAR_STUMPS quota is exhaustive (`10_000`, timeout 0). Full-chain
  complete requires stones=0, large_rocks=0, stumps=0. Skipping one
  chunk keeps the farm red.
- Quota smash with `farm_bounds` does not walk to the plant notch
  (pocket approach stays CLEAR_PLOT only). Fence stone dump passes
  `farm_bounds` into `FenceClearLoopTask`.
- Unit: `tests/test_d2_farm_chunks.py` + leftover/quota/clearer/glance
  190 passed. Did not re-run a live leftover pin.

### Exact next action

Live leftover is still last-cell work. Run one chunk, not `--section all`.
Last boulder is SE; last stone was SW. From the leftover partial pin:

```bash
HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe \
  --section rocks --chunk se --state Y1_D2_Leftover_Partial \
  --timeout 80000 --out recordings/d2_leftover_rocks_se.json
```

Then stumps by chunk from a hammer-done pin:

```bash
HEADLESS=1 uv run python -m harvest.scripts.d2_leftover_probe \
  --section stumps --chunk nw --state Y1_D2_After_Spa \
  --timeout 80000 --out recordings/d2_leftover_stumps_nw.json
```

Human inspect if a chunk stalls:

```bash
uv run python -m harvest.runtime.harvest_bot play \
  --state Y1_D2_Leftover_Partial --no-day-plan --record leftover_rocks_se
```

Do not start from `Y1_D2_Morning_After_D1`. Do not 400k `--section all`.

### Non-claims

- No STATUS promotion
- No natural power-on Day 2 completion
- CLEAR_STONES green is leftover-pin continuation, not power-on
- Live farm still has leftover smash (last boulder + stumps)
- Chunked unit empty ≠ live pin empty
