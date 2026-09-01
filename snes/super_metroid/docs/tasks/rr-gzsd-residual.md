## Residual — rr-gzsd Ceres Elevator TAS-speed ascent

**Continue (rr-gzsd):** Occupancy is locked. The 363 left face latches.
Do not remap the shaft. Do not retry `chain.py`. Do not retry a
walk-off 475-side kick.

**Pin in:** `scratch/ceres_elev_wj/elev_entry.state` (cached fast entry).
**Living checkbox:** named-face wall jumps between 475 and 363, then plant.
**Probe:** `PYTHONPATH=snes uv run python -m super_metroid.scratch.ceres_elev_wj.climb 363`
(`--no-video`). Proof is `latch_363.json` + `decode.py`, not an MP4.

Did not STATUS-promote. Did not change `DEFAULT_CONTINUOUS_TIP`. Did not
overwrite `recordings/<tip>.json`. Did not wire `magnet.py`.

### Already green (do not re-prove)

| Layer | Evidence |
|-------|----------|
| Occupancy tests (no emu) | `uv run pytest snes/super_metroid/tests/test_ceres_elev_climb.py -q` — 16 passed. Half-tile col 13, 363 left face, 475/363 floors, editor clip/BTS, ROM shape-1 `8×16+8×0` at `$94:8B2B`. |
| Product 475 WJ (control) | face 216, pose 132. `our_wram.json`. |
| **363 left face latch** | face **160**, y=384–399, pose **132**. `scratch/ceres_elev_wj/latch_363.json`. |

**363 latch row** (decode, WRAM after the step):

| f | x.sub | y | pose | mt | xr | yr | a96 | buttons |
|---|-------|---|------|----|----|----|-----|---------|
| 24 | 151.255 | 399 | 26 | 3 | 5 | 12 | **11** (`$0A96=0x0B`) | LEFT |
| **25** | **152.255** | **399** | **132** | 20 | 5 | 12 | 0 (cleared) | LEFT A |

Recipe: from 475 plant, walk launch x=137, 2f RIGHT, RIGHT+A until first
frame in y∈[384,399] x∈[147,155], 2f LEFT A-off, LEFT+A. Carry peaked
y=323 still pose 132 — overshot the 363 seat, short of 267. Not wired.

### Closed miss

**475 box right side, walk-off.** `latch_475right.json`. in_band=True at
(165, 497) but pose 42/136 falling, `$0A96` never reached 0x0B (peak 3).
Walk-off is not a spin. One miss of this class; do not repeat it.

### Next claim (one run)

Spin CWJ off a 475 box side — the 363 takeoff (ground spin into the face,
2f A-off) applied to face 96 (wj 83–91) or face 160 (wj 165–173) at
y=496–511. Halt at the first miss. Then figure a plant after the 363 WJ
(y=323 midair is not a seat).

```bash
uv run pytest snes/super_metroid/tests/test_ceres_elev_climb.py -q
PYTHONPATH=snes uv run python snes/super_metroid/scratch/ceres_elev_wj/decode.py \
  snes/super_metroid/scratch/ceres_elev_wj/latch_363.json 22 26
```
