# Metal stroke-play calibration memory

Durable notes for Amateur stroke play with the **METAL PLAY** bag.
Route source of truth remains `hals_golf/tasks/routes/metal.py` and
`REST_BAND_OVERRIDES` in `shot_policy.py`. This file records what worked,
what blew up, and what to try next.

## Best known results

| Date | Scope | Scorecard / hole | Notes |
|------|-------|------------------|-------|
| 2026-07-20 | H1–H12 partial | `[3,3,8,2,3,3,14,5,66,25,6,4]` | Only H1/H2/H4/H6 calibrated |
| 2026-07-21 | H7–H11 singles/clear | 5 / 4 / 2 / 7 / 5 | Worst-hole pass |
| 2026-07-21 | H12–H18 singles | **3 / 3 / 4 / 3 / 5 / 3 / 3** | Full back-nine overlays |
| 2026-07-21 | H17/H18 timeline fix | **2 / 4** | Eagle + par; putt `13→20`; tee `42/-5`→`44/-5` |
| 2026-07-21 | H12–H18 live clear | **`[3,3,4,3,5,2,4]`** (−3) | `over_par=[]`, 23,025 frames |
| 2026-07-21 | **Title metal video** | **`[3,3,4,6,4,3,5,4,2,7,5,3,3,4,3,5,2,4]`** | total=**70** (−2), `record_metal_clear.sh` |

Verified Title→`course_complete` (video `recordings/metal_stroke_clear.ogv`):

```text
[3, 3, 4, 6, 4, 3, 5, 4, 2, 7, 5, 3, 3, 4, 3, 5, 2, 4]
```

total **70** (−2), over-par **H4 (+3)** and **H10 (+3)**. H3 finish fixed
(soft driver from ~62y) for the recording run.

Composite best-known per-hole stitch (singles; H4 still 2 offline):

```text
[3, 3, 4, 2, 3, 3, 5, 4, 2, 7, 5, 3, 3, 4, 3, 5, 2, 4]
```

## Priority (worst first)

1. **H4** — Title clear scored 6; single-hole metal still 2
2. **H10** — 7 is playable but still +3
3. Re-record after H4/H10 improve: `./record_metal_clear.sh`

## Provenance map (stroke ← live search)

| Hole | Overlay source | Confidence |
|------|----------------|------------|
| 1–2, 4, 6 | Live stroke calibration | High |
| 3 | MetalTee3 `44/-2` + VS HAL water finish | Medium |
| 5 | VS HAL metal soft SW | Medium |
| 7 | `MetalH7_fw254/144/23` | High |
| 8 | Bunker-avoid tee + `MetalH8_fw104` | High |
| 9 | MetalTee9 PW `38/0` green | High |
| 10 | VS HAL metal corridor (stroke scored 7) | Medium |
| 11 | `MetalH11_fw219/58` | High |
| 12 | MetalTee12 `44/-4` + 7I `38/-2` | High |
| 13 | MetalTee13 8I `38/-2` green | High |
| 14 | Amateur metal fallback (scored par) | Medium |
| 15 | MetalTee15 `44/-8` + 3W `42/-4` | High |
| 16 | `MetalH16_fw276/149/22` four-shot | High |
| 17 | MetalTee17 7I `34/-4` + putt `13→20` (eagle) | High |
| 18 | Post-H17 MetalTee18 `42/-5` + `44/-5` (`MetalH18_fw169`) | High |

**Timeline note:** metal tees rewrite under `--tee-state-prefix`. After an H17
eagle, `MetalTee18` rejects the older `44/0` opener — always re-search the
live tee after upstream birdies/eagles.

## Calibration workflow

```bash
HEADLESS=1 PYTHONUNBUFFERED=1 ./run_bot.sh clear \
  --state Title --club-set metal --max-frames 250000 \
  --tee-state-prefix MetalTee

HEADLESS=1 ./run_bot.sh search-hio \
  --state MetalTeeN --club-set metal --max-candidates 60 \
  --club-deltas 0,1,2 --power-deltas 0,-2,2,-4,4 \
  --aim-deltas 0,-4,4,-8,8
```

Always record **end lie** when promoting a tee winner. Straight `42/0`
driver often **fails to move** on metal stroke tees (H12/H15) — prefer a
verified aim/club from `search-hio` before trusting Amateur power.

## Memory rules

1. **Code tables win** — never rely on agent transcripts as the only record.
2. **Named `Metal*` save states** are replay evidence, not planner input.
3. **VS HAL metal** is a prior only; always verify on stroke `MetalTeeN`.
4. Leave-shaped REST bands use `requires_vs_hal=None` (or False for stroke-only).
5. Empty overlay = intentional Amateur fallback — do not invent Pro numbers.
