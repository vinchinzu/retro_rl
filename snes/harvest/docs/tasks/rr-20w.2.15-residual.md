## Residual — rr-20w.2.15 D2 leftover section-first lift

**Status:** planner GREEN, live NW bushes RED (boxed stone). Do not STATUS.
No `--video`. No full-day soak.

### Verified this session

- `next_d2_spec` is section-first for hands work: weeds → fences → stones
  in one quadrant before the next. Hammer/axe still chain by chunk after.
- Unit: `tests.test_d2_farm_chunks.SectionFirstLiftTests` plus leftover
  order tests. Local NW stone beats distant SE bush.
- Dump `Y1_After_Buy_Potato` `recordings/d2_leftover_section_dump.json`:

  | chunk | weeds | fences | stones | rocks | stumps |
  |-------|------:|-------:|-------:|------:|-------:|
  | nw    | 79    | 63     | 25     | 5     | 4      |
  | ne    | 96    | 17     | 63     | 15    | 13     |
  | sw    | 168   | 0      | 51     | 18    | 10     |
  | se    | 163   | 0      | 46     | 13    | 11     |

- Live `--section bushes --chunk nw`: only NW weeds moved **79→42**
  (37 lifted). Other chunks unchanged. Then goal-stall at stone `(5,5)`
  (`Skipping lift thrash`), 35509f, `recordings/d2_leftover_bushes_nw.json`.

### Exact next action

FarmClearer boxed-weed opener at `(5,5)`: lift the blocking stone instead
of skip-thrash, or mark it failed and pick the next reachable weed.
Do not re-soak the whole farm. Pin remains `--section bushes --chunk nw`.

### Non-claims

- No STATUS promotion
- Did not start from `Y1_D2_Morning_After_D1`
- Did not record a BFS-closable walk
- Did not treat CrossMap origin-return as shop success
- Did not claim a 40% full-clear Δ (no whole-farm re-bench)
