## Residual — rr-20w.2.3 D2 field clearing

**Status:** IN PROGRESS. 4/4 boulders and 2/2 quota stumps are pin-green.
CLEAR_STONES is exhaustive in catalog (unit-green) but not empty on the
live leftover pin. Two live leftover-stone windows from
`Y1_D2_Stones_Frontier` are red. Natural-entry Day 2 field clearing is
not.
**Natural entry:** power-on. Named states below are diagnostic pins and
do not promote STATUS.

### Verified this session

- Trimmed `horse_barn_edges` in place: dropped the idle prefix (no
  input). 107066 → 9438 frames (~16.9MB, still gitignored). Start state
  `Y1_D2_Stones_Frontier`. First step off `(17,20)` `0xA3` is south onto
  `(17,21)`, then west around y=23. Do not start from
  `Y1_D2_Morning_After_D1`.
- Occupancy from the trimmed slice is **push-into cells the farmer never
  stood on**, not stand-on stasis. Horse-barn no-go:
  `(16,18)/(17,18)/(18,19)/(19,18)` plus leftover rams `(18,20)/(18,21)`.
  Cow-barn body is **x=30 y=18–21**, not the dirt column x=31 the tape
  stood on. `(16,20)` and `(17,21)` stay open. Treating stand-on stasis
  as no-go boxed `(16,20)` against `0xD8` and blocked the human leave.
- North-of-barn leftover dumps at `(46,16)` face-up into `0xFA`
  (`east_spur_fa`). Tape lifted at `(43,12)`/`(43,11)` held=13 and tossed
  there (held 13→0 twice). Do not haul that cluster around the cow barn
  to F0. Live pin path from `(17,20)`: 59 adjacent tiles, leave
  `(17,21)`, ends at FA, no sprite cells. Units: `test_map_config`,
  `test_carry_toss`, `test_crop_skills` (58 passed).
- Pin still 45 stones at 18:13, stamina 76, axe+hoe. Remaining samples
  are the far-east north cluster (x≈40–59, y≈3–8) plus `(19,7)`. AFTER
  45→32 / 200001f still stand. Do not spend a third 200k.
- `d2_leftover_probe --headed` is watch-only. Human inspect is
  `harvest.runtime.harvest_bot play` (`A+S+TAB` / `L+R+SELECT`, `P`
  session no-go, `F5` save).

### Exact next action

Leftover stones from `Y1_D2_Stones_Frontier` toward the NE cluster, dump
at `(46,16)` `0xFA`. Optional headed watch under 8k to see takeoff leave
south. Then a stones window well under 200k, or a human `play --record
leftover_stones_ne` from that pin. Do not treat pond-lip / A-lift stasis
from this tape as farm no-go.

```bash
uv run python -m harvest.runtime.harvest_bot play \
  --state Y1_D2_Stones_Frontier --no-day-plan --record leftover_stones_ne
```

### Non-claims

- No STATUS promotion
- No natural power-on Day 2 completion
- No claim that remaining stones are gone
- No live leftover-stone proof after the FA dump patch
- No third 200k
