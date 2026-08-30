# Residual — rr-t4s2 Alleycat Clean 0x5E pack

**Status:** suite **2/4** (LATE+Boss2 `stage_advance`). Stage2 and
stage1_clear still KO. Do not STATUS.

## Scratch `probe_clean --stage 2 --suite` (2026-08-30)

| Entry | Outcome | Frames | Min HP | Max hit | Notes |
|-------|---------|--------|--------|---------|-------|
| Stage2 | ko | 5035 | 8 | 24 | 24@x=193, 12@175, 24@179, 12@164 |
| LATE | stage_advance | 4234 | 37 | 12 | fade HP 37→80 is the game, not pizza |
| Boss2 | stage_advance | 4093 | 23 | 12 | same fade restore 79→80 |
| stage1_clear | ko | 4532 | 8 | 24 | pizza then 24@x=176 and 24@x=153 |

Pizza-only. No emergency HP. Last-life LATE/Boss2 (`lives=0`) are not
the Clean gate.

## What shipped

- `snes/tmnt_iv/tactics/alleycat.py` — stage-1 pack overrides (no B, no A)
- Hooked in `Stage1Policy.tick` after `PizzaSeek`
- Tests: `snes/tmnt_iv/tests/test_tactics.py` (pack right-exit + kicker)

KEEP geometry:

- Overlap plant-poke (`min_range` 0). LEFT-forever on overlap freezes wave 1.
- 0x5E jump-kick from the right (adx≤32) → LEFT (24-dmg at x=199 / 226)
- 0x68 jumper: poke instead of `align_up`
- 0x76: plant, do not walk into grab
- Dense Y only on **0x5E** sandwich/overlap. Dense Y on 0x60 sandwich
  froze opening wave at x=113.

## Burned (do not re-open)

- Dense Y on all sandwiches
- Wall-only releft (stuck mashing Y at x=113)
- Hold-and-poke at adx≤80 vs left 0x5E clump (24-dmg came back earlier)
- LEFT+Y close into left 0x5E clump (two 24s)
- Pack jump-hop / 0x5E jump-slash / global min_range (older playbook)
- TTC / hy≥180 sewer hop (never reached Rat King)
- Treating 0x0B fade HP restore as unlabeled pizza

## Exact next action

Stage2 still dies on 24s at x≈193 then x=164. `alley_right_exit` is
necessary but not sufficient. Wave dumpster DOWN stays; skip dumpster
on Metalhead and on 0x0B fade. Pizza-only. Do not clobber
`tmnt_iv_full_hard_*`.
