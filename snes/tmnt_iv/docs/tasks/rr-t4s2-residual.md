# Residual — rr-t4s2 Alleycat Clean 0x5E pack

**Status:** REACH improved; CKPT / BRIDGE / suite still RED. Do not STATUS.
BOSS + LATE were already green — re-check them after this pack tactic
before claiming 3/4.

## Stage2 heal=none (dirty tree + AlleycatPackTactics)

| Policy | Outcome | Frames | Min HP | Max hit | Notes |
|--------|---------|--------|--------|---------|-------|
| before pack tactic | life_loss | ~4456 | 2 | 12 | died on 0x68/0x76; never saw 0x5E |
| pack tactic KEEP | life_loss | ~5405 | 8 | 24 | **REACH** 0x5E at HP 80; first kick 24→12 |
| suite-of-record (2026-08-02) | life_loss | 5825 | 20 | 24 | pizza then 0x5E pile at ~21.6k |

Pizza-only. No emergency HP.

## What shipped

- `snes/tmnt_iv/tactics/alleycat.py` — stage-1 pack overrides (no B, no A)
- Hooked in `Stage1Policy.tick` after `PizzaSeek`
- Tests: `snes/tmnt_iv/tests/test_alleycat_tactics.py`

KEEP geometry:

- Overlap plant-poke (`min_range` 0). LEFT-forever on overlap freezes wave 1.
- 0x5E jump-kick from the right (adx≤32) → LEFT (24-dmg at x=199 / 226)
- 0x68 jumper: poke instead of `align_up`
- 0x76: plant, do not walk into grab
- Dense Y only on **0x5E** sandwich/overlap. Dense Y on 0x60 sandwich
  froze opening wave at x=113.

## Burned this sitting (do not re-open)

- Dense Y on all sandwiches
- Wall-only releft (stuck mashing Y at x=113)
- Hold-and-poke at adx≤80 vs left 0x5E clump (24-dmg came back earlier)
- LEFT+Y close into left 0x5E clump (two 24s)
- Pack jump-hop / 0x5E jump-slash / global min_range (older playbook)

## Exact next action

Residual 24-dmg while `alley_releft` into a **left** 0x5E clump
(~progress 21423 / player_x=164 / enemies at x≈69–96). Need a way off
the right shoulder that is not walk-through, not 80px hold, not LEFT+Y.
Then CKPT `stage_advance`, BRIDGE, suite. Pizza-only. Do not clobber
`tmnt_iv_full_hard_*`.
