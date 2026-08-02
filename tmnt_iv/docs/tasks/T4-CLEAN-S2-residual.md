# Residual — T4-CLEAN-S2

## Outcome
RED (suite still open)

## Suite (2026-08-01, production policy)

| Entry | Outcome | Frames | Damage | Min HP | Pizza |
|-------|---------|--------|--------|--------|-------|
| `Stage2` | **life_loss** | 8,391 | 104 | 8 | 1 |
| `Stage2_Clear_w17_cam27882` | stage_advance | 4,331 | 29 | 51 | 1 |
| `Boss2` | stage_advance | 3,826 | 38 | 26 | 0 |
| Stage1_Clear bridge | **life_loss** | 7,721 | 134 | 10 | 1 |

Metalhead + pre-boss remain Clean. Early/mid Foot packs remain the gate.

## Failure window (Stage2 checkpoint)

Post-pizza pile-ons after the single pizza at f4396 (HP 48→80, x≈129):

| Frame | Hit | HP after | player_x | progress |
|-------|-----|----------|----------|----------|
| 4634 | **24** | 56 | 204 | 21126 |
| 4790 | 12 | 44 | 162 | 21282 |
| 4972 | **24** | 20 | 179 | 21464 |
| 5363–6677 | 4+4+4 | →8 | ~128–151 | 21855–23169 |

Then life_loss before Metalhead. Max hit 24 = 0x5E pack trade.

## Rejected knobs (do not re-try without new theory)

| Knob | Result |
|------|--------|
| `_ALLEY_MIN_RANGE` 8→20 | **worse** 7,310f / 130 dmg |
| `_ALLEY_STANDOFF` 36→48 | **no-op** identical 8,391 / 104 |
| `AlleycatPackSpace` LEFT when ≥2 near (adx≤48, HP≤56) | **worse** 5,349f / 88 (desync death) |

Playbook bans still hold: no mid-wave far pizza chase, no pack jump-hop thrash,
no elev≥44 generic jump on Alleycat.

## Next card ID
`T4-CLEAN-S2-PACK` (or continue S2)

## One change (next)
**Critical mid-wave pizza seek scoped only for Alleycat:** when `stage==1`,
`HP ≤ 24`, nearest pizza `dist ≤ 100`, allow seek **even with living enemies**
(survival grab only — not full mid-wave chase). A/B vs baseline 8391/104;
if emergency dry-run Stage2 damage rises >20% reject.

Rationale: chip-spacing knobs desync wave timing; a second pizza after the
post-pizza 24+24 dump is the STATUS-suggested alternate path ("second safe
pizza without mid-wave desync"). Keep underfoot always; do **not** reopen
global mid-wave far seek.

## Evidence
- `recordings/stage2_clean_track/clean_suite.json`
- `recordings/stage2_clean_track/clean_stage2.json`
