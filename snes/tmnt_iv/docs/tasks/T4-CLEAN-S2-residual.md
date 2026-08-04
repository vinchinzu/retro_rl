# Residual — T4-CLEAN-S2

## Outcome
RED (suite still open — **2/4**)

## Suite (2026-08-03, regenerated `clean_suite.json` this session)

| Entry | Outcome | Frames | Damage | Min HP | Pizza | Max hit |
|-------|---------|--------|--------|--------|-------|---------|
| `Stage2` | **life_loss** | 5,825 | 96 | 20 | 1 | 24 |
| `Stage2_Clear_w17_cam27882` | stage_advance | 4,592 | 40 | 40 | 1 | 12 |
| `Boss2` | stage_advance | 3,502 | 28 | 36 | 0 | 8 |
| Stage1_Clear bridge | **timeout** | 28,604 | 40 | 40 | 0 | 12 |

Suite `passed=2/4`, `all_passed=false`. Stage2 still life_loss (**max_hit 24**, the
24-dmg pack hit at `player_x=199`, progress ~21,611, remains); Stage1_Clear bridge
**timeout** (soft-lock, damage 40, lives 1->1). Metalhead + pre-boss (w17, Boss2)
remain Clean stage_advance.

## Kept landings

| Knob | Effect |
|------|--------|
| `_ALLEY_MIN_RANGE` 8→0 | **KEEP** — eliminates `space_left` retreats during point-blank Foot attacks; Stage2 damage 104→96; bridge damage 134→92 |

## Rejected knobs (do not re-try without new theory)

| Knob | Result |
|------|--------|
| `_ALLEY_EDGE_WAIT` 3f (progress 20800–21500, x∈[160,180], stop walk/approach) | **NO-OP** — byte-identical to baseline (7,239f / 96 dmg / min 4 / max_hit 24). Trajectory never passes x∈[160,180] with a walk/approach reason during the band: the turtle is already at x=202 at the 24-dmg hit (prog≈20,995), so the micro-pause window is unreachable |
| `_ALLEY_MIN_RANGE` 8→20 | **worse** 7,310f / 130 dmg |
| `_ALLEY_MIN_RANGE` 8→4 | **worse** 7,938f / 112 dmg (worse than 0) |
| `_ALLEY_STANDOFF` 36→48 / 28 / 44 | **no-op** identical 8,391 / 104 (or 7,239 / 96 with min_range=0) |
| `AlleycatPackSpace` LEFT when ≥2 near (adx≤48, HP≤56) | **worse** 5,349f / 88 (desync death) |
| `PizzaSeek` mid-wave survival grab (HP≤24, dist≤100) | **no-op** identical 8,391 / 104 (no pizza post-f4396) |
| `_alley_pack_fight_action` UP/DOWN shift on `0x5E` (`adx≤64`, `dy<12`) | **worse** 5,304f / 104 (earlier wave desync death) |
| `0x5E` jump-slash `B+Y` on `adx≤56` | **worse** 6,593f / 110 dmg (earlier desync death) |
| `_ALLEY_ATTACK_GAP` 2→1 | **worse** 6,947f / 130 dmg |
| `_ALLEY_ATTACK_GAP` 2→3 | **worse** 7,529f / 120 dmg |
| `_ALLEY_Y_TOLERANCE` 6→12 / 8 | **worse** 5,776f / 114 dmg (Boss2 dmg 38→94) |
| `_ALLEY_ATTACK_HOLD` 1→2 | **worse** 4,706f / 76 dmg (skipped pizza) |
| `_ALLEY_ATTACK_RANGE` 65→75 / 60 | **worse** 5,626f / 78 (skipped pizza) or 4,304f / 124 |

Playbook bans still hold: no mid-wave far pizza chase, no pack jump-hop thrash,
no elev≥44 generic jump on Alleycat.

## Next card ID
`T4-CLEAN-S2-REACH` (EDGE rejected as no-op; metric win on full Stage2)

Thin ladder: [CLEAN_LADDER.md](CLEAN_LADDER.md). Epic `T4-CLEAN-S2` is not
an executor ticket. BOSS+LATE already done; do not claim SUITE until CKPT+BRIDGE.

**Note (2026-08-03):** Executor session restored the **proven KEEP baseline** —
reverted unproven churn (`_ALLEY_Y_TOLERANCE` 6→18 and the `align_up/align_down→
attack` branch) leaving only documented KEEP `_ALLEY_MIN_RANGE = 0`. Verified
Stage2 baseline matches residual: **7,239f / 96 dmg / min 4 / max_hit 24**. Then
ran `T4-CLEAN-S2-EDGE` (micro-pause) — **NO-OP, REJECTED** (see table). Suite
of-record remains **2/4** (`clean_stage2.json` = clean Stage2 entry).

## One change (next)
**Full Stage2 REACH** ([T4-CLEAN-S2-REACH](T4-CLEAN-S2-REACH.md)) — no new
Alleycat knob; re-measure Stage2 and seek a REACH metric win (lower `max_hit`
below 24, lower `damage_taken` <96, or higher `frames`/`min_hp`). EDGE micro-pause
is dead (trajectory never enters x∈[160,180] with a walk/approach reason at
prog 20800–21500). Any future Alleycat knob must target the actual hit location
(prog ≈20,995, x≈202) or a different primitive.

**Note:** Suite-of-record is **2/4** (regenerated 2026-08-03; `stage1_clear`
soft-lock timeout, `Stage2` life_loss) — see Suite table above.

## Evidence
- `recordings/stage2_clean_track/clean_suite.json`
- `recordings/stage2_clean_track/clean_stage2.json` (2026-08-03)

