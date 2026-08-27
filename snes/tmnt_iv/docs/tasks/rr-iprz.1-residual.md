# rr-iprz.1 residual — Slash jump-over behind-combo (Raph Hard)

Pin KEEP. Do **not** STATUS / BASELINE from this probe. Park for a
full continuous dry-run (`record_full_hard_run --dry-run`) because Slash
path changes later-stage RNG (spin-40 lesson).

## Probe (`RaphFullHardBoss5`, heal=emergency, `--max-frames 40000 --stop-stage-gt 4`)

| Run | Outcome | Frames | Damage | Heals | Boss HP |
|-----|---------|-------:|-------:|------:|---------|
| Production | — | 11,386 | 478 | 6 | 160→0 |
| Probe 1 | stage_advance | **9,595** | **435** | 6 | 160→0 |
| Probe 2 | stage_advance | **9,595** | **435** | 6 | 160→0 |

Both CLEAR. Frames **and** damage beat production. Heals tied at 6.
`slash_spin_dodge_adx` stayed **52** (did not port 40). Never A.

## What landed

- Extracted `SlashTactics` → `tmnt_iv/tactics/slash.py` (re-export from
  `policy.py`). `policy.py` 2026 → 1889 LOC.
- Raphael (char 8): B-only jump-over at mid adx, grounded combo when
  behind, hop away after the string, space instead of walking into his
  body, B+Y only to meet an **elevated** Slash. Spin/claw/punish
  overrides unchanged.
- Leo: hybrid whiplash unchanged.

## Rejected mid-sitting (worse than 11,386 / 478)

Same-Y jump-kick (B+Y while crossing) + aggressive hop/bait:

| Variant | Frames | Damage | Heals |
|---------|-------:|-------:|------:|
| Jump-kick + bait dwell | 21,440 | 1,270 | 18 |
| Jump-kick, shorter hop | 19,297 | 1,175 | 17 |
| Jump-kick, combo toward live dx | 27,818 | 1,673 | 24 |
| Control (legacy thrash on Raph) | 11,386 | 478 | 6 |

Mid-air Y on the same-Y cross is the regression; KEEP uses B-only over.

## Follow-up sitting 2026-08-27 — REJECT (do not reopen)

Four isolated algorithms + KEEP trace + three parent patches. None
beat **9,595f / 435 / 6** on `RaphFullHardBoss5`. Production
`tactics/slash.py` restored to this KEEP. spin stays 52.

| Algorithm | Emergency | vs KEEP |
|-----------|-----------|---------|
| `vuln_reactive` status FSM | 10,260f / **342** / 5 CLEAR | −93 dmg, **+665f** (lab only) |
| Bait-jump (honest) | timeout or 15.9k+ | reject |
| Kick-punish opener | 29,910f / 1,668 CLEAR | reject (`player_y` is lane) |
| Claw-mash ablation | timeout, Slash HP 40 | reject |
| Parent claw-80 + `0xB6` | 13,141f / 706 | reject |
| Parent Y-while-walking-in | 15,157f / 715 | reject |
| Parent re-glue combo | 17,034f / 1,113 | reject |

KEEP first-connect is frame **910**; last 1 HP costs ~2,024f. Jump
height is entity `+0x10` / pose `+0x16`, not `player_y`. `0xB6` is
29% of chip **and** the walk cycle.

## Next

`rr-iprz.5` Starbase stall (continuous +5:11). Do not port spin-40.
Do not STATUS from this pin.
