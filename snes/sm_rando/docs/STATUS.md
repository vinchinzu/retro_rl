# SM Rando — Status

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M1 |
| Best verified result | Power-on → FirstPlay (Ceres elevator, game_state 8) on vanilla SM ROM; `./play` + record |
| Last verification | 2026-08-06 |
| Runtime class | Bronze |
| Intervention class | Clean |

## Role

**Single-game Super Metroid randomizer** — simpler solver ground than SMZ3.
Build item-logic + room skill hooks + seed-robust spine here, then extend to
combined SMZ3. Vanilla skills live in `snes/super_metroid/` (do not fork).

Program stack: `docs/SOLVER_ARCHITECTURE.md`.

## Checklist

| Item | State |
|------|--------|
| Package `sm_rando/` | scaffolded |
| Seed package schema | done (offline fixture + `seeds/demo_seed/`) |
| Early logic graph | done (coarse, planned edges) |
| Integration ROM (vanilla SM) | done (`setup_rom` → SMRando-Snes/rom.sfc) |
| Boot → FirstPlay | done (`make_boot`, Ceres `0xDF45`) |
| Play/record spine | done (`./play`, record default, F5 → package integration) |
| Patched seed ROM / generator | open |
| Multi-seed S/T dry-run | open |

## Next

1. Wire real rando generator or IPS patch into seed packages.
2. Map early graph edges to vanilla pure skills; promote verification.
3. Multi-seed report via shared seed-robust harness (`rr-gbd.11`).
