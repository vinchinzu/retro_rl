# Status — TMNT II (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M3 |
| Best verified result | First Stage 1 wave: score 0→5 from `Level1` (~814f, HP 59) |
| Last verification | 2026-07-27 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **first wave segment clear (M3)** |
| Integration | `TeenageMutantNinjaTurtlesII-Nes` |
| ROM zip | `roms/Nintendo/NES/Teenage Mutant Ninja Turtles II - The Arcade Game.zip` |
| Ready frame (probe) | ~1030 |
| Checkpoint | `Level1.state` |
| Clear state | `Stage1_Clear` / `Stage1_Clear_sc5_hp59` |
| Evidence | [stage1_clear.png](../recordings/stage1_clear.png), [stage1_segment.json](../recordings/stage1_segment/stage1_segment.json) |

## Done

- Directory layout and NES integration stubs
- `scripts/setup_rom.py` wiring via `retro_harness.env` (`.nes`)
- Deterministic reset → first controllable play (`scripts/boot_probe.py`)
- Early readiness RAM + unit tests
- Instrumentation: health, lives, score, OAM player screen X/Y
- **Stage1Policy** open → face-LEFT lock → push (`policy.py`)
- **M3 segment:** `scripts/run_stage1_segment.py` score≥5, 3/3 deterministic

## Segment metrics (Level1 → score≥5)

| Metric | Value |
|--------|------:|
| Frames | 814 |
| Final HP | 59 (start 60) |
| Final score | 5 |
| Trials | 3/3 |

## Not done

- Full M2 enemy/camera map
- Score≥8+ continuous packs / natural-entry M4
- Stage chain or full-game run

## Next

1. Extend policy past score 5 (right-edge unlock + next packs).
2. Map enemy slots / screen-lock flag.
3. Natural-entry segment from boot (M4).
