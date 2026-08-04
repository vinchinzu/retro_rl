# Status — DuckTales (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M1 |
| Best verified result | Controllable first playable frame (title → land select controllable) |
| Last verification | 2026-07-27 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **boot verified** |
| Integration | `DuckTales-Nes` |
| ROM zip | `roms/Nintendo/NES/Duck Tales.zip` |
| Ready frame (probe) | ~298 |
| Checkpoint | `Level1.state` |
| Evidence | [boot_level1.png](../recordings/boot_level1.png) |

## Done

- Directory layout and NES integration stubs
- `scripts/setup_rom.py` wiring via `retro_harness.env` (`.nes`)
- Deterministic reset → first controllable play (`scripts/boot_probe.py`)
- Early readiness RAM + unit tests

## Not done

- Broader instrumentation (M2)
- Segment policies / behavior tree
- Continuous multi-segment or full-game runs

## Next

enter a land stage and clear first segment.
