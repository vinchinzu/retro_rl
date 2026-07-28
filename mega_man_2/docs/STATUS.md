# Status — Mega Man 2 (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M1 |
| Best verified result | Controllable first playable frame (title/stage select → Air Man stage playable) |
| Last verification | 2026-07-27 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **boot verified** |
| Integration | `MegaMan2-Nes` |
| ROM zip | `roms/Nintendo/NES/Mega Man II.zip` |
| Ready frame (probe) | ~1200 |
| Checkpoint | `Level1.state` |
| Evidence | [boot_level1.png](../recordings/boot_level1.png) |

## Done

- Directory layout and NES integration stubs
- `scripts/setup_rom.py` wiring via `snes_oneshot.rom_setup` (`.nes`)
- Deterministic reset → first controllable play (`scripts/boot_probe.py`)
- Early readiness RAM + unit tests

## Not done

- Broader instrumentation (M2)
- Segment policies / behavior tree
- Continuous multi-segment or full-game runs

## Next

first Robot Master stage segment clear.
