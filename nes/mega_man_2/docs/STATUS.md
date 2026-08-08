# Status — Mega Man 2 (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M3 (isolated segment) |
| Best verified result | Air Man first screen clear from `Level1` (camera X screen ≥ 1; 3/3) |
| Last verification | 2026-08-08 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **isolated Air Man screen-1 clear** |
| Integration | `MegaMan2-Nes` |
| ROM zip | `roms/Nintendo/NES/Mega Man II.zip` |
| Ready frame (probe) | ~1204 |
| Checkpoints | `Level1.state` (Air Man playable), `AirScreen1.state` |
| Policy | `AirScreen1Policy` (RIGHT + jump 50/12 + shoot pulse) |
| Evidence | [air_screen1.png](../recordings/air_screen1.png), [air_segment/](../recordings/air_segment/) |

## Done

- Directory layout and NES integration stubs
- `scripts/setup_rom.py` wiring via `retro_harness.env` (`.nes`)
- Deterministic reset → first controllable play (`scripts/boot_probe.py`)
- M2 RAM: camera X/screen, player X/Y, health/lives, tile feet, invuln, weapons, boss HP
- **M3 segment:** `scripts/run_air_segment.py` camera screen ≥ 1, 3/3 deterministic (~248f, HP 26)

## Segment metrics (Level1 → camera screen ≥ 1)

| Metric | Value |
|--------|------:|
| Frames | 248 |
| Final HP | 26 (start 28) |
| Camera screen | 1 |
| Trials | 3/3 |

## Not done

- Full Robot Master stage clear (Air Man boss)
- Natural-entry M4 from power-on through screen-1
- Stage select other masters / weapon routing

## Next

1. Extend Air Man past screen 1 (gaps / fans) toward boss door.
2. Natural-entry: power-on → Level1 pose → screen-1 without loading `Level1`.
