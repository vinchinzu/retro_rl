# Status — Mega Man 2 (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M3 (isolated segment) |
| Best verified result | Air Man camera screen ≥ 2 from `Level1` (3/3) |
| Last verification | 2026-08-08 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **isolated Air Man screen-2 clear** |
| Integration | `MegaMan2-Nes` |
| ROM zip | `roms/Nintendo/NES/Mega Man II.zip` |
| Ready frame (probe) | ~1204 |
| Checkpoints | `Level1`, `AirLanded` (grounded scr1), `AirScreen2` |
| Policy | `AirManPolicy` (early 50/12 → land jump → mid 50/12 → gap@142) |
| Evidence | [air_segment/](../recordings/air_segment/), [air_landed.png](../recordings/air_landed.png) |

## Done

- Directory layout and NES integration stubs
- Deterministic reset → first controllable play (`scripts/boot_probe.py`)
- M2 RAM: camera X/screen, player X/Y, health/lives, tile feet, invuln, weapons, boss HP
- **M3 screen-1:** camera ≥ 1 (~248f) via `AirScreen1Policy`
- **M3 mid-stage:** camera ≥ 2 from `Level1` (~522f, HP 22, 3/3) and from `AirLanded` (~226f, 3/3)

## Segment metrics

### Level1 → camera screen ≥ 2

| Metric | Value |
|--------|------:|
| Frames | 522 |
| Final HP | 22 (start 28) |
| Camera screen | 2 |
| Progress X | 513 |
| Trials | 3/3 |

### AirLanded → camera screen ≥ 2

| Metric | Value |
|--------|------:|
| Frames | 226 |
| Final HP | 22 |
| Camera screen | 2 |
| Trials | 3/3 |

## Not done

- Full Robot Master stage clear (Air Man boss)
- Natural-entry M4 from power-on through screen-2
- Stage select other masters / weapon routing

## Next

1. Extend past screen 2 (fans / longer gaps) toward boss door.
2. Natural-entry: power-on → screen-2 without loading `Level1`.
