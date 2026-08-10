# Status — Mega Man 2 (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M3 (isolated segment) |
| Best verified result | Air Man camera screen ≥ 4 from `AirScreen2` (3/3) |
| Last verification | 2026-08-09 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **isolated Air Man screen-4 clear (from AirScreen2)** |
| Integration | `MegaMan2-Nes` |
| ROM zip | `roms/Nintendo/NES/Mega Man II.zip` |
| Ready frame (probe) | ~1204 |
| Checkpoints | `Level1`, `AirLanded` (grounded scr1), `AirScreen2`, `AirScreen3`, `AirScreen4` |
| Policy | `AirManPolicy` (Level1/landed mid-stage; `start=screen2` late: 45/16 → fan 145–180 → late 40/16) |
| Evidence | [air_segment/](../recordings/air_segment/), [air_landed.png](../recordings/air_landed.png) |

## Done

- Directory layout and NES integration stubs
- Deterministic reset → first controllable play (`scripts/boot_probe.py`)
- M2 RAM: camera X/screen, player X/Y, health/lives, tile feet, invuln, weapons, boss HP
- **M3 screen-1:** camera ≥ 1 (~248f) via `AirScreen1Policy`
- **M3 mid-stage:** camera ≥ 2 from `Level1` (~522f, HP 22, 3/3) and from `AirLanded` (~226f, 3/3)
- **M3 late-stage (fans/gaps):** camera ≥ 3 and ≥ 4 from `AirScreen2` (3/3 each; 2026-08-09)

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

### AirScreen2 → camera screen ≥ 3

| Metric | Value |
|--------|------:|
| Frames | 241 |
| Final HP | 20 (start 22) |
| Camera screen | 3 |
| Progress X | 768 |
| Trials | 3/3 |

### AirScreen2 → camera screen ≥ 4

| Metric | Value |
|--------|------:|
| Frames | 502 |
| Final HP | 16 |
| Camera screen | 4 |
| Progress X | 1024 |
| Trials | 3/3 |

## Not done

- Full Robot Master stage clear (Air Man boss door / fight)
- Natural-entry M4 from power-on through screen-2+
- Stage select other masters / weapon routing
- Past screen 4 (still dies ~prog 1047 on open-sky platforms)

## Next

1. Extend past screen 4 toward boss door (more fans / precision platforms).
2. Natural-entry: power-on → screen-2+ without loading `Level1`.
