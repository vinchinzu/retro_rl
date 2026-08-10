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
| Status | **isolated Air Man screen-4 clear (from AirScreen2); post-s4 blocked** |
| Integration | `MegaMan2-Nes` |
| ROM zip | `roms/Nintendo/NES/Mega Man II.zip` |
| Ready frame (probe) | ~1204 |
| Checkpoints | `Level1`, `AirLanded` (grounded scr1), `AirScreen2`, `AirScreen3`/`AirScreen4` (mid-air), `AirFanPlatform` (grounded scr3 prog~949), `AirLeftPlatform` (grounded scr3 prog~902 left of Goblin) |
| Policy | `AirManPolicy` (Level1/landed mid-stage; `start=screen2` late: 45/16 → fan 145–180 → late 40/16) |
| Evidence | [air_segment/](../recordings/air_segment/), [air_fan_probe/](../recordings/air_fan_probe/), [air_boost/](../recordings/air_boost/) |

## Done

- Directory layout and NES integration stubs
- Deterministic reset → first controllable play (`scripts/boot_probe.py`)
- M2 RAM: camera X/screen, player X/Y, health/lives, tile feet, invuln, weapons, boss HP
- **M3 screen-1:** camera ≥ 1 (~248f) via `AirScreen1Policy`
- **M3 mid-stage:** camera ≥ 2 from `Level1` (~522f, HP 22, 3/3) and from `AirLanded` (~226f, 3/3)
- **M3 late-stage (fans/gaps):** camera ≥ 3 and ≥ 4 from `AirScreen2` (3/3 each; 2026-08-09)
- **Post-s4 probe (rr-54ui, open):** geometry + checkpoints; no camera ≥5 / boss door yet

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

## Post-s4 probe (2026-08-09, rr-54ui)

**Not cleared.** Camera stays at 4; pure late RIGHT dies ~prog 1047–1073. Bird-boost press ~prog 1085–1086 still screen 4, airborne.

### Geometry (corrected)

| Observation | Detail |
|-------------|--------|
| Baseline death | AirScreen2 + late 40/16 → die f≈519, prog≈1047, HP16, fallen |
| AirScreen4.state | Mid-air (feet=0, sy≈89); freefall death ~17f — **do not start here** |
| Last solid land | f≈437, scr=3, prog≈949, sx≈53, sy=84, HP16 → `AirFanPlatform` |
| Pink head | **Goblin / Air Tikki** (standable head when spikes down / 5px corner), **not** an updraft fan |
| Left of Goblin | Short platform → `AirLeftPlatform` (prog~902, sx~6); walk left continues to pit ~prog 792 |
| “Ladder” bar | Visible left of Goblin; `tile_feet` never becomes 2 in Air Man path (not usable ladder state) |
| Right of platform | Pipi / small cloud; collision bounce reaches min_sy≈23–26 with HP damage |
| Jump height | A rising edge required; hold≥12 apex sy34; continuous A from load does not jump |
| Best press | Bird-boost from right edge ~prog **1085–1086** scr4, still pit death |

### Swept (2026-08-09 night)

- Pure RIGHT period/hold from `AirFanPlatform` / AirScreen2 continue
- Left cross over Goblin → left platform (stable); ladder seek (no feet=2)
- Dense jump grids onto Goblin head / small clouds (0 elevated lands)
- Edge wait for spawn; shoot-then-jump; bird-boost timing variants
- No camera ≥ 5; no grounded land with sy&lt;82 past `AirFanPlatform`

## Not done

- Past screen 4 / boss door (Goblin head pixel land, Lightning Lord cloud ride, or bird-boost land)
- Full Robot Master stage clear (Air Man boss door / fight)
- Natural-entry M4 from power-on through screen-2+
- Stage select other masters / weapon routing

## Next

1. Pixel-precise Goblin head land (5px toe / spikes-down window) or kill+ride Lightning Lord cloud.
2. Controlled Pipi bounce → solid cloud land; save grounded s4/s5 checkpoint.
3. Freeze frame recipe → AirScreen2→target 5 (3/3); then boss door.
4. Natural-entry: power-on → screen-2+ without loading `Level1`.
