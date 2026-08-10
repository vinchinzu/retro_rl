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

**Not cleared.** No camera ≥ 5; no boss door. M3 screen-4 from AirScreen2 still GREEN.

### Geometry (verified overnight)

| Observation | Detail |
|-------------|--------|
| Baseline death | AirScreen2 + late 40/16 → die f≈519, prog≈1047, HP16, fallen |
| AirScreen4.state | Mid-air (feet=0, sy≈89); freefall death ~17f — **do not start here** |
| Last solid land | f≈437, scr=3, prog≈949, sx≈53, sy=84, HP16 → `AirFanPlatform` |
| Platform extent | **Grounded prog 937–984** (left fall walk~14; right fall walk~33, sx~41–88) |
| Pink head | **Goblin / Air Tikki** obj slot14 type36 @~(39,49) — **not** updraft fan |
| Goblin top land | Dense hop/wait grids (spike-cycle waits 0–200+, both sides): **0** feet=1 in prog (906,936) or sy&lt;82 |
| Left of Goblin | `AirLeftPlatform` short ledge prog **902–905** only (~9f walk right then air) |
| “Ladder” bar | Never `tile_feet==2` on this path |
| Right of platform | Type35 eggs @y~84; Pipi bounce min_sy≈23–26 with damage |
| Best press | Shoot+bird-boost ~prog **1086** scr4 min_sy~23, still pit (pure jump ~1064–1072) |
| False saves | Pruned `AirGoblinHead*` / `AirPastFan*` (were left ledge or same-platform) |

### Swept (2026-08-09 overnight)

- Platform edge map; strict land criteria (prog≤935 goblin / prog≥988 past)
- Goblin 5px + spike-cycle waits from fan + left; long hop only reaches left ledge
- Pipi/edge waits; shoot-then-jump; adaptive post-bounce steer
- No camera ≥ 5; no new grounded s4/s5 checkpoint

## Not done

- Past screen 4 / boss door (true Goblin solid window still unfound; Lightning Lord not reached)
- Full Robot Master stage clear (Air Man boss door / fight)
- Natural-entry M4 from power-on through screen-2+
- Stage select other masters / weapon routing

## Next

1. Re-check Goblin solidity (animation RAM / RTA 5px setup may need different approach X).
2. Find Lightning Lord spawn / cloud ride past this island (may need earlier route fork).
3. Controlled Pipi bounce → solid cloud if one exists off current trajectories.
4. Only after grounded s4/s5: freeze recipe → AirScreen2→target 5 (3/3) → boss door.
