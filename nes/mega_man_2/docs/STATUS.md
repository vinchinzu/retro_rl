# Status — Mega Man 2 (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M3 (isolated segment) |
| Best verified result | Air Man camera screen ≥ 4 from `AirScreen2` (3/3) |
| Last verification | 2026-08-10 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **isolated Air Man screen-4 clear (from AirScreen2); post-s4 blocked (~296px pit)** |
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
- **Post-s4 probe (rr-54ui, open):** type36 not solid; ~296px pit; no camera ≥5 / boss door yet

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

## Post-s4 probe (2026-08-09/10, rr-54ui)

**Not cleared.** No camera ≥ 5; no boss door. M3 screen-4 from AirScreen2 still GREEN.

### Geometry (verified)

| Observation | Detail |
|-------------|--------|
| Baseline death | AirScreen2 + late 40/16 → die f≈519, prog≈1047, HP16, fallen |
| AirScreen4.state | Mid-air (feet=0, sy≈89); freefall death ~17f — **do not start here** |
| Last solid land | f≈437, scr=3, prog≈949, sx≈53, sy=84, HP16 → `AirFanPlatform` |
| Platform extent | **Grounded prog 937–984** (left fall walk~14; right fall walk~33, sx~41–88) |
| Solids are tiles | `tile_feet`/`tile_center==1` on all grounded poses; **not** object collision |
| Pink head type36 | Damage enemy (slot14). Attack cycle ~111f: when inv=0, teleports to player and hits (−2 HP). **Not landable** |
| "On goblin" at s2 | Visual only — Mega Man stands on **tile** platform at y=52 (`tile_c=1`); can walk far past type36 x while feet=1 |
| Goblin top land | 1000+ phase/top-down hops both sides: **0** feet=1 in prog (906,936) or sy<82 |
| Left of Goblin | `AirLeftPlatform` short ledge prog **902–905**; further left returns prior y84 chain (~865) |
| "Ladder" bar | Never `tile_feet==2` (UP/UP+dir grids) |
| Wind / cam Y | Walk speed ~1px/f same as s2; `camera_y` always 0 through death |
| Right of platform | Type35 eggs/birds; freefall tile sample past 984: **0** solid; no new types in 400–600f camp |
| Jump envelope | Pure RIGHT max prog **~1065–1071** scr4 min_sy~34 (edge walk~34–38 + jh≥12) |
| Best press | Shoot+Pipi boost still ~prog **1086** min_sy~23 (prior); damage-boost grids ≤1065 |
| Gap math | Screen5 @ prog 1280; last solid 984 → **~296px** open. One jump covers ~75–90px only |
| False saves | Pruned `AirGoblin*`, `AirPast*`, `AirHigh*`, `AirFurtherLeft*` probe states |

### Map-match (2026-08-10 night4)

| Progress | Visual / objects | Wiki section guess |
|----------|------------------|--------------------|
| 0–200 | Long platforms, type2 early, then Pipi35 | Stage start / pre-goblin |
| 514 AirScreen2 | y52 tile + type36 goblin + Pipi | Goblin chain (A / late) |
| 629 | y68 short stripe platform between goblins | Still goblin hops |
| 689–984 | y84 stripe platforms + goblin + Pipi | **Not** Matasaburo E (no wind, no fan robots) |
| >984 pit | Open sky; objects stay 1/35/36 only | Suspected LL (B) sky — **LL never spawns** |

Object types seen Level1→death hybrid: **{1, 2, 35, 36}** only (2 early-only). No Kaminari Goro / cloud type.

### Swept (2026-08-09 + 08-10 overnight ×2)

- Platform edge map; Goblin 5px + phase-cycle + top-down phase hops; ladder UP grids
- Pipi/edge waits; shoot-spam; damage-boost; high-path period variants from AirScreen2
- Right-edge camp 400f + idle 600f spawn watch (types stay 1/35/36 only)
- **Night4 (LL fork tip):** high forks before y84 descent; descent interrupts;
  AirFan gap micro-hops (186, 0 land >984); edge void shoot 500f; slow late waits;
  Level1 hybrid→screen2 death ~1029; WRAM novelty after prog800 — **no new 0x400 types**
- No camera ≥ 5; no new grounded s4/s5 checkpoint
- Evidence: `recordings/air_post4_night3/`, `recordings/air_post4_night4/` (+ RED_PIN.txt)

## Not done

- Past screen 4 / boss door (**intermediate solid missing** for ~296px gap; Lightning Lord cloud **never observed** under Clean play)
- Full Robot Master stage clear (Air Man boss door / fight)
- Natural-entry M4 from power-on through screen-2+
- Stage select other masters / weapon routing

## Next

1. **ROM / TAS spawn path** — Air Man stage enemy placement data or human TAS compare
   at prog≥1000 (why LL/cloud never enter object table 0x400).
2. Nametable/tile platforms past 984 that may load only under specific scroll/camera
   state (not more pure-RIGHT / goblin-solid grids).
3. Only after grounded s4/s5: freeze recipe → AirScreen2→target 5 (3/3) → boss door.
