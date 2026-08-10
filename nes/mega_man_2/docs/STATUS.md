# Status — Mega Man 2 (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M3 (isolated segment) |
| Best verified result | Air Man camera screen ≥ 4 from `AirScreen2` (3/3) |
| Last verification | 2026-08-10 (rr-54ui cloud land probe) |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **isolated Air Man screen-4 clear (from AirScreen2); post-s4 open (rider kill OK; object-solid land residual)** |
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
- **Post-s4 (rr-54ui, open):** LL spawns; **rider kill Clean** (0x3D pulse-B); cloud land still RED; no camera ≥5

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
| >984 pit | Open sky; first LL body at y≈32–36 | **Section B LL sky — LL spawns (rr-fpd6)** |

Object types (corrected 2026-08-10 fpd6): goblin **0x40**, pipi **0x37**, LL **0x3D/0x3E**, plus low IDs/projectiles.
Night3–5 “only 1/2/35/36” was under-read of `$0400`.

### Swept (2026-08-09 + 08-10 overnight ×3)

- Platform edge map; Goblin 5px + phase-cycle + top-down phase hops; ladder UP grids
- Pipi/edge waits; shoot-spam; damage-boost; high-path period variants from AirScreen2
- Right-edge camp 400f + idle 600f spawn watch (types stay 1/35/36 only)
- **Night4 (LL fork tip):** high forks before y84 descent; descent interrupts;
  AirFan gap micro-hops (186, 0 land >984); edge void shoot 500f; slow late waits;
  Level1 hybrid→screen2 death ~1029; WRAM novelty after prog800 — **no new 0x400 types**
- **Night5 (ROM/TAS + nametable tip):**
  - Map-match: prog~950 = pre-LL (A/late); open sky after 984 = expected B/LL start
  - Type36 = indestructible damage Air Tikki (f420 64→128 teleport-hit); stands are
    **tiles** adjacent/under sprite (AirScreen2 sx~130 vs goblin x~119)
  - Freefall collision grid: max feet=1 prog **980**; **0** tile hits prog>984
  - Shoot-camp / edge / policy-camp / WRAM $0400–$06FF: still types {1,35,36} only
  - ROM: property rows with type IDs 0x22–0x25; **no** spawn-list decode yet
  - Smoke AirScreen2→4 still GREEN (502f). Units 10/10
- No camera ≥ 5; no new grounded s4/s5 checkpoint
- Evidence: `recordings/air_post4_night3/`, `night4/`, `night5/` (+ RED_PIN.txt)

## rr-fpd6 decode (2026-08-10) — CLOSED

Lightning Lord spawn **decoded + live-confirmed**:

| Field | Value |
|-------|-------|
| Type IDs | `0x3E` `objects_kaminari_goro` (+ `0x3D` move, `0x3F` bolt) |
| First placement | mapset **4**, x=`0xC0`, y=`0x20` (ROM objects_set) |
| Live spawn | prog **~961** (still scr3 cam_x~193) → slots with t=61/62, obj scr=4 |
| Source | lsmmega/mm2 `airman_wily2_objects_set.asm` + live `ll_spawn_probe.py` |
| Evidence | `docs/LL_SPAWN_DECODE.md`, `probe_*.json`, `summary.json` |

Cloud altitude y≈32–36 matches pure-jump min_sy~34. fpd6 closest Clean approach
was ~**28px short in X**; **rr-54ui night closed X** (Y-meet after kill dx≈5–10)
but still **no stand** — object-solid residual deeper than X gap.

## rr-54ui cloud land (2026-08-10) — OPEN (partial)

| Field | Value |
|-------|-------|
| Rider kill | **Yes** — `0x3D` HP 20→13→6→despawn via pulsed B (period 3–8) in air |
| Body | `0x3E` stays; on kill flash types 6 + 118 |
| Best Y-meet | dx≈5–10, \|dy\|≤4 after kill — still freefall `ft=0` |
| Pitfall | Kill while dy≳20 → player+cloud sink same rate (gap frozen) |
| Cam ≥5 | **No** |
| Evidence | `recordings/air_post4_cloud/RED_PIN.md` + `cloud_land_*.json` / v2–v7 |

## Not done

- Past screen 4 / boss door (**kill OK; Thunder Chariot object-solid stand residual**)
- Full Robot Master stage clear (Air Man boss door / fight)
- Natural-entry M4 from power-on through screen-2+
- Stage select other masters / weapon routing

## Next

1. **rr-54ui:** Decode empty-cloud solid (`aobject_tsa=$4E0`, flag 128→192, type 118);
   stand from above with feet-on-top geometry (not sy==by alone). Probe:
   `scripts/cloud_land_probe.py` + `recordings/air_post4_cloud/RED_PIN.md`.
2. Chain mapset 5–6 LLs → camera ≥5 → boss door; freeze AirScreen2→5 (3/3).
3. Do **not** re-sweep goblin-solid, “LL never spawns”, or hold-B without pulse.
