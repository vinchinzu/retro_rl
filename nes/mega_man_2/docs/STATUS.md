# Status — Mega Man 2 (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M3 (isolated segment) |
| Best verified result | Air Man camera screen ≥ 4 from `AirScreen2` (3/3); Heat Man camera ≥ 7 pre-boss from `HeatScreen5Ground` (3/3) |
| Last verification | 2026-08-10 (rr-809 Heat late dual-green cam ≥7) |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **Air s4 clear; post-s4 cloud RED; Heat dual-green cam ≥7 pre-boss; s7 wall-lock climb residual (rr-809 PARTIAL)** |
| Integration | `MegaMan2-Nes` |
| ROM zip | `roms/Nintendo/NES/Mega Man II.zip` |
| Ready frame (probe) | ~1204 Air / ~926 Heat |
| Checkpoints | Air: `Level1`, `AirLanded`, `AirScreen2`–`4`, `AirFanPlatform`, `AirLeftPlatform`. Heat: `Heat1`, `HeatScreen1`–`HeatScreen7`, `HeatScreen5Ground`, `HeatScreen7Mid` |
| Policy | `AirManPolicy` (mid/late); `HeatManPolicy` multi-phase (early/s2/s3/s4/s5) |
| Evidence | [air_segment/](../recordings/air_segment/), [heat_boot/](../recordings/heat_boot/), [heat_segment/](../recordings/heat_segment/), [heat_s7_seg/](../recordings/heat_s7_seg/), [heat_s7_climb_residual/](../recordings/heat_s7_climb_residual/) |

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
| Body | `0x3E` stays; type 6 = `objects_killed` (~12f); type 118 **not** seen this session |
| Best Y-meet | dx≈5–10, \|dy\|≤4 after kill — still freefall `ft=0` |
| feet-on-top | **feet_dy=0 @ dx≤2** (MM_H=24) still freefall — residual ≠ X / ≠ feet align alone |
| Solid decode | `aobject_tsa` = AI timer (not solid); flag 128→192 = facing `objects_right`; body never becomes platform type |
| Pitfall | Co-sink (matched vertical rate) locks feet_dy≈−3…−4; kill high still no arm |
| Cam ≥5 | **No** |
| Evidence | `docs/CLOUD_LAND_RED_PIN.md`, `recordings/air_post4_cloud_solid/`, v2–v7 |

### Disasm + screen-align (same night, later) — residual sharpened

| Field | Value |
|-------|-------|
| Body AI (lsmmega `14_19`) | Spawns rider `0x3D`; **no solid-arm rewrite** when child dies |
| Object solid path | Only decoded path = `appearing_block` flag `$10` — never on empty `0x3E` |
| PRG | 4× `CMP #$3E` (AI only); 0× `CMP #$3D` — no type solid whitelist |
| Cloud top | OAM y≈by−16; prior feet_dy=0 was body **center** not top |
| Screen-align | Kill window cam=3 / body scr=4; cam=4 arrives with top_dy≈−19 |
| Diag fall_top | top_dy≈+1 force-place still freefall (solid path inactive) |
| Evidence | `scripts/cloud_screen_align.py`, `recordings/air_post4_screen_align/` |

### Alt path + appear-mask (2026-08-10) — still OPEN (PARTIAL)

| Field | Value |
|-------|-------|
| Cam ≥5 | **No** |
| Human path | Cloud ride ×5 after kill (walkthroughs) |
| TAS alt | Item-1 (Heat-first) — `weapons=$00` on AirFan → not Air-first Clean |
| Jump skip | **No** — gap ~296px after prog 984 |
| Appear arm | Sole `LDA #$90` = appear-block AI `14_23`; body never arms |
| Zero-mask force | Global solid under fceumm (path works when configured) |
| Localized masks | Still freefall after kill |
| Residual child | **rr-f3nr** (Heat→Air Item-1 scaffold PARTIAL; FCEUX pin protocol documented) |
| Evidence | `docs/CLOUD_LAND_RED_PIN.md`, `recordings/air_post4_altpath/` |

## rr-f3nr Heat→Item-1 scaffold (2026-08-10) — PARTIAL

Air-first cloud path blocked overnight; preferred alt = Heat→Air Item-1.

| Milestone | Result |
|-----------|--------|
| Inventory | Heat chain pinned `Heat1`…`HeatScreen5`; Air-only still `weapons=$00` |
| Stage-select decode | `$002A`: Wily=0, Air=2, Heat=8; password→select at Wily; LEFT→Heat |
| Heat1 entry | **GREEN** — `boot_to_heat_man_script` + `boot_heat_probe.py` |
| Heat screens 1–5 | **GREEN** — multi-phase `HeatManPolicy` (rr-808 PARTIAL) |
| Heat clear / Item-1 | **Not done** — s7 wall-lock (sx152); ladder x192 unreachable; no boss_hp |
| Air + Item-1 past s4 | **Not done** |
| FCEUX stick pin | Protocol only (`docs/HEAT_ITEM1_PATH.md`); no human run this session |
| Evidence | `docs/HEAT_ITEM1_PATH.md`, `recordings/heat_boot/`, `heat_segment/`, `heat_s7_climb_residual/` |

### Heat Man segment metrics

| Segment | Frames | HP | Cam | Prog | Trials |
|---------|-------:|---:|----:|-----:|-------:|
| Heat1 → screen ≥1 | ~244 | 24 | 1 | 256 | **3/3** |
| HeatScreen1 → ≥2 | ~194 | 24 | 2 | 512 | **3/3** |
| HeatScreen2 → ≥3 | ~302 | 28 | 3 | 768 | **3/3** |
| HeatScreen3 → ≥4 | ~161 | 28 | 4 | 1024 | **3/3** |
| HeatScreen4 → ≥5 | ~131 | 28 | 5 | 1280 | **3/3** |

Grounded pins: `HeatScreen2` prog513, `HeatScreen3` prog819 sy116, `HeatScreen4`
prog1110 sy148, `HeatScreen5` prog1473 sy124 (reload may show feet=0). Best late
press from s4: prog ~1504–1507 then pit (not dual-green past s5).

### HeatScreen5Ground → camera ≥7 (pre-boss)

| Metric | Value |
|--------|------:|
| Frames | ~293 |
| Final HP | 22 |
| Camera screen | 7 |
| Progress X | 1792 |
| Trials | **3/3** |

Policy `start=screen5`: idle2 → j1/20 → LEFT4 → j2/24 → hop9/gw3 (A-edge).
Pins: `HeatScreen5Ground`, `HeatScreen6`, `HeatScreen7`, `HeatScreen7Mid`.

## rr-808 Heat mid/late (2026-08-10) — PARTIAL (screens 2–5 dual-green)

Superseded for late path by rr-809 pre-boss dual-green; bead closed/partial as noted.

## rr-809 Heat boss + Item-1 (2026-08-10) — PARTIAL

| Field | Value |
|-------|-------|
| Dual-green | cam ≥7 from `HeatScreen5Ground` 3/3 ~293f (Clean Bronze) |
| s7 climb | **Blocked** — wall sx152; mapset7 ladder x192 unreachable |
| Boss / Item-1 | **Not done** (no boss_hp, weapons/items still 0) |
| Pin | `HeatScreen7Mid` sx152 sy124 under Telly |
| Residual detail | `docs/HEAT_ITEM1_PATH.md` + `recordings/heat_s7_climb_residual/` |
| Next | Past s7 wall → scroll_down shaft → boss clear → Item-1 pin |

## Not done

- Past Air screen 4 / boss door (**kill OK; cloud solid RED; Item-1 chain open**)
- Heat Man s7 climb / boss door / boss clear + Item-1 unlock pin
- Air with Item-1 past camera ≥5
- Full Robot Master stage clear
- Natural-entry M4 from power-on through screen-2+

## Next

1. **s7 climb past wall** — reach mapset7 ladder (x192+) or alt vertical entry;
   dual-green into Sniper Armor shaft / boss door.
2. **Item-1 pin** — Heat clear → `$009B\|$01` + Atomic Fire `$009A\|$01` (**rr-809**).
3. **Air + Item-1** — stage select to Air with items set; deploy platforms past s4
   (cam ≥5) (**rr-810**). Doc: `docs/HEAT_ITEM1_PATH.md`.
4. Optional parallel: FCEUX/human empty-cloud RAM pin (protocol in HEAT_ITEM1_PATH).
5. Do **not** re-sweep goblin-solid, LL-absent, hold-B only, feet_dy grids,
   screen-align-only, fall_top/appear/flag08, zero-mask global solid, or s7
   RIGHT-wall hop-only / UP-DOWN feet=2 spam without a new route hypothesis.
