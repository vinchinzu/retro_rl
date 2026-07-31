# Status — Super Metroid


## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M5 |
| Best verified result | Continuous power-on → Warehouse Entrance (KPDR K2.6) |
| Last verification | 2026-07-31 |
| Runtime class | Bronze |
| Intervention class | Resource-assisted |

| Field | Value |
|-------|-------|
| Status | **Continuous power-on → Warehouse Entrance verified** (KPDR K2.6 tip) |
| Target | Continuous assisted power-on → ending/credits |
| Current assists | Current energy on Zebes + naturally unlocked current ammo |
| Shared ROM SHA-256 | `12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72` |
| Acceptance result | Natural Warehouse Entrance `0xA6A1` after Below Spazer tunnel chain |
| Video | re-encode optional; machine report is authority |
| Machine report | `recordings/start_to_warehouse.json` (**83,512** frames) |
| Save-state loads | 0 |
| Progression/capacity writes | 0 |

## Verified baseline

### Continuous power-on → Warehouse Entrance / KPDR K2.6 (2026-07-31)

`recordings/start_to_warehouse.json`: power-on through verified Below Spazer
prefix, then natural West Tunnel → Glass → East → **Warehouse Entrance**
`0xA6A1`. Controllers: `routes/kpdr/red_tower.py`
(`play_below_spazer_to_west` … `play_east_to_warehouse`).

| Metric | Value |
|--------|------:|
| Total frames | **83,512** (~23.2 min @ 60 fps) |
| Super collect | 73,169 |
| Red Tower entry | 80,267 |
| Bat Room entry | 81,512 |
| Below Spazer entry | 82,180 |
| West Tunnel entry | 82,670 |
| Glass entry | 82,942 |
| East Tunnel entry | 83,164 |
| Warehouse entry | **83,391** |
| Final room | `0xA6A1` ordinary gameplay |
| Below Spazer dwell (entry→exit split) | **490** |
| State loads / progression writes | 0 / 0 |
| Outcome | `warehouse_entry` |

Reproduce:

```bash
uv run python super_metroid/scripts/record/continuous.py --no-video
uv run python super_metroid/scripts/record/continuous.py --to warehouse --no-video --room-timing
```

Prefix milestones: `--to below_spazer|bat|red_tower|supers|spore|bombs|morph`.

### Continuous power-on → Below Spazer / KPDR K2.1 (2026-07-30)

`recordings/start_to_below_spazer.json`: power-on through verified Bat prefix,
then natural Bat Room three-platform dry crossing into **Below Spazer**
`0xA408`. Controllers: `routes/kpdr/red_tower.py` (`play_bat_to_below_spazer`;
high-sill and low-ledge entry paths for continuous door-settle variance).

| Metric | Value |
|--------|------:|
| Total frames | **82,300** (~22.9 min @ 60 fps) |
| Super collect | 73,169 |
| Red Tower entry | 80,267 |
| Bat Room entry | 81,512 |
| Below Spazer entry | **82,180** |
| Final room | `0xA408` ordinary gameplay |
| Bat dwell (entry→exit split) | **668** |
| State loads / progression writes | 0 / 0 |
| Outcome | `below_spazer_entry` |

Reproduce:

```bash
uv run python super_metroid/scripts/record/continuous.py --no-video
uv run python super_metroid/scripts/record/continuous.py --to below_spazer --no-video --room-timing
```

Prefix milestones: `--to bat|red_tower|supers|spore|bombs|morph` on the same CLI.

### Continuous power-on → Bat Room / KPDR K2.0 (2026-07-30)

`recordings/start_to_bat.json`: power-on with `retro.State.NONE` through the
verified K1 Red Tower prefix, then natural Red Tower descent (zigzag + timed
bomb floor + bottom exit) into **Bat Room** `0xA3DD`. Controllers:
`routes/kpdr/red_tower.py` (`play_red_tower_to_bat`).

| Metric | Value |
|--------|------:|
| Total frames | **81,652** (~22.7 min @ 60 fps) |
| Super collect | 73,169 |
| Red Tower entry | 80,267 |
| Bat Room entry | **81,512** |
| Final room | `0xA3DD` ordinary gameplay |
| Red Tower dwell (entry→exit split) | **1,245** |
| State loads / progression writes | 0 / 0 |
| Outcome | `bat_room_entry` |

Reproduce:

```bash
uv run python super_metroid/scripts/record/continuous.py --to bat --no-video
uv run python super_metroid/scripts/record/continuous.py --to bat --no-video --room-timing
```

### Continuous power-on → Red Tower / KPDR K1 (2026-07-30)

`recordings/start_to_red_tower.json`: power-on with `retro.State.NONE` through
Spore Super collect, then natural Super exit → farming → Big Pink main shaft →
GHZ → Noob → **Red Tower** `0xA253`. Charge Beam side trip is **not** on this
chain. Controllers: `routes/kpdr/` (same as prior controller-dev K1).

| Metric | Value |
|--------|------:|
| Total frames | **80,445** (~22.3 min @ 60 fps) |
| Super collect | 73,169 |
| Big Pink main | 76,967 |
| GHZ entry | 77,804 |
| Noob entry | 79,410 |
| Red Tower entry | **80,267** |
| Final room | `0xA253` ordinary gameplay |
| State loads / progression writes | 0 / 0 |
| Outcome | `red_tower_entry` |

Reproduce:

```bash
uv run python super_metroid/scripts/record/continuous.py --to red_tower --no-video
uv run python super_metroid/scripts/record/continuous.py --to red_tower --no-video --room-timing
```

### Continuous power-on → Spore Super Missiles (prefix; 2026-07-30)

`recordings/start_to_supers.json`: power-on with `retro.State.NONE`, full
accepted prefix through Spore Spawn exit into Super room `0x9B5B`, then
natural Super Missile collect (capacity **0 → 5**) via
`kpdr.play_super_room_collect`. Includes Spore fight rewrite + Climb early-fall
splice (below). Prior 2026-07-28 report was 92,424 frames.

| Metric | 2026-07-28 | Spore re-record | **+ Climb splice** |
|--------|-----------:|----------------:|-------------------:|
| Total frames | 92,424 | 74,421 | **73,251** (~20.3 min) |
| Spore fight (activate→HP0) | 23,173 | 5,170 | 5,170 |
| Climb→Parlor dwell | 4,339 | 4,339 | **3,169** |
| Super collect frame | 92,342 | 74,339 | **73,169** |
| State loads / progression writes | 0 / 0 | 0 / 0 | 0 / 0 |
| Outcome | supers | supers | **`spore_supers_collected`** |

Reproduce:

```bash
uv run python super_metroid/scripts/record/continuous.py --to supers --no-video
uv run python super_metroid/scripts/record/continuous.py --to supers
# Opt-in per-room timing (separate artifact; does not change integrity):
uv run python super_metroid/scripts/record/continuous.py --to supers --no-video --room-timing
```

### Room timing baseline (2026-07-30, post Climb splice)

Opt-in `RoomTimer` on continuous power-on → Supers (same integrity contract;
no door warps / progression writes). Artifacts:

| Artifact | Path |
|----------|------|
| Continuous report | `recordings/start_to_supers.json` |
| Room timing | `recordings/room_timings/start_to_supers_room_timing.json` |

| Metric | Value |
|--------|-------|
| Outcome | `spore_supers_collected` (integrity green) |
| Total frames | **73,251** |
| Visits timed | 39 hops |
| Total dwell frames | **51,138** |

Slowest hops by **dwell** (controllable room time):

| Rank | Room | Hop | Dwell | Notes |
|------|------|-----|------:|-------|
| 1 | Spore Spawn `0x9DC7` | → Super room | 6,777 | Fight + exit (was 24,780) |
| 2 | Spore Kihunters `0x9D9C` | → Spore | 4,968 | Pre-boss clear |
| 3 | Bomb Torizo `0x9804` | → Flyway | 3,993 | Boss fight |
| 4 | Ceres Ridley `0xE0B5` | → Flat | 3,453 | Scripted fight / escape |
| 5 | Parlor `0x92FD` | → Terminator | 3,350 | Post-Torizo left exit |
| 6 | Climb `0x96BA` | → Parlor | **3,169** | was 4,339; early fall loops removed |

### Climb early-fall splice (2026-07-30) — **done**

Profiled `pit_to_post_torizo` Climb ascent: second half (steady platform/wall
hops up the shaft) was fine; early segment had **repeated left-wall attempts**
that peaked ~y=1970 then fell back to the y=2067 ledge (and occasionally the
floor), burning ~1,170 policy frames.

| Fix | Detail |
|-----|--------|
| Policy | `policies/early_game/pit_to_post_torizo.json` |
| Splice | keep `[0:2138)` + resume at `3308` (drop thrash loop) |
| Frames removed | **1,170** (14,313 → 13,143) |
| Climb dwell | **4,339 → 3,169** (−1,170) |
| Continuous check | full power-on → Supers integrity green; bombs + Parlor exit OK |
| Backup | `pit_to_post_torizo.json.bak_pre_climb` (local; pre-splice) |

Metadata on the policy records `climb_early_fall_splice` with the cut indices.

**Next pure-nav targets** (after bosses/pre-boss): Parlor→Terminator (3,350),
Parlor→Flyway (2,627), Green Elev→Dachora (2,660).

### Continuous power-on → Spore Spawn (re-verified 2026-07-29)

Report-only re-run after Spore fight policy rewrite (expanded mouth-open
spritemaps + multi-missile open windows). Integrity green; no video re-encode
this pass.

| Metric | Old (2026-07-24) | New (2026-07-29) |
|--------|-----------------:|-----------------:|
| Total frames | 91,220 | **73,216** |
| Fight (activate→HP0) | 23,173 (~386 s) | **5,170 (~86 s)** |
| Exit frame | 90,802 | 72,798 |
| Speedup (fight) | — | **~4.5×** |

Root cause of the old ~6–10 min fight: controller only treated open/close
transition spritemaps as vulnerable and missed the long fully-open holds
(`0xEF3D` / `0xEF4F` / `0xEF61`), so most open windows landed zero missiles.
New policy still floor-bounces under the core (floor shots do not hit) and
fires every other frame while any open spritemap is active.

Reproduce: `uv run python super_metroid/scripts/record/continuous.py --to spore --no-video`.
See [START_TO_SPORE_SPAWN.md](routes/START_TO_SPORE_SPAWN.md).

Supers continuous baseline (2026-07-30) now embeds this fight.

## Full-room development infrastructure

On 2026-07-25, the research topology and isolated room-development loop were
validated:

- 261 vanilla reference rooms plus one editor-only unused room;
- 300 physical connections expanded to 583 directed traversals, retaining 17
  forward-only connections;
- 262 canonical room problems, with 69 initially classified as easy;
- successful bulk generation of 262 explicitly unverified starter policies;
- a 23-anchor completion sequence whose 22 legs all have a capability-aware
  room path;
- save-state capture/teleport validation and natural target-room settlement;
- two passing queue-1 policies: Green Brinstar Missile Station `0x9C89` →
  Fireflea `0x9C5E`, and Brinstar Map Room `0x9C35` → Pre-Map `0x9B9D`;
- one extra passing traversal: Flyway `0x9879` → Parlor `0x92FD`;
- item-objective validation that rejects an exit when the expected capacity or
  equipment delta did not occur.

These are development-state results and do not change the accepted continuous
prefix. See [ROOM_PROBLEM_CATALOG.md](research/ROOM_PROBLEM_CATALOG.md).

## Definition of done

The project is not a full clear yet. Completion still requires one emulator
session that naturally acquires required progression, defeats required bosses,
finishes the endgame escape, and reaches verified ending/credits state. The
resource assists may not write route progress.

## Next milestone

**Play KPDR by room — no door-warp route evidence.** Continuous Bat Room entry is
verified (first K2 hop). Authoritative order:

**[ROUTE_KPDR.md](routes/ROUTE_KPDR.md)** · hop topology:
**[PATH_ROOM_BOARD.md](research/PATH_ROOM_BOARD.md)** · legacy Pink-PB notes:
**[ROUTE_SUPERS_TO_PHANTOON.md](routes/ROUTE_SUPERS_TO_PHANTOON.md)**

| Layer | Furthest played |
|-------|-----------------|
| Continuous | **Warehouse Entrance `0xA6A1`** (`start_to_warehouse`, **83,512f**) — K2.1 Below Spazer prefix + West/Glass/East tunnels |
| Controller (dev) | **Warehouse→Hi-Jump→Warehouse→natural Kraid entry** (15,356f). The E-Tank and Boots are real PLM collects; the return uses intended Hi-Jump ledges and ordinary tunnel bombs, not an IBJ |
| Dev topology | **24/24 hops** Big Pink → Hi-Jump room (`kpdr.py route-to-hijump`); `dev_hijump_room_entry` + granted boots `dev_hijump_collected_dev` |
| ★ Next hop | Attach Warehouse→Hi-Jump→Kraid to continuous Warehouse predecessor, then `kraid_entry_to_varia` (boss-only Varia closeout already proven from doorway entry). Charge return remains a separate K1 gap |

Progress chart: [KPDR_TRACKER.md](routes/KPDR_TRACKER.md) · CSV
[KPDR_TRACKER.csv](routes/KPDR_TRACKER.csv) · JSON `maps/kpdr_tracker.json`.

```bash
uv run python super_metroid/scripts/export/path_room_board.py
uv run python super_metroid/scripts/probe/post_spore_pb.py --to main
```

Path status (unique rooms on research completion path): **31 continuous**,
**6 controller_dev**, **6 boss_deferred**, **64 open** (107 total / 199 hops).

Topology door-warps (`probe_route.py full` / hybrid) remain diagnostic only —
they do not count as room clearance.

Still blocked for *played* KPDR spine:

| Gap | Why it matters |
|-----|----------------|
| Continuous Super → Red Tower | **Done** (`start_to_red_tower`) |
| Charge / Big Pink return | Charge collects naturally; conventional return is not route-ready. Continuous K1 uses the direct Big Pink→GHZ line (no IBJ) |
| Continuous Warehouse → Hi-Jump → Kraid | Warehouse continuous green; Hi-Jump→Kraid still controller-only (15,356f) |
| ★ Kraid + Varia | Boss-only **closeout** from doorway entry: fight + rear door + real Varia PLM (`play_kraid_fight_to_varia`, ~1908f collect / ~2388f w/ fanfare on `eye_hj_kraid_entry`; `debug/kraid_varia_run.json`). Not continuous until composed after natural K2 entry |
| Alpha PB (not Pink PB) | First PB on competitive KPDR after Ice |
| Ship / Phantoon / … | After Alpha PB; warp entry is not continuous |
| Escape → credits | after MB by play |

Immediate next:

1. **Promote K2 remainder:** attach Warehouse→Hi-Jump collect→return→Kraid
   to the continuous Warehouse predecessor (`run_to("warehouse")` end state).
2. **Compose K3 on continuous:** run `kraid_entry_to_varia` after natural
   `play_eye_to_kraid` — boss-only fight + rear door + Varia PLM is already
   proven from doorway entry (`kraid_combat.py varia`).
3. **Optional K1 side trip:** Charge Beam conventional return (no IBJ).
4. **Parked:** pure Pink PB; ship-first Phantoon skip.

```bash
uv run python super_metroid/scripts/record/continuous.py --to warehouse --no-video
uv run python super_metroid/scripts/probe/kpdr.py pure warehouse-hijump-kraid \
  --source super_metroid/custom_integrations/SuperMetroid-Snes/scratch/red_to_warehouse_controller.state
uv run python super_metroid/scripts/export/kpdr_tracker.py
```


## Midgame / late dev furthest (not continuous)

| Checkpoint | State / evidence |
|------------|------------------|
| Spore Super room | `natural_post_spore_spawn` (no Supers yet on continuous) |
| Supers + Red Tower / GHZ / Noob / Warehouse | many `dev_*` states, items `0x1004`, supers 5 |
| Kraid Eye Door | `dev_kraid_eye_at_eye` |
| **Kraid defeated** | doorway policy ~1520f (`eye_hj_kraid_entry`); also `dev_kraid_defeated` |
| **Varia collected (boss-only)** | `eye_hj_kraid_varia_collected` items `0x1105`; also dev equip `dev_varia_equipped_dev` |
| **Power Bombs** | `dev_b1_pb_natural` / probe `--to pb-collect` pb `5/5` (sill+maze place bridges) |
| **Phantoon entry** | `dev_phantoon_entry` room `0xCD13` |
| **Ridley entry** | `dev_route_ridley_entry` room `0xB32E` (fights skipped) |
| **Mother Brain entry** | `dev_route_mother_brain_entry` room `0xDD58` (fights skipped) |
| **Full 22-leg finish** | door-warp chain ends Landing Site `0x91F8` (`probe_route.py full`) |
| **Late finish** | same via late 9-leg subset (`probe_route.py late-full`) |

## Endgame development track (not continuous evidence)

Mother Brain room is now reachable via the full late route skeleton (not only
the old direct teleport). Remaining fight/escape blockers:

- Zebetites regen 1 HP/frame until properly killed.
- Escape-room geometry needs pipe-corridor placement (air near y≈100).
- Escape timer needs full engine init to tick; credits evidence still open.
- Bank `$7E` WRAM must be used for events/boss bits (`read_bank7e_wram`).
