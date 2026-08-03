# Status — Super Metroid


## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M5 |
| Best verified result | Continuous power-on → Frog Savestation (KPDR K4.0) |
| Last verification | 2026-08-01 |
| Runtime class | Bronze |
| Intervention class | Resource-assisted |
| Milestone board | [routes/MILESTONES.md](routes/MILESTONES.md) |
| Backlog | [routes/BACKLOG.csv](routes/BACKLOG.csv) (~308 tickets → M8 + CLEAN + Cathedral) |
| Parallel track | **Clean** (no energy/ammo) → Bomb Torizo — [CLEAN_TRACK.md](CLEAN_TRACK.md) |

| Field | Value |
|-------|-------|
| Status | **Continuous power-on → Frog Savestation verified** (KPDR K4.0) |
| Target | Continuous assisted power-on → ending/credits |
| Current assists | Current energy on Zebes + naturally unlocked current ammo |
| Clean track | Morph **green** (27,074f); ★ next bombs/Torizo Clean — [CLEAN_TRACK.md](CLEAN_TRACK.md); does **not** change this program gate |
| Shared ROM SHA-256 | `12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72` |
| Acceptance result | Natural Varia return spine + Warehouse reverse stack + Business elevator descent → Frog Save |
| Video | No-video dual verification (first video still open) |
| Machine report | `recordings/start_to_frog_save.json` + `_reverify.json` (**114,923f** each) |
| Save-state loads | 0 |
| Progression/capacity writes | 0 |

## Verified baseline

### Continuous power-on → Frog Savestation / KPDR K4.0 (approved 2026-08-01)

Two matching `--to frog --no-video` runs reached ordinary Frog Savestation
`0xB167`. The full K3 return spine, Business elevator descent, and closed
blue-door exit all passed with known transitions, ordered splits, and **0**
state loads / progression writes / capacity writes / deaths.

| Metric | Value |
|--------|------:|
| Total frames | **114,923** (~31.9 min @ 60 fps), twice |
| Business return | 113,530 |
| Frog Save entry | **114,776** |
| Final room | `0xB167` ordinary gameplay |
| Checkpoint | `scratch/post_frog_continuous.state` |
| Outcome | `frog_save_reached` |

Machine reports: `recordings/start_to_frog_save.json` and
`recordings/start_to_frog_save_reverify.json`. The default continuous CLI tip
is now `frog`; first Bubble path is **Cathedral climb** (CATH-01/02 pure green;
★ next `SM-K4-CATH-03`). Frog Speedway is post-Speed only.

### Continuous power-on → Business Center return / KPDR K3→K4 (approved 2026-08-01)

Two matching `--to business --no-video` runs reached ordinary Business Center
`0xA7DE` after the natural Varia return spine and the right-ledge Warehouse
reverse stack. Both reports are integrity-green: all transitions known,
required splits present and ordered, natural Business endpoint, and **0** state
loads / progression writes / capacity writes / deaths.

| Metric | Value |
|--------|------:|
| Total frames | **113,723** (~31.6 min @ 60 fps), twice |
| Varia collect | 104,382 |
| Business return | **113,723** |
| Final room | `0xA7DE` ordinary gameplay |
| Checkpoint | `scratch/post_business_continuous.state` |
| Outcome | `business_return` |

Machine reports: `recordings/start_to_business.json` and
`recordings/start_to_business_reverify.json`. This milestone advanced the
default tip to `business`; Frog Save later superseded it as the current default.

### Continuous power-on → Varia Suit / KPDR K3 (approved 2026-08-01)

**Integrity re-verify GREEN** (post Wave-10 dual-track room farm; spine
controllers unchanged by farm). Power-on → natural Kraid fight + rear door +
real Varia PLM in `0xA6E2`. Controllers: `routes/kpdr/` + `combat/kraid.py`
(`play_kraid_entry_to_varia`).

| Metric | Best published (2026-07-30) | Re-verify (2026-07-31 / 2026-08-01) |
|--------|----------------------------:|------------------------------------:|
| Total frames | **101,954** | **104,382** (+2,428; no savings) |
| Hi-Jump collect | 87,696 | 87,696 |
| Warehouse return (business exit) | 92,241 | 91,940 |
| Kraid entry (`eye_to_kraid`) | 97,051 | 96,805 |
| Varia collect | **101,954** | **104,382** |
| Final room | `0xA6E2` ordinary | `0xA6E2` ordinary |
| State loads / prog / capacity | 0 / 0 / 0 | 0 / 0 / 0 |
| Outcome | `varia_collected` | `varia_collected` |

**Frame-total policy:** keep **101,954** as best published tip time (video-
matched). Multi-run re-verifies land at **104,382f** with full integrity green
— do **not** promote the slower total. Climb dwells multi-run match Wave-6:
`business_to_warehouse` **2,006f**, `hj_shaft` return band **1,835f**.

Latest machine report: `recordings/start_to_varia.json` (and
`recordings/start_to_varia_reverify_20260801.json`, same totals). Integrity:
`state_loads_zero`, `progression_writes_zero`, `capacity_writes_zero`,
`deaths_zero`, `natural_varia_room`, `varia_collected` all true.

Reproduce:

```bash
uv run python super_metroid/scripts/record/continuous.py --to business
uv run python super_metroid/scripts/record/continuous.py --no-video  # default tip
```

### Continuous power-on → Kraid entry / KPDR K2.18 (2026-07-30)

`recordings/start_to_kraid.json`: power-on through verified Hi-Jump collect,
Business Center return climb (continuous-hardened), Warehouse → Zeela → …
→ natural **Kraid's Room** `0xA59F`. Controllers: `routes/kpdr/` (including
`play_business_to_warehouse` grounded hop gates + 987→907 run-up 14 / floor
retry). Integrity green. Video: `recordings/start_to_kraid.mp4`.

| Metric | Value |
|--------|------:|
| Total frames | **97,170** (~27.0 min @ 60 fps) |
| Hi-Jump collect | 87,696 |
| Warehouse return | 92,241 |
| Eye door entry | 96,331 |
| Kraid entry | **97,051** (split) / report end **97,170** |
| Final room | `0xA59F` ordinary gameplay |
| State loads / progression writes | 0 / 0 |
| Outcome | `kraid_entry` |

Reproduce:

```bash
uv run python super_metroid/scripts/record/continuous.py --to kraid
```

Business climb continuous fixes (for the prior 1339 lip / 907 miss / 779 lip
blockers): standing gates before charged hops; 987→907 uses 14f run-up first
(pure re-climbs with 8f); 779→elevator setup band x≤80; floor recover + one
full re-climb on miss.

### Continuous power-on → Hi-Jump Boots / KPDR K2.10 (2026-07-30)

`recordings/start_to_hijump.json`: power-on through verified Warehouse
prefix, then natural elevator → Business Center → Hi-Jump shaft → **Hi-Jump
Room** `0xA9E5` with real boots PLM (`collected_items` gains `0x0100`).
Controllers: `routes/kpdr/` (`play_warehouse_to_business` …
`play_hj_room_collect`). Integrity green.

| Metric | Value |
|--------|------:|
| Total frames | **87,696** (~24.4 min @ 60 fps) |
| Warehouse entry | 83,391 |
| Business entry | 83,720 |
| HJ shaft entry | 85,161 |
| HJ room entry | 86,519 |
| Hi-Jump collect | **87,696** |
| Final room | `0xA9E5` ordinary gameplay |
| State loads / progression writes | 0 / 0 |
| Outcome | `hijump_collected` |

Reproduce:

```bash
uv run python super_metroid/scripts/record/continuous.py --to hijump --no-video
```

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
uv run python super_metroid/scripts/record/continuous.py --to warehouse --no-video --room-timing
```

Prefix milestones: `--to below_spazer|bat|red_tower|supers|spore|bombs|morph`.

Architecture / contracts: [`ARCHITECTURE.md`](ARCHITECTURE.md),
`routes/segment.py`.

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

**Play KPDR by room — no door-warp route evidence.** Continuous power-on →
Frog Savestation is the verified tip (K4.0); K4 forward remains pure-first.
Authoritative order:

**[ROUTE_KPDR.md](routes/ROUTE_KPDR.md)** · hop topology:
**[PATH_ROOM_BOARD.md](research/PATH_ROOM_BOARD.md)** · process:
**[tasks/PROCESS.md](tasks/PROCESS.md)** · architecture:
**[ARCHITECTURE.md](ARCHITECTURE.md)** · legacy Pink-PB notes:
**[ROUTE_SUPERS_TO_PHANTOON.md](archive/routes/ROUTE_SUPERS_TO_PHANTOON.md)** (archived; not KPDR)

| Layer | Furthest played |
|-------|-----------------|
| Continuous | **Frog Savestation `0xB167`** tip **integrity GREEN** twice at **114,923f**. Prefixes: Business 113,723f, Varia 104,382f, Kraid entry ~97k, Hi-Jump 87,696f, Warehouse 83,512f |
| Controller (dev) | Cathedral pure **CATH-01/02 GREEN**; CATH-03 scaffold open. First Bubble = Cathedral (no Speed). Speedway pure green but **parked** post-Speed |
| Dev topology | **24/24 hops** Big Pink → Hi-Jump room (`kpdr.py route-to-hijump`); full 22-leg door-warp tour exists (`developmentOnly`) |
| ★ Next hop | **`SM-K4-CATH-03`** Cathedral → Rising Tide from `post_cathedral_entrance_to_cathedral_pure`; then Bubble → Speed → Wave → Ice pure before continuous tips |

Progress chart: [KPDR_TRACKER.md](routes/KPDR_TRACKER.md) · CSV
[KPDR_TRACKER.csv](routes/KPDR_TRACKER.csv) · JSON `maps/kpdr_tracker.json`.

```bash
uv run python super_metroid/scripts/export/path_room_board.py
uv run python super_metroid/scripts/export/kpdr_tracker.py
```

Path status (unique rooms on research completion path): continuous coverage is
the early KPDR spine through Frog Save; topology has **~107 rooms / 199 hops**
identified. Exact continuous vs controller_dev counts live in the path board
export — do not invent hop tallies here.

Topology door-warps (`probe_route.py full` / hybrid) remain diagnostic only —
they do not count as room clearance.

Still open for *played* KPDR spine:

| Gap | Why it matters |
|-----|----------------|
| Continuous Super → Red Tower → Warehouse → Hi-Jump → Kraid → Varia → Business → Frog Save | **Done** (two 114,923f Frog Save returns; see baseline) |
| K4 forward (Cathedral → Bubble → Speed → Wave → Ice) | First missing natural segment; CATH-01/02 green; ★ CATH-03 |
| Speedway → Farm → Bubble | Parked until post-Speed (Boost Blocks) |
| Alpha PB (not Pink PB) | First PB on competitive KPDR after Ice |
| Ship / Phantoon / Botwoon / Draygon / Ridley / MB | Sequential per [`BOSS_PIPELINE.md`](BOSS_PIPELINE.md); warp entry is not continuous |
| Escape → credits | After MB by play; ending evidence open |
| Charge / Big Pink return | Optional K1 side trip; continuous K1 uses direct Big Pink→GHZ (no IBJ) |

Immediate next (continuous tip + structure):

1. **★ K4 Cathedral pure stack:** CATH-03 → Bubble → Speed → Wave → Ice;
   graph/compose/continuous only after pure green; stabilize after each tip.
2. **Parallel:** Clean bombs tip · room farm · 1–2 ARCH (hops extract, RAM gate).
3. **Optional tighten** high-dwell continuous hops offline first
   (`split_dwell.py` on green reports) — secondary to pure stack.
4. **K5–K6** Alpha PB → ship / natural Phantoon after Ice continuous.
5. **Code-plan leverage** (does not replace pure-first): selective-RAM /
   StateCache enforcement, hop-table extract, richer pure RED
   diagnostics, graph-first hop ranking — see [`plan.md`](plan.md) and
   [`ARCHITECTURE.md`](ARCHITECTURE.md).
7. **Dual-track:** room practice / combat unit scaffolds in parallel via
   `farm_room_waves.sh` while spine advances.
8. **Parked:** pure Pink PB; ship-first Phantoon skip; Charge conventional
   return; vision BC in `legacy/`.

```bash
uv run python super_metroid/scripts/record/continuous.py --to business
uv run python super_metroid/scripts/record/continuous.py --to kraid --no-video
uv run python super_metroid/scripts/export/split_dwell.py \
  super_metroid/recordings/start_to_business.json --top 15
# First K4 forward door (pure / source-backed)
uv run python super_metroid/scripts/probe/kpdr.py suggest-source \
  --room 0xA7DE --segment business-to-frog-save
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

## Maturity ladder (this game)

| Gate | Target | Status (2026-07-31) |
|------|--------|---------------------|
| **M5** | Bronze observation; resource-assisted continuous tip | **Current** — power-on → Varia |
| **M6** | Complete route graph with owners/predicates for critical path | In progress — ~107/199 hops identified; early KPDR continuous |
| **M7** | Continuous dry-run invariants (power-on → credits path; resource assists only) | Open — needs spine through MB + escape |
| **M8** | Verified capture + ending/credits evidence | Open |

Observation-class migration (Bronze → Silver) is a **separate** workstream after
continuous reliability.

**Clean intervention track (parallel):** early continuous tips with **no**
energy or ammo assists, targeting Bomb Torizo first. Process, artifact
isolation, and tickets: [`CLEAN_TRACK.md`](CLEAN_TRACK.md). Primary program
gate above stays **Resource-assisted** until Clean tips are green and
explicitly documented as a secondary claim — Clean never overwrites assisted
baselines or demotes Frog Save.

Program process: pure-first + one-knob + residual schema + dual-track
([`tasks/PROCESS.md`](tasks/PROCESS.md)). Do not relax those rules for speed.
