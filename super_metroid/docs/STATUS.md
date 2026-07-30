# Status — Super Metroid


## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M5 |
| Best verified result | Continuous power-on → Spore Super collect |
| Last verification | 2026-07-29 |
| Runtime class | Bronze |
| Intervention class | Resource-assisted |

| Field | Value |
|-------|-------|
| Status | **Continuous power-on → Spore Super Missiles verified** |
| Target | Continuous assisted power-on → ending/credits |
| Current assists | Current energy on Zebes + naturally unlocked current ammo |
| Shared ROM SHA-256 | `12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72` |
| Acceptance result | Super capacity 0→5 in room `0x9B5B`; prior Spore clear intact |
| Video | `recordings/start_to_supers.mp4` (92,425 frames, ~25.7 min) |
| Machine report | `recordings/start_to_supers.json` |
| Save-state loads | 0 |
| Progression/capacity writes | 0 |

## Verified baseline

### Continuous power-on → Spore Super Missiles (2026-07-28)

`recordings/start_to_supers.json` + `.mp4`: power-on with
`retro.State.NONE`, full accepted prefix through Spore Spawn exit into Super
room `0x9B5B`, then natural Super Missile collect (capacity **0 → 5**) via
`post_spore_controller.play_super_room_collect`.

| Metric | Value |
|--------|-------|
| Total frames | 92,425 @ 60 fps (~25.7 min) |
| Super collect frame | ~92,342 |
| Final room | `0x9B5B` ordinary gameplay |
| State loads | 0 |
| Progression / capacity writes | 0 |
| Outcome | `spore_supers_collected` |

Reproduce:

```bash
uv run python super_metroid/scripts/record/start_to_supers.py --no-video
uv run python super_metroid/scripts/record/start_to_supers.py
# Opt-in per-room timing (separate artifact; does not change integrity):
uv run python super_metroid/scripts/record/start_to_supers.py --no-video --room-timing
```

### Room timing baseline (2026-07-29)

Opt-in `RoomTimer` on continuous power-on → Supers (same integrity contract;
no door warps / progression writes). Artifacts:

| Artifact | Path |
|----------|------|
| Continuous report | `recordings/start_to_supers_room_timing_baseline.json` |
| Room timing | `recordings/room_timings/start_to_supers_room_timing.json` |

| Metric | Value |
|--------|-------|
| Outcome | `spore_supers_collected` (integrity green) |
| Total frames | 92,424 |
| Visits timed | 39 hops |
| Total dwell / room / door | 70,311 / 77,426 / 7,115 emulator frames |
| Discontinuities | 2 (`boot_or_menu` at Ceres ship cutscene; `session_end` open Super room) |

Slowest hops by **dwell** (controllable room time):

| Rank | Room | Hop | Dwell | Notes |
|------|------|-----|------:|-------|
| 1 | Spore Spawn `0x9DC7` | → Super room | 24,780 | Boss fight (expected) |
| 2 | Spore Kihunters `0x9D9C` | → Spore | 4,968 | Pre-boss clear |
| 3 | **Climb `0x96BA`** | → Parlor | **4,339** | **Largest early nav hop** (post-Pit bombs path) |
| 4 | Bomb Torizo `0x9804` | → Flyway | 3,993 | Boss fight |
| 5 | Parlor `0x92FD` | → Terminator | 3,350 | Post-Torizo left exit |

Aggregate dwell by room (nav-relevant): Parlor 6,775f (3 visits), Climb
4,995f (2 visits). **Next experiment (not done):** re-record a tighter
Climb→Parlor slice inside `policies/early_game/pit_to_post_torizo.json`
(segment frames ~34,598–39,107), then re-run
`start_to_supers.py --no-video --room-timing` and require lower Climb dwell
with integrity still green. No policy edit applied from this measurement
alone (unsafe without a re-verified splice).

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

Reproduce: `uv run python super_metroid/scripts/record/start_to_spore_spawn.py --no-video`.
See [START_TO_SPORE_SPAWN.md](routes/START_TO_SPORE_SPAWN.md).

Supers continuous baseline above still embeds the old fight until re-recorded
(`start_to_supers.py`); expected total after re-run ≈ 92k − ~18k frames.

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

**Play KPDR by room — no door-warp route evidence.** Continuous Super collect is
verified. Authoritative order:

**[ROUTE_KPDR.md](routes/ROUTE_KPDR.md)** · hop topology:
**[PATH_ROOM_BOARD.md](research/PATH_ROOM_BOARD.md)** · legacy Pink-PB notes:
**[ROUTE_SUPERS_TO_PHANTOON.md](routes/ROUTE_SUPERS_TO_PHANTOON.md)**

| Layer | Furthest played |
|-------|-----------------|
| Continuous | Super collect `0x9B5B` (`start_to_supers`) |
| Controller (dev) | **Natural Big Pink main→GHZ→Noob→Red** (3,478f); **Red→Warehouse Entrance** (2,929f); **Warehouse→Hi-Jump→Warehouse→natural Kraid entry** (15,356f). The E-Tank and Boots are real PLM collects; the return uses intended Hi-Jump ledges and ordinary tunnel bombs, not an IBJ |
| Dev topology | **24/24 hops** Big Pink → Hi-Jump room (`kpdr.py route-to-hijump`); `dev_hijump_room_entry` + granted boots `dev_hijump_collected_dev` |
| ★ Next hop | Compose the Kraid fight from the natural controller entry, take the rear door, and collect Varia; Charge return remains a separate K1 gap |

Progress chart: [KPDR_TRACKER.md](routes/KPDR_TRACKER.md) · CSV
[KPDR_TRACKER.csv](routes/KPDR_TRACKER.csv) · JSON `maps/kpdr_tracker.json`.

```bash
uv run python super_metroid/scripts/export/path_room_board.py
uv run python super_metroid/scripts/probe/post_spore_pb.py --to main
```

Path status (unique rooms on research completion path): **20 continuous**,
**17 controller_dev**, **6 boss_deferred**, **64 open** (107 total / 199 hops).

Topology door-warps (`probe_route.py full` / hybrid) remain diagnostic only —
they do not count as room clearance.

Still blocked for *played* KPDR spine:

| Gap | Why it matters |
|-----|----------------|
| Continuous Super collect | **Done** |
| Super → farming → main shaft | controller_dev; not continuous power-on yet |
| Charge / Big Pink → GHZ | Direct Big Pink→GHZ is controller-complete; Charge collects naturally, but its conventional return is not route-ready. No IBJ is required on the active suffix |
| Warehouse → Hi-Jump → Kraid | **Controller-complete from the natural Warehouse predecessor:** real E-Tank/Boots PLMs, reverse traversal, three-Super wall, and natural Kraid-room entry |
| ★ Kraid + Varia | Kraid combat exists only as a dev probe; compose it from natural entry, take the rear door, and collect the real Varia PLM |
| Alpha PB (not Pink PB) | First PB on competitive KPDR after Ice |
| Ship / Phantoon / … | After Alpha PB; warp entry is not continuous |
| Escape → credits | after MB by play |

Immediate next:

1. **Pure K3:** run the Kraid fight from the natural K2 entry, exit through
   the rear door, and collect Varia from its real PLM.
2. **Finish K1 safety side trip:** Charge Beam collects naturally; implement a
   conventional return to the direct Big Pink→GHZ line. Do not pursue an IBJ.
3. **Promote composition:** attach Red→Warehouse→Hi-Jump→Kraid to the real K1
   predecessor, then to the continuous power-on prefix.
4. **Parked:** pure Pink PB; ship-first Phantoon skip.

```bash
uv run python super_metroid/scripts/probe/kpdr.py route-to-hijump --grant-hijump
uv run python super_metroid/scripts/export/kpdr_tracker.py
uv run python super_metroid/scripts/probe/post_spore_pb.py --to main
```


## Midgame / late dev furthest (not continuous)

| Checkpoint | State / evidence |
|------------|------------------|
| Spore Super room | `natural_post_spore_spawn` (no Supers yet on continuous) |
| Supers + Red Tower / GHZ / Noob / Warehouse | many `dev_*` states, items `0x1004`, supers 5 |
| Kraid Eye Door | `dev_kraid_eye_at_eye` |
| **Kraid defeated** | `dev_kraid_defeated`, boss bit 0 set ~frame 2100 |
| Varia room (dev equip) | `dev_varia_equipped_dev` items `0x1005` |
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
