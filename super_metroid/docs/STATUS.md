# Status — Super Metroid


## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M5 |
| Best verified result | Continuous power-on → Spore Super collect |
| Last verification | 2026-07-27 |
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

### Continuous power-on → Spore Super Missiles (2026-07-27)

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
uv run python super_metroid/scripts/record_start_to_supers.py --no-video
uv run python super_metroid/scripts/record_start_to_supers.py
```

### Prior continuous power-on → Spore Spawn (2026-07-24)

Still valid prefix evidence: Morph, Missiles, Bombs/Torizo, Terminator E-Tank,
Spore Spawn 960→0, natural exit to Super room (no Supers collected on that
run). Video `recordings/start_to_spore_spawn.mp4` (91,220 frames). See
[START_TO_SPORE_SPAWN.md](START_TO_SPORE_SPAWN.md).

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
prefix. See [ROOM_PROBLEM_CATALOG.md](ROOM_PROBLEM_CATALOG.md).

## Definition of done

The project is not a full clear yet. Completion still requires one emulator
session that naturally acquires required progression, defeats required bosses,
finishes the endgame escape, and reaches verified ending/credits state. The
resource assists may not write route progress.

## Next milestone

**Play the path — no door-warp route evidence.** Continuous Super collect is
verified. Furthest *played* progress and the full 107-room plan:

**[PATH_ROOM_BOARD.md](PATH_ROOM_BOARD.md)** · post-Super detail:
**[ROUTE_SUPERS_TO_PHANTOON.md](ROUTE_SUPERS_TO_PHANTOON.md)**

| Layer | Furthest played |
|-------|-----------------|
| Continuous | Super collect `0x9B5B` (`start_to_supers`) |
| Controller (dev) | Big Pink main shaft `0x9D19` ~(746,1465); PB sill entry + mid-maze collect |
| ★ Next hop | Sill approach (wall@613) + maze past wall@437 → pure `0x9E11`+PB |

```bash
uv run python super_metroid/scripts/export_path_room_board.py
uv run python super_metroid/scripts/probe_post_spore_pb.py --to main
```

Path status (unique rooms on research completion path): **20 continuous**,
**2 controller_dev**, **6 boss_deferred**, **79 open** (107 total / 199 hops).

Topology door-warps (`probe_route.py full` / hybrid) remain diagnostic only —
they do not count as room clearance.

Still blocked for *played* spine:

| Gap | Why it matters |
|-----|----------------|
| Continuous Super collect | **Done** |
| Super → farming → main shaft | controller_dev; not continuous power-on yet |
| ★ Shaft → Pink PB door + collect | **partial**: sill entry + morph-bomb collect green; approach/maze bridges remain |
| All later path hops | open — must be played, not warped |
| Boss fights | deferred until natural entry on chain |
| Escape → credits | after MB by play |

Immediate next:

1. Finish Big Pink → PB **without place bridges**: (a) main/intercept onto sill past wall@613, (b) spawn past maze wall@437, then existing sill entry + morph-bomb collect.
2. Continuous power-on → PB.
3. Next open hop on the board; promote status; regenerate PATH_ROOM_BOARD.

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
