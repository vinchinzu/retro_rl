# Status — Super Metroid


## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M5 |
| Best verified result | Continuous power-on → Spore Spawn |
| Last verification | 2026-07-25 |
| Runtime class | Bronze |
| Intervention class | Resource-assisted |

| Field | Value |
|-------|-------|
| Status | **Continuous power-on → Bomb Torizo → Spore Spawn verified** |
| Target | Continuous assisted power-on → ending/credits |
| Current assists | Current energy on Zebes + naturally unlocked current ammo |
| Shared ROM SHA-256 | `12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72` |
| Acceptance result | Spore Spawn 960→0; natural exit to `0x9B5B` |
| Video | `recordings/start_to_spore_spawn.mp4` |
| Machine report | `recordings/start_to_spore_spawn.json` |
| Independent verification | `recordings/start_to_spore_spawn.verify.json` |
| Save-state loads | 0 |
| Progression/capacity writes | 0 |

## Verified baseline

The 2026-07-24 acceptance run starts with `retro.State.NONE`, selects a fresh
file from the title flow, traverses Ceres in both directions, escapes to Zebes,
collects Morph Ball, naturally collects the First Missile and Blue Brinstar
Missile expansions, doubles back through Morph Ball Room, climbs from Pit Room
to Parlor, opens the Flyway red door, collects Bombs, defeats Bomb Torizo, and
exits naturally back through Flyway. It then collects the Terminator Energy
Tank, descends into Green Brinstar, crosses Dachora and Big Pink, clears the
Spore Kihunters, defeats Spore Spawn, and exits naturally to the Super room.

The H.264 recording is 91,220 frames at 60 fps (1,520.333 seconds), 512×448.
The report proves:

- Missile capacity changes naturally `0 → 5 → 10`.
- Collected/equipped item masks end at `0x1004` (Morph Ball + Bombs).
- Bomb Torizo activates at 800 HP, reaches zero HP, and the locked room exit
  transition completes naturally.
- The Terminator Energy Tank changes maximum energy naturally `99 → 199`.
- Spore Spawn activates at 960 HP, reaches zero, completes its death
  animation, and the run takes the natural exit to room `0x9B5B`.
- All 40 observed room transitions match typed graph edges, including every
  edge in the editor-precalculated Spore route.
- No save state is loaded after power-on.
- Energy assist is suspended on Ceres, then performs 252 current-energy writes
  up to natural max energy on Zebes. Ammo performs 497 current-Missile writes
  after natural unlock.
- Progression-write and capacity-write counters remain zero.

See [START_TO_SPORE_SPAWN.md](START_TO_SPORE_SPAWN.md) for reproduction,
planning provenance, and evidence.

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

Collect the Spore Super naturally, traverse `0xA0A4 → 0x9D19 → 0x9E11`, and
collect the first Power Bomb expansion. Use the pre-calculated room catalog to
develop the three room clears independently, then promote only transitions
observed in the continuous emulator session.
