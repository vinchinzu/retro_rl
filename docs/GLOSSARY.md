# Glossary

## Ladders (keep separate)

| Name | Labels | Meaning |
|------|--------|---------|
| Runtime observation class | Gold, Silver, Bronze | What the live agent may observe |
| Intervention class | Clean, Survival-/Resource-/Protection-/Progression-assisted | What may be written mid-run |
| Completion maturity | M0–M8 | Engineering gate for one game |
| Capability phase | Phase 0–7 | Program-level genre track priority |
| Capability track | Named genre tracks | Parallel competency lines (combat, platforming, …) |
| Automation class | See below | How much game-specific privilege the policy uses |
| Project state | planned … archived | Workspace lifecycle |

Never use unqualified **tier**, **class**, or **rank** when another label fits.

## Capability tracks

- Pipeline and menus
- Cursor and peripheral control
- Beat-’em-up combat / linear combat
- Platforming
- Continuous vehicle control
- Top-down navigation
- Metroidvania navigation
- Graph navigation (overworld / cave / stage graphs; NES Zelda family)
- Fighting-game policies
- RPG and dialogue
- Tactical planning
- Simulation and scheduling
- Procedural adaptation

## Automation class

| Class | Meaning |
|-------|---------|
| Replay | Playback of recorded inputs |
| RAM script | Hand-authored policy using game-specific RAM |
| Hybrid vision | Mix of vision and limited internals |
| Autonomous discovery | Agent discovers map/RAM without handcrafted scripts |
| Unseen-game generalization | Policy adapts to a game not seen in development |

## Project states

```text
planned
scaffolded
bootable
instrumented
segmenting
route-building
continuous-candidate
verified
blocked
archived
```

## Continuous clear vocabulary

| Term | Meaning |
|------|---------|
| Segment clear | Completes from a development checkpoint |
| Natural-entry clear | Completes from the state produced by the real predecessor |
| Segmented ending | Ending reached with checkpoint cuts between segments |
| Continuous clear / dry run | One reset-to-ending session without state loads |
| Verified capture | Continuous clear plus published audiovisual artifact |
| Assisted continuous | Continuous clear with disclosed, counted RAM writes |
| Seed-robust clear | Clears S of T random seeds within budget (randomizer class) |
| Mod-robust clear | Same idea over a published set of edited ROMs (later) |

## Solver vocabulary

| Term | Meaning |
|------|---------|
| Solver | Reactive loop: observe → plan → invoke skill → replan |
| Skill library | Seed-invariant low-level controllers (rooms, bosses, menus) |
| Input tape | Precomputed input sequence; CI regression + imitation demo only |
| Item-logic graph | Transitions annotated with prerequisite item/capability sets |
| Logic-graph planner | Inventory-aware search over the item-logic graph (Layer 4) |
| Online world model | Runtime-discovered rooms/doors/items (Layer 3) |
| Observation bootstrap | Per-run RAM or vision semantics discovery (Layer 2) |
| Skill synthesis | RL / neuroevo / optimizers that create skills when the library lacks one |
| Flagship triangle | Super Metroid + ALTTP substrate + SMZ3 seed-abstract proof |
| Harness fixture | Great Waldo Search — deterministic pipeline M8 |

Solver layers L0–L4 are defined in [SOLVER_ARCHITECTURE.md](SOLVER_ARCHITECTURE.md)
and are orthogonal to M0–M8 completion gates.

## Directory names (authoritative)

| Directory | Title |
|-----------|-------|
| `SMW/` | Super Mario World (SNES) |
| `smb/` | Super Mario Bros. (NES) |
| `smb3/` | Super Mario Bros. 3 (NES) |
| `super_metroid/` | Super Metroid |
| `harvest/` | Harvest Moon |
| `alttp/` | The Legend of Zelda: A Link to the Past (Zelda 3) |
| `sm_rando/` | Super Metroid Randomizer (single-game) |
| `alttp_rando/` | ALTTP Randomizer (single-game) |
| `smz3/` | SMZ3 Super Metroid + ALTTP combined randomizer |
| `tmnt_i/` … `tmnt_iv/` | Teenage Mutant Ninja Turtles I–IV |
| `zelda_i/` / `zelda_ii/` | The Legend of Zelda / Zelda II (NES) |
| `great_waldo_search/` | The Great Waldo Search |
| `retro_harness/` | Shared emulator harness + scripted-completion helpers |

Stale names that must not appear as live workspace paths:

- `super_metroid_rl/` → use `super_metroid/`
- `super_mario_bros/` → use `SMW/` for Super Mario World; use `smb/` for NES SMB
- `snes_oneshot/` → folded into `retro_harness/` (see `docs/REPO_HYGIENE.md`)
