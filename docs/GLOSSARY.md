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

## Directory names (authoritative)

| Directory | Title |
|-----------|-------|
| `SMW/` | Super Mario World |
| `super_metroid/` | Super Metroid |
| `harvest/` | Harvest Moon |
| `tmnt_iv/` | Teenage Mutant Ninja Turtles IV |
| `great_waldo_search/` | The Great Waldo Search |
| `snes_oneshot/` | Shared scripted-completion helpers (historical package name) |

Stale names that must not appear as live workspace paths:

- `super_metroid_rl/` → use `super_metroid/`
- `super_mario_bros/` → use `SMW/` for Super Mario World work in this repo
- `alttp/` → A Link to the Past opening-route workspace

## Package name note

`snes_oneshot` is the historical package name for shared scripted-completion
policies. Prefer human-facing terms: scripted completion, full-game automation,
continuous clear, reset-to-ending evaluation.
