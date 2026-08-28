# RAM map — Super Metroid

ROM SHA-256:
`12b77c4bc9c1832cee8881244659065ee1d84c70c3d29e6eaf92e6798cc2ca72`.
Addresses are WRAM offsets.

| Confidence | Meaning |
|------------|---------|
| Source-confirmed | Named in the local reverse-engineered source |
| Differential | Isolated by controlled before/after behavior |
| Route-verified | Correct throughout a continuous natural-entry route |
| Freeze-tested | A permitted local write produced only the predicted effect |

## Core state

| Meaning | Address/type | Confidence | Probe/evidence |
|---------|--------------|------------|----------------|
| Game/menu/control mode | `0x0998 u16` | Route-verified | Observed reset/menu, gameplay `8`, transitions `9–11`, Ceres success `32–34`; enum cross-checked with `ida_types.h` |
| Area index | `0x079F u16` | Source-confirmed | `area_index` in `variables.h`; parsed into named areas |
| Room pointer/ID | `0x079B u16` | Route-verified | 40 continuous transitions from Ceres through the post-Spore room match typed edges |
| Door/elevator transition | `0x0797 u16` | Route-verified | Becomes nonzero across doors/elevator and guards ordinary gameplay |
| Door direction | `0x0791 u16` | Route-verified | Direction changes across accepted transitions |
| Player X/Y | `0x0AF6/0x0AFA u16` | Route-verified | Movement and Landing Site ship-settle predicate |
| Player X/Y subpixel | `0x0AF8/0x0AFC u16` | Source-confirmed | Door-clip / TAS subpixel windows (`door_kinematics`) |
| Player velocity X/Y | `0x0B42/0x0B2E` | Source-confirmed | Horizontal speed + vertical speed (pixels); subpixels at `0x0B44` / `0x0B2C` |
| Horizontal momentum X | `0x0B46/0x0B48` | Source-confirmed | Separate from speed; mockball / dash carry across doors |
| Speed-booster counter | `0x0B3E` hi byte | Source-confirmed | Echo / blue-suit charge (0–4+); `speed_boosting` when ≥4 |
| Speed-check flag | `0x0B3C u16` | Source-confirmed | Gates temp→permanent blue suit conversion |
| Vertical direction | `0x0B36 u16` | Source-confirmed | 0 ground, 1 up, 2 down. Moonfall keeps this at 0 while airborne (uncapped fall). |
| Moonwalk option | `0x09E4 u16` | Source-confirmed | Special Setting Mode copy. 0 = off (new-file default), 1 = on. Required for moonwalk / moonfall. PJBoy RAM map. Poke via `ram.set_moonwalk`; not a progression write. |
| Facing / movement type | `0x0A1E u8` / `0x0A1F u8` | Source-confirmed | Facing 4=left, 8=right |
| Shine-spark timer | `0x0A68 u16` | Source-confirmed | Shared with crystal flash |
| Door definition pointer | `0x078D u16` | Source-confirmed | Active DDB; leave/entry reports + door-warp |
| Player pose | `0x0A1C u16` | Route-verified | Ceres ledge alignment and room navigation |
| Grounded/control subflags | — | — | pending; coarse control currently uses game/door state |

Door leave/entry kinematics (speed + position + pose through transitions) are
first-class on `SuperMetroidState` and `door_kinematics.DoorKinematics`.
In-room takeoff windows reuse the same `DoorKinematicsRequirement` matcher
via `takeoff.TakeoffWindow` (x / x_sub / `|momentum|` / facing). Continuous
`RouteSession` attaches leave/entry snapshots on each
`ObservedTransition`. See `door_kinematics.py`, `takeoff.py`, and ARCHITECTURE.
| Current/max energy | `0x09C2/0x09C4 u16` | Route-verified | natural Ceres damage; Terminator Energy Tank changes max `99 → 199`; assist writes current only on Zebes |
| Reserve current/max | `0x09D6/0x09D4 u16` | Source-confirmed | zero throughout accepted prefix |
| Death/game over | `0x0998 = 19–26,29,35–37` | Source-confirmed | source enum; no death in accepted prefix |
| Ending/credits | `0x0998 = 38–39` | Source-confirmed | source enum; full-run live evidence pending |

## Inventory, resources, and progress

| Meaning | Address/type | Confidence | Probe/evidence |
|---------|--------------|------------|----------------|
| Missile current/capacity | `0x09C6/0x09C8 u16` | Route-verified | capacity changes naturally `0 → 5 → 10`; assist writes current only |
| Super Missile current/capacity | `0x09CA/0x09CC u16` | Route-verified | locked at `0/0` through post-Torizo Parlor |
| Power Bomb current/capacity | `0x09CE/0x09D0 u16` | Route-verified | locked at `0/0` through post-Torizo Parlor |
| Selected weapon/item | `0x09D2 u16` | Route-verified | beam `0`, Missiles `1`; natural Select input normalizes the Pit replay boundary |
| Equipped/collected items | `0x09A2/0x09A4 u16` | Route-verified | both change naturally `0 → 0x0004 → 0x1004` at Morph/Bombs |
| Equipped/collected beams | `0x09A6/0x09A8 u16` | Route-verified | remain zero in accepted prefix |
| Global event bytes | `0xD820..0xD827 u8[8]` | Source-confirmed | `events_that_happened`; read via bank `$7E` WRAM block (`read_bank7e_wram`), not raw `env.get_ram()` high offsets. Event `0x0E` = Mother Brain defeated / escape started |
| Per-area boss bits | `0xD828..0xD82F u8[8]` | Source-confirmed | `boss_bits_for_area`; same `$7E` read path as events. Tourian byte bit `0x02` set on MB death |
| Door definition pointer | `0x078D u16` | Source-confirmed | `door_def_ptr`; write + game state `9` door-warps for development teleports |
| Samus invincibility timer | `0x18A8 u16` | Source-confirmed | development spray-and-pray i-frames |
| Morph Ball ownership | collected items bit `0x0004` | Route-verified | natural pickup and item banner in acceptance video |
| Bomb ownership | collected items bit `0x1000` | Route-verified | natural pickup, `BOMB` banner, Torizo activation, and final `0x1004` mask |
| Enemy count/kills | `0x0E4E/0x0E50 u16` | Route-verified | sampled across early rooms for combat-aware progress vectors |
| Enemy slot 0 X/Y/HP | `0x0F7A/0x0F7E/0x0F8C u16` | Route-verified | Bomb Torizo reaches 800→0; Spore Spawn reaches 960→0 before natural exits |
| Enemy slot 0 spritemap | `0x0F8E u16` | Route-verified | Spore Spawn mouth-open states `0xEE79/0xEE8B/0xEE9D/0xEEAF/0xEEC1/0xEED3/0xEEE5` plus fully-open holds `0xEF3D/0xEF4F/0xEF61` gate controller fire |
| Ceres timer type | `0x0943 u8` | Route-verified | type `3` marks the natural evacuation countdown |
| Escape timer | `0x0945..0x0947 u8` | Route-verified | observed during Ceres; Tourian semantics pending |

The typed parser is in `ram.py`. Current ammo and capacity are intentionally
separate. The resource controller may write current energy only up to natural
max energy on Zebes, and current ammo only up to a nonzero natural capacity
during ordinary gameplay. Progression, capacity, area, room, event, boss,
position, and timer fields are read-only.
