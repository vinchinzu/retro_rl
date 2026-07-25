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
| Player velocity X/Y | `0x0B42/0x0B2E i16` | Source-confirmed | `variables.h`, parsed for navigation |
| Player pose | `0x0A1C u16` | Route-verified | Ceres ledge alignment and room navigation |
| Grounded/control subflags | — | — | pending; coarse control currently uses game/door state |
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
| Global event bytes | `0xD820..0xD827 u8[8]` | Source-confirmed | `events_that_happened`; live milestone mapping pending |
| Per-area boss bits | `0xD828..0xD82F u8[8]` | Source-confirmed | `boss_bits_for_area`; parsed Brinstar byte stayed zero across the natural Spore clear, so live mapping remains pending and is not acceptance evidence |
| Morph Ball ownership | collected items bit `0x0004` | Route-verified | natural pickup and item banner in acceptance video |
| Bomb ownership | collected items bit `0x1000` | Route-verified | natural pickup, `BOMB` banner, Torizo activation, and final `0x1004` mask |
| Enemy count/kills | `0x0E4E/0x0E50 u16` | Route-verified | sampled across early rooms for combat-aware progress vectors |
| Enemy slot 0 X/Y/HP | `0x0F7A/0x0F7E/0x0F8C u16` | Route-verified | Bomb Torizo reaches 800→0; Spore Spawn reaches 960→0 before natural exits |
| Enemy slot 0 spritemap | `0x0F8E u16` | Route-verified | Spore Spawn mouth-open states `0xEEAF/0xEEC1/0xEED3/0xEEE5` gate controller fire |
| Ceres timer type | `0x0943 u8` | Route-verified | type `3` marks the natural evacuation countdown |
| Escape timer | `0x0945..0x0947 u8` | Route-verified | observed during Ceres; Tourian semantics pending |

The typed parser is in `ram.py`. Current ammo and capacity are intentionally
separate. The resource controller may write current energy only up to natural
max energy on Zebes, and current ammo only up to a nonzero natural capacity
during ordinary gameplay. Progression, capacity, area, room, event, boss,
position, and timer fields are read-only.
