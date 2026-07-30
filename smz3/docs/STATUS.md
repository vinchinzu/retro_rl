# SMZ3 — Status

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M2 |
| Best verified result | Power-on → Landing Site → Parlor (`0x92FD`) controllable with room-timeout watchdog; portal catalog + partial Z3 residue (`cave $0122`) |
| Last verification | 2026-07-29 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Item | State |
|------|--------|
| Directory `smz3/` | done |
| Integration `SMZ3-Snes` | boots |
| Seed + combo ROM builder | done (test seed 1337) |
| Room timeout 3× | unit-tested + wired in early segment |
| Power-on → SM controllable (M1) | done |
| World detect WRAM heuristic | done |
| Multi-room natural segment | **done** — Landing Site → Parlor |
| Early portal catalog | done (`portals.py` / `docs/EARLY_ROOMS.md`) |
| Clean Z3 controllable via portal | not yet (map path needs missiles + red door; dev place hangs at module `$0F`) |
| Dual-bot race + video | scaffold only |

## Current milestone

### M2 — first rooms + portal map (this slice)

- Natural movement: ship Landing Site → bottom-left blue door → Parlor
- `early_route.run_landing_to_parlor` + room timeout baselines
- Fixed portals documented (Crateria Map `$8976` → Fortune Teller `$0122`)
- Probe: `scripts/probe_early_rooms.py`

## Next

1. Parlor descent → red door (missiles) → Pre-Map → Map → **settled** Z3 Link.
2. One-bot session past Parlor (morph path and/or portal) with video.
3. Dual-bot race harness on the same seed.
