# Status: SMZ3

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M2 |
| Best verified result | Landing Site to Parlor on the combo ROM. `PortalSettled` save state to Link's House chest on seed 1337 (heart container). Live red door uses missile assist. Fixture dry S/T is 3/3, not shuffled ROMs. |
| Last verification | 2026-08-09 |
| Runtime class | Bronze |
| Intervention class | Missile **assist** on the live red door. Z3 outdoor is flee-only (no sword). Dry S/T is Clean synthetic. Not a continuous Clean power-on. |

The matrix sticker is M2. `PortalSettled` is a save state. Do not read the
house chest or the dry campaign as an M3 or continuous Clean clear.

| Item | State |
|------|--------|
| Directory `smz3/` | done |
| Integration `SMZ3-Snes` | boots |
| Seed + combo ROM builder | done (test seed 1337); hash-validates vanilla ROMs |
| Vanilla Z3 JP 1.0 at `roms/zelda3_jp.sfc` | OK (`0x8AC8FD15`) |
| Super Metroid vanilla hash | OK (`0xCADB4883`) |
| Room timeout 3× | unit-tested + wired in the early segment |
| Power-on to SM controllable | done (SM side) |
| World detect WRAM heuristic | done |
| Landing Site to Parlor | done (natural segment) |
| Parlor red door to portal start | done (`portal_route.py`, missile assist) |
| Z3 controllable via portal | done on JP 1.0 after the settle wait (module `$09`, OW `$35`) |
| Fortune Teller to Link's House, then chest | done from `PortalSettled` (no sword; heart container on 1337) |
| Multi-seed portal-to-house dry-run | done 2026-08-09 on fixtures (below) |
| Dual-bot race + video | scaffold only |

## Solver note

SMZ3 is the combined-randomizer proof target. Single-seed parlor-to-house
is development evidence. The dry report is fixture substrate, not
shuffled-seed robustness. Program stack: `docs/SOLVER_ARCHITECTURE.md`.

## Multi-seed portal to house (rr-gbd.13)

| Field | Value |
|-------|-------|
| Goal | `portal_to_house` (`PortalSettled` to Link's House chest) |
| Seeds (T) | fixture `1337` / `1338` / `1339` |
| Threshold (S) | 2 of 3 (actual **3/3**) |
| Claimable | yes (no INFRA_ERROR) |
| Spoiler oracle | false (layout-fixed outdoor + morph-original settings) |
| Substrate | fixture (offline packages; not shuffled combo ROMs) |
| Intervention (dry) | Clean synthetic envs |
| Live note | resource assist `missile_red_door` until natural morph to missiles |
| Reports | `docs/portal_house_seed_campaign_dry.json` (committed). Runtime copies under `recordings/`. |
| CLI | `uv run python snes/smz3/scripts/run_portal_house_campaign.py --mode dry --publish-docs` |

Room path, settle frames, and clip names: [EARLY_ROOMS.md](EARLY_ROOMS.md).
Code: `portal_route.py`, `outdoor_route.py`, `house_route.py`.

## Next

1. Drop missile assist once natural morph to missiles is on the combo path.
2. Live multi-seed portal-to-house (`--mode live`) once each fixture seed has a combo ROM.
3. Dual-bot race harness on one seed (`quest` + graph).
4. Longer Z3 outdoor / SM legs as new graph edges (uncle sword and the rest).
