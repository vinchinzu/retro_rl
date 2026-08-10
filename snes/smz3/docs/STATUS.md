# SMZ3 — Status

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M2 → M3 |
| Best verified result | PortalSettled → Link's House interior + chest open (heart container on seed 1337); multi-seed dry S/T claimable 3/3 |
| Last verification | 2026-08-09 |
| Runtime class | Bronze |
| Intervention class | Clean SM path + missile **assist** for red door; Z3 outdoor flee-only (no sword); dry S/T = Clean synthetic |

| Item | State |
|------|--------|
| Directory `smz3/` | done |
| Integration `SMZ3-Snes` | boots |
| Seed + combo ROM builder | done (test seed 1337); **hash-validates vanilla ROMs** |
| Vanilla Z3 JP 1.0 at `roms/zelda3_jp.sfc` | **OK** (`0x8AC8FD15`, densetsu dump) |
| Super Metroid vanilla hash | **OK** (`0xCADB4883`) |
| Room timeout 3× | unit-tested + wired in early segment |
| Power-on → SM controllable (M1) | done (SM side) |
| World detect WRAM heuristic | done |
| Multi-room natural segment | **done** — Landing Site → Parlor |
| Early portal catalog | done (`portals.py` / `docs/EARLY_ROOMS.md`) |
| Parlor → red door → portal start | **done** (`portal_route.py`, missile assist) |
| Clean Z3 controllable via portal | **done** — idle ~300f after `$0F` → module `$09` OW screen `$35` |
| Fortune Teller → Link's House (no sword) | **done** — `$35`→`$2D`→`$2C`, flee side-steps |
| Enter Link's House + open chest | **done** — map path west-ramp → door; chest @ vanilla XY |
| Multi-seed portal→house S/T dry-run | **done** 2026-08-09 — see below |
| Dual-bot race + video | scaffold only |

## Solver flagship note

SMZ3 is the combined-randomizer **proof target** for the program solver stack
(vanilla SM + ALTTP skills → shared planner/discovery → seed-abstract S/T).
Single-seed parlor→house is development evidence; the multi-seed dry report
below is **fixture-substrate** harness evidence (not shuffled-seed robustness).
See `docs/SOLVER_ARCHITECTURE.md` and `docs/BENCHMARK_SPEC.md` (seed-robustness).

## Multi-seed portal→house (rr-gbd.13)

| Field | Value |
|-------|-------|
| Goal | `portal_to_house` (PortalSettled → Link's House chest) |
| Seeds (T) | fixture `1337` / `1338` / `1339` |
| Threshold (S) | 2 of 3 (actual **3/3**) |
| Claimable | yes (no INFRA_ERROR) |
| Spoiler oracle | **false** (path is layout-fixed outdoor + morph-original settings) |
| Substrate | **fixture** (offline packages; not shuffled combo ROMs) |
| Intervention (dry) | Clean synthetic envs |
| Live note | resource assist `missile_red_door` until natural morph→missiles |
| Reports | `docs/portal_house_seed_campaign_dry.json` (committed); runtime `recordings/portal_house_seed_campaign.json` + classic projection |
| CLI | `uv run python snes/smz3/scripts/run_portal_house_campaign.py --mode dry --publish-docs` |
| Pattern | mirrors `sm_rando.early_tip_campaign` / `SeedCampaignRunner` |

Not shuffled-seed robustness until a rando generator/patch is wired per seed.

## Current milestone

### M3 — portal settle + first Z3 outdoor leg

- Natural SM path: Parlor red door `$8976` = first combo portal (no Pre-Map)
- Checkpoint: `PortalRedDoor` (still SM) via `probe_portal.py` / `play_portal.py`
- **Portal settle (2026-07-30):** with JP 1.0 combo, natural red-door portal
  reaches module `$0F` then settles after ~300 idle frames to module `$09`
  overworld screen `$35` (Fortune Teller exterior). Earlier probes stopped on
  first `$0F` and looked hung. `data.json` addresses rewritten to absolute
  `$7Exxxx` form (stable-retro SNES convention) with fuller SM+Z3 fields.
- **Z3 outdoor (2026-07-30):** `outdoor_route.py` from `PortalSettled` walks
  `$35`→`$2D`→`$2C` without sword (corridor X≈2704; UP+LEFT on `$2D`).
  Side-step flee only — no combat. Video:
  `recordings/fortune_to_links_house.mp4` (~14s). PNG:
  `recordings/m3_links_house_ow.png`.
- **Link's House chest (2026-07-30):** map-driven entry via Yaze entrance
  `(2224, 2800)` — west flank under-house Y≈2846 → door X → UP. Chest open
  at vanilla `(2491, 8632)` face UP + A; test seed grants **heart container**
  (max HP 24→32), flag `$0403`. Implementation: `house_route.py`.

## Architecture (combinatorial readiness)

- Shared control: `control.py` (hold / wait_z3_control / go_xy)
- Segment base: `segment.py`; assist contracts: `assist.py`
- Capability graph: `route_graph.py` (missiles gate red door)
- Composer: `quest.run_early_quest(stop=...)` for parlor → chest chain
- New legs register graph edges + controllers; avoid clone `*_route.py` sprawl

## Next

1. Drop missile assist once natural morph → missiles is on the combo path.
2. Live multi-seed portal→house (`--mode live`) once combo ROMs exist per fixture seed.
3. Dual-bot race harness on the same seed (use `quest` + graph).
4. Longer Z3 outdoor / SM legs with video (uncle sword, etc.) as new edges.
