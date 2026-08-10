# ALTTP Rando — Status

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M1 |
| Best verified result | JP FirstPlay → uncle fighter sword (`house_to_uncle` natural_entry) |
| Last verification | 2026-08-09 |
| Runtime class | Bronze |
| Intervention class | Clean |

## Role

**Single-game ALTTP randomizer** — simpler solver ground than SMZ3 (no SM
portals). Build item-logic + dungeon/OW skill hooks + seed-robust spine here,
then extend to combined SMZ3. Vanilla skills live in `snes/alttp/`.

Program stack: `docs/SOLVER_ARCHITECTURE.md`.

## Verified boot (M1)

| Field | Value |
|-------|-------|
| ROM | `roms/zelda3_jp.sfc` (JP 1.0, xxh32 `0x8AC8FD15`) |
| Method | `alttp.startup` (name entry + load; mash fallback present) |
| Module | `0x07` (indoor) |
| Room | `0x04` Link's House |
| Control | `has_control` true |
| State | `custom_integrations/ALTTPRando-Snes/FirstPlay.state` |

## Checklist

| Item | State |
|------|--------|
| Package `alttp_rando/` | done |
| JP 1.0 ROM wiring | done (`setup_rom`) |
| Seed package schema | done (offline fixture + `demo_seed`) |
| Early logic graph | done (opening → Eastern tip; `house_to_uncle` natural_entry) |
| Play/record spine | done (`./play` + MP4/JSON + F5) |
| FirstPlay boot | **done** (M1) |
| House → uncle skill bind | **done** (natural_entry; vanilla `alttp` opening skills) |
| Multi-seed opening S/T dry-run | **done** (fixture S=3/T=3 claimable; `substrate=vanilla`) |
| Patched seed ROM integration | open |
| Seed generator (ALTTPR / API) | open |

## House → uncle (natural_entry)

| Field | Value |
|-------|-------|
| Edge | `house_to_uncle` (`z3_links_house` → `z3_uncle_sword`) |
| Skill | `z3.house_to_uncle.vanilla_opening` → `play_house_to_uncle` |
| Predecessor | `FirstPlay` (M1 Link's House control, JP 1.0) |
| Composition | wake + lamp + house exit + OW to castle + `castle_to_sword` |
| Evidence | `recordings/house_to_uncle.json` + `.evidence.json` |
| Intervention | Clean (0 progression writes; 1 predecessor state load) |
| Last verification | 2026-08-09 |

## Multi-seed opening tip (S/T dry-run)

| Field | Value |
|-------|-------|
| Edge / goal | `house_to_uncle` |
| Seeds | fixture `1337` / `1338` / `1339` |
| S/T | **3/3** claimable (threshold 2) |
| Mode | dry (audited synthetic); live path wired, needs ROM + FirstPlay |
| Substrate | **vanilla** JP 1.0 FirstPlay (honest label; not shuffled ROMs) |
| Seed source | fixture packages (`alttp_rando.fixture`) |
| Spoiler oracle | false (seed-agnostic uncle-sword placement) |
| Intervention | Clean |
| Skill | `z3.house_to_uncle.vanilla_opening` |
| Consumer | `alttp_rando.opening_tip_campaign` → `SeedCampaignRunner` |
| Published report | `docs/opening_tip_seed_campaign_dry.json` |
| CLI | `uv run python -m alttp_rando.scripts.run_opening_tip_campaign --mode dry --publish-docs` |
| Last verification | 2026-08-09 |

## Next

1. ALTTPR / patch fixture seed → same FirstPlay path (unlock live multi-seed on real patches).
2. Bind next early-graph edges (`uncle_to_yard`, …) as skills clear.
3. Live multi-seed opening tip once generator/patch is wired (still fail-closed INFRA without ROM).
