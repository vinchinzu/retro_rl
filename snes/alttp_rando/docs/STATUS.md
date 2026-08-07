# ALTTP Rando — Status

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M1 |
| Best verified result | JP 1.0 power-on → Link's House controllable (`FirstPlay.state`) |
| Last verification | 2026-08-06 |
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
| Early logic graph | done (opening → Eastern tip, planned) |
| Play/record spine | done (`./play` + MP4/JSON + F5) |
| FirstPlay boot | **done** (M1) |
| Patched seed ROM integration | open |
| Seed generator (ALTTPR / API) | open |
| Multi-seed S/T dry-run | open |

## Next

1. Bind opening graph edges to `alttp` natural-entry skills from FirstPlay.
2. ALTTPR / patch fixture seed → same FirstPlay path.
3. Multi-seed report via shared seed-robust harness (`rr-gbd.11`).
