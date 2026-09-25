# Plan: SMZ3

Verified facts: [STATUS.md](STATUS.md). Room notes: [EARLY_ROOMS.md](EARLY_ROOMS.md).

## North star

Roll an SMZ3 seed and have two bots race it end to end, with video as a
quest artifact. Vanilla Super Metroid and ALttP primitives stay in those
trees. This folder is the randomizer and race layer.

Prove patterns on `sm_rando` / `alttp_rando` first when the logic is
simpler. Program stack: `docs/SOLVER_ARCHITECTURE.md`.

## Closed

Foundation, boot, Landing to Parlor, portal settle, Fortune Teller to
Link's House, the house chest on seed 1337, and the fixture dry
portal-to-house S/T (2026-08-09) are closed. Evidence: [STATUS.md](STATUS.md),
[EARLY_ROOMS.md](EARLY_ROOMS.md),
[portal_house_seed_campaign_dry.json](portal_house_seed_campaign_dry.json).
The live red door still uses missile assist. `PortalSettled` is a save
state, not a continuous power-on.

## Open

1. Longer single-bot SM or Z3 segment, with video, as a new graph edge.
2. Room standard times from vanilla timers or human refs.
3. Live multi-seed S/T once each fixture seed has a combo ROM. Spoiler
   oracle stays off for claimed solver runs. Not shuffled-seed robustness
   until a generator patch is wired per seed.
4. Shared item-logic planner (`retro_harness.adventure`) instead of a hand
   route per seed.
5. Race: two bots, same seed, parallel sessions, video pair plus report.
6. Replace the 3× room timeout with a progress or softlock metric.

## Stop rule (provisional)

`dwell_frames > standard_frames * 3` in a settled room ends that bot.
See `room_timeout.py`.

## Shape

```
seed (samus.link / optional local CLI)
  → seed package (meta, spoiler, patch)
  → combo ROM (SM + Z3 + zsm.ips + seed patch)
  → stable-retro SMZ3-Snes
  → world detect → dispatch
        ├─ super_metroid.*  (when in SM)
        └─ alttp.*          (when in Z3)
  → room_timeout
  → race harness (2 emulators, same ROM, videos)
```

Early legs go through `route_graph.py` and `quest.run_early_quest`.
Register a graph edge instead of adding a seed-specific route file.
Assists stay explicit in `assist.py`.

## Reuse

- Do not copy `super_metroid/` or `alttp/` trees.
- Import parsers, timers, and controllers. Wrap only combo-specific seams.
- Item logic comes from the seed package.

## Dependencies

| Need | Source |
|------|--------|
| Super Metroid ROM | `roms/SuperMetroid.sfc` |
| ALttP JP 1.0 ROM | `roms/zelda3_jp.sfc` (not USA `zelda3.sfc`) |
| Base combo IPS | `refs/zsm.ips.gz` |
| Seed generation | samus.link API (`pyz3r`) |
| Optional offline CLI | `tewtal/SMZ3Randomizer` (dotnet) |

## Out of scope

- Multiworld multi-player SMZ3
- Cas' Randomizer tracker integration
- Seed-robustness claimed from one seed, a fixed tape, or a fixture dry-run
