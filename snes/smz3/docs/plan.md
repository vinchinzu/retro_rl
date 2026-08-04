# SMZ3 — Plan

## North star

Roll an SMZ3 seed and have **two bots race** it end-to-end, with **video** as a
first-class quest artifact. That requires strong vanilla Super Metroid and
ALttP primitives first; this folder is the randomizer + race layer on top.

## Architecture

```
seed (samus.link / optional local CLI)
  → seed package (meta, spoiler, patch)
  → combo ROM (SM + Z3 + zsm.ips + seed patch)
  → stable-retro SMZ3-Snes
  → world detect → dispatch
        ├─ super_metroid.*  (when in SM)
        └─ alttp.*          (when in Z3)
  → room_timeout (3× baseline → game over)
  → race harness (2 emulators, same ROM, videos)
```

## Phases

1. **Foundation** — seed tooling, ROM build, timeout rule, docs. *(done)*
2. **Boot (M1)** — power-on → RAM-verified SM controllable. *(done)*
3. **Early rooms (M2)** — Landing → Parlor + portal catalog + timeout. *(done)*
4. **Portal fire (M3a)** — Parlor red door `$8976` → cave `$0122` residue. *(done; missile assist)*
5. **Portal settle (M3b)** — finish `transition_to_zelda` under stable-retro (module `$0F` → controllable Link). *(done; JP 1.0 + settle wait)*
6. **Z3 outdoor (M3c)** — Fortune Teller → overworld → Link's House without sword. *(done; video)*
7. **Link's House chest (M3d)** — enter house + open chest (map-driven). *(done; heart container on 1337)*
8. **Single-bot play** — longer SM segment and/or Z3 segment + video.
9. **Baselines** — room standard times from vanilla timers / human refs.
10. **Race** — two bots, same seed, parallel sessions, video pair + report.
11. **Smarter stop** — replace 3× room rule with progress / softlock metrics.

## Stop rule (provisional)

`dwell_frames > standard_frames * 3` in a settled room → game over for that
bot. Documented in `room_timeout.py`. Intended only until better stuck
detection exists.

## Reuse policy

- Do not copy `super_metroid/` or `alttp/` trees.
- Import parsers, timers, and controllers; wrap only combo-specific seams.
- Item logic / spoilers come from the seed package, not reimplemented logic.

## Dependencies

| Need | Source |
|------|--------|
| Super Metroid ROM | `roms/SuperMetroid.sfc` |
| ALttP JP 1.0 ROM | `roms/zelda3_jp.sfc` (not USA `zelda3.sfc`) |
| Base combo IPS | `refs/zsm.ips.gz` (tewtal SMZ3Randomizer) |
| Seed generation | samus.link API (`pyz3r`) |
| Optional offline CLI | clone `tewtal/SMZ3Randomizer` (dotnet) |

## Progression model (combinatorial readiness)

Early legs are composed via `route_graph.py` + `quest.run_early_quest`:

- Nodes: rooms / OW screens / portal / chest
- Edges: `requires` (e.g. missiles for parlor red door) + verification provenance
- Controllers stay in `*_route.py`; do **not** add a new seed-specific route
  file per leg without registering the edge on the graph
- Assists are explicit (`assist.py`), not buried RAM writes without a contract
- Shared control loops live in `control.py` (text / death / hold-up)

Full spoiler-driven auto-routing remains deferred; the graph is the seam it
will plug into.

## Out of scope (for now)

- Multiworld multi-player SMZ3
- Cas' Randomizer tracker integration
- Full spoiler-driven auto-routing
