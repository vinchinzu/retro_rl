# Status — Metroid (NES)

## Program gate

| Field | Value |
|-------|-------|
| Current maturity | M5 |
| Best verified result | Power-on → Morph Ball → three east doors → west-shaft upper platform |
| Last verification | 2026-07-28 |
| Runtime class | Bronze |
| Intervention class | Clean |

| Field | Value |
|-------|-------|
| Status | **chained first-missiles prefix** (boot → west shaft) |
| Integration | `Metroid-Nes` |
| ROM zip | `roms/Nintendo/NES/Metroid.zip` |
| Ready frame (probe) | ~550–600 |
| Checkpoint | `Level1.state`, `AfterMorph.state` |
| Evidence | [first_missiles_natural.json](../recordings/first_missiles_natural.json), [first_missiles_after_morph.json](../recordings/first_missiles_after_morph.json), [morph_ball_natural.json](../recordings/morph_ball_natural.json) |

## Verified segments

| Segment | Entry | Result | Frames (typ.) | Evidence |
|---------|-------|--------|---------------|----------|
| Maru Mari | `Level1.state` (isolated) | equipment `$6878 & 0x10` | ~358 | `morph_ball_isolated.json` |
| Maru Mari | power-on boot (natural) | equipment `$6878 & 0x10` | ~358 (+boot) | `morph_ball_natural.json` |

Natural-entry and isolated both use the same `MorphBallController` (no RAM writes).

## In progress — first missiles

The route is naturally chained to the upper west-shaft platform, but the
missile pickup is not route-cleared:

| Piece | Status |
|-------|--------|
| Stop predicate `$687A > 0` | `is_missiles_obtained` in `ram.py` |
| Graph east cells (3,14)→(5,14) | continuous probe edges |
| Graph morph→missiles legs | planned |
| `FirstMissilesController` | natural verified prefix; terminal `FRONTIER` |
| Morph → real start return | **verified** via low morph tunnel; no state load |
| East blue doors | **verified**: all three transitions through map x=11 |
| Long morph tunnel | **verified** with 14 energy at exit |
| West shaft | first three stable landings verified at map (11,13), x≈106/y=225 |
| Missile pickup | **blocked** above current west-shaft platform |

## Done

- Directory layout and NES integration
- `scripts/setup_rom.py` / `scripts/boot_probe.py`
- **M2 instrumentation** — mode, map cell, Samus room x/y, equipment WRAM, missiles, tanks
- **Shared graph core** — `adventure_common` (`RouteGraph`, milestones, capability BFS); second consumer after `zelda_i`
- **Early Brinstar graph** — start (3,14) → morph (1,14); east probe to (5,14)
- **M3–M5 morph segment** — west corridor climb, pedestal collect
- Natural first-missiles prefix through the third west-shaft platform

## Not done

- Upper west shaft → bridge → east shaft → first missiles — **active blocker**
- Bombs / Long Beam / broader Brinstar graph
- Continuous multi-item dry run (M6–M8)

## Next

1. Clear the enemy-populated upper west shaft from the verified map (11,13)
   x≈106/y=225 natural frontier.
2. Cross the bridge, descend the east shaft, and stop on `$687A > 0`.
3. Promote the existing power-on → morph → west-shaft prefix to a full
   no-state-load missiles result.
