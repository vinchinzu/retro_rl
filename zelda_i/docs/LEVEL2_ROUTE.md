# Level 2 route — The Moon (overworld approach)

Planning source:
[Zelda Dungeon — Level 2: The Moon](https://www.zeldadungeon.net/the-legend-of-zelda-walkthrough/level-2-the-moon/).
Walkthrough directions from the start screen (right 4, up 2, right 2, up, left,
up) correlate to overworld screen **0x3C** as the Moon door. Claims below that
are emulator-verified are marked; walkthrough-only claims stay labeled.

## Post-Triforce return (verified)

Collecting shard 1 sets `ADDR_TRIFORCE & 0x01` and enters **mode 18** (fanfare).
After ~535 idle frames the engine transitions (modes 2→3→4) and places Link on
**overworld screen 0x37** at ~(112, 125) around frame **704**. This is
engine-driven, not a save-state warp.

- Live settle works from `Level1HeartCollected` → triforce controller → idle.
- Reloading `Level1Complete.state` mid-fanfare can freeze mode 18; prefer
  `Level1ExitOverworld.state` (captured after settle) or a live settle.

## Verified walk prefix (0x37 → 0x4A)

```text
0x37 ─E@y≈140─► 0x38 ─S@x≈120─► 0x48 ─S@x≈112─► 0x58
  ─E@y148–162─► 0x59 ─N@x≈112─► 0x49 ─E@y≈141─► 0x4A
```

Stop predicate: `level2_path_prefix_success` — overworld play, screen 0x4A,
sword ≥ 1, triforce & 0x01.

Probe extension (not yet controller-stable): `0x4A→0x4B→0x5B` then bush-east.

## Traps

| Trap | Detail |
|------|--------|
| 0x79 rocky dead-end | Enterable from 0x78 east@y≈180; **no east exit**. Do not use naive “right four from start”. |
| 0x37 east lane | Only **y≈140** exits east; y≈125 re-enters Level 1. |
| 0x5A | High damage corridor; prefix goes **north via 0x49** instead. |
| Health | Prefix arrives on 0x4A with ~empty hearts; further screens need combat/heal work. |

## Planned suffix (walkthrough → 0x3C)

Not yet 2/2. Intended continuation:

```text
0x4A ─E─► 0x4B ─S─► 0x5B ─E bush─► 0x5C ─E─► 0x5D
  ─N─► 0x4D ─W─► 0x4C ─N─► 0x3C (Moon door)
```

**Blocker:** overworld health management past 0x4A.

## Controllers / runner

- Hop geometry: `overworld.LEVEL2_PATH_HOPS` / `LEVEL2_PATH_SCREENS` (single source)
- Shared movement helpers: `nav_common` (swing, stuck, edge recovery, align)
- `level2_overworld.PostTriforceSettleController`
- `level2_overworld.OverworldToLevel2Controller` (default stop 0x4A)
- `scripts/run_to_level2_prefix.py`

```bash
uv run python zelda_i/scripts/run_to_level2_prefix.py --trials 2
uv run python zelda_i/scripts/run_to_level2_prefix.py --from-heart --trials 2
```
