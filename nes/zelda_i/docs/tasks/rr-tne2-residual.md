# Residual — rr-tne2 L6 Survival handoff

**Status:** Recovered power-on tip is cleared L6 `0x29` west leftover `(56,157)`
after `--through level6-clear29` 1/1 (`l6_clear29_west_recompose`). Fresh
`--through level6-south29` occupancy-DOWN from that leftover is **red 1/1**
(`l6_south29_west_recompose`). East-leftover south29 remains **BLOCKED 4/4**.
Bead `rr-tne2` stays open. Do not STATUS-promote.

## Recovered prefix (unchanged through south19)

Prefix through skip-Map `0x09` clear was already 1/1. Stairs09 east-of-block
through south19 still 1/1.

| through | tag | leftover | keys |
|---------|-----|----------|------|
| level6-stairs09 | l6_stairs09_recompose | mode 9 `0x75` `(208,93)` | 3 |
| level6-rod | l6_rod_recompose | mode 9 `0x75` `(136,141)` Rod=1 | 3 |
| level6-exit75 | l6_exit75_recompose | play `0x09` `(192,141)` | 3 |
| level6-south09 | l6_south09_recompose | play `0x19` `(120,77)` | 3 |
| level6-south19 | l6_south19_recompose | dark play `0x29` `(120,77)` | 3→2 |
| level6-clear29 | l6_clear29_west_recompose | play `0x29` `(56,157)` | 2→3 |

Census/inventory still match: TF `0x1F`, bombs=8, Bow=1, Rod=1, map `0x0A`,
health `0x66` lo==hi, deaths/state-load/progression/capacity writes 0. Key
deficit is still one versus historical; do not top up. East `0x29` sealed
(`open_doorway_mask` 12). PNG shows Link west of the center block at the SW
face. Do not rerun the east leftover `(184,144)`.

## clear29 west leftover — 1/1

`--through level6-clear29 --tag l6_clear29_west_recompose` 1/1, hop 2,194f,
end 206,889f. West-aisle patrol + LEFT peel; fight only x<64. Final play
`0x29` `(56,157)` tile 244 at hop start of south29, keys=3, bombs=8, Bow=1,
Rod=1, TF=`0x1F`, health `0x66` lo==hi, deaths/state-load/progression/capacity
writes 0, `status_claim=false`. Occupancy 449 misses / 13 blocked during
clear. Do not retouch this cleanup.

## south29 — east leftover BLOCKED 4/4 (do not retry)

All four east-leftover trials `--through level6-south29 --tag l6_south29_recompose`
0/1 from leftover `(184,144)`. Do not retry any of these, and do not retry
the prior-sitting RIGHT+DOWN clip from leftover `(55,133)`.

| # | policy | leftover / reason | what it killed |
|---|--------|-------------------|----------------|
| prior | RIGHT+DOWN clip, occupancy x=120 @ y=141 | `(184,144)` `south_up_halt` 4000f misses 0 | clip assumed west leftover |
| 1 | LEFT on y=144 to x=120 | `(184,141)` `south_up_halt`; `miss_f2_LEFT_184_142` then f3/f4 at `(184,141)` tile 117 misses 3 | LEFT is the center block (tile 244). Wall-slide 144→141 |
| 2 | DOWN clip to south band y=181 | `(184,165)` `south_peel` 4000f tile 244 misses 0 | east aisle **does** walk 144→165 (~1px/f) then **boxes**. Clip never LEFT |
| 3 | occupancy dest `(120,189)` | `(184,149)` `south_up_halt`; `miss_f3_DOWN_184_147` (2px step), `miss_f6_DOWN_184_149`, `miss_f7_LEFT_184_149` tile 116 misses 3 | occupancy false-misses the same DOWN the clip completed to 165 |
| 4 | DOWN to measured box y=165, then LEFT | `(184,165)` `south_up_halt`; `miss_f19_LEFT_184_165`, tile 244, misses 1 | DOWN works 144→165; y=165 is **not** a west aisle and the first LEFT is solid |

## south29 — west leftover red 1/1 (do not retry this policy)

Parked `SOUTH29_SPEC` is occupancy to `(120,189)` / never UP from leftover
`(56,157)`. Live `--through level6-south29 --tag l6_south29_west_recompose`
0/1, hop 4000f timeout, end 210,889f. Final still play `0x29` `(57,157)`,
tile 119, keys=3, bombs=8, Bow=1, Rod=1, TF=`0x1F`, health `0x66` lo==hi,
deaths/state-load/progression/capacity writes 0, `status_claim=false`.

| # | policy | leftover / reason | what it killed |
|---|--------|-------------------|----------------|
| west 1 | occupancy dest `(120,189)`, forbid UP | `(57,157)` `south_up_halt` 4000f; `miss_f2_DOWN_56_157`, `miss_f4_DOWN_56_157`, `miss_f6_RIGHT_59_157`, `miss_f7_DOWN_57_157` tile 119 misses 4 | leftover is the **SW face** of the center block. DOWN and RIGHT are solid. Occupancy then wanted UP around the block; `forbid_up` stood |

Do not rerun occupancy-DOWN from `(56,157)`. Do not restore east-box LEFT,
occupancy-DOWN from `(184,144)`, or the `(55,133)` RIGHT+DOWN clip. Next open
geometry: peel **UP** the west aisle from `(56,157)` (the remaining live
cardinal) and form a south-door policy from that measured north leftover.

```bash
# After changing only SOUTH29_SPEC to UP-peel from leftover (56,157):
QT_QPA_PLATFORM=offscreen uv run python \
  nes/zelda_i/scripts/run_survival_spine.py \
  --through level6-south29 --no-video --trials 1 \
  --tag l6_south29_west_up
```

## Key deficit (still open; do not top-up)

Ended L6 compass keys=4 vs historical 5. Gap already at L4 `0x40`. clear58
had no natural key. room09 KEY-UP spent 4→3. south19 KEY-DOWN spent 3→2
(historical 4→3). clear29 floor key 2→3 (historical 3→4). Do not hide it
with a key top-up.

## Remaining L6 after south29 (not this sitting)

Phase 1: south29 (west leftover red 1/1; east leftover BLOCKED 4/4) →
settle39 → clear39 → east39 → settle3a → clear3a.

Phase 2: un-dedicate `stairs3a-warp` / `cellar08` / `south1d` / `west2d` /
`north2c` so they compose from clear `0x3A`. Do not start Phase 2 until
clear3a is green.

Phase 3: natural heart in `0x1C`, north `0x0C`, TF `0x20`.

Phase 4–5: measured `set_state` audit, then `--through level6`.

## Parallel L7/L8/L9/OW (fixture-live only)

Wave A handoffs: `docs/tasks/{l7,l8,l9,ow}-handoff.md`. Public `--through`
names unchanged. Do not attach `continue_level7/8/9_spine` until L6 leave is
measured. `route_eligible=false`. Post-dungeon OW leftovers remain
UNMEASURED (do not assume L6 leave is OW `0x22`).

## Non-claims

- Did not STATUS-promote or overwrite Clean M5.
- Did not close `rr-tne2` or reach Gohma / TF `0x20`.
- Did not poke doors, TF, bow, Rod, Map, Whistle, or capacity.
- Did not retouch maze-west, 0x40, stairs09, or the cellar walk.
- Did not retry west-aisle stairs09.
- Did not retry the `(55,133)` RIGHT+DOWN clip or any east-leftover south29.
- Did not rerun occupancy-DOWN from `(56,157)` after the red.
- Did not push.
