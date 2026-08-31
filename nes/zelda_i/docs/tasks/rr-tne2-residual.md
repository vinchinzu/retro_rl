# Residual — rr-tne2 L6 Survival handoff

**Status:** Recovered power-on tip is still cleared L6 `0x29` leftover `(184,144)`
after `--through level6-clear29` 1/1. `--through level6-south29` is **BLOCKED
4/4** from this east leftover (plus the prior-sitting RIGHT+DOWN 0/1). Bead
`rr-tne2` stays open. Do not STATUS-promote.

## Recovered prefix (unchanged this sitting)

Prefix through skip-Map `0x09` clear was already 1/1. Stairs09 east-of-block
through clear29 still 1/1 from the previous sitting.

| through | tag | leftover | keys |
|---------|-----|----------|------|
| level6-stairs09 | l6_stairs09_recompose | mode 9 `0x75` `(208,93)` | 3 |
| level6-rod | l6_rod_recompose | mode 9 `0x75` `(136,141)` Rod=1 | 3 |
| level6-exit75 | l6_exit75_recompose | play `0x09` `(192,141)` | 3 |
| level6-south09 | l6_south09_recompose | play `0x19` `(120,77)` | 3 |
| level6-south19 | l6_south19_recompose | dark play `0x29` `(120,77)` | 3→2 |
| level6-clear29 | l6_clear29_recompose | play `0x29` `(184,144)` | 2→3 |

Census/inventory still match: TF `0x1F`, bombs=8, Bow=1, Rod=1, map `0x0A`,
health `0x66` lo==hi, deaths/state-load/progression/capacity writes 0. Key
deficit is still one versus historical; do not top up. East `0x29` sealed
(`open_doorway_mask` 12). Center block is immediately west of leftover; PNG
shows Link east-center of dark `0x29`.

## south29 — BLOCKED 4/4

All four east-leftover trials `--through level6-south29 --tag l6_south29_recompose`
0/1, hop 4000f timeout, end ~209,989f, leftover still play `0x29`, keys=3,
rod=1, deaths/state-load/progression/capacity 0, `status_claim=false`. Do
not retry any of these three, and do not retry the prior-sitting RIGHT+DOWN
clip from leftover `(55,133)`.

| # | policy | leftover / reason | what it killed |
|---|--------|-------------------|----------------|
| prior | RIGHT+DOWN clip, occupancy x=120 @ y=141 | `(184,144)` `south_up_halt` 4000f misses 0 | clip assumed west leftover |
| 1 | LEFT on y=144 to x=120 | `(184,141)` `south_up_halt`; `miss_f2_LEFT_184_142` then f3/f4 at `(184,141)` tile 117 misses 3 | LEFT is the center block (tile 244). Wall-slide 144→141 |
| 2 | DOWN clip to south band y=181 | `(184,165)` `south_peel` 4000f tile 244 misses 0 | east aisle **does** walk 144→165 (~1px/f) then **boxes**. Clip never LEFT |
| 3 | occupancy dest `(120,189)` | `(184,149)` `south_up_halt`; `miss_f3_DOWN_184_147` (2px step), `miss_f6_DOWN_184_149`, `miss_f7_LEFT_184_149` tile 116 misses 3 | occupancy false-misses the same DOWN the clip completed to 165 |
| 4 | DOWN to measured box y=165, then LEFT | `(184,165)` `south_up_halt`; `miss_f19_LEFT_184_165`, tile 244, misses 1 | DOWN works 144→165; y=165 is **not** a west aisle and the first LEFT is solid |

Trial 4 was the parked `SOUTH29_SPEC`; do not rerun it. Report
`l6_south29_recompose.json` ended at frame 209,989 with play `0x29`
`(184,165)`, health `0x66` lo==hi, keys=3, bombs=8, Bow=1, Rod=1, TF=`0x1F`,
deaths/state-load/progression/capacity writes 0, `status_claim=false`.

Next open geometry task: change `clear29` combat cleanup so the natural clear
leftover is west of x=64, then take one screenshot-first south-door trial from
that measured predecessor. Do not assume or rerun the old `(55,133)`
RIGHT+DOWN policy; form a fresh policy from the new PNG/RAM trace. Do not
occupancy-DOWN (red 3), LEFT at y=144 (red 1), LEFT at y=165 (red 4), or clip
DOWN toward y=181 (red 2).

```bash
# After changing only the clear29 cleanup policy to target x<64:
QT_QPA_PLATFORM=offscreen uv run python \
  nes/zelda_i/scripts/run_survival_spine.py \
  --through level6-clear29 --no-video --trials 1 \
  --tag l6_clear29_west_recompose
```

## Key deficit (still open; do not top-up)

Ended L6 compass keys=4 vs historical 5. Gap already at L4 `0x40`. clear58
had no natural key. room09 KEY-UP spent 4→3. south19 KEY-DOWN spent 3→2
(historical 4→3). clear29 floor key 2→3 (historical 3→4). Do not hide it
with a key top-up.

## Remaining L6 after south29 (not this sitting)

Phase 1: south29 (BLOCKED 3/3) → settle39 → clear39 → east39 → settle3a →
clear3a.

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
- Did not retry the `(55,133)` RIGHT+DOWN clip.
- Live-ran the parked y=165 clip exactly once; it failed at the first LEFT.
- Did not rerun after that red.
- Did not push.
