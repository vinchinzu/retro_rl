# Residual — rr-tne2 L6 Survival handoff

**Status:** Recovered power-on tip is L6 compass `0x68`. `--through
level6-compass` 1/1. Next is Zol clear / compass bit (`level6-clear68`).
Bead `rr-tne2` stays open. Do not STATUS-promote.

## Recovered — power-on through L5 TF, L6 west, and compass enter

Did not retouch maze-west or 0x40. L5 TF suffix leftover 0x66 needed a
west-aisle prefight (same occupancy as the first 0x66 clear).

| through | tag | frames | leftover | keys |
|---------|-----|--------|----------|------|
| level5-entry | l5_entry_recompose | 116,491 | play `0x76` (120,205) | 4 |
| level5-clear66 | l5_clear66_recompose | 117,649 | play `0x66` (40,159) | 4→5 |
| level5-east77 | l5_east77_recompose | 125,560 | play `0x77` (136,165) | 6 |
| level5-whistle | l5_whistle_recompose | 145,760 | mode 9 `0x04` (135,141) | 5 |
| level5 | l5_tf_recompose | 159,070 | mode 18 `0x14` (120,149) TF `0x1F` | 4 |
| level6-entry | l6_entry_recompose | 164,891 | play `0x79` (120,205) | 4 |
| level6-east-key | l6_east_key_recompose | — | play `0x7a` | 5 |
| level6-west | l6_west_recompose | 169,088 | play `0x78` (104,149) | 4 |
| level6-compass | l6_compass_recompose | 169,403 | play `0x68` (120,205) | 4 |

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  nes/zelda_i/scripts/run_survival_spine.py \
  --through level6-compass --no-video --trials 1 \
  --tag l6_compass_recompose
```

`l6_compass_recompose` **1/1**, 169,403f, hop `level6_north_0x68` 315f,
play `0x68` `(120,205)`, TF=`0x1F`, keys=4, bombs=8, bow=1, health `0x66`
lo==hi. `mid_run_state_load=false`, deaths 0, progression/capacity writes 0.
`status_claim=false`. PNG: Link on the south mouth of compass room; Zols
live; north door visible. Glance: room `0x68`, mode 5, south-door band,
TF bits, owned keys/bombs, hearts lo==hi.

Policy: west-pocket DOWN to y=189, RIGHT to historical x=144, occupancy UP
(21 miss-blocks on the x=144 statue column: `miss_f61_UP_144_187` …
`miss_f79_UP_144_162`, `arrived_68`). Historical green was occupancy from
`(144,141)` 221f / 8 miss-blocks; this peel starts from west leftover
`(104,149)` and does not retry x=120 UP.

`l6_west_recompose` **1/1**, 169,088f, play `0x78` `(104,149)`, TF=`0x1F`,
keys=4. `l5_tf_recompose` **1/1**, 159,070f, TF `0x0F→0x1F`.

## Dated — compass 0x78 UP (`level6_north_0x68`)

West-clear leftover this prefix is `(104,149)` (west statue pocket), not
historical `(144,141)`. v1–v4 failed; v5 greens.

| trial | leftover | wrong belief |
|-------|----------|----------------|
| 1 | `(104,149)` stand 4000f | occupancy UP from west-clear leftover (first miss UP) |
| 2 | `(104,158)` stand 4000f | y≤157 is south of the west statue |
| 3 | `(104,173)` stand 4000f, 56 misses | CLIP_CLEAR_Y is south of the SW statue (RIGHT still boxed) |
| 4 | `(120,149)` north_path 4000f, 12 misses | x=120 then occupancy UP threads the north door |
| 5 | play `0x68` `(120,205)` **1/1** | RIGHT to x=144 then occupancy UP |

Do not retouch the 0x78 peel. Do not retry x=120 UP.

## Next sitting — `--through level6-clear68`

Recompose the L6 body from this leftover (plan step 2). First checkbox is
Zol clear + `ADDR_COMPASS|0x20` in `0x68`. One `--through level6-clear68`
`--no-video --trials 1`. Stop at the first red. Do not retouch 0x40,
maze-west, L5, or the compass peel. Do not skip to Gohma.

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  nes/zelda_i/scripts/run_survival_spine.py \
  --through level6-clear68 --no-video --trials 1 \
  --tag l6_clear68_recompose
```

## Key deficit (still open; do not top-up)

Ended L6 compass keys=4 vs historical 5. Gap already at L4 `0x40`
(`keys_before` 4 vs 5). Do not hide it with a key top-up.

## L6 defects already audited (Gohma / TF `0x20`)

Even after L6-west greens, current `--through level6-gohma` cannot finish L6:

- Dedicated composition skips stairs `0x3A`, cellar `0x08`, south `0x1D`,
  and west `0x2D`; `_gohma_stages()` starts NORTH2C from the wrong predecessor.
- There is no `level6` / TF `0x20` endpoint. Gohma success only proves the
  body is gone in `0x1C`; add natural heart pickup, north into `0x0C`, and TF.
- `level4/boss_combat.py` has an `em.set_state()` fallback while the spine
  hardcodes `mid_run_state_load=false`. Fail closed and measure state loads.

Required chain:

`clear3A → stairs-position assist → cellar08 → south1D → west2D → north2C →`
`wooden-arrow assist/Gohma → natural heart → north 0x0C → natural TF 0x20`.

## Non-claims

- Did not STATUS-promote or overwrite Clean M5.
- Did not close `rr-tne2` or reach Gohma / TF `0x20`.
- Did not poke doors, TF, bow, Rod, Map, Whistle, or capacity.
- Did not retouch maze-west or 0x40.
- Did not push. This sitting’s L6 files are the peel + residual.
