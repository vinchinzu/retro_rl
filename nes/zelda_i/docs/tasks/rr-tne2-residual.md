# Residual — rr-tne2 L6 Survival handoff

**Status:** Recovered power-on tip is L6 `0x19` east mouth after the Gleeok
suffix. `--through level6-room19` 1/1. Next is `level6-clear19`. Bead
`rr-tne2` stays open. Do not STATUS-promote.

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
| level6-clear68 | l6_clear68_recompose | 172,546 | cleared play `0x68` (120,133) | 4 |
| level6-keese | l6_keese_recompose | 172,742 | play `0x58` (120,205) | 4 |
| level6-clear58 | l6_clear58_recompose | 173,388 | cleared play `0x58` (77,109) | 4 |
| level6-room48 | l6_room48_recompose | 173,598 | play `0x48` (120,205) | 4 |
| level6-room38 | l6_room38_recompose | 173,859 | play `0x38` (120,189) | 4 |
| level6-clear38 | l6_clear38_recompose | 177,444 | cleared play `0x38` (102,125) | 4 |
| level6-room28 | l6_room28_recompose | 178,288 | play `0x28` (120,189) | 4 |
| level6-clear28 | l6_clear28_recompose | 179,762 | cleared play `0x28` (120,181) | 4 |
| level6-room18 | l6_room18_recompose | 180,039 | play `0x18` (120,189) | 4 |
| level6-settle18 | l6_settle18_recompose | 180,551 | settled play `0x18` (120,189) | 4 |
| level6-gleeok18 | l6_gleeok18_recompose | 183,766 | body-gone play `0x18` (121,133) | 4 |
| level6-postgleeok18 | l6_postgleeok18_recompose | 183,958 | settled play `0x18` (121,133) | 4 |
| level6-room19 | l6_room19_recompose | 184,238 | play `0x19` (16,141) | 4 |

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

## Recovered — `--through level6-clear68`

`l6_clear68_recompose` **1/1**, 172,546f. The five Zol/gels in `0x68` cleared
and the natural Compass reward collected; final state is play L6 `0x68`
`(120,133)`, TF=`0x1F`, keys=4, bombs=8, Bow=1, health=`0x66` with full
hearts. The `level6_clear_0x68` controller completed in 3,143f with seven
maximum live enemies and zero remaining. Final PNG agrees with the cleared
room and north doorway. Deaths=0, post-reset state loads=0,
progression/capacity writes=0, and `status_claim=false`.

## Recovered — `--through level6-keese` through `level6-room19`

Eleven fresh power-on Survival boundaries passed after clear `0x68`, all with
deaths 0, post-reset state loads 0, progression/capacity writes 0, full
`0x66` health, TF=`0x1F`, bombs=8, Bow=1, and `status_claim=false`. Final
PNG/report evidence is under `recordings/l6_*_recompose.{json,final.png}`.

- `level6-keese` reached `0x58` in 196f; eight Keese were live.
- `level6-clear58` cleared all eight in 646f, but did **not** collect a key:
  actual keys remain 4 (not the historical 5). Do not top up or assume a
  drop.
- `room48`, `room38`, `room28`, and `room18` all entered naturally. The
  `0x38` seven-enemy clear took 3,585f; the `0x28` two-Wizzrobe clear took
  1,474f.
- The `0x18` census observed live `0x44` Gleeok plus `0x56`; Gleeok body-gone
  passed in 3,215f, and post-Gleeok waited out the head/fireball residual.
- `level6-room19` entered naturally from the east door in 280f at `(16,141)`.

## Next sitting — `--through level6-clear19`

Recompose the next L6 body boundary from the natural `0x19` entry: clear two
Zols and two Like-Likes. Run exactly one fresh power-on `level6-clear19`
trial with no video; stop at the first red. Do not retouch `0x40`, maze-west,
L5, or the Compass peel. Do not skip to Gohma.

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  nes/zelda_i/scripts/run_survival_spine.py \
  --through level6-clear19 --no-video --trials 1 \
  --tag l6_clear19_recompose
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
