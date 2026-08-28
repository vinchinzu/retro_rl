# Residual — rr-tne2 L6 Survival (Gohma wired; bow-splice leftover L3 0x5b)

**Status:** `--through level6-gohma` is wired (dedicated hop, wooden-arrow
poke). Live bow-on-spine re-run **cleared `bomb_north_1e`** (`l6_gohma_bow_v11`
skip-to-stand FACE/PLACE) then **failed** at `level3_boss_tf` (`clear_5b_return`
6000f). Leftover play `0x5b` `(47,93)`. Bow=1 on HUD. Do not STATUS. Bead
`rr-tne2` stays open until TF `0x20`.

## What is green

| Stage | Result |
|-------|--------|
| `--through level1-bow-pickup` | **1/1** `ADDR_BOW=1` |
| `--through level2-entry` | **1/1** `l1_bow_splice_l2_entry_v14` bow=1 arrows=0 |
| Gohma hop wiring | dedicated `--through level6-gohma`; `unarmed_no_bow` if bow=0 |
| Arrow poke | `poke_wooden_arrows` `$0659=1` B=2; no `ADDR_BOW`; contract 2026-08-28 |
| Bow-splice through L2 boom | `l6_gohma_bow_v5` 1/1 through `clear2e`; boom+compass on HUD |
| **`enter_1e`** | **1/1** `l6_gohma_bow_v8` 242f, 33 occupancy misses, play `0x1e` |
| **`clear1e`** | **1/1** last_live=0 max_live=5 |
| **`bomb_north_1e`** | **1/1** `l6_gohma_bow_v11` 323f; skip_to_stand; bomb 16→15; entered `0x0e` |
| **`fight_dodongo`** | **1/1** 1815f `dodongo_dead` (no poke) |
| **`collect_tf`** | **1/1** 432f L2 TF; tape `tf=3` |

Old `l6_north2c_continuous` Gohma enter is **1/1** on the **pre-bow** tape
(bow=0). That pin cannot fight. Do not poke bow.

`Level2Enter1eController` stays: occupancy RIGHT out of the `(96,141)`
gutter, LEFT+UP clip when BFS is not RIGHT, then occupancy to `(120,93)`.
`BombWallController` SOUTH_BAND now FACE/PLACE when `_at_stand` (stand_tol
can hold while approach_tol still hunts the last waypoint). Isolated specs
unchanged.

## Bow-splice L2 (closed this session)

| tag | failed | leftover | wrong belief |
|-----|--------|----------|--------------|
| v5–v7 | `enter_1e` **BLOCKED** | `0x2e` `(96,141)` | cardinal LEFT/DOWN/UP in the diamond gutter |
| v8 | `bomb_north_1e` **1/3** | `0x1e` `(120,117)` | `enter_1e` **green**; waypoint `(120,93)` is the closed bomb wall |
| v9 | `bomb_north_1e` **2/3** | `0x1e` `(96,101)` | west peel `(96,117)→(96,101)` **worked**; cardinal RIGHT to stand |
| v10 | `bomb_north_1e` **3/3 BLOCKED** | `0x1e` `(120,93)` | RIGHT+UP clip overshoots stand Y to the door alcove; SOUTH_BAND hunts `(120,101)` at approach_tol=4 and holds DOWN |
| **v11** | `bomb_north_1e` **green** | — | skip-to-stand FACE/PLACE when `_at_stand`; bomb 16→15; 323f to `0x0e` |

v11 notes: `approach_1`, `approach_2`, `skip_to_stand`, `placed_bomb`,
`entered_0x0e`. Did not v11 on RIGHT+UP/DOWN. Did not DOWN from `(120,93)`.

## Leftover — `level3_boss_tf` `clear_5b_return`

`l6_gohma_bow_v11` 0/1, 77278f, `failed_stage=level3_boss_tf`. PNG
`l6_gohma_bow_v11_final.png`. RAM leftover play `0x5b` `(47,93)` mode 5
keys=3 bombs=7 bow=1 arrows=0 tf=3 raft=1. 1 Darknut live (type `0x0b` hp=64
at `(110,93)`). `manhandla_confirmed=false`, `reached_4d=false`.
deaths/progression/capacity 0. New checkbox (1/1), not 3-red BLOCKED.

Path log: `passage_exit` `69_up` `59_bomb_right` `5a_right` **ok**;
`inspect_5b_return` live_darknuts=0 (three type `0x0b` hp=0); then
`clear_5b_return` 6000f `ok=false`. Comment in `level3_boss_path`: return-visit
sprites can precede reliable HP bytes, so clear is unconditional.

Next: diagnose `clear_5b_return` at `0x5b` `(47,93)`. Do not retouch
`bomb_north_1e`. Do not poke bow.

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-gohma --no-video --trials 1 --tag l6_gohma_bow_v12
```

Only after a `clear_5b_return` policy. Then Manhandla / L3 TF / the same
`--through level6-gohma` tape can apply the arrow poke and fight.

## Non-claims

- Did not buy arrows. Did not collect L1 `0x72`. Did not poke bow.
- Did not overwrite Clean M5. Wooden arrows at Gohma `0x1C` are a disclosed
  Survival exception; not applied yet (never reached 0x1C on the bow tape).
- Did not close `rr-tne2`, start L7, or push.
- Did not STATUS-promote.
