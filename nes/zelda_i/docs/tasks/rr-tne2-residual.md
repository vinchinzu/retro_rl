# Residual — rr-tne2 L6 Survival (Gohma wired; bow-splice BLOCKED at L2 bomb_north_1e)

**Status:** `--through level6-gohma` is wired (dedicated hop, wooden-arrow
poke). Live bow-on-spine re-run **cleared `enter_1e`** then **BLOCKED** at
`bomb_north_1e` after 3 serial reds on that checkbox
(`l6_gohma_bow_v{8,9,10}`). Leftover play `0x1e` `(120,93)` in the closed
north-door alcove. Bow=1 on HUD. Do not STATUS. Bead `rr-tne2` stays open
until TF `0x20`.

## What is green

| Stage | Result |
|-------|--------|
| `--through level1-bow-pickup` | **1/1** `ADDR_BOW=1` |
| `--through level2-entry` | **1/1** `l1_bow_splice_l2_entry_v14` bow=1 arrows=0 |
| Gohma hop wiring | dedicated `--through level6-gohma`; `unarmed_no_bow` if bow=0 |
| Arrow poke | `poke_wooden_arrows` `$0659=1` B=2; no `ADDR_BOW`; contract 2026-08-28 |
| Bow-splice through L2 boom | `l6_gohma_bow_v5` 1/1 through `clear2e`; boom+compass on HUD |
| **`enter_1e`** | **1/1** `l6_gohma_bow_v8` 242f, 33 occupancy misses, play `0x1e` |
| **`clear1e`** | **1/1** last_live=0 max_live=5 (same tape) |

Old `l6_north2c_continuous` Gohma enter is **1/1** on the **pre-bow** tape
(bow=0). That pin cannot fight. Do not poke bow.

`Level2Enter1eController` stays: occupancy RIGHT out of the `(96,141)`
gutter, LEFT+UP clip when BFS is not RIGHT, then occupancy to `(120,93)`.
`enter_2e` is still `Level2SouthBandUpController`. Isolated specs unchanged.

## Bow-splice L2 (this session)

| tag | failed | leftover | wrong belief |
|-----|--------|----------|--------------|
| v5–v7 | `enter_1e` **BLOCKED** | `0x2e` `(96,141)` | cardinal LEFT/DOWN/UP in the diamond gutter |
| v8 | `bomb_north_1e` **1/3** | `0x1e` `(120,117)` | `enter_1e` **green**; waypoint `(120,93)` is the closed bomb wall |
| v9 | `bomb_north_1e` **2/3** | `0x1e` `(96,101)` | west peel `(96,117)→(96,101)` **worked**; cardinal RIGHT to stand |
| v10 | `bomb_north_1e` **3/3 BLOCKED** | `0x1e` `(120,93)` | RIGHT+UP clip overshoots stand Y to the door alcove |

## BLOCKED — `bomb_north_1e` door alcove

PNG `l6_gohma_bow_v10_final.png`: Link in the closed north doorway at
`(120,93)`. Goriya still on the north band. bombs=16 keys=3 bow=1 arrows=0.
`bombs_before_place` is null — never FACEd. Notes: `approach_1`,
`approach_2` (west peel hit). 12000f timeout.

Stand is `(120,101)` with `stand_tol=12`. Leftover dy=8 **is already
inside stand_tol**. SOUTH_BAND still hunts waypoint `(120,101)` at
`approach_tol=4`, so it holds DOWN from the alcove. DOWN is solid.

Next offline (do not v11 on this checkbox): when `_at_stand`, skip the
remaining approach waypoint and FACE/PLACE. Do not DOWN from `(120,93)`.
Do not extend 12000f.

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level6-gohma --no-video --trials 1 --tag l6_gohma_bow_v11
```

Only after that skip-to-stand. Then Dodongo / L2 TF / the same
`--through level6-gohma` tape can apply the arrow poke and fight.

## Non-claims

- Did not buy arrows. Did not collect L1 `0x72`. Did not poke bow.
- Did not overwrite Clean M5. Wooden arrows at Gohma `0x1C` are a disclosed
  Survival exception; not applied yet (never reached 0x1C on the bow tape).
- Did not close `rr-tne2`, start L7, or push.
- Did not STATUS-promote.
- Halted after 3 serial reds on `bomb_north_1e`.
