# Residual — rr-tne2 L6 Survival handoff

**Status:** L5 entry recovered from leftover L4 TF (`--through level5-entry`
1/1, play `0x76` `(120,205)`, TF `0x0F`). Bead `rr-tne2` stays open. Do not
STATUS-promote or claim Level 5 / Level 6 complete.

## Recovered — L5 entry from L4 leftover

`--through level5-entry` from mode 18 room `0x03` `(120,149)`. Existing
`PostL4TriforceSettleController` + `POST_L4_TO_LEVEL5_HOPS` (not old At4A).
Did not retouch maze-west or 0x40. Did not edit L5 overworld policy.

```bash
QT_QPA_PLATFORM=offscreen uv run python \
  nes/zelda_i/scripts/run_survival_spine.py \
  --through level5-entry --no-video --trials 1 \
  --tag l5_entry_recompose
```

`l5_entry_recompose` **1/1**, 116,491f, play `0x76` `(120,205)`, TF=`0x0F`,
keys=4, bombs=13, bow=1, map=`0x0A`, health `0x55` lo==hi. settle_l4_tf
283f (`post_l4_ow_ready` island `0x45`). enter_level5 5,282f: hops
`0x55…0x1B`, pocket DOWN then already-free, hills_ups=3 then door `0x0B`,
entered L5. `mid_run_state_load=false`, deaths 0, progression/capacity
writes 0. Glance empty. PNG: `recordings/l5_entry_recompose_final.png`
(L5 south mouth, 3 Zols live). `status_claim=false`.

Historical `l5_entry_continuous_v1` was 134,393f keys=5 hop 5,138f. This
tape is the recovered L4 prefix (110,926f keys=4) plus 5,565f settle+path.

## Key deficit (still open; do not top-up)

L4-entry keys 3 vs historical 4; L4 TF and L5 entry ended keys=4 vs
historical 5. Deficit is already at L4 `0x40` (`keys_before` 4 vs 5).
Bow splice KEY-LEFT spends the 0x23 key then `SPINE_L1_KEY_RETOPUP` writes
0→1 at `backtrack44`; L2 still pokes keys 0→2, so that L1 spend is not
the live L4 gap. Do not hide it with another key top-up.

## Next knob — L5 0x66 from this leftover

`--through level5-clear66` from play `0x76` `(120,205)`. Historical green
is occupancy `l5_clear66_continuous_v2` leftover `(32,101)` keys 5→6.
This leftover has keys=4. Do not retouch maze-west, 0x40, or L5 entry.

## L6 defects already audited

Even after L4/L5-entry greens, current `--through level6-gohma` cannot
finish L6:

- Dedicated composition skips stairs `0x3A`, cellar `0x08`, south `0x1D`,
  and west `0x2D`; `_gohma_stages()` starts NORTH2C from the wrong predecessor.
- There is no `level6` / TF `0x20` endpoint. Gohma success only proves the
  body is gone in `0x1C`; add natural heart pickup, north into `0x0C`, and TF.
- `level4/boss_combat.py` has an `em.set_state()` fallback while the spine
  hardcodes `mid_run_state_load=false`. Fail closed and measure state loads.
- Arrow assist telemetry is not merged into the whole-run inventory audit.
  Add an L6 endpoint validator.

Required chain:

`clear3A → stairs-position assist → cellar08 → south1D → west2D → north2C →`
`wooden-arrow assist/Gohma → natural heart → north 0x0C → natural TF 0x20`.

## Non-claims

- Did not STATUS-promote or overwrite Clean M5.
- Did not close `rr-tne2` or reach Gohma / TF `0x20`.
- Did not poke doors, TF, bow, Rod, Map, Whistle, or capacity.
- Did not retouch maze-west or 0x40.
- Did not push or commit.
