# Residual — rr-tne2 L6 Survival (L3 dest_6b-clear 1/1; L4 0x40 key BLOCKED)

**Status:** dest_6b first-visit 0x5b clear is on the spine. `south_darknuts_0x69`
from that predecessor is **1/1**. `--through level3` **1/1** OW `0x74` TF `0x07`.
`--through level6-gohma` / `level4-room40-key` then **BLOCKED 3/3** at
`level4_key_0x40`. Do not STATUS. Bead `rr-tne2` stays open until TF `0x20`.
Did not retouch `bomb_north_1e`. Did not restore `clear_5b_return`.

## What is green

| Stage | Result |
|-------|--------|
| `--through level1-bow-pickup` | **1/1** `ADDR_BOW=1` |
| **`bomb_north_1e`** | **1/1** 323f; entered `0x0e` |
| **`north_chain` 0x5b clear** | **1/1** 3898f; spawn → occupancy → `darknuts_cleared` |
| **`south_darknuts_0x69`** | **1/1** `l3_dest_clear5b_v3` 2760f; `entered_0x69` |
| **`--through level3`** | **1/1** `l3_dest_clear5b_v3` 73229f; OW `0x74` `(128,130)` mode 4; `tf=7`; bow=1 arrows=0 keys=3 bombs=0; deaths/progression/capacity 0 |

Gohma hop wiring and wooden-arrow poke stay; not applied (never reached 0x1C).

## Policy that greened L3

`down_to_69` drops to the lower aisle from **either** diamond side:

- east: `x >= 176` (unchanged)
- west: `x <= 64` — dest_6b-clear leftover `(48,141)` where RIGHT no-op'd

v2 notes were `cleared_59_doors=5_alldead=9` then `down_69_timeout`.
`doors=5` is RIGHT|DOWN — kill-open already fired. v3 same clear, then
`entered_0x69`. Do not restore `fight_clear` on 0x5b return.

## BLOCKED — `level4_key_0x40` (3/3)

dest_6b-clear predecessor reaches play `0x40` and clears Zols. Key is on the
floor in the center opening; Link cannot collect. PNG
`l4_key40_pocket_v3_final.png`.

| Trial | Leftover | Notes |
|-------|----------|-------|
| `l6_dest_clear5b_v1` | `(120,149)` keys=4 | `path_done` + `key_hunt_timeout` (scripted hunt) |
| `l4_key40_pocket_v1` | `(120,149)` 25000f | greedy UP to `(120,141)` no-op |
| `l4_key40_pocket_v2` | `(128,149)` 8 misses | occupancy RIGHT; dest unreachable |
| `l4_key40_pocket_v3` | `(128,149)` 8 misses | unblock dest + fallback `(120,117)`; same leftover |

Play `0x40` `(128,149)` mode 5 keys=4 bombs=15 bow=1 arrows=0 tf=7. Key sprite
sits ~north of the south pocket; UP is solid. Occupancy wrapper lives in
`level4_key40.py` (did not grow 1128-line `level4_maze_path`). Halt.

Do not rerun the same hunt. Next knob is a **maze occupancy seed** or the
already-verified v7 coordinate thread `(160,177)→(116,181)→(112,124)→(128,103)`
then UP/LEFT from leftover `(128,149)`. Do not poke bow. Wooden arrows only at
Gohma `0x1C`.

```bash
uv run python nes/zelda_i/scripts/run_survival_spine.py --through level4-room40-key --no-video --trials 1 --tag l4_key40_pocket_v4
```

Only after a **new** 0x40 path policy (not another dest-tile tweak).

## Non-claims

- Did not buy arrows. Did not collect L1 `0x72`. Did not poke bow.
- Did not overwrite Clean M5. Wooden arrows at Gohma `0x1C` not applied.
- Did not close `rr-tne2`, start L7, or push.
- Did not STATUS-promote.
- Did not restore `clear_5b_return`. Did not retouch `bomb_north_1e`.
