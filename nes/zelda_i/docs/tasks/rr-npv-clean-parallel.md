> Historical lane note. The live sitting is the gathering prefix in [PRE_L1.md](../PRE_L1.md). This file stays because the clean-tip ladder or a route doc still cites it. It is not the current plan.

# Parallel Clean campaign — Zelda I (`rr-ps7` → `rr-npv`)

Implementation plan; no STATUS claim. Survival power-on → credits is green
(`recordings/level9_credits.json`, 354346f). Clean STATUS is still M5 L1 only.

Full sitting plan lives in the session `plan.md`. This file is the lane
contract for parallel agents.

## Spine lock

One spine bead: `rr-ps7.3` (L2 `0x4C` east mouth). Living residual:
[`rr-8t4.4-residual.md`](rr-8t4.4-residual.md).

Fixture-live Clean beads are `zelda_i` + `clean`, **not** `spine`.

## Evidence ladder

1. Hypothesis
2. Fixture-live (`route_eligible=false`)
3. Natural-segment
4. Spine-green (power-on, `set_state=0`, `--no-infinite-life --no-pokes`)

Parallel agents stop at fixture-live. Integrator promotes.

## Reactive hop contract

Gohma (`level6/gohma.py`) is the gold standard: leftover pose and entry
frame may change; dest is RAM. Occupancy miss → block cell → replan; no
path → stand. No `idle(n)` as a hop. Leave is `screen_glance` bands.

## Beads

| ID | Lane | Exclusive writes |
|----|------|------------------|
| `rr-ps7.3` | spine (claimed) | L2 OW walk (`overworld/graph.py` hops, `overworld/path.py`) |
| `rr-npv.7` | Wave 0 shared | `dungeon/hop_controller.py`, `dungeon/door_hop.py`, `spine/survival.py`, CLI |
| `rr-npv.1` | L3 | `level3/**` |
| `rr-npv.2` | L5 | `level5/**` |
| `rr-npv.3` | L7 | `level7/**` |
| `rr-npv.4` | L8 | `level8/**` |
| `rr-npv.5` | L9 | `level9/**`, `door_graph/level9_exits.py` |
| `rr-npv.6` | OW | `overworld/{rupee_farm,heart_farm,stitch}.py` (not the L2 walk while `rr-ps7.3` is live) |
| `rr-4oz` / `rr-bxzj` / `rr-d6v` | L2 / L4 / L6 Clean | existing children |

Integrator-only: `spine/survival.py`, `assist.py`, `ram.py`, `STATUS.md`,
`.beads/issues.jsonl`. Never `git add -A`. Do not overwrite
`recordings/BASELINE_20260907.json`.
