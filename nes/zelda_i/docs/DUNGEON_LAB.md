# Zelda I dungeon laboratory

The dungeon lab turns room discovery into a repeatable checkpoint-isolated
experiment. Room behavior stays in `zelda_i/dungeon.py` as data until another
game proves a shared API.

## Quick start

```bash
uv run python zelda_i/scripts/dungeon_lab.py \
  --state Level1Cleared53 \
  --door east \
  --expected-room 0x54 \
  --enemy-type 0x1b \
  --alive-by type \
  --reward auto \
  --attack-phases 0,2,4,6 \
  --engage-distances 48,56 \
  --trials 2 \
  --jobs 4 \
  --save-state Level1Cleared54
```

This runs 16 isolated trials (eight configurations × two repeats) in separate
processes. Tune from a naturally produced predecessor checkpoint, then run the
winning controller through the full natural chain:

```bash
uv run python zelda_i/scripts/run_level1_clear54.py --trials 2
uv run python zelda_i/scripts/run_level1_clear54.py \
  --natural-entry --trials 2 --save-state
```

## Outputs

Each lab directory contains:

- `summary.json`: ranked policies, phase RAM deltas, reward analysis, exit
  probes, trace comparison, and artifact paths
- `report.md`: generated human-readable handoff
- `room_spec_suggestion.json`: measured fields for promoting the next room
  specification
- `trial_NNN.trace.jsonl`: every action, policy reason, phase, compact RAM
  snapshot, and live object
- `trial_NNN.failure_tail.jsonl`: last 120 frames for failed trials
- `trial_NNN.final.png`: final evidence frame
- `exit_<direction>.png`: physical exit-probe evidence

Promoted states receive a sibling `.provenance.json` containing SHA-256 hashes,
the source checkpoint, the request, and the selected trial. Natural captures
record that no source state was loaded.

## Trace comparison

```bash
uv run python zelda_i/scripts/dungeon_lab.py --diff-traces \
  left.trace.jsonl right.trace.jsonl
```

The output is the first frame whose phase, reason, action, or compact state
differs. This is the primary tool for deterministic timing failures.

## Room specification rules

`DungeonRoomSpec` records:

- predecessor room and door waypoints
- enemy type IDs and expected count
- liveness rule (`hp` or type-only)
- chase distance, attack cadence, and patrol geometry
- clear-only or fixed-inventory reward contract
- known exit routes

Stalfos (`0x2A`) require positive HP. Keese (`0x1B`) keep HP at zero while
alive, so they use type-only liveness. Unknown IDs remain explicitly named
`unknown_*` in `dungeon_ids.py`; do not promote guesses into verified labels.

## Acceptance boundary

Parallel lab trials are isolated development evidence. A room becomes
route-ready only after the winning specification clears from the real
predecessor in a no-state-load natural-entry run.

## Door graph (pathfinding primitive)

Shared dungeon door-graph template: `zelda_i/door_graph.py` encodes per-room
exits with gate kinds (`OPEN` / `KILL_CLEAR` / `KEY` / `BOMB` / `SEALED`) and
inventory-aware BFS. Seed `LEVEL_2_DOOR_GRAPH` matches the verified L2 interior
in `LEVEL2_ROUTE.md` (rooms `0x7d`–`0x5e`). Pure tests:
`zelda_i/tests/test_door_graph.py`. Feeds future `RouteGraph` edges; not a
STATUS promote.
