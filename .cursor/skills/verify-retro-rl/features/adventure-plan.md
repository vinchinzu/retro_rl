# Plan an adventure path

The adventure planner finds a capability-aware path over a route graph without booting an emulator.

## Sub-features

- `plan-found` returns `FOUND` with edge ids `collect` then `open` on the morph-door fixture.
- `plan-compatible` makes `inventory_aware_path` return the same edge sequence as `plan`.
- `plan-unreachable` (optional extra) returns `UNREACHABLE` when the only edge requires `bombs` the agent does not have.

## How to get to it (user POV)

- Call the public library: `from retro_harness.adventure import GraphEdge, PlanRequest, plan, inventory_aware_path`.
- Run the skill helper that exercises that API: `uv run python .cursor/skills/verify-retro-rl/scripts/plan_demo.py`.
- Run the packaged tests: `uv run pytest retro_harness/adventure/tests/test_planner.py -q` (supporting check, not a substitute for the helper JSON).

## Driving it with verify-retro-rl

Preconditions:

- Doctor reports `ok=true` and `harness_ok=true`.
- No ROM is required.

- **Morph-door plan.** Run `.cursor/skills/verify-retro-rl/scripts/cli --out $RUN/adventure-plan -- uv run python .cursor/skills/verify-retro-rl/scripts/plan_demo.py`. `exit.txt` is `0`. `stdout.txt` is JSON with `"status": "FOUND"`, `"path_edge_ids": ["collect", "open"]`, `"total_cost": 2.0`, and `"inventory_aware_path_ids": ["collect", "open"]`.
- **Proof.** Keep the JSON. The helper calls `plan` and `inventory_aware_path` on `retro_harness.adventure` (public names). A passing pytest of an unrelated module is not this feature.

## Gotchas

- `inventory_aware_path` is the compatibility adapter; `plan` is the bounded explainable planner. Assert both ids match on this fixture.
- Default expansion budget is 500. Long synthetic chains can return `BUDGET_EXHAUSTED` instead of `FOUND`.
- This is not a Zelda or Super Metroid route clear. Game graphs live under `nes/zelda_i` and `snes/super_metroid` and need their own skills once a ROM path is in play.
- `plan_demo.py` is verification scaffolding. Proof is the JSON it prints from the public API, not the existence of the script.
