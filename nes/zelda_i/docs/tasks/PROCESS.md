# Zelda I process (SM-lessons applied)

Work tracker: **`bd ready -l zelda_i`**. Product board: `docs/STATUS.md` +
`docs/plan.md`. Do not invent a second ticket system.

## Dual track

| Track | When | STATUS promote? |
|-------|------|-----------------|
| **Assisted first pass** | `--infinite-life` (Survival assist) | No Clean claim; graph + room IDs only |
| **Clean** | default runners, no RAM write | Yes, after natural-entry 2/2 |

Heart death on 0x5C blocked Clean door-path progress. **First-pass strategy:**

1. Infinite life so agents **finish the game** (path + puzzles + items).
2. Track damage (`total_damage`, `damage_by_location`) without stopping the run.
3. Only then tune: Clean heart-farm / combat harden for hottest rooms.

**Priority within a bead:** route geometry and puzzle mechanics over combat
polish. A room that opens the next door with sloppy sword work is green for
first-pass; residual combat harden is a later child bead.

Assist contract: [`docs/ASSIST_CONTRACT.md`](../ASSIST_CONTRACT.md).

## Pure-first wave (per dungeon)

Borrowed from Super Metroid `docs/tasks/PROCESS.md`:

1. **Recon** — probe room IDs, object types, doors (lab / short scripts).
2. **Isolated pure** — controller from a checkpoint state; 2/2 green.
3. **Natural-entry** — from real predecessor (no mid-run state load).
4. **Compose** — named route + `RouteGraph` edge promotion.
5. **Stabilize** — re-verify pure + natural before stacking more knobs.

One-knob: one geometry constant group or one combat policy per bead when a
room stays PARTIAL.

## Bead grain (~100 to credits)

Do **not** open 100 leaf beads up front. Expand as the tip advances:

| Layer | Count (approx) | Examples |
|-------|----------------|----------|
| Epics | ~12 | Full clear; L2–L9; OW prep; Death Mountain; graph |
| Active tip tasks | 5–15 | Current dungeon rooms / OW hops |
| Discovered residuals | as needed | hard room splits, door bits |
| Closed history | grows | do not archive as living cards |

Target total closed+open by credits: **~80–120 beads**. Ready queue stays
small (`bd ready -l zelda_i`).

### Per-dungeon template (spawn children when L_n is tip)

```text
Z-Ln-OW        overworld door path (assisted → Clean)
Z-Ln-ENTRY     entry room ready predicate
Z-Ln-KEY*      key rooms (one bead each when known)
Z-Ln-ITEM      dungeon item (raft, ladder, …)
Z-Ln-BOSS      boss policy
Z-Ln-TF        triforce bit 0x.. natural-entry
Z-Ln-GRAPH     RouteGraph + NamedRoute milestones
```

## Adventure harness integration

Keep game-local: RAM, controllers, stop predicates, combat.

Use shared `retro_harness.adventure` for:

- `RouteGraph` / `GraphEdge` / capability BFS (`overworld.py`, future dungeon graph)
- `NamedRoute` / `RouteMilestone` (`routes.py`)
- `RouteLeg` plans (`route_legs.py`)
- `Waypoint` / `WaypointFollower` (`nav_common` / overworld)
- `sha256_file` provenance (dungeon lab checkpoints)

Promote richer shared APIs only when a **second** adventure consumer needs them
(Metroid / ALTTP already share the graph core).

## Hot modules (serialize edits)

- `level2_overworld.py`, `overworld.py` hop tables
- `chain.py` natural-entry compose
- `dungeon.py` / room specs
- `docs/STATUS.md` (planner-only promotes)
- `assist.py` / ASSIST_CONTRACT (contract changes)

## Force-pass ban

Same honesty rules as SM: no gate self-weakening, no fixture-as-natural,
no STATUS promote from assisted-only evidence, no scaffold-as-green.

## Session loop

```bash
bd ready -l zelda_i
bd update <id> --status in_progress
# implement one bead …
uv run pytest zelda_i/tests retro_harness/adventure/tests -q
bd close <id> --reason "…"
bd sync   # commit issues.jsonl with code
```

## Next tip (live)

**`rr-0fx` Z4.1** — assisted L4 live entry from `Level3Complete` (Raft owned).
See `bd ready -l zelda_i` and `QUEUE.md` architecture block.
