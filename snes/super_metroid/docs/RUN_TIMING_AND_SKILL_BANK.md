# Run timing + skill bank (room PBs + hop skills)

Record **start → credits** in segments with **full button tapes**, keep
**room-by-room** times, fold them into **item → item** and **boss start →
finish**, then optimize **one hop at a time** from its live pin. Multi-room
replay is **hop-compose** (pin → body → leave pin), not multi-minute power-on
open-loop for pin recovery (that is how we lose pins).

**Hop** = one settled room visit (enter pin → leave); bank key includes
direction + items. **Open-loop** = fixed button body from a pin with no
mid-body branching; safe only hop-from-pin. Full glossary:
[tasks/HUMAN_TAPE_PIPELINE.md](tasks/HUMAN_TAPE_PIPELINE.md). Product epic
library→bank→reactive script: `rr-nzrg`.

Canonical room names: **Map Rando / sm-json-data**
([maprando.com/logic](https://maprando.com/logic)), vendored at
`refs/sm-json-data/`. On-disk index:

| Artifact | Role |
|----------|------|
| `maps/maprando_room_catalog.json` | 261 rooms: name, maprandoId, roomId hex, area |
| `maps/maprando_room_names.json` | Fast lookups by room hex / id / name |
| `maps/maprando_tech_catalog.json` | ~242 Map Rando techs + difficulty + bot builders |
| `rooms/canonical_names.py` | Room name loader + rebuild |
| `rooms/tech_catalog.py` | Tech tree loader + builder coverage |
| `scripts/export/maprando_catalog.py` | Room catalog CLI rebuild |
| `scripts/export/maprando_tech_catalog.py` | Tech catalog CLI rebuild |
| `docs/TECH_TREE.md` | Tech → bot builder matrix (Basic/Medium first) |

```bash
uv run python snes/super_metroid/scripts/export/maprando_catalog.py --summary
```

**IDs:** `maprandoId` = `/logic/room/<id>` (e.g. Landing Site = 8).
`roomId` / `0x91F8` = SNES pointer low 16 bits = RAM `room_id`.

---

## Hierarchy (one clock)

**Any% KPDR RTA zero** = first Ceres Elevator ordinary control (not title/menu).
The kennycason tracker default auto-start is title Start (gs 2→31) and
**includes** the intro cinematic (~40s); we do not, so HUD/PB is gameplay RTA.
`human_tape.rta_clock.resolve_rta_clock` folds archived power-on→seam segments
so `./play morph` / `./play bomb` HUD shows full-run time (Morph ~3–4 min on a
casual take; Bombs follows). Stitched PB tables rezero the same way
(`./play --pb` → persistent ``*_pb_board.json`` with PB/avg/sd/Δ across all
segment samples; product RTA uses seam-deduped chain). The same command also
prints a **KPDR Any%** table in [kennycason auto-tracker](https://github.com/kennycason/super_metroid_auto_tracker)
layout: BEST (product line) / Best +/- / TIME (hop-PB gold). **Ceres Station**
is ASL `ceresEscape` (leave ordinary in Ceres Elevator, gs 8→32), not first
Landing Site settle. Later KPDR rows are still room-entry proxies.

```
leaf:     room visit / hop          dwell_frames, room_frames
  ↑ fold
item:     item_delta A → B          sum of leaves with entry in [A,B)
boss:     boss_start → boss_finish  fight span (or boss-room leaf)
segment:  named anchors             e.g. "varia" → "speed"
full:     Ceres control → credits   verified continuous OR theoretical PB sum
```

Implementation:

- Leaves (human): `build_room_hops` → **`settle_room_hops`** (RoomTimer-aligned
  ordinary entry; transition leading edge stored as `transition_frames`)
- Leaves (continuous live): `room_timer.RoomTimer` (same settle rule)
- Fold-up: `run_splits.build_run_timing` + `events_from_task_payload`
  (trace item deltas + anchors + boss-room proxies)
- Bank: `skill_bank.records_from_hops_and_anchors` → `SkillBank`
- **One entrypoint:** `materialize.materialize_take(task_path)`

```bash
# After any guided_human take (also runs by default on F5 unless --no-materialize)
uv run python snes/super_metroid/scripts/tools/extract_human_tape.py \
  snes/super_metroid/tasks/my_run.json --materialize --summary

# Optional: merge dual_green=False candidates into recordings/skill_bank/bank.json
uv run python snes/super_metroid/scripts/tools/extract_human_tape.py \
  snes/super_metroid/tasks/my_run.json --materialize --bank
```

```python
from super_metroid.materialize import materialize_take

result = materialize_take("snes/super_metroid/tasks/my_run.json", write=True)
# result.hops_settled, result.run_timing, result.bank_records
# sidecars: <task>_extract.json, <task>_run_timing.json
```

---

## Why room name alone is not a PB key

Same room is traversed many ways:

| Visit | Direction / goal | Inventory |
|-------|------------------|-----------|
| Red Tower | up to Hellway | pre-Varia |
| Red Tower | down to Warehouse | post-Varia |
| Landing Site | ship ending | G4 clear |

**hop_key:**

```text
{room}:{from}->{to|goal}:{items}
# e.g. 0xA253:0xA322->0xA6A1:0x1005
```

`skill_bank.make_hop_key(...)`.

---

## Skill bank + combining runs

```text
Run A ──┐
Run B ──┼─► SkillBank ── best(hop_key) ── theoretical route PB
Run C ──┘         │
                  └─ compose_plan → ordered anchors + bodies
```

- **Ingest** many human / continuous / pure takes: `merge_runs_into_bank`
- **PB** = min frames among dual-green records for that hop_key
- **Frankenstein PB** = sum of hop PBs along a route — **theoretical** until
  natural-entry compose is dual-green (label it; do not STATUS-promote)

---

## Hill-climb / GA (per hop)

| Do | Don't |
|----|--------|
| Boot **one** hop from its live `entry_anchor` | Mutate multi-room tapes as one open-loop |
| Score exit predicate only | Score full-run IGT while mutating mid-route |
| Write improved body + keep pin | Overwrite end pins by multi-minute pin recovery |
| Re-pin next hop if leave kinematics change | Assume frame-append across seams is sound |
| Compose multi-hop via pin→body chain | Treat timing stitch as button replay |

`HopOptimizeJob` (`skill_bank.py`) is the optimizer boundary:

```text
entry_anchor + seed_body + exit_predicate + max_frames
→ hill-climb / GA / pure rewrite
→ dual-green hop-replay
→ bank.add(HopSkillRecord(..., dual_green=True))
```

Reuse human_tape pipeline:

```text
./play / guided_human (anchors ON, materialize ON, --bank)
  → archive prior segment if same --name
  → materialize_take (settled hops + <stem>_hops/ bodies + run_timing + bank)
  → hop-replay dual green
  → compose_human_hops for multi-room seams
  → trim (safe contiguous for open-loop seeds)
  → mid_lockstep if RED (not full-tape invent mid)
  → bank.add(..., dual_green=True) / export skill → autopilot
```

Details: [tasks/HUMAN_TAPE_PIPELINE.md](tasks/HUMAN_TAPE_PIPELINE.md).

Platformer optimizers (`retro_harness.platformer.hillclimb`, `genetic`) are the
pattern for **body search**; wrap them in **single-hop** `HopOptimizeJob` after
dual-green hop records exist. Multi-room **compose** is separate
(`human_tape.compose`).

---

## Segmented start→end recording recipe

1. **Segments** with **full button tapes** (human or continuous). Prefer seams
   at items/bosses/doors; one 2-hour take is ok but cancel without F5 loses the
   body for that segment.
2. Each session: `./play` / `guided_human` with anchors ON (default). Same
   `--name` **archives** the previous tape under `tasks/<name>_segments/sN/`.
3. After each segment: materialize (hop bodies + timing) → bank ingest →
   hop-replay dual green on new hops.
4. **Stitch points** = live pins at segment boundaries (item collect, boss dead,
   area door). Timing stitch (`*_stitched.json`) is the RTA clock; **replay**
   is hop-compose from those pins + bodies, not button concat across seams.
5. When re-recording a slow room: only that hop_key; re-verify next hop entry.

Suggested segment cuts (KPDR-ish):

| Segment | From | To |
|---------|------|-----|
| Ceres | power-on | Zebes landing |
| Morph–Bombs | Landing | Bombs + Alcatraz out |
| Early Brinstar | … | Supers / Charge |
| Kraid | Warehouse | Varia out |
| Speed–Ice | Business / Frog | Ice / Wave |
| Moat–Phantoon | Red elev | Phantoon dead |
| Maridia | Tube | Draygon / Space Jump |
| Norfair low | LN elev | Ridley / Screw |
| Tourian | G4 | Credits |

Exact cut list can live as `segment` events in the timing report.

---

## Modules

| Module | Role |
|--------|------|
| `rooms/canonical_names.py` | Map Rando names |
| `room_timer.py` | Live leaf timing (continuous) |
| `run_splits.py` | Fold room → item / boss / segment |
| `skill_bank.py` | Hop PB bank (`recordings/skill_bank/`), stitch plan |
| `materialize.py` | Post-take spine: settle + bodies + timing + bank |
| `human_tape/` | Anchors, hop extract, compose, archive, dual-green |

Artifacts per take (next to task JSON):

| File | Role |
|------|------|
| `tasks/<name>.json` | frames + trace (full button tape) |
| `tasks/<name>_anchors.json` | live pins index |
| `tasks/<name>_hops/` | per-hop SNES-12 bodies (hill-climb / bank seeds) |
| `tasks/<name>_segments/sN/` | immutable archived prior takes (same name) |
| `tasks/<name>_extract.json` | settled hop board |
| `tasks/<name>_run_timing.json` | room / item / boss folds |
| `tasks/<name>_stitched.json` | multi-session RTA clock (timing only) |
| `recordings/skill_bank/bank.json` | aggregate hop PB bank (`--bank` / `./play`) |

---

## Acceptance for “room PB” claims

1. Named with Map Rando canonical name + hop_key
2. Entry anchor path + fingerprint stored
3. Dual-green hop-replay (or continuous integrity for full-run only)
4. Folded item/boss times derived from the same leaf timeline (no second clock)
5. Theoretical multi-hop PB clearly labeled until compose green
