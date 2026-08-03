# Bot Architecture Plan

Goal: finish the game through scriptable, inspectable, RAM-verified tasks. The
bot should treat ROM/decomp data and verified RAM fields as the foundation, with
recordings used as temporary bridges only where the task model is not strong yet.

**Authoritative structure / API direction:** [PLANNING_STACK.md](PLANNING_STACK.md).  
**Verified facts / gates:** [STATUS.md](STATUS.md).  
**Future work queue:** [plan.md](plan.md).

This file tracks **layer ownership** and **migration status**. Prefer
PLANNING_STACK for skill composition, contracts, and advisor policy.

## Target Layers

### 1. RAM catalog — **done / maintain**

- `harvest/core/ram_catalog.py` is the source of truth for named RAM fields.
- Every field carries address, kind, section, source, aliases, stable-retro
  data key when available, and display scaling when RAM stores a compressed
  value such as money.
- Live hot edits go through `LiveRamEditor`; save-state edits go through the
  same specs with raw storage writes.
- D1 town event mask: `d1_town_event_mask` (`0x11F74` live mirror).

### 2. ROM / map model — **done / maintain**

- `harvest/runtime/rom_tools.py` owns ROM extraction and map scene data.
- `harvest/core/tile_catalog.py` is the source of truth for tile IDs, walkability,
  debris, crop/grass/water classification, and live/save metatile reads.
- `harvest/maps/map_config.py` owns map exits, named landmarks, and named routes
  (farm, path, town, shop, coop, mountain spa corridors, D1 town stands).
- Keep runtime viewport caveats explicit: live BFS still needs RAM tile
  observations because the SNES updates visible tiles as the viewport moves.
- `densify_waypoints(max_hop_tiles=7)` densifies long same-map hops for viewport BFS.

### 3. World model — **done / maintain**

- `harvest/core/world_snapshot.py` exposes a `WorldSnapshot` facade over RAM.
- `harvest/core/world_context.py` caches hot fields per frame for builders/skills.
- `harvest/core/npc_catalog.py` decodes the WRAM game-object table, dialogue
  groups, romance tiers, marriage bits, and event flag banks.
- `harvest/core/scene.py` classifies normal maps, dialogue/menu/locks,
  transitions, sleep/wake, endings, unknown tilemaps, invalid coordinates.
- Task code should stop reading raw offsets directly except in low-level
  pathfinding/scanning modules.

### 4. Verified task primitives + skills — **partial**

| Piece | Status |
|-------|--------|
| `tasks/primitives.py` (sequence, press-and-verify, RAM wait, retry) | Done |
| `tasks/skills.py` (`NavSkill`, interact, coop/farm/talk factories) | Boundary done |
| `TaskContract` + `evaluate_task_contract` | Done; production crop/coop/sleep wired |
| Domain mono FSMs (coop ~1.3k, cow ~2.4k, crop_planter ~2.3k, D1 handoff ~0.9k) | Still production path |
| Full skill extraction + multi-adult coop | Open |

Each autonomous task should declare (via phase contract and/or task docs):

- preconditions: named RAM fields / maps / tools
- observations: fields/tile regions to watch while acting
- success criteria: named RAM deltas or exact values
- fallback/retry budget
- documented `failure_modes`

### 5. Planner — **done / extend**

```text
day_plan.py                 # compatibility barrel
day_plan_orchestrator.py    # multi-day + phase schedule execution
day_plan_phases.py          # dynamic phase lists (build_day_phases)
day_phase_catalog.py        # static PhaseSpec + contracts
day_phase_registry.py       # PhaseKind → TaskBuildContext builders
day_plan_decision.py        # serializable plan facts + deferrals
local_llm.py                # advisory; gated optional apply
crop_planner.py             # seasonal crop policy
```

- Multi-day owns return/sleep (`include_end_day=False` on day task).
- Failure policy: `required` / `optional` / `opportunistic`.
- Advisor: notes+defer default; `HARVEST_PLAN_LLM_APPLY=1` for schema-validated
  optional reorder/append only.

### 6. Tooling — **done / maintain**

- Recorder + pinned task JSON under `tasks/`.
- Editor uses RAM catalog; state profiles for fast setup.
- Probes: `boot_probe`, `run_to_day2`, `town_day1_recon`, spa validators,
  `harvest_bot world|npc|dialogue|ram-fields`.
- Power-on bootstrap: `PowerOnStartTask` → Spring D1 07:00 town gate (Clean).

## Migration checklist (historical → current)

| Step | Status |
|------|--------|
| 1. Centralize field constants in `ram_catalog` | Done |
| 2. Catalog reads in major tasks / day plan / bot | Mostly done; residual raw offsets in scanners |
| 3. Task preconditions via WorldSnapshot / WorldContext | Partial |
| 4. Tile logic → `tile_catalog` + map registry | Done for new work; residual in old mono tasks |
| 5. Named NPC/schedule from sprite IDs | Partial (catalog exists; gift/talk routes open) |
| 6. `VerifiedActionTask` / skill factories for feed/ship/talk | Boundary factories done; not yet production |
| 7. Barn chores with milk/brush verification | Partial (feed loop stronger than milk/brush) |
| 8. Seasonal rotations + relationship routes | Spring path strong; summer/fall/social open |
| 9. Phase contracts on production catalog | **Done 2026-08-01** for crop/coop/sleep/hot-spring |
| 10. Skill composition of coop/cow/harvest mono files | Open (next architecture tax) |

## Architecture next (do in order)

1. **Domain close-loop (M3→M4 money):** empty-can refill → water → multi-day
   growth → harvest → ship (assert money after **5pm**).
2. **Coop skill extraction:** use factories in `skills.py`; fix multi-adult /
   dynamic egg tiles; restore to daily plan with replay.
3. **Contract preflight in probes:** **Done 2026-08-03** —
   `preflight_phase_contract` / `tool_tags_from_ram` in
   `day_phase_types.py`; day-plan probe emits `contract_preflight` events and
   planned-phase summary on start; `day_plan_debug_snapshot` attaches soft
   fail notes when map/tool mismatch.
4. **D1 pure routes:** replace rest-recording dependency for power-on→D2; keep
   recording as regression oracle.
5. **Recording distillation:** `spring_festival.json`, `fix_rainy_day.json`.
6. **Promote** Pathfinder / planning primitives only after a second consumer.

## Hot edit / inspect entry points

```bash
uv run python -m harvest.runtime.harvest_bot ram-fields
uv run python -m harvest.runtime.harvest_bot world --state latest --compact
uv run python -m harvest.runtime.harvest_bot npc --state TMP_Town_From_GoToShop --compact
uv run python -m harvest.runtime.harvest_bot dialogue --npc maria --compact
uv run python -m harvest.core.npc_catalog flags --state latest

# Hot edit live RAM every frame while autoplay runs:
uv run python -m harvest.runtime.harvest_bot play --state latest --autoplay \
  --ram-set day=28 --ram-set weather=rain --ram-set money=7000
```

Use `FIELD:raw=VALUE` when the raw stored value matters (e.g. `money:raw=700`).
