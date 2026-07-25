# Bot Architecture Plan

Goal: finish the game through scriptable, inspectable, RAM-verified tasks. The
bot should treat ROM/decomp data and verified RAM fields as the foundation, with
recordings used as temporary bridges only where the task model is not strong yet.

## Target Layers

1. **RAM catalog**
   - `harvest/core/ram_catalog.py` is the source of truth for named RAM fields.
   - Every field carries address, kind, section, source, aliases, stable-retro
     data key when available, and display scaling when RAM stores a compressed
     value such as money.
   - Live hot edits go through `LiveRamEditor`; save-state edits go through the
     same specs with raw storage writes.

2. **ROM/map model**
   - `harvest/runtime/rom_tools.py` owns ROM extraction and map scene data.
   - `harvest/core/tile_catalog.py` is now the source of truth for tile IDs, walkability,
     debris, crop/grass/water classification, and live/save metatile reads.
   - `harvest/maps/map_config.py` owns map exits, named landmarks, and named routes. Farm,
     path, town, shop, coop, and provisional mountain landmarks are registered
     there instead of being spread through task modules.
   - Keep runtime viewport caveats explicit: live BFS still needs RAM tile
     observations because the SNES updates visible tiles as the viewport moves.

3. **World model**
   - `harvest/core/world_snapshot.py` exposes a `WorldSnapshot` facade over `WorldState.ram`.
   - It exports date/weather, scalar RAM fields, player pixel/tile/tilemap,
     tile histograms, interesting map objects, crop plots/stages, animals,
     relationship fields, decoded status flags, dynamic game-object/NPC
     candidates, dialogue registers, and nearest registered landmarks.
   - `harvest/core/npc_catalog.py` decodes the WRAM game-object table (`0x019C` slots), ROM
     text pointer table, decoded `UnlinkedText.txt` dialogue groups, romance
     heart tiers, marriage bits, and known relationship/event flag banks.
   - Task code should stop reading raw offsets directly except in low-level
     pathfinding/scanning modules.

4. **Verified task primitives**
   - Each autonomous task should declare:
     - preconditions: named RAM fields required before starting
     - observations: fields/tile regions to watch while acting
     - success criteria: named RAM deltas or exact values
     - fallback/retry budget
   - Use `RamExpectation` for exact field checks and add delta expectations for
     actions such as shipping, feeding, brushing, milking, gifting, buying, and
     sleeping.

5. **Planner**
   - `harvest/planner/day_plan.py` should assemble phases from `WorldSnapshot` facts, not from
     scattered helper functions.
   - Multi-day planning should be policy driven: chores, crop season, weather,
     festivals, relationships, livestock state, and money goals.

6. **Tooling**
   - Recorder stays, but every new recording should ship with RAM watches and a
     postcondition file.
   - Editor should use the RAM catalog for all scalar fields and animal slots.
   - Add state profiles for fast setup: date/weather, money, chickens, cows,
     relationships, flags, and inventory.
   - Dialogue/NPC inspection should use `harvest_bot.py npc` and
     `harvest_bot.py dialogue` before adding gift/talk routes.

## Migration Order

1. Centralize all field constants in `harvest/core/ram_catalog.py`.
2. Convert `harvest/tasks/coop_task.py`, `harvest/tasks/harvest_task.py`, `harvest/planner/day_plan.py`, and `harvest/runtime/harvest_bot.py`
   to use catalog reads and named fields.
3. Convert task preconditions to `WorldSnapshot`, then verify loops.
4. Finish replacing duplicated tile logic with `harvest/core/tile_catalog.py`, then backfill
   map registry entries from ROM/decomp where recordings are only provisional.
5. Promote candidate NPC game objects to named NPC/schedule entries by replaying
   town/interior recordings and matching sprite IDs to dialogue handlers.
6. Add a generic `VerifiedActionTask` base for small interactions:
   feed, collect egg, incubate, ship, brush, milk, talk, gift, buy, sleep.
7. Build barn chores using the coop pattern but with cow records and milk/brush
   RAM verification.
8. Expand the planner for summer/fall rotations and relationship routes.

## Current Hot Edit Entry Points

List available named fields:

```bash
uv run python -m harvest.runtime.harvest_bot ram-fields
```

Hot edit live RAM every frame while autoplay runs:

```bash
uv run python -m harvest.runtime.harvest_bot play --state latest --autoplay \
  --ram-set day=28 \
  --ram-set weather=rain \
  --ram-set money=7000 \
  --ram-set num_cows=2 \
  --ram-set num_chickens=12 \
  --ram-set eve_hearts=999 \
  --ram-set incubator_flags=0x2000
```

Use `FIELD:raw=VALUE` when the raw stored value matters, for example:

```bash
uv run python -m harvest.runtime.harvest_bot play --state latest --autoplay --ram-set money:raw=700
```

Export the current known world from a save state:

```bash
uv run python -m harvest.runtime.harvest_bot world --state latest --compact
uv run python -m harvest.runtime.harvest_bot world --state latest --bounds 0,0,63,63 --grid --out debug_alignment/world_latest.json
```

Export dynamic game objects/NPC candidates and decoded flags:

```bash
uv run python -m harvest.runtime.harvest_bot npc --state TMP_Town_From_GoToShop --compact
uv run python -m harvest.runtime.harvest_bot dialogue --npc maria --compact
uv run python -m harvest.core.npc_catalog flags --state latest
```
