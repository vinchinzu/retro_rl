# Room engine — low-context room work

SM-style **data + generic player** for dungeon rooms. Agents should **not** load
full segment source for every B1 door.

## Authority

| Layer | Owns |
|-------|------|
| `alttp/maps/room_XX.json` | **Geometry authority:** points, doors, clear policy |
| `alttp.room_map` | Load/save map schema (`load_room_map`) |
| `alttp.room_sense` | Sprite AABBs, edge detect, overlay (re-exports map load) |
| `alttp.opening_route.room_engine` | clear + combat-aware path + door push |
| `opening_route/*_to_*.py` | Thin segment glue + multi-hop acceptance only |
| `escape_graph` | Capability hops + verification + `map_id`/`door_label` |

**Do not** redeclare coords in Python (segments, anchors, graph meta). Edit the
JSON map; re-run. Approach/trigger anchors derive door xy from the map.

**Isolated edge success:** `run_room_edge` → `ok=True` when door dest is
reached. `castle_dungeon.DungeonRoomEdge` composes measured first-dungeon
doors (`0x61→0x60→0x50`); higher-level aggregates such as
`main_hall_to_zelda` apply their own Zelda-follower acceptance.

## Agent recipe (small context)

```bash
# 1. What rooms exist / doors / approaches (~1 screen of text)
uv run python alttp/scripts/room_engine.py list
uv run python alttp/scripts/room_engine.py show room_61
uv run python alttp/scripts/room_engine.py show room_61 --json

# 2. Isolated play from save-state
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/room_engine.py run room_61 \
  --edge west_to_0x60 --state CastleMain --overlay

# 3. Segment still works (wraps room_engine)
SDL_VIDEODRIVER=dummy uv run python alttp/scripts/main_hall_to_zelda.py --overlay
```

Read: this doc + `show` output + `docs/TRIGGER_HANDOFF.md` open section.
Skip: entire `main_hall_to_zelda.py` history, probe PNG dumps, unrelated docs.

## Maps present (Sanctuary-path seed)

| Map | Room | Notes |
|-----|------|--------|
| `room_55` | 0x55 | Secret entrance (continuous stairs clear already scripted) |
| `room_61` | 0x61 | Main hall; west continuous prefix |
| `room_60` | 0x60 | Main west; north→0x50 continuous prefix |
| `room_50` | 0x50 | NW chamber; east→0x01 measured |
| `room_01` | 0x01 | North connector |
| `room_51` | 0x51 | Throne / mantle approach |
| `room_52` | 0x52 | NE chamber |
| `room_62` | 0x62 | Main east |
| `room_71`–`room_72`, `room_80`–`room_82` | B1 / Zelda | Geometry seeds; doors partial |

`z3Label` on maps is optional randomizer logic text (US/JP vanilla same room
ids for these chambers). Geometry authority is still the measured JSON.

## Add a B1 room (copy this checklist)

1. Measure from a save-state; write `alttp/maps/room_XX.json` (points + doors + path).
2. `show` validates load; unit-test load if non-trivial.
3. `run room_XX --edge <label> --state <State>` until isolated green.
4. Graph: add node/edge when isolated (`verification=isolated`) with
   `map_id` + `door_label` only — **no** copied approach/landing xy.
5. If a real predecessor wedges on an otherwise measured door, record its
   alternate measured points as that door's map-only `recoveryPath`; do not
   add room-specific coordinates to Python.
6. Optional thin segment only if continuous spine needs multi-room acceptance.
7. STATUS fact + TRIGGER_HANDOFF row — no Zelda claim without `$F3CC==1`.

## Map schema (minimal)

```json
{
  "schemaVersion": 1,
  "roomBaseId": 97,
  "name": "…",
  "sourceState": "CastleMain",
  "points": [{"label": "west_door_approach", "xy": [520, 3320], "role": "approach"}],
  "doors": [{
    "label": "west_to_0x60",
    "direction": "LEFT",
    "toRoom": 96,
    "approachXy": [520, 3320],
    "landingXy": [511, 3320],
    "role": "zelda_path",
    "path": ["south_mid", "hall_corridor", "west_mid", "west_door_approach"],
    "recoveryPath": ["natural_clear_left", "natural_clear_north"],
    "pathTolerances": {"south_mid": 16, "default": 12}
  }],
  "clearPolicy": {"maxDistance": 180, "skirmishMaxDistance": 90}
}
```

## Randomizer note

JSON geometry is **vanilla measured**. For seeds, keep sense + door labels;
swap or regenerate maps per layout. Do not hardcode coords in Python segments.
