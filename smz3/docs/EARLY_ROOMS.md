# SMZ3 early rooms and portals

## Start

Combo always boots into **Super Metroid**. After file select on test seed 1337:

| Field | Value |
|-------|--------|
| Room | Landing Site `0x91F8` |
| Area | Crateria (0) |
| Health | 99 |
| Ship settle XY | ~(1152, 1088) |

## First natural segment (M2)

```
Landing Site (0x91F8)
  → bottom-left blue door (no items)
Parlor and Alcatraz (0x92FD)
  entry XY ~(1240, 139) top-right
```

Reproduce:

```bash
SDL_VIDEODRIVER=dummy uv run python smz3/scripts/probe_early_rooms.py --save-png
```

Implementation: `smz3/early_route.py` (room timeout 3× provisional baselines).

### Landing Site notes

- Early controllable frames may still ignore input (pose 0 on ship).
- Door to Parlor is **bottom-left** (graph block `[0, 71]`), not right.
- Open blue door with shot (`LEFT`+`X`), then hold `LEFT` through transition.

## Path to first Zelda portal

Fixed portal (not randomized): **Crateria Map Room ↔ Lake Hylia Fortune Teller**.

```
Landing Site (0x91F8)
  → Parlor (0x92FD)
  → bottom-right RED door (needs missiles)  [31, 55]
Pre-Map Flyway (0x98E2)
  → right blue door
Crateria Map Room (0x9994)
  → door ptr $8976  (combo sm_teleport_table)
ALttP cave $0122  Fortune Teller (light world)
```

| Side | ID |
|------|-----|
| SM door pointer | `$8976` |
| Z3 cave / exit | `$0122` |
| Return SM door | `$8BCE` → Parlor |

Catalog: `smz3/portals.py`. Other portals: Norfair map ↔ Old Man cave,
Maridia missile refill ↔ DW ice rod cave, LN refill ↔ Mire fairy.

### Portal status

- Natural red-door navigation from Parlor top entry is not scripted yet
  (tall multi-shaft room).
- Dev place near red door can trip WRAM into Z3 with `room_id=$0122` and
  module `$0F`, but **does not settle** to controllable Link — proper door
  transition / SPC handoff still required.
- World detect treats cave `$0122` + non-SM game_state as `ALTTP`.

## Room timeout

| Room key | Provisional standard frames |
|----------|----------------------------|
| `0x91F8` | 90s |
| `0x92FD` | 60s |
| `0x98E2` | 30s |
| `0x9994` | 20s |

Game over at **3×** standard (`smz3.room_timeout`).

## Reuse

While SM is active, call `super_metroid.ram.parse_state` on combo `get_ram()`.
While Z3 is controllable, use `alttp.ram` (pending clean portal settle).
